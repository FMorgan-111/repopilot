"""Deterministic integration coverage for success-first model escalation."""

from types import SimpleNamespace

import pytest

from src import graph, model_policy, new_agent, run_store
from src.evidence import EvidenceStore
from src.nodes import plan as plan_node
from src.nodes import reflect as reflect_node
from src.reasoning_loop import (
    NEW_EVIDENCE_SECTION,
    prompt_with_new_evidence,
    response_tool_intent,
    validate_reasoning_response,
)
from src.state import (
    Evidence,
    FileInfo,
    PatchEdit,
    RepairPlan,
    VerifiedEdit,
    VerifiedEditBatch,
)
from src.tool_router import ToolRouteResult


def _plan_response() -> dict:
    return {
        "kind": "plan",
        "plan": "Return the new sentinel.",
        "patch": "",
        "patch_edits": [
            {
                "file": "src/widget.py",
                "search": "return 'old-sentinel'",
                "replace": "return 'new-sentinel'",
            }
        ],
        "files": ["src/widget.py"],
        "test_command": "pytest tests/test_widget.py -q",
        "decision_frame": {
            "stage": "plan",
            "summary": "Update the sentinel.",
            "recommended_action": "execute",
            "risk": "low",
            "confidence": 0.9,
        },
    }


def _state(**updates):
    values = {
        "issue_url": "https://github.com/acme/widget/issues/7",
        "issue_title": "Return the sentinel value",
        "issue_body": "widget must return new-sentinel",
        "current_phase": new_agent.Phase.PLAN,
        "repo_ref": "a" * 40,
        "relevant_files": [
            FileInfo(
                path="src/widget.py",
                content="def widget():\n    return 'old-sentinel'\n",
            )
        ],
        "skip_commit": True,
    }
    values.update(updates)
    return new_agent.AgentState(**values)


def test_escalated_reflect_prompt_declares_three_exclusive_variants():
    prompt = reflect_node.ESCALATED_REFLECT_SYSTEM

    assert "kind='tool'" in prompt
    assert "kind='reflect'" in prompt
    assert "kind='stop'" in prompt
    assert "optional tool_intent" not in prompt
    assert "must not mix" in prompt.lower()


def _enable_escalation(monkeypatch):
    monkeypatch.setattr(model_policy, "escalation_is_configured", lambda: True)
    monkeypatch.setattr(
        model_policy,
        "get_model_config",
        lambda provider: SimpleNamespace(
            model=(
                "claude-opus-4-8:stable"
                if provider == "escalation"
                else "gemini-3.5-flash:stable"
            )
        ),
    )


def _repair_result():
    return (
        RepairPlan(
            root_cause="The function returns the stale sentinel.",
            target_files=["src/widget.py"],
            target_symbols=["widget"],
            required_behavior="Return the new sentinel.",
            regression_test_strategy="Run the sentinel unit test.",
        ),
        VerifiedEditBatch(
            edits=[
                VerifiedEdit(
                    file_path="src/widget.py",
                    search="return 'old-sentinel'",
                    replace="return 'new-sentinel'",
                    intent="Update the sentinel.",
                )
            ]
        ),
    )


def _accept_repair(state, repair_plan, verified_batch):
    edits = [
        PatchEdit(
            file_path="src/widget.py",
            search="return 'old-sentinel'",
            replace="return 'new-sentinel'",
        )
    ]
    state.patch_edits = edits
    return SimpleNamespace(accepted=True, edits=edits, issues=[])


@pytest.mark.parametrize(
    "invalid_response",
    [
        {},
        {
            "kind": "tool",
            "tool_intent": {"action": "search_text", "args": "invalid"},
        },
    ],
    ids=["empty", "invalid"],
)
async def test_configured_invalid_primary_counts_then_next_full_plan_uses_opus(
    exact_repair_state, monkeypatch, invalid_response
):
    _enable_escalation(monkeypatch)
    calls = []
    responses = [invalid_response, _plan_response()]

    async def fake_llm(system, user, model=None, *, provider="primary", **_kwargs):
        calls.append((system, user, model, provider))
        return responses.pop(0)

    monkeypatch.setattr(plan_node, "llm_call", fake_llm)
    first = await plan_node.plan_fix(exact_repair_state)

    assert first.current_phase == new_agent.Phase.PLAN
    assert first.retry_count == 1
    assert first.primary_failed_repair_rounds == 1
    assert first.escalated is True
    assert first.current_repair_round_id == 0

    second = await plan_node.plan_fix(first)

    assert second.current_phase == new_agent.Phase.EXECUTE
    assert [call[3] for call in calls] == ["primary", "escalation"]
    assert [call[2] for call in calls] == [
        "gemini-3.5-flash:stable",
        "claude-opus-4-8:stable",
    ]
    assert all(call[0] == plan_node.PLAN_SYSTEM for call in calls)
    assert second.repair_round_sequence == 2
    assert second.authorized_repair_provider == "escalation"


async def test_escalated_plan_prompt_bounds_and_redacts_adversarial_context(
    exact_repair_state, monkeypatch
):
    exact_repair_state.active_provider = "escalation"
    exact_repair_state.active_model = "claude-opus-4-8:stable"
    exact_repair_state.escalated = True
    exact_repair_state.escalation_reason = "primary_repair_round_limit"
    exact_repair_state.relevant_files = [
        FileInfo(
            path="src/widget.py",
            content=(
                "distinct-safe-source-sentinel\n"
                "sk-BBBBBBBBBBBBBBBB\n"
                "RePaIrPlAn forbidden-source-sentinel\n" + "x" * 20_000
            ),
        )
    ]
    exact_repair_state.repair_correction_context = (
        "distinct-safe-correction-sentinel\n"
        "VeRiFiEdEdItBaTcH forbidden-correction-sentinel\n" + "c" * 6_000
    )
    calls = []

    async def fake_llm(system, user, model=None, *, provider="primary", **_kwargs):
        calls.append((system, user, model, provider))
        return _plan_response()

    monkeypatch.setattr(plan_node, "llm_call", fake_llm)
    result = await plan_node.plan_fix(exact_repair_state)

    assert result.current_phase == new_agent.Phase.EXECUTE
    [call] = calls
    prompt = call[1]
    assert "distinct-safe-source-sentinel" in prompt
    assert "distinct-safe-correction-sentinel" in prompt
    for forbidden in (
        "sk-bbbbbbbb",
        "repairplan",
        "verifiededitbatch",
        "forbidden-source-sentinel",
        "forbidden-correction-sentinel",
    ):
        assert forbidden not in prompt.casefold()
    assert len(prompt) < 20_000


@pytest.mark.parametrize("denied_term", ["RePaIrPlAn", "VeRiFiEdEdItBaTcH"])
def test_plan_prompt_resanitizes_the_rolling_summary_for_protocol_terms(
    denied_term,
):
    state = _state(
        attempt_outcome_summary=(
            "safe-summary-sentinel\n"
            "sk-AAAAAAAAAAAAAAAA\n"
            f"{denied_term} forbidden-summary-sentinel"
        )
    )

    prompt = plan_node.build_plan_user_prompt(state)
    folded = prompt.casefold()

    assert "safe-summary-sentinel" in prompt
    assert "completed attempts (rolling summary)" in folded
    assert denied_term.casefold() not in folded
    assert "forbidden-summary-sentinel" not in prompt
    assert "sk-aaaaaaaa" not in folded


def test_fresh_evidence_suffix_sanitizes_persisted_adversarial_ids_and_bounds_all():
    base_prompt = "base-plan-prompt-sentinel"
    malicious_id = (
        "ev_legacy-safe-id-sentinel-"
        + "safe-id-padding-" * 32
        + "\nRePaIrPlAn-sk-AAAAAAAAAAAAAAAA"
    )
    persisted = Evidence(
        evidence_id=malicious_id,
        tool="read_symbol",
        summary="persisted-safe-summary-sentinel",
        content="persisted-safe-content-sentinel",
        fingerprint="legacy-fingerprint",
    )
    state = _state(evidence=[persisted])
    complete_suffix_limit = len(EvidenceStore.render_for_prompt([persisted]))

    prompt = prompt_with_new_evidence(
        base_prompt,
        state,
        (malicious_id,),
        denied_literals=plan_node.MODEL_CONTEXT_DENIED_LITERALS,
        max_evidence_chars=complete_suffix_limit,
    )
    suffix = prompt[len(base_prompt) :]
    folded = prompt.casefold()

    assert prompt.startswith(base_prompt)
    assert len(suffix) <= complete_suffix_limit
    assert malicious_id not in prompt
    assert "repairplan" not in folded
    assert "sk-aaaaaaaa" not in folded


async def test_escalated_tool_reprompt_keeps_baseline_source_and_fresh_evidence(
    exact_repair_state, monkeypatch
):
    exact_repair_state.active_provider = "escalation"
    exact_repair_state.active_model = "claude-opus-4-8:stable"
    exact_repair_state.escalated = True
    exact_repair_state.escalation_reason = "primary_repair_round_limit"
    calls = []
    responses = [
        {
            "kind": "tool",
            "tool_intent": {
                "action": "read_symbol",
                "args": {"path": "src/widget.py", "symbol": "widget"},
                "reason": "Confirm the exact body.",
                "expected_evidence": "Current widget source.",
            },
        },
        _plan_response(),
    ]

    async def fake_llm(system, user, model=None, *, provider="primary", **_kwargs):
        calls.append((system, user, model, provider))
        return responses.pop(0)

    async def fake_route(state, intent, **_kwargs):
        added = EvidenceStore(state).add(
            tool=intent.action,
            summary="fresh-safe-evidence-summary",
            content=(
                "fresh-tool-evidence-sentinel\n"
                "sk-BBBBBBBBBBBBBBBB\n"
                "VeRiFiEdEdItBaTcH forbidden-tool-evidence-sentinel\n" + "x" * 20_000
            ),
            file_path="src/widget.py",
            symbol="widget",
        )
        assert added.evidence is not None
        return ToolRouteResult(
            action=intent.action,
            status="ok",
            args_fingerprint="fresh-tool-args",
            evidence_id=added.evidence.evidence_id,
            made_progress=True,
        )

    monkeypatch.setattr(plan_node, "llm_call", fake_llm)
    monkeypatch.setattr(plan_node, "route_tool_intent", fake_route)
    result = await plan_node.plan_fix(exact_repair_state)

    assert result.current_phase == new_agent.Phase.EXECUTE
    assert len(calls) == 2
    assert "def widget():" in calls[1][1]
    assert "fresh-tool-evidence-sentinel" in calls[1][1]
    fresh_section = calls[1][1].split(NEW_EVIDENCE_SECTION, 1)[1]
    assert len(fresh_section) <= 12_500
    for forbidden in (
        "sk-bbbbbbbb",
        "verifiededitbatch",
        "forbidden-tool-evidence-sentinel",
    ):
        assert forbidden not in fresh_section.casefold()
    assert all(call[0] == plan_node.PLAN_SYSTEM for call in calls)
    assert all(call[3] == "escalation" for call in calls)


async def test_escalated_plan_two_no_progress_tools_stay_in_the_open_round(
    exact_repair_state, monkeypatch
):
    exact_repair_state.active_provider = "escalation"
    exact_repair_state.active_model = "claude-opus-4-8:stable"
    exact_repair_state.escalated = True
    exact_repair_state.escalation_reason = "primary_repair_round_limit"
    tool_response = {
        "kind": "tool",
        "tool_intent": {
            "action": "search_text",
            "args": {"text": "missing-two-tool-sentinel"},
            "reason": "Check the same missing symbol.",
            "expected_evidence": "A source location.",
        },
    }
    responses = [tool_response, tool_response, _plan_response()]
    routed = []

    async def fake_llm(system, user, model=None, *, provider="primary", **_kwargs):
        return responses.pop(0)

    async def no_progress_route(state, intent, *, calls_this_round):
        routed.append((intent.action, calls_this_round))
        return ToolRouteResult(
            action=intent.action,
            status="duplicate",
            args_fingerprint=f"duplicate-{calls_this_round}",
            made_progress=False,
        )

    monkeypatch.setattr(plan_node, "llm_call", fake_llm)
    monkeypatch.setattr(plan_node, "route_tool_intent", no_progress_route)

    result = await plan_node.plan_fix(exact_repair_state)

    assert result.current_phase == new_agent.Phase.EXECUTE
    assert routed == [("search_text", 0), ("search_text", 1)]
    assert result.repair_round_sequence == 1
    assert result.current_repair_round_id == 1
    assert result.authorized_repair_round_id == 1
    assert result.retry_count == 0
    assert result.last_counted_repair_round_id == 0
    assert result.primary_failed_repair_rounds == 0
    assert result.opus_no_progress_rounds.get("plan_fix", 0) == 0
    assert not any(
        item.get("event") == "opus_no_progress" for item in result.node_diagnostics
    )


async def test_escalated_plan_no_progress_stream_stops_only_at_shared_tool_cap(
    monkeypatch,
):
    state = _state(
        active_provider="escalation",
        active_model="claude-opus-4-8:stable",
        escalated=True,
        escalation_reason="primary_repair_round_limit",
    )
    routed = []

    async def fake_llm(system, user, model=None, *, provider="primary", **_kwargs):
        return {
            "kind": "tool",
            "tool_intent": {
                "action": "search_text",
                "args": {"text": "missing-eight-tool-sentinel"},
                "reason": "Check the missing symbol.",
                "expected_evidence": "A source location.",
            },
        }

    async def no_progress_route(current, intent, *, calls_this_round):
        routed.append(calls_this_round)
        return ToolRouteResult(
            action=intent.action,
            status="duplicate",
            args_fingerprint=f"duplicate-{calls_this_round}",
            made_progress=False,
        )

    monkeypatch.setattr(plan_node, "llm_call", fake_llm)
    monkeypatch.setattr(plan_node, "route_tool_intent", no_progress_route)

    result = await plan_node.plan_fix(state)

    assert routed == list(range(8))
    assert result.current_phase == new_agent.Phase.FAILURE
    assert result.failure_reason == "tool_round_limit"
    assert result.repair_round_sequence == 1
    assert result.current_repair_round_id == 1
    assert result.retry_count == 0
    assert result.last_counted_repair_round_id == 0
    assert result.primary_failed_repair_rounds == 0
    assert result.opus_no_progress_rounds.get("plan_fix", 0) == 0
    assert not any(
        item.get("event") == "opus_no_progress" for item in result.node_diagnostics
    )


async def test_graph_replays_an_already_escalated_saved_state(tmp_path, monkeypatch):
    state = _state(
        trace_id="already-escalated",
        active_provider="escalation",
        active_model="claude-opus-4-8:stable",
        escalated=True,
        escalation_reason="repeated_plan",
    )
    run_store.save_run(state, root_dir=tmp_path)
    loaded = run_store.load_run(state.trace_id, root_dir=tmp_path)
    observed = []

    async def replayed_plan(current):
        observed.append(
            (current.active_provider, current.active_model, current.escalated)
        )
        current.current_phase = new_agent.Phase.DONE
        return current

    monkeypatch.setattr(new_agent, "StateGraph", None)
    monkeypatch.setattr(new_agent, "plan_fix", replayed_plan)

    result = await graph.run_graph(
        new_agent.build_agent_graph(start_phase=new_agent.Phase.PLAN),
        loaded,
    )

    assert observed == [("escalation", "claude-opus-4-8:stable", True)]
    assert result.current_phase == new_agent.Phase.DONE


def test_discriminated_tool_response_rejects_mixed_legacy_patch_payload():
    with pytest.raises(ValueError, match="mixed tool and outcome"):
        response_tool_intent(
            {
                "kind": "tool",
                "tool_intent": {
                    "action": "search_text",
                    "args": {"text": "sentinel"},
                    "reason": "find source",
                    "expected_evidence": "source location",
                },
                "patch": "evaluator-payload-must-not-cross-the-tool-boundary",
            }
        )


@pytest.mark.parametrize(
    ("response", "outcome_kind"),
    [
        (
            {
                "kind": "plan",
                "tool_intent": {
                    "action": "search_text",
                    "args": {"text": "sentinel"},
                    "reason": "find source",
                    "expected_evidence": "source location",
                },
            },
            "plan",
        ),
        (
            {
                "kind": "stop",
                "stop_reason": "no safe repair",
                "plan": "mixed outcome",
            },
            "plan",
        ),
        (
            {
                "kind": "plan",
                "plan": "nominal plan",
                "root_cause": "reflection field mixed into plan",
            },
            "plan",
        ),
        ({"kind": "unknown", "plan": "ambiguous"}, "plan"),
    ],
)
def test_discriminated_reasoning_response_rejects_ambiguous_variants(
    response,
    outcome_kind,
):
    with pytest.raises(ValueError, match="structured response"):
        validate_reasoning_response(response, outcome_kind=outcome_kind)


@pytest.mark.parametrize(
    "response",
    [
        {"root_cause": "safe", "test_patch": "evaluator-only"},
        {"root_cause": "FAIL_TO_PASS evaluator marker"},
        {"root_cause": "raw HTTP response payload: secret"},
        {
            "root_cause": "safe",
            "tool_intent": {
                "action": "search_text",
                "args": {"text": "sentinel"},
                "reason": "mixed",
                "expected_evidence": "source",
            },
        },
        {"root_cause": "safe", "stop_reason": "mixed"},
    ],
)
def test_legacy_reflect_response_rejects_evaluator_raw_or_mixed_payload(response):
    with pytest.raises(ValueError, match="structured response"):
        validate_reasoning_response(response, outcome_kind="reflect")


@pytest.mark.parametrize(
    "extra",
    [
        {"what_went_wrong": "mixed"},
        {"decision_frame": {"stage": "reflect"}},
        {"target_files": ["src/widget.py"]},
        {"plan": "mixed"},
        {"stop_reason": "mixed"},
    ],
)
def test_untagged_legacy_tool_requires_exactly_one_top_level_key(extra):
    response = {
        "tool_intent": {
            "action": "search_text",
            "args": {"text": "sentinel"},
            "reason": "find source",
            "expected_evidence": "source location",
        },
        **extra,
    }

    with pytest.raises(ValueError, match="mixed tool and outcome"):
        validate_reasoning_response(response, outcome_kind="reflect")


async def test_plan_persists_fixed_model_stop_code_without_secret_reason(monkeypatch):
    async def stopped(*args, **kwargs):
        return {
            "kind": "stop",
            "stop_reason": "Bearer sk-model-stop-secret-sentinel",
        }

    monkeypatch.setattr(plan_node, "llm_call", stopped)
    state = _state(
        active_provider="escalation",
        active_model="claude-opus-4-8:stable",
        escalated=True,
        escalation_reason="repeated_plan",
    )

    result = await plan_node.plan_fix(state)

    assert result.current_phase == new_agent.Phase.PLAN
    assert result.failure_reason == "model_stop"
    assert result.retry_count == 1
    assert result.last_counted_repair_round_id == 1
    assert "model-stop-secret-sentinel" not in result.model_dump_json()


def test_evaluator_only_payload_never_enters_safe_evidence_or_prompt():
    state = _state()
    added = EvidenceStore(state).add(
        tool="read_range",
        summary="safe summary\nFAIL_TO_PASS: evaluator-only-sentinel",
        content=(
            "safe source prefix\n"
            '"test_patch": "evaluator-test-patch-sentinel"\n'
            '"patch": "evaluator-patch-sentinel"\n'
            "PASS_TO_PASS: evaluator-pass-sentinel"
        ),
    )

    rendered = EvidenceStore.render_for_prompt([added.evidence])
    dumped = state.model_dump_json()

    assert added.evidence.summary == "safe summary"
    assert added.evidence.content == "safe source prefix"
    for forbidden in (
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "test_patch",
        "evaluator-test-patch-sentinel",
        "evaluator-patch-sentinel",
        "evaluator-pass-sentinel",
    ):
        assert forbidden not in rendered
        assert forbidden not in dumped


@pytest.mark.parametrize(
    "marker",
    [
        '"gold_patch": "evaluator-gold-sentinel"',
        "gold_patch: evaluator-gold-sentinel",
    ],
)
def test_gold_patch_payload_never_enters_safe_evidence(marker):
    state = _state()
    added = EvidenceStore(state).add(
        tool="read_range",
        summary=f"safe summary\n{marker}",
        content=f"safe source prefix\n{marker}",
    )

    assert added.evidence.summary == "safe summary"
    assert added.evidence.content == "safe source prefix"
    assert "evaluator-gold-sentinel" not in state.model_dump_json()
