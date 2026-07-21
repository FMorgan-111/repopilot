"""Deterministic integration coverage for success-first model escalation."""

from types import SimpleNamespace

import pytest

from src import graph, model_policy, new_agent, run_store
from src.evidence import EvidenceStore
from src.http_client import LLMResponseError
from src.nodes import plan as plan_node
from src.reasoning_loop import response_tool_intent, validate_reasoning_response
from src.state import FileInfo, PatchEdit, RepairPlan, VerifiedEdit, VerifiedEditBatch


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


async def test_empty_gemini_completion_escalates_to_accepted_opus_edit_and_passes(
    monkeypatch,
):
    _enable_escalation(monkeypatch)

    async def empty_primary(*args, **kwargs):
        raise LLMResponseError("empty chat completion after retries")

    async def repair(*args, **kwargs):
        return _repair_result()

    monkeypatch.setattr(plan_node, "llm_call", empty_primary)
    monkeypatch.setattr(plan_node, "generate_opus_repair", repair)
    monkeypatch.setattr(plan_node, "validate_patch_batch", _accept_repair)

    state = await plan_node.plan_fix(_state())

    assert state.escalated is True
    assert state.active_provider == "escalation"
    assert state.current_phase == new_agent.Phase.EXECUTE
    assert state.patch_edits[0].replace == "return 'new-sentinel'"

    state.fix_attempts.append(
        new_agent.FixAttempt(
            patch_edits=state.patch_edits,
            test_result="one passed",
            success=True,
        )
    )
    state.current_phase = new_agent.Phase.VERIFY
    state = await new_agent.verify_fix(state)
    assert state.current_phase == new_agent.Phase.DONE


async def test_reserve_boundary_uses_opus_without_calling_primary(monkeypatch):
    _enable_escalation(monkeypatch)

    async def primary(*args, **kwargs):
        raise AssertionError("primary model must not run past its reserve boundary")

    async def repair(*args, **kwargs):
        return _repair_result()

    monkeypatch.setattr(plan_node, "llm_call", primary)
    monkeypatch.setattr(plan_node, "generate_opus_repair", repair)
    monkeypatch.setattr(plan_node, "validate_patch_batch", _accept_repair)

    state = await plan_node.plan_fix(_state(token_usage=55_000))

    assert state.escalation_reason == "primary_budget_reserve"
    assert state.current_phase == new_agent.Phase.EXECUTE


async def test_empty_gemini_completion_without_escalation_key_keeps_compatible_failure(
    monkeypatch,
):
    monkeypatch.setattr(model_policy, "escalation_is_configured", lambda: False)

    async def empty_primary(*args, **kwargs):
        raise LLMResponseError("empty chat completion after retries")

    monkeypatch.setattr(plan_node, "llm_call", empty_primary)

    state = await plan_node.plan_fix(_state())

    assert state.active_provider == "primary"
    assert state.escalated is False
    assert state.current_phase == new_agent.Phase.FAILURE
    assert state.failure_reason == "Failed to generate fix plan: LLMResponseError"


async def test_two_opus_no_progress_rounds_stop_with_stable_reason(monkeypatch):
    _enable_escalation(monkeypatch)

    async def invalid_repair(*args, **kwargs):
        raise ValueError("sentinel invalid repair")

    monkeypatch.setattr(plan_node, "generate_opus_repair", invalid_repair)
    state = _state(
        active_provider="escalation",
        active_model="claude-opus-4-8:stable",
        escalated=True,
        escalation_reason="repeated_plan",
    )

    state = await plan_node.plan_fix(state)

    assert state.current_phase == new_agent.Phase.FAILURE
    assert state.failure_reason == "opus_no_progress_limit"
    assert state.no_progress_rounds == 2


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
        observed.append((current.active_provider, current.active_model, current.escalated))
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


async def test_exhausted_opus_patch_gate_stops_after_two_rejected_rounds(
    monkeypatch,
):
    _enable_escalation(monkeypatch)

    async def repair(*args, **kwargs):
        return _repair_result()

    issue = SimpleNamespace(code="search_missing")

    def reject(*args, **kwargs):
        return SimpleNamespace(accepted=False, edits=[], issues=[issue])

    async def unchanged(*args, **kwargs):
        return _repair_result()[1]

    monkeypatch.setattr(plan_node, "generate_opus_repair", repair)
    monkeypatch.setattr(plan_node, "validate_patch_batch", reject)
    monkeypatch.setattr(plan_node, "request_verified_edit_correction", unchanged)
    state = _state(
        active_provider="escalation",
        active_model="claude-opus-4-8:stable",
        escalated=True,
        escalation_reason="repeated_plan",
    )

    state = await plan_node.plan_fix(state)
    assert state.current_phase == new_agent.Phase.REFLECT

    state.current_phase = new_agent.Phase.PLAN
    state = await plan_node.plan_fix(state)

    assert state.current_phase == new_agent.Phase.FAILURE
    assert state.failure_reason == "opus_no_progress_limit"
