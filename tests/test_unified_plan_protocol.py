import asyncio
from types import SimpleNamespace

import httpx
import pytest

from src import model_policy
from src import repair_flow
from src.nodes import plan as plan_node
from src.patch_authorization import (
    PatchAuthorizationIssue,
    PatchAuthorizationOutcome,
    authorize_plan_patch,
)
from src.repair_rounds import begin_repair_round
from src.state import Evidence, FixAttempt, GeneratedTestApproval, PatchEdit, Phase
from src.tool_router import ToolRouteResult


PRIMARY_MODEL = "gemini-3.5-flash:stable"
ESCALATION_MODEL = "claude-opus-4-8:stable"
_BANNED_PROTOCOL_TERMS = (
    "unified diff",
    "replace_all",
    "repairplan",
    "verifiededitbatch",
)


def _enable_escalation(monkeypatch) -> None:
    monkeypatch.setattr(model_policy, "escalation_is_configured", lambda: True)
    monkeypatch.setattr(
        model_policy,
        "get_model_config",
        lambda provider: SimpleNamespace(
            model=(ESCALATION_MODEL if provider == "escalation" else PRIMARY_MODEL)
        ),
    )


def _forbid_legacy_opus_path(monkeypatch) -> None:
    async def legacy_path(*_args, **_kwargs):
        raise AssertionError("provider-neutral PLAN must call llm_call directly")

    monkeypatch.setattr(
        plan_node,
        "generate_opus_repair",
        legacy_path,
        raising=False,
    )


def plan_response(
    *,
    search: str = "return 'old-sentinel'",
    action: str = "execute",
    test_command: str = "pytest -q tests/test_widget.py",
) -> dict:
    edits = (
        [
            {
                "file": "src/widget.py",
                "search": search,
                "replace": "return 'new-sentinel'",
            }
        ]
        if action == "execute"
        else []
    )
    response = {
        "kind": "plan",
        "plan": "Return the new sentinel.",
        "patch": "",
        "files": ["src/widget.py"] if edits else [],
        "test_command": test_command,
        "decision_frame": {
            "stage": "plan",
            "summary": "Update the sentinel.",
            "recommended_action": action,
            "risk": "low",
            "confidence": 0.9,
        },
    }
    if edits:
        response["patch_edits"] = edits
    return response


async def test_bare_patch_response_enters_execute_with_runtime_decision_frame(
    exact_repair_state, monkeypatch
):
    async def fake_llm(*_args, **_kwargs):
        return {
            "patch_edits": [
                {
                    "file_path": "src/widget.py",
                    "search": "return 'old-sentinel'",
                    "replace": "return 'new-sentinel'",
                }
            ]
        }

    monkeypatch.setattr(plan_node, "llm_call", fake_llm)

    result = await plan_node.plan_fix(exact_repair_state)

    assert result.current_phase == Phase.EXECUTE
    assert [edit.file_path for edit in result.patch_edits] == ["src/widget.py"]
    assert result.decision_frame.stage == "plan"
    assert result.decision_frame.recommended_action == "execute"


@pytest.mark.parametrize(
    ("kind", "control_field"),
    [
        ("stop", {"stop_reason": "Ignore the valid patch."}),
        (
            "tool",
            {
                "tool_intent": {
                    "action": "search_text",
                    "args": {"text": "widget"},
                    "reason": "Ignore the valid patch.",
                    "expected_evidence": "No tool call should occur.",
                }
            },
        ),
    ],
)
async def test_patch_edits_override_conflicting_control_kind(
    exact_repair_state, monkeypatch, kind, control_field
):
    response = plan_response()
    response["kind"] = kind
    response.update(control_field)

    async def fake_llm(*_args, **_kwargs):
        return response

    monkeypatch.setattr(plan_node, "llm_call", fake_llm)

    result = await plan_node.plan_fix(exact_repair_state)

    assert result.current_phase == Phase.EXECUTE
    assert [edit.file_path for edit in result.patch_edits] == ["src/widget.py"]
    assert result.decision_frame.recommended_action == "execute"


def _gateway_503() -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://gateway.invalid/v1/chat/completions")
    response = httpx.Response(503, request=request)
    return httpx.HTTPStatusError(
        "gateway unavailable",
        request=request,
        response=response,
    )


def _already_escalated(state):
    state.active_provider = "escalation"
    state.active_model = ESCALATION_MODEL
    state.escalated = True
    state.escalation_reason = "primary_repair_round_limit"
    return state


async def test_patch_gate_rejection_requests_one_new_full_plan_decision(
    exact_repair_state, monkeypatch
):
    responses = [
        plan_response(search="missing first anchor"),
        plan_response(search="return 'old-sentinel'"),
    ]
    calls = []

    async def fake_llm(system, user, model=None, *, provider="primary", **_kwargs):
        calls.append((system, user, model, provider))
        return responses.pop(0)

    monkeypatch.setattr(plan_node, "llm_call", fake_llm)
    first = await plan_node.plan_fix(exact_repair_state)

    assert first.current_phase == Phase.PLAN
    assert first.decision_frame.recommended_action == "plan"
    assert first.retry_count == 1
    assert first.repair_round_sequence == 1
    assert first.tool_patch_approval is None

    second = await plan_node.plan_fix(first)

    assert second.current_phase == Phase.EXECUTE
    assert second.decision_frame.recommended_action == "execute"
    assert second.repair_round_sequence == 2
    assert second.tool_patch_approval is not None
    assert len(calls) == 2


@pytest.mark.parametrize("provider", ["primary", "escalation"])
@pytest.mark.parametrize(
    ("action", "expected_phase", "expected_retry", "keeps_open_round"),
    [
        ("execute", Phase.EXECUTE, 0, True),
        ("collect_more_context", Phase.LOCATE, 0, True),
        ("ask_user", Phase.WAITING_FOR_USER, 0, True),
        ("stop", Phase.PLAN, 1, False),
    ],
)
async def test_provider_neutral_plan_action_matrix_preserves_round_and_decision_fields(
    exact_repair_state,
    monkeypatch,
    provider,
    action,
    expected_phase,
    expected_retry,
    keeps_open_round,
):
    _forbid_legacy_opus_path(monkeypatch)
    state = exact_repair_state
    if provider == "escalation":
        _already_escalated(state)
    calls = []

    async def fake_llm(system, user, model=None, *, provider="primary", **_kwargs):
        calls.append((system, user, model, provider))
        return plan_response(action=action)

    monkeypatch.setattr(plan_node, "llm_call", fake_llm)
    result = await plan_node.plan_fix(state)

    expected_model = ESCALATION_MODEL if provider == "escalation" else PRIMARY_MODEL
    assert calls == [(plan_node.PLAN_SYSTEM, calls[0][1], expected_model, provider)]
    assert result.current_phase == expected_phase
    assert result.retry_count == expected_retry
    assert result.repair_round_sequence == 1
    assert result.current_repair_round_id == (1 if keeps_open_round else 0)
    assert result.test_command == (
        "" if action == "execute" else "pytest -q tests/test_widget.py"
    )
    assert result.decision_frame.summary == (
        "Apply 1 validated structured edit(s)."
        if action == "execute"
        else "Update the sentinel."
    )
    assert result.decision_frame.recommended_action == (
        "plan" if action == "stop" else action
    )
    assert bool(result.tool_patch_approval) is (action == "execute")


@pytest.mark.parametrize("provider", ["primary", "escalation"])
async def test_provider_neutral_tool_reprompt_keeps_open_round(
    exact_repair_state, monkeypatch, provider
):
    _forbid_legacy_opus_path(monkeypatch)
    state = exact_repair_state
    if provider == "escalation":
        _already_escalated(state)
    calls = []

    async def fake_llm(system, user, model=None, *, provider="primary", **_kwargs):
        calls.append((system, user, model, provider, state.current_repair_round_id))
        if len(calls) == 1:
            return {
                "kind": "tool",
                "tool_intent": {
                    "action": "read_symbol",
                    "args": {"path": "src/widget.py", "symbol": "widget"},
                    "reason": "Confirm the exact implementation.",
                    "expected_evidence": "The current function body.",
                },
            }
        return plan_response()

    async def fake_route(current, intent, **_kwargs):
        current.evidence.append(
            Evidence(
                evidence_id="ev_tool",
                tool=intent.action,
                summary="Exact widget source",
                content="def widget(): return 'old-sentinel'",
                fingerprint="tool-fingerprint",
            )
        )
        return ToolRouteResult(
            action=intent.action,
            status="ok",
            args_fingerprint="fingerprint",
            evidence_id="ev_tool",
            made_progress=True,
        )

    monkeypatch.setattr(plan_node, "llm_call", fake_llm)
    monkeypatch.setattr(plan_node, "route_tool_intent", fake_route)
    result = await plan_node.plan_fix(state)

    assert [item[4] for item in calls] == [1, 1]
    assert [item[3] for item in calls] == [provider, provider]
    assert all(item[0] == plan_node.PLAN_SYSTEM for item in calls)
    assert result.retry_count == 0
    assert result.current_repair_round_id == 1
    assert result.current_phase == Phase.EXECUTE


@pytest.mark.parametrize(
    "outcome",
    ["stop", "raw_rejection", "gate_rejection", "post_gate_rejection"],
)
async def test_new_full_transaction_atomically_retires_stale_authorization(
    exact_repair_state, monkeypatch, outcome
):
    state = exact_repair_state
    begin_repair_round(state)
    accepted = authorize_plan_patch(state, plan_response())
    assert accepted.status == "accepted"
    assert state.tool_patch_approval is not None
    state.generated_test_approvals = [
        GeneratedTestApproval(
            path="tests/generated_test.py",
            content_sha256="0" * 64,
            patch_gate_fingerprint="1" * 64,
        )
    ]

    if outcome == "stop":
        response = {"kind": "stop", "stop_reason": "No patch."}
    elif outcome == "raw_rejection":
        response = plan_response()
        response["patch_edits"][0]["node_target"] = "widget"
    elif outcome == "gate_rejection":
        response = plan_response(search="missing gate anchor")
    else:
        response = plan_response()
        state.fix_attempts = [
            FixAttempt(
                patch_edits=[
                    PatchEdit(
                        file_path="src/widget.py",
                        search="return 'old-sentinel'",
                        replace="return 'new-sentinel'",
                    )
                ],
                success=False,
            )
        ]

    async def fake_llm(*_args, **_kwargs):
        return response

    monkeypatch.setattr(plan_node, "llm_call", fake_llm)
    result = await plan_node.plan_fix(state)

    assert result.current_phase in {Phase.PLAN, Phase.FAILURE}
    assert result.patch_content == ""
    assert result.patch_edits == []
    assert result.active_repair_plan is None
    assert result.tool_patch_approval is None
    assert result.generated_test_approvals == []
    assert result.authorized_repair_round_id == 0
    assert result.authorized_repair_provider is None
    assert result.authorized_repair_model == ""


async def test_patchgate_environment_failure_routes_infra_without_retry_or_correction(
    exact_repair_state, monkeypatch
):
    state = exact_repair_state
    state.repo_ref = "f" * 40
    calls = 0

    async def fake_llm(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return plan_response()

    monkeypatch.setattr(plan_node, "llm_call", fake_llm)
    result = await plan_node.plan_fix(state)

    assert calls == 1
    assert result.current_phase == Phase.FAILURE
    assert result.repair_correction_context == ""
    assert result.retry_count == 0
    assert result.primary_failed_repair_rounds == 0
    assert result.last_counted_repair_round_id == 0
    assert result.tool_patch_approval is None


async def test_mixed_patchgate_issues_prefer_environment(
    exact_repair_state, monkeypatch
):
    async def fake_llm(*_args, **_kwargs):
        return plan_response()

    def mixed_outcome(state, _response):
        return PatchAuthorizationOutcome(
            status="environment",
            issues=(
                PatchAuthorizationIssue(
                    code="search_missing",
                    file_path="src/widget.py",
                    message="The search anchor is absent.",
                    failure_class="model_correctable",
                ),
                PatchAuthorizationIssue(
                    code="target_missing",
                    file_path="src/widget.py",
                    message="The exact checkout is unavailable.",
                    failure_class="environment",
                ),
            ),
        )

    monkeypatch.setattr(plan_node, "llm_call", fake_llm)
    monkeypatch.setattr(
        plan_node,
        "authorize_plan_patch",
        mixed_outcome,
        raising=False,
    )
    result = await plan_node.plan_fix(exact_repair_state)

    assert result.current_phase == Phase.FAILURE
    assert result.retry_count == 0
    assert result.primary_failed_repair_rounds == 0
    assert result.repair_correction_context == ""


async def test_primary_503_fallback_rebinds_opus_in_same_round(
    exact_repair_state, monkeypatch
):
    _enable_escalation(monkeypatch)
    calls = []

    async def fake_llm(system, user, model=None, *, provider="primary", **_kwargs):
        calls.append((provider, model, exact_repair_state.current_repair_round_id))
        if len(calls) == 1:
            raise _gateway_503()
        return plan_response()

    monkeypatch.setattr(plan_node, "llm_call", fake_llm)
    result = await plan_node.plan_fix(exact_repair_state)

    assert calls == [
        ("primary", PRIMARY_MODEL, 1),
        ("escalation", ESCALATION_MODEL, 1),
    ]
    assert result.retry_count == 0
    assert result.primary_failed_repair_rounds == 0
    assert result.repair_round_sequence == 1
    assert result.authorized_repair_provider == "escalation"
    assert result.authorized_repair_model == ESCALATION_MODEL
    assert result.escalation_reason == "primary_gateway_unavailable_after_retries"
    assert [item.status for item in result.model_history] == ["error", "ok"]


@pytest.mark.parametrize("extra_field", ["tool_intent", "stop_reason"])
async def test_patch_response_ignores_untrusted_control_fields(
    exact_repair_state, monkeypatch, extra_field
):
    response = plan_response()
    response[extra_field] = (
        {
            "action": "search_text",
            "args": {"text": "sentinel"},
            "reason": "mixed payload",
            "expected_evidence": "source",
        }
        if extra_field == "tool_intent"
        else "mixed stop"
    )

    async def fake_llm(*_args, **_kwargs):
        return response

    monkeypatch.setattr(plan_node, "llm_call", fake_llm)
    result = await plan_node.plan_fix(exact_repair_state)

    assert result.current_phase == Phase.EXECUTE


async def test_primary_token_reserve_binds_opus_before_first_call(
    exact_repair_state, monkeypatch
):
    _enable_escalation(monkeypatch)
    exact_repair_state.token_usage = 55_000
    calls = []

    async def fake_llm(system, user, model=None, *, provider="primary", **_kwargs):
        calls.append((provider, model, exact_repair_state.current_repair_round_id))
        return plan_response()

    monkeypatch.setattr(plan_node, "llm_call", fake_llm)
    result = await plan_node.plan_fix(exact_repair_state)

    assert calls == [("escalation", ESCALATION_MODEL, 1)]
    assert result.retry_count == 0
    assert result.authorized_repair_provider == "escalation"
    assert result.escalation_reason == "primary_budget_reserve"


async def test_tool_reprompt_crossing_token_reserve_rebinds_opus_in_same_round(
    exact_repair_state, monkeypatch
):
    _enable_escalation(monkeypatch)
    calls = []

    async def fake_llm(system, user, model=None, *, provider="primary", **_kwargs):
        calls.append((provider, model, exact_repair_state.current_repair_round_id))
        if len(calls) == 1:
            return {
                "kind": "tool",
                "tool_intent": {
                    "action": "read_symbol",
                    "args": {"path": "src/widget.py", "symbol": "widget"},
                    "reason": "Check the implementation.",
                    "expected_evidence": "The current function body.",
                },
            }
        return plan_response()

    async def reserve_crossing_route(current, intent, **_kwargs):
        current.token_usage = 55_000
        return ToolRouteResult(
            action=intent.action,
            status="ok",
            args_fingerprint="reserve-crossing",
            made_progress=False,
        )

    monkeypatch.setattr(plan_node, "llm_call", fake_llm)
    monkeypatch.setattr(plan_node, "route_tool_intent", reserve_crossing_route)
    result = await plan_node.plan_fix(exact_repair_state)

    assert calls == [
        ("primary", PRIMARY_MODEL, 1),
        ("escalation", ESCALATION_MODEL, 1),
    ]
    assert result.retry_count == 0
    assert result.repair_round_sequence == 1
    assert result.authorized_repair_provider == "escalation"


async def test_escalation_gateway_exhaustion_routes_environment_without_retry(
    exact_repair_state, monkeypatch
):
    _forbid_legacy_opus_path(monkeypatch)
    state = _already_escalated(exact_repair_state)
    calls = 0

    async def fake_llm(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise _gateway_503()

    monkeypatch.setattr(plan_node, "llm_call", fake_llm)
    result = await plan_node.plan_fix(state)

    assert calls == 1
    assert result.current_phase == Phase.FAILURE
    assert result.retry_count == 0
    assert result.primary_failed_repair_rounds == 0
    assert result.last_counted_repair_round_id == 0
    assert result.repair_correction_context == ""


@pytest.mark.parametrize("mode", ["gateway", "reserve"])
async def test_unconfigured_gateway_or_token_fallback_never_spins_or_claims_opus(
    exact_repair_state, monkeypatch, mode
):
    calls = 0
    if mode == "reserve":
        exact_repair_state.token_usage = 55_000

    async def fake_llm(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if mode == "gateway":
            raise _gateway_503()
        return plan_response()

    monkeypatch.setattr(plan_node, "llm_call", fake_llm)
    result = await plan_node.plan_fix(exact_repair_state)

    assert calls == (1 if mode == "gateway" else 0)
    assert result.current_phase == Phase.FAILURE
    assert result.active_provider == "primary"
    assert result.active_model == PRIMARY_MODEL
    assert result.escalated is False
    assert result.escalation_reason == ""
    assert result.retry_count == 0
    assert result.primary_failed_repair_rounds == 0


async def test_unconfigured_threshold_never_claims_opus(
    exact_repair_state, monkeypatch
):
    state = exact_repair_state
    state.retry_count = 2
    state.primary_failed_repair_rounds = 2
    state.repair_round_sequence = 2
    state.last_counted_repair_round_id = 2
    calls = []

    async def fake_llm(system, user, model=None, *, provider="primary", **_kwargs):
        calls.append((provider, model))
        return plan_response()

    monkeypatch.setattr(plan_node, "llm_call", fake_llm)
    result = await plan_node.plan_fix(state)

    assert calls == [("primary", PRIMARY_MODEL)]
    assert result.current_phase == Phase.EXECUTE
    assert result.escalated is False
    assert result.escalation_reason == ""
    assert result.authorized_repair_provider == "primary"


async def test_correction_context_is_preserved_for_tools_then_replaced_or_cleared(
    exact_repair_state, monkeypatch
):
    responses = [
        plan_response(search="first missing anchor"),
        {
            "kind": "tool",
            "tool_intent": {
                "action": "read_symbol",
                "args": {"path": "src/widget.py", "symbol": "widget"},
                "reason": "Read the exact source.",
                "expected_evidence": "The current function body.",
            },
        },
        plan_response(),
    ]
    prompts = []

    async def fake_llm(system, user, model=None, *, provider="primary", **_kwargs):
        prompts.append(user)
        return responses.pop(0)

    async def fake_route(current, intent, **_kwargs):
        return ToolRouteResult(
            action=intent.action,
            status="ok",
            args_fingerprint="correction-tool",
            made_progress=False,
        )

    monkeypatch.setattr(plan_node, "llm_call", fake_llm)
    monkeypatch.setattr(plan_node, "route_tool_intent", fake_route)
    first = await plan_node.plan_fix(exact_repair_state)
    original = first.repair_correction_context
    assert original

    result = await plan_node.plan_fix(first)

    assert original in prompts[1]
    assert original in prompts[2]
    assert result.repair_correction_context == ""
    assert result.current_phase == Phase.EXECUTE


async def test_second_patch_rejection_replaces_correction_instead_of_appending(
    exact_repair_state, monkeypatch
):
    responses = [plan_response(), plan_response()]
    outcomes = [
        PatchAuthorizationOutcome(
            status="model_correctable",
            issues=(
                PatchAuthorizationIssue(
                    code="search_missing",
                    message="FIRST correction sentinel",
                    correction_context="FIRST bounded window",
                    failure_class="model_correctable",
                ),
            ),
        ),
        PatchAuthorizationOutcome(
            status="model_correctable",
            issues=(
                PatchAuthorizationIssue(
                    code="search_missing",
                    message="SECOND correction sentinel",
                    correction_context="SECOND bounded window",
                    failure_class="model_correctable",
                ),
            ),
        ),
    ]

    async def fake_llm(*_args, **_kwargs):
        return responses.pop(0)

    def reject(*_args, **_kwargs):
        return outcomes.pop(0)

    monkeypatch.setattr(plan_node, "llm_call", fake_llm)
    monkeypatch.setattr(plan_node, "authorize_plan_patch", reject)

    first = await plan_node.plan_fix(exact_repair_state)
    first_correction = first.repair_correction_context
    second = await plan_node.plan_fix(first)

    assert "FIRST" in first_correction
    assert "SECOND" in second.repair_correction_context
    assert "FIRST" not in second.repair_correction_context


@pytest.mark.parametrize(
    ("action", "expected_phase"),
    [
        ("collect_more_context", Phase.LOCATE),
        ("ask_user", Phase.WAITING_FOR_USER),
    ],
)
async def test_nonexecute_decision_preserves_seeded_correction_suffix(
    exact_repair_state, monkeypatch, action, expected_phase
):
    exact_repair_state.repair_correction_context = "seeded correction sentinel"

    async def fake_llm(*_args, **_kwargs):
        return plan_response(action=action)

    monkeypatch.setattr(plan_node, "llm_call", fake_llm)
    result = await plan_node.plan_fix(exact_repair_state)

    assert result.current_phase == expected_phase
    assert result.repair_correction_context == "seeded correction sentinel"
    assert result.retry_count == 0
    assert result.current_repair_round_id == 1


async def test_accepted_plan_clears_seeded_correction_suffix(
    exact_repair_state, monkeypatch
):
    exact_repair_state.repair_correction_context = "obsolete correction sentinel"

    async def fake_llm(*_args, **_kwargs):
        return plan_response()

    monkeypatch.setattr(plan_node, "llm_call", fake_llm)
    result = await plan_node.plan_fix(exact_repair_state)

    assert result.current_phase == Phase.EXECUTE
    assert result.repair_correction_context == ""


@pytest.mark.parametrize(
    "denied_term",
    ["UnIfIeD DiFf", "RePlAcE_AlL", "rEpAiRpLaN", "VeRiFiEdEdItBaTcH"],
)
def test_previous_failure_context_redacts_each_mixed_case_denied_term(denied_term):
    state = plan_node.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        issue_title="Safe issue",
        issue_body="Safe body",
        fix_attempts=[
            FixAttempt(
                test_result=(
                    "safe-result-sentinel "
                    "sk-AAAAAAAAAAAAAAAA "
                    f"{denied_term} forbidden-result-sentinel"
                ),
                error_log="safe-error-sentinel",
                success=False,
            )
        ],
    )

    prompt = plan_node.build_plan_user_prompt(state)
    folded = prompt.casefold()

    assert "safe-result-sentinel" in prompt
    assert "safe-error-sentinel" in prompt
    assert denied_term.casefold() not in folded
    assert "forbidden-result-sentinel" not in prompt
    assert "sk-aaaaaaaa" not in folded


def test_previous_failure_context_has_individual_and_total_bounds():
    state = plan_node.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        issue_title="Safe issue",
        issue_body="Safe body",
        fix_attempts=[
            FixAttempt(
                test_result=f"result-{index}-sentinel " + "r" * 4_000,
                error_log=f"error-{index}-sentinel " + "e" * 4_000,
                success=False,
            )
            for index in range(40)
        ],
    )

    prompt = plan_node.build_plan_user_prompt(state)

    assert "result-0-sentinel" in prompt
    assert "error-0-sentinel" in prompt
    assert len(prompt) <= plan_node.PLAN_PREVIOUS_FAILURES_TOTAL_LIMIT + 5_000


@pytest.mark.parametrize("failure_mode", ["checkout_drift", "read_io"])
async def test_post_gate_symbol_resolution_environment_failure_spends_no_retry(
    exact_repair_state, monkeypatch, failure_mode
):
    exact_repair_state.assertion_diversity_required = True
    exact_repair_state.fix_attempts = [
        FixAttempt(
            patch_edits=[
                PatchEdit(
                    file_path="src/widget.py",
                    search="return 'old-sentinel'",
                    replace="return 'wrong-sentinel'",
                )
            ],
            failure_kind="assertion_failure",
            error_log="AssertionError: wrong sentinel",
            success=False,
        )
    ]
    response = plan_response()
    response["patch_edits"][0]["node_target"] = "widget"
    response["patch_edits"][0].pop("search")
    real_authorize = plan_node.authorize_plan_patch

    def authorize_then_break_environment(state, raw):
        outcome = real_authorize(state, raw)
        assert outcome.status == "accepted"
        if failure_mode == "checkout_drift":
            state.repo_ref = "f" * 40
        else:
            monkeypatch.setattr(
                repair_flow,
                "_read_regular_no_follow",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("read failed")),
            )
        return outcome

    async def fake_llm(*_args, **_kwargs):
        return response

    monkeypatch.setattr(plan_node, "llm_call", fake_llm)
    monkeypatch.setattr(
        plan_node, "authorize_plan_patch", authorize_then_break_environment
    )
    result = await plan_node.plan_fix(exact_repair_state)

    assert result.current_phase == Phase.FAILURE
    assert result.retry_count == 0
    assert result.primary_failed_repair_rounds == 0
    assert result.last_counted_repair_round_id == 0
    assert result.patch_edits == []
    assert result.tool_patch_approval is None
    assert result.repair_correction_context == ""


@pytest.mark.parametrize("provider", ["primary", "escalation"])
async def test_plan_prompts_never_advertise_legacy_patch_protocol(
    exact_repair_state, monkeypatch, provider
):
    _forbid_legacy_opus_path(monkeypatch)
    state = exact_repair_state
    if provider == "escalation":
        _already_escalated(state)
    state.fix_attempts = [
        FixAttempt(
            patch_content="legacy patch",
            test_result="patch_apply_failed",
            failure_kind="patch_apply_failed",
            error_log=(
                "old instructions mentioned unified diff replace_all RepairPlan "
                "VerifiedEditBatch"
            ),
            success=False,
        )
    ]
    responses = [
        plan_response(search="missing prompt anchor"),
        plan_response(),
    ]
    prompts = []

    async def fake_llm(system, user, model=None, *, provider="primary", **_kwargs):
        prompts.append(f"{system}\n{user}".casefold())
        return responses.pop(0)

    monkeypatch.setattr(plan_node, "llm_call", fake_llm)
    first = await plan_node.plan_fix(state)
    await plan_node.plan_fix(first)

    assert len(prompts) == 2
    for prompt in prompts:
        assert not any(term in prompt for term in _BANNED_PROTOCOL_TERMS)


@pytest.mark.parametrize(
    "cancellation",
    [asyncio.CancelledError("cancel plan"), pytest.param(None, id="drain")],
)
async def test_plan_cancellation_preserves_identity_without_telemetry_or_debit(
    exact_repair_state, monkeypatch, cancellation
):
    if cancellation is None:
        from src.async_safety import CancellationDrainError

        underlying = asyncio.CancelledError("cancel tool")
        cancellation = CancellationDrainError(
            "plan model",
            underlying,
            RuntimeError("cleanup failed"),
        )
    starting_tokens = exact_repair_state.token_usage

    async def cancelled(*_args, **_kwargs):
        raise cancellation

    monkeypatch.setattr(plan_node, "llm_call", cancelled)
    with pytest.raises(type(cancellation)) as raised:
        await plan_node.plan_fix(exact_repair_state)

    assert raised.value is cancellation
    assert exact_repair_state.model_history == []
    assert exact_repair_state.token_usage == starting_tokens
    assert exact_repair_state.retry_count == 0
    assert exact_repair_state.last_counted_repair_round_id == 0
    assert exact_repair_state.repair_round_sequence == 1
    assert exact_repair_state.current_repair_round_id == 1
