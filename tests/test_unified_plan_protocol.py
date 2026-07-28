import asyncio
from types import SimpleNamespace

import httpx
import pytest

from src import model_policy
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
    return {
        "kind": "plan",
        "plan": "Return the new sentinel.",
        "patch": "",
        "patch_edits": edits,
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
    assert result.test_command == "pytest -q tests/test_widget.py"
    assert result.decision_frame.summary == "Update the sentinel."
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
