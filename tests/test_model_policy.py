import hashlib
import json

import pytest

from src import model_policy
from src.model_provider import ModelConfig
from src.state import AgentState, NoProgressEvent


def make_state(**updates):
    defaults = {
        "issue_url": "https://github.com/acme/widget/issues/7",
        "token_budget": 100_000,
    }
    defaults.update(updates)
    return AgentState(**defaults)


def enable_escalation(monkeypatch):
    monkeypatch.setattr(model_policy, "escalation_is_configured", lambda: True)
    monkeypatch.setattr(
        model_policy,
        "get_model_config",
        lambda provider: ModelConfig(
            provider=provider,
            model=(
                "claude-opus-4-8:stable"
                if provider == "escalation"
                else "gemini-3.5-flash:stable"
            ),
            base_url="https://models.invalid/v1",
            api_key="test-only-key",
        ),
    )


@pytest.mark.parametrize(
    "reason",
    [
        "empty_completion_after_retries",
        "invalid_structured_response_after_retries",
    ],
)
def test_approved_immediate_failures_escalate(monkeypatch, reason):
    enable_escalation(monkeypatch)

    decision = model_policy.should_escalate(
        make_state(), immediate_reason=reason
    )

    assert decision.escalate is True
    assert decision.reason == reason


def test_unapproved_immediate_failure_does_not_escalate(monkeypatch):
    enable_escalation(monkeypatch)

    decision = model_policy.should_escalate(
        make_state(), immediate_reason="rate_limit"
    )

    assert decision.escalate is False
    assert decision.reason == ""


@pytest.mark.parametrize(
    "reason",
    [
        "repeated_plan",
        "repeated_context",
        "repeated_test",
        "repeated_edit",
        "test_generation_retry",
    ],
)
def test_later_escalation_reasons_are_approved(reason):
    decision = model_policy.EscalationDecision(escalate=True, reason=reason)

    assert decision.reason == reason


def test_first_no_progress_signature_starts_streak_at_one(monkeypatch):
    enable_escalation(monkeypatch)
    state = make_state()

    model_policy.record_no_progress(
        state,
        kind="unchanged_context",
        fingerprint={"files": ["b.py", "a.py"]},
        node="locate_code",
    )

    assert state.no_progress_rounds == 1
    assert model_policy.should_escalate(state).escalate is False


def test_two_consecutive_same_signatures_escalate(monkeypatch):
    enable_escalation(monkeypatch)
    state = make_state()
    fingerprint = {"path": "a.py", "search": "x"}

    model_policy.record_no_progress(
        state, kind="repeated_edit", fingerprint=fingerprint, node="plan_fix"
    )
    model_policy.record_no_progress(
        state, kind="repeated_edit", fingerprint=fingerprint, node="plan_fix"
    )

    decision = model_policy.should_escalate(state)
    assert decision.escalate is True
    assert decision.reason == "repeated_edit"
    assert state.no_progress_rounds == 2


def test_changed_signature_resets_streak_without_escalating(monkeypatch):
    enable_escalation(monkeypatch)
    state = make_state()

    model_policy.record_no_progress(
        state, kind="unchanged_plan", fingerprint="plan-a", node="plan_fix"
    )
    model_policy.record_no_progress(
        state, kind="unchanged_plan", fingerprint="plan-b", node="plan_fix"
    )

    assert state.no_progress_rounds == 1
    assert model_policy.should_escalate(state).escalate is False


def test_same_signature_is_not_consecutive_across_another_signal(monkeypatch):
    enable_escalation(monkeypatch)
    state = make_state()

    model_policy.record_no_progress(
        state, kind="unchanged_plan", fingerprint="plan-a", node="plan_fix"
    )
    model_policy.record_no_progress(
        state, kind="unchanged_context", fingerprint="context-a", node="locate_code"
    )
    model_policy.record_no_progress(
        state, kind="unchanged_plan", fingerprint="plan-a", node="plan_fix"
    )

    assert state.no_progress_rounds == 1
    assert model_policy.should_escalate(state).escalate is False


def test_real_progress_resets_consecutive_rounds_but_keeps_history(monkeypatch):
    enable_escalation(monkeypatch)
    state = make_state()
    model_policy.record_no_progress(
        state, kind="unchanged_plan", fingerprint="same plan", node="plan_fix"
    )

    model_policy.record_progress(state)

    assert state.no_progress_rounds == 0
    assert state.last_plan_signature == ""
    assert state.last_context_fingerprint == ""
    assert state.last_test_failure_signature == ""
    assert len(state.no_progress_history) == 1
    assert model_policy.should_escalate(state).escalate is False

    model_policy.record_no_progress(
        state, kind="unchanged_plan", fingerprint="same plan", node="plan_fix"
    )

    assert state.no_progress_rounds == 1
    assert model_policy.should_escalate(state).escalate is False


def test_unknown_no_progress_kind_breaks_consecutive_sequence(monkeypatch):
    enable_escalation(monkeypatch)
    state = make_state()
    model_policy.record_no_progress(
        state, kind="unchanged_plan", fingerprint="same plan", node="plan_fix"
    )

    model_policy.record_no_progress(
        state, kind="unapproved_signal", fingerprint="x", node="plan_fix"
    )

    assert state.no_progress_rounds == 0
    assert len(state.no_progress_history) == 1
    assert model_policy.should_escalate(state).escalate is False


@pytest.mark.parametrize(
    "hostile_value",
    ["CredentialSentinel", '"quoted-sentinel"', "FAIL_TO_PASS", "unknown_value"],
)
def test_hostile_no_progress_values_never_enter_state(hostile_value):
    state = make_state()

    model_policy.record_no_progress(
        state,
        kind=hostile_value,
        fingerprint="sentinel-fixture",
        node=hostile_value,
    )

    assert state.no_progress_history == []
    assert hostile_value not in state.model_dump_json()


@pytest.mark.parametrize("node", ["plan", "reflect", "coverage", "test_generation"])
def test_later_routing_nodes_are_approved(node):
    event = NoProgressEvent(kind="repeated_edit", fingerprint="safe-sha", node=node)

    assert event.node == node


def test_signatures_are_stable_sha256_over_canonical_json():
    plan = {"edits": [{"replace": "y", "search": "x", "path": "a.py"}]}
    context = {"files": ["a.py"], "symbols": {"a.py": ["f", "g"]}}
    failure = {"failed": ["test_a"], "exit_code": 1}
    plan_state = make_state()
    context_state = make_state()
    failure_state = make_state()

    model_policy.record_no_progress(
        plan_state, kind="unchanged_plan", fingerprint=plan, node="plan_fix"
    )
    model_policy.record_no_progress(
        context_state,
        kind="unchanged_context",
        fingerprint=context,
        node="locate_code",
    )
    model_policy.record_no_progress(
        failure_state,
        kind="unchanged_test_failure",
        fingerprint=failure,
        node="verify_fix",
    )

    def digest(value):
        canonical = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    assert plan_state.last_plan_signature == digest(plan)
    assert context_state.last_context_fingerprint == digest(context)
    assert failure_state.last_test_failure_signature == digest(failure)


@pytest.mark.parametrize(
    ("budget", "expected"),
    [(100_000, 55_000), (80_000, 40_000), (40_000, 0), (20_000, 0)],
)
def test_primary_budget_limit_reserves_escalation_capacity(budget, expected):
    assert model_policy.primary_budget_limit(make_state(token_budget=budget)) == expected


def test_primary_budget_boundary_escalates_when_configured(monkeypatch):
    enable_escalation(monkeypatch)
    state = make_state(token_usage=55_000)

    decision = model_policy.should_escalate(state)

    assert decision.escalate is True
    assert decision.reason == "primary_budget_reserve"


@pytest.mark.parametrize("configured", [False, True])
def test_no_escalation_without_complete_configuration(monkeypatch, configured):
    monkeypatch.setattr(
        model_policy, "escalation_is_configured", lambda: configured
    )
    state = make_state(token_usage=55_000)

    if configured:
        monkeypatch.setattr(model_policy, "escalation_is_configured", lambda: False)

    assert model_policy.should_escalate(state).escalate is False


def test_apply_escalation_is_idempotent_and_emits_safe_diagnostic(monkeypatch):
    enable_escalation(monkeypatch)
    state = make_state(no_progress_rounds=2)
    decision = model_policy.EscalationDecision(
        escalate=True, reason="repeated_edit"
    )

    model_policy.apply_escalation(state, decision)
    model_policy.apply_escalation(state, decision)

    assert state.active_provider == "escalation"
    assert state.active_model == "claude-opus-4-8:stable"
    assert state.escalated is True
    assert state.escalation_reason == "repeated_edit"
    assert state.node_diagnostics == [
        {
            "event": "model_escalated",
            "from": "gemini-3.5-flash:stable",
            "to": "claude-opus-4-8:stable",
            "reason": "repeated_edit",
            "round": 2,
        }
    ]


def test_apply_escalation_never_downgrades_existing_escalation(monkeypatch):
    enable_escalation(monkeypatch)
    state = make_state(
        active_provider="escalation",
        active_model="custom-opus",
        escalated=True,
        escalation_reason="repeated_plan",
    )

    model_policy.apply_escalation(
        state,
        model_policy.EscalationDecision(escalate=True, reason="repeated_context"),
    )

    assert state.active_provider == "escalation"
    assert state.active_model == "custom-opus"
    assert state.escalation_reason == "repeated_plan"
    assert state.node_diagnostics == []


@pytest.mark.parametrize(
    "hostile_reason",
    ["CredentialSentinel", '"quoted-sentinel"', "FAIL_TO_PASS", "unknown_value"],
)
def test_hostile_escalation_reason_is_rejected_before_state_and_diagnostics(
    monkeypatch, hostile_reason
):
    enable_escalation(monkeypatch)
    state = make_state()
    decision = model_policy.EscalationDecision(
        escalate=True,
        reason=hostile_reason,
    )

    model_policy.apply_escalation(state, decision)

    assert state.escalated is False
    assert state.escalation_reason == ""
    assert state.node_diagnostics == []
    assert hostile_reason not in state.model_dump_json()
