import hashlib
import json

import pytest

from src import model_policy
from src.model_provider import ModelConfig
from src.state import AgentState


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


def test_two_consecutive_approved_no_progress_events_escalate(monkeypatch):
    enable_escalation(monkeypatch)
    state = make_state()

    model_policy.record_no_progress(
        state, kind="unchanged_context", fingerprint={"files": ["b.py", "a.py"]}, node="locate_code"
    )
    assert model_policy.should_escalate(state).escalate is False

    model_policy.record_no_progress(
        state, kind="repeated_edit", fingerprint={"path": "a.py", "search": "x"}, node="plan_fix"
    )

    decision = model_policy.should_escalate(state)
    assert decision.escalate is True
    assert decision.reason == "repeated_edit"
    assert state.no_progress_rounds == 2


def test_real_progress_resets_consecutive_rounds_but_keeps_history(monkeypatch):
    enable_escalation(monkeypatch)
    state = make_state()
    model_policy.record_no_progress(
        state, kind="unchanged_plan", fingerprint="same plan", node="plan_fix"
    )

    model_policy.record_progress(state)

    assert state.no_progress_rounds == 0
    assert len(state.no_progress_history) == 1
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
    assert len(state.no_progress_history) == 2
    assert model_policy.should_escalate(state).escalate is False


def test_signatures_are_stable_sha256_over_canonical_json():
    state = make_state()
    plan = {"edits": [{"replace": "y", "search": "x", "path": "a.py"}]}
    context = {"files": ["a.py"], "symbols": {"a.py": ["f", "g"]}}
    failure = {"failed": ["test_a"], "exit_code": 1}

    model_policy.record_no_progress(
        state, kind="unchanged_plan", fingerprint=plan, node="plan_fix"
    )
    model_policy.record_progress(state)
    model_policy.record_no_progress(
        state, kind="unchanged_context", fingerprint=context, node="locate_code"
    )
    model_policy.record_progress(state)
    model_policy.record_no_progress(
        state,
        kind="unchanged_test_failure",
        fingerprint=failure,
        node="verify_fix",
    )

    def digest(value):
        canonical = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    assert state.last_plan_signature == digest(plan)
    assert state.last_context_fingerprint == digest(context)
    assert state.last_test_failure_signature == digest(failure)


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
        escalation_reason="first_reason",
    )

    model_policy.apply_escalation(
        state,
        model_policy.EscalationDecision(escalate=True, reason="later_reason"),
    )

    assert state.active_provider == "escalation"
    assert state.active_model == "custom-opus"
    assert state.escalation_reason == "first_reason"
    assert state.node_diagnostics == []
