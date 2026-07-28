from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src import model_policy
from src.repair_rounds import (
    RepairRetryDecision,
    begin_repair_round,
    bind_repair_round_author,
    freeze_authorized_repair_round,
    record_failed_repair_round,
    retire_patch_authorization,
    validate_repair_round_state,
)
from src.state import AgentState, FixAttempt, PatchEdit, Phase, ToolPatchApproval


PRIMARY_MODEL = "gemini-3.5-flash:stable"
ESCALATION_MODEL = "claude-opus-4-8:stable"


def enable_escalation(monkeypatch):
    monkeypatch.setattr(model_policy, "escalation_is_configured", lambda: True)
    monkeypatch.setattr(
        model_policy,
        "get_model_config",
        lambda provider: SimpleNamespace(
            model=(
                ESCALATION_MODEL
                if provider == "escalation"
                else PRIMARY_MODEL
            )
        ),
    )


def disable_escalation(monkeypatch):
    monkeypatch.setattr(model_policy, "escalation_is_configured", lambda: False)


def make_state(**updates):
    defaults = {
        "issue_url": "https://github.com/acme/widget/issues/7",
        "max_retries": 4,
        "active_model": PRIMARY_MODEL,
    }
    defaults.update(updates)
    return AgentState(**defaults)


def gate_approval() -> ToolPatchApproval:
    return ToolPatchApproval(
        base_ref="a" * 40,
        patch_sha256="b" * 64,
        patch_gate_fingerprint="c" * 64,
        changed_manifest=(),
        manifest_fingerprint="d" * 64,
    )


def prepare_authorized_patch(state: AgentState) -> None:
    state.patch_content = "diff --git a/widget.py b/widget.py"
    state.patch_edits = [
        PatchEdit(file_path="widget.py", search="old", replace="new")
    ]
    state.tool_patch_approval = gate_approval()
    freeze_authorized_repair_round(state)


def test_failed_rounds_are_counted_once_and_switch_after_two_primary_failures(
    monkeypatch,
):
    enable_escalation(monkeypatch)
    state = make_state()

    first = begin_repair_round(state)
    bind_repair_round_author(state)
    one = record_failed_repair_round(
        state,
        round_id=first,
        provider="primary",
        model=PRIMARY_MODEL,
        failure_reason="invalid_edit",
        retry_phase=Phase.PLAN,
    )
    before_duplicate = state.model_copy(deep=True)
    duplicate = record_failed_repair_round(
        state,
        round_id=first,
        provider="primary",
        model=PRIMARY_MODEL,
        failure_reason="invalid_edit",
        retry_phase=Phase.PLAN,
    )

    assert one == RepairRetryDecision(counted=True, retry_allowed=True, round_id=1)
    assert duplicate == RepairRetryDecision(
        counted=False, retry_allowed=True, round_id=1
    )
    assert state == before_duplicate
    assert state.retry_count == 1
    assert state.primary_failed_repair_rounds == 1
    assert state.active_provider == "primary"
    assert state.node_diagnostics[-1]["failure_reason"] == "invalid_edit"

    second = begin_repair_round(state)
    bind_repair_round_author(state)
    record_failed_repair_round(
        state,
        round_id=second,
        provider="primary",
        model=PRIMARY_MODEL,
        failure_reason="tests_failed",
        retry_phase=Phase.REFLECT,
    )

    assert second == 2
    assert state.retry_count == 2
    assert state.primary_failed_repair_rounds == 2
    assert state.active_provider == "escalation"
    assert state.active_model == ESCALATION_MODEL
    assert state.escalation_reason == "primary_repair_round_limit"


@pytest.mark.parametrize("max_retries", [0, 1, 3, 4])
def test_max_retry_values_allocate_exact_transaction_sequences(
    monkeypatch, max_retries
):
    disable_escalation(monkeypatch)
    state = make_state(max_retries=max_retries)
    allocated = []

    for expected_id in range(1, max_retries + 2):
        round_id = begin_repair_round(state)
        allocated.append(round_id)
        decision = record_failed_repair_round(
            state,
            round_id=round_id,
            provider="primary",
            model=PRIMARY_MODEL,
            failure_reason="tests_failed",
            retry_phase=Phase.REFLECT,
        )
        assert decision.retry_allowed is (expected_id <= max_retries)

    assert allocated == list(range(1, max_retries + 2))
    assert state.retry_count == max_retries
    assert state.repair_round_sequence == max_retries + 1
    assert state.last_counted_repair_round_id == max_retries + 1
    assert state.current_phase == Phase.FAILURE

    with pytest.raises(ValueError, match="budget is exhausted"):
        begin_repair_round(state)
    assert state.repair_round_sequence == max_retries + 1


def test_context_and_tool_reentry_reuse_open_round(monkeypatch):
    disable_escalation(monkeypatch)
    state = make_state()

    first = begin_repair_round(state)
    second = begin_repair_round(state)
    third = begin_repair_round(state)

    assert (first, second, third) == (1, 1, 1)
    assert state.repair_round_sequence == 1
    assert state.retry_count == 0
    assert state.last_counted_repair_round_id == 0


def test_gateway_fallback_rebinds_same_round_without_counter_change(monkeypatch):
    disable_escalation(monkeypatch)
    state = make_state()
    round_id = begin_repair_round(state)

    state.active_provider = "escalation"
    state.active_model = ESCALATION_MODEL
    bind_repair_round_author(state)

    assert begin_repair_round(state) == round_id
    assert state.current_repair_provider == "escalation"
    assert state.current_repair_model == ESCALATION_MODEL
    assert state.repair_round_sequence == 1
    assert state.retry_count == 0
    assert state.primary_failed_repair_rounds == 0


def test_patch_authorization_retirement_is_atomic_and_test_only_can_preserve_attribution(
    monkeypatch,
):
    disable_escalation(monkeypatch)
    state = make_state()
    begin_repair_round(state)
    prepare_authorized_patch(state)

    retire_patch_authorization(state, preserve_authorized_attribution=True)

    assert state.patch_content == ""
    assert state.patch_edits == []
    assert state.active_repair_plan is None
    assert state.tool_patch_approval is None
    assert state.generated_test_approvals == []
    assert state.authorized_repair_round_id == 1
    assert state.authorized_repair_provider == "primary"
    assert state.authorized_repair_model == PRIMARY_MODEL

    state.patch_content = "diff --git a/widget.py b/widget.py"
    state.patch_edits = [
        PatchEdit(file_path="widget.py", search="old", replace="new")
    ]
    state.tool_patch_approval = gate_approval()
    retire_patch_authorization(state)

    assert state.patch_content == ""
    assert state.patch_edits == []
    assert state.tool_patch_approval is None
    assert state.authorized_repair_round_id == 0
    assert state.authorized_repair_provider is None
    assert state.authorized_repair_model == ""


def test_record_progress_does_not_reset_primary_failed_rounds():
    state = make_state(primary_failed_repair_rounds=2, no_progress_rounds=2)

    model_policy.record_progress(state)

    assert state.primary_failed_repair_rounds == 2
    assert state.no_progress_rounds == 0


def test_historical_empty_ledger_validates_with_defaults_and_migrates_retry_count():
    historical = AgentState.model_validate(
        {
            "issue_url": "https://github.com/acme/widget/issues/7",
            "retry_count": 2,
            "max_retries": 4,
        }
    )

    validate_repair_round_state(historical)

    assert historical.repair_round_sequence == 2
    assert historical.last_counted_repair_round_id == 2
    assert historical.current_repair_round_id == 0
    assert historical.primary_failed_repair_rounds == 0
    assert historical.repair_correction_context == ""


def test_direct_historical_state_object_without_ledger_fields_migrates():
    historical = AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        retry_count=2,
        max_retries=4,
    )
    assert not historical.model_fields_set & {
        "repair_round_sequence",
        "current_repair_round_id",
        "last_counted_repair_round_id",
    }

    validate_repair_round_state(historical)

    assert historical.repair_round_sequence == 2
    assert historical.last_counted_repair_round_id == 2


@pytest.mark.parametrize(
    "updates",
    [
        pytest.param(
            {"retry_count": 5, "max_retries": 4},
            id="retry-count-beyond-budget",
        ),
        pytest.param(
            {
                "retry_count": 0,
                "last_counted_repair_round_id": 2,
                "repair_round_sequence": 2,
            },
            id="retry-last-difference",
        ),
        pytest.param(
            {
                "retry_count": 1,
                "last_counted_repair_round_id": 1,
                "repair_round_sequence": 3,
            },
            id="sequence-last-difference",
        ),
        pytest.param(
            {
                "retry_count": 1,
                "last_counted_repair_round_id": 1,
                "repair_round_sequence": 2,
                "current_repair_round_id": 1,
                "current_repair_provider": "primary",
                "current_repair_model": PRIMARY_MODEL,
            },
            id="current-id-mismatch",
        ),
        pytest.param(
            {
                "retry_count": 1,
                "last_counted_repair_round_id": 1,
                "repair_round_sequence": 2,
                "current_repair_round_id": 2,
            },
            id="current-missing-attribution",
        ),
        pytest.param(
            {
                "retry_count": 1,
                "last_counted_repair_round_id": 1,
                "repair_round_sequence": 1,
                "primary_failed_repair_rounds": 2,
            },
            id="primary-counter-beyond-counted",
        ),
        pytest.param(
            {
                "retry_count": 1,
                "last_counted_repair_round_id": 1,
                "repair_round_sequence": 1,
                "authorized_repair_round_id": 1,
                "tool_patch_approval": gate_approval(),
            },
            id="authorized-missing-attribution",
        ),
        pytest.param(
            {
                "retry_count": 1,
                "last_counted_repair_round_id": 1,
                "repair_round_sequence": 1,
                "authorized_repair_round_id": 1,
                "authorized_repair_provider": "primary",
                "authorized_repair_model": PRIMARY_MODEL,
            },
            id="authorized-missing-approval",
        ),
        pytest.param(
            {"retry_count": 1, "repair_round_sequence": 0},
            id="explicit-default-partial-ledger",
        ),
        pytest.param(
            {"retry_count": 1, "primary_failed_repair_rounds": 1},
            id="legacy-shape-with-primary-counter",
        ),
        pytest.param(
            {
                "retry_count": 1,
                "authorized_repair_round_id": 1,
                "authorized_repair_provider": "primary",
                "authorized_repair_model": PRIMARY_MODEL,
                "tool_patch_approval": gate_approval(),
            },
            id="legacy-shape-with-complete-authorization",
        ),
        pytest.param(
            {
                "retry_count": 1,
                "authorized_repair_round_id": 2,
                "authorized_repair_provider": "primary",
                "authorized_repair_model": PRIMARY_MODEL,
                "tool_patch_approval": gate_approval(),
            },
            id="migration-must-not-write-before-later-error",
        ),
        pytest.param(
            {"retry_count": 1, "current_repair_provider": "primary"},
            id="orphaned-current-provider",
        ),
        pytest.param(
            {"retry_count": 1, "authorized_repair_model": PRIMARY_MODEL},
            id="orphaned-authorized-model",
        ),
    ],
)
def test_invalid_ledger_relationships_fail_closed_without_mutation(updates):
    state = make_state(**updates)
    before = state.model_dump(mode="json")
    before_fields_set = state.model_fields_set.copy()

    with pytest.raises(ValueError):
        validate_repair_round_state(state)

    assert state.model_dump(mode="json") == before
    assert state.model_fields_set == before_fields_set


def test_positive_fix_attempt_attribution_requires_runtime_provider_and_model():
    assert FixAttempt().repair_provider is None

    with pytest.raises(ValidationError, match="runtime repair attribution"):
        FixAttempt(repair_round_id=1)

    attempt = FixAttempt(
        repair_round_id=1,
        repair_provider="escalation",
        repair_model=ESCALATION_MODEL,
    )
    assert attempt.repair_round_id == 1
    assert attempt.repair_provider == "escalation"
    assert attempt.repair_model == ESCALATION_MODEL


def test_fix_attempt_assignment_cannot_create_orphaned_positive_attribution():
    attempt = FixAttempt()
    before = attempt.model_dump(mode="json")

    with pytest.raises(ValidationError, match="runtime repair attribution"):
        attempt.repair_round_id = 1

    assert attempt.model_dump(mode="json") == before

    attempt.repair_provider = "primary"
    attempt.repair_model = PRIMARY_MODEL
    attempt.repair_round_id = 1
    assert attempt.repair_round_id == 1
    assert attempt.repair_provider == "primary"
    assert attempt.repair_model == PRIMARY_MODEL
    assert FixAttempt.model_validate(attempt.model_dump()) == attempt

    for field, value in (("repair_provider", None), ("repair_model", "")):
        valid = attempt.model_dump(mode="json")
        with pytest.raises(ValidationError, match="runtime repair attribution"):
            setattr(attempt, field, value)
        assert attempt.model_dump(mode="json") == valid


def test_new_failure_rejects_mismatched_bound_author_without_accounting(monkeypatch):
    disable_escalation(monkeypatch)
    state = make_state()
    round_id = begin_repair_round(state)
    before = state.model_copy(deep=True)

    with pytest.raises(ValueError, match="author attribution"):
        record_failed_repair_round(
            state,
            round_id=round_id,
            provider="escalation",
            model=ESCALATION_MODEL,
            failure_reason="tests_failed",
            retry_phase=Phase.REFLECT,
        )

    assert state == before


def test_duplicate_terminal_failure_remains_terminal_without_mutation(monkeypatch):
    disable_escalation(monkeypatch)
    state = make_state(max_retries=0)
    round_id = begin_repair_round(state)
    record_failed_repair_round(
        state,
        round_id=round_id,
        provider="primary",
        model=PRIMARY_MODEL,
        failure_reason="tests_failed",
        retry_phase=Phase.REFLECT,
    )
    before = state.model_copy(deep=True)

    duplicate = record_failed_repair_round(
        state,
        round_id=round_id,
        provider="primary",
        model=PRIMARY_MODEL,
        failure_reason="different",
        retry_phase=Phase.PLAN,
    )

    assert duplicate == RepairRetryDecision(
        counted=False, retry_allowed=False, round_id=1
    )
    assert state == before


def test_partial_future_ledger_is_rejected_without_legacy_repair():
    state = make_state(
        retry_count=4,
        repair_round_sequence=6,
        current_repair_round_id=6,
        current_repair_provider="primary",
        current_repair_model=PRIMARY_MODEL,
        last_counted_repair_round_id=5,
    )

    with pytest.raises(ValueError, match="transaction cap"):
        validate_repair_round_state(state)

    assert state.repair_round_sequence == 6
    assert state.last_counted_repair_round_id == 5
