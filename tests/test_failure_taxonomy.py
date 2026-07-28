"""Failure taxonomy classifier — the fine-grained mapping is the whole value,
so pin each category to a representative error log."""

from types import SimpleNamespace

import pytest

from eval.failure_taxonomy import (
    classify_attempt,
    classify_sample,
    summarize,
)
from eval.safe_contracts import has_structured_model_gateway_failure
from src import model_policy, new_agent
from src.model_policy import record_progress
from src.patch_gate import validate_patch_batch
from src.repair_rounds import (
    begin_repair_round,
    bind_repair_round_author,
    record_failed_repair_round,
    retire_patch_authorization,
)
from src.state import RepairPlan, VerifiedEdit, VerifiedEditBatch


def _append_authorized_failure(
    state,
    *,
    failure_kind: str,
    error_log: str,
    legacy_attribution: bool = False,
):
    if state.tool_patch_approval is not None:
        retire_patch_authorization(state)
    plan = RepairPlan(
        root_cause="widget returns the old sentinel",
        target_files=["src/widget.py"],
        target_symbols=["widget"],
        required_behavior="widget returns the new sentinel",
        regression_test_strategy="run the focused widget test",
    )
    state.active_repair_plan = plan
    begin_repair_round(state)
    bind_repair_round_author(state)
    edit = VerifiedEdit(
        file_path="src/widget.py",
        search="return 'old-sentinel'",
        replace="return 'new-sentinel'",
        intent="return the new sentinel",
    )
    assert validate_patch_batch(
        state,
        plan,
        VerifiedEditBatch(edits=[edit]),
    ).accepted
    round_id = state.authorized_repair_round_id
    attempt = new_agent.FixAttempt(
        patch_content=state.patch_content,
        patch_edits=[item.model_copy(deep=True) for item in state.patch_edits],
        file_path=state.patch_edits[0].file_path,
        failure_kind=failure_kind,
        error_log=error_log,
        success=False,
        **(
            {}
            if legacy_attribution
            else {
                "repair_provider": state.authorized_repair_provider,
                "repair_model": state.authorized_repair_model,
                "repair_round_id": round_id,
            }
        ),
    )
    state.fix_attempts.append(attempt)
    state.current_phase = new_agent.Phase.VERIFY
    return attempt


def verified_coverage():
    return {
        "coverage_status": "generated_verified",
        "coverage_proof": {
            "source": "generated",
            "status": "generated_verified",
            "test_files": ["tests/test_auth.py"],
            "fixed_runs": [
                {
                    "outcome": "pass",
                    "failing_test_ids": [],
                    "assertion_fingerprint": "",
                },
                {
                    "outcome": "pass",
                    "failing_test_ids": [],
                    "assertion_fingerprint": "",
                },
            ],
            "base_runs": [
                {
                    "outcome": "assertion_failure",
                    "failing_test_ids": ["tests/test_auth.py::test_login"],
                    "assertion_fingerprint": "a" * 64,
                },
                {
                    "outcome": "assertion_failure",
                    "failing_test_ids": ["tests/test_auth.py::test_login"],
                    "assertion_fingerprint": "a" * 64,
                },
            ],
        },
    }


def test_wrong_file_path():
    assert (
        classify_attempt(
            "patch_apply_failed",
            "Search/replace edit failed: edit 1 target file was not found: lib/x.py.",
        )
        == "wrong_file_path"
    )


def test_empty_patch_not_invalid_diff():
    # "No valid patches in input" is git apply on an EMPTY patch (a gate cleared
    # the edits), NOT the model emitting a bad diff. Must not inflate invalid_diff.
    assert (
        classify_attempt(
            "patch_apply_failed",
            "Patch preflight check failed:\nerror: No valid patches in input",
        )
        == "empty_patch"
    )


def test_invalid_diff_real_hunks():
    assert (
        classify_attempt(
            "patch_apply_failed",
            "Patch preflight check failed:\ndiff --git a/x b/x\n@@ -1,3 +1,4 @@\ncorrupt",
        )
        == "invalid_diff"
    )


def test_search_not_found():
    assert (
        classify_attempt(
            "patch_apply_failed",
            "Search/replace edit failed: edit 1 search block was not found in a.py.",
        )
        == "search_not_found"
    )


def test_test_failed():
    assert (
        classify_attempt(
            "test_failed", "===== test session starts =====\nFAILED tests/test_x.py"
        )
        == "test_failed"
    )


def test_infra_timeout():
    assert classify_attempt("", "httpx.ReadTimeout") == "infra"
    assert (
        classify_attempt("infra_error", "Infrastructure error during execution")
        == "infra"
    )


def test_budget():
    assert (
        classify_attempt("", "Token budget exceeded during verification.") == "budget"
    )


def test_sample_decisive_is_last_attempt():
    sample = {
        "id": "x/y#1",
        "success": False,
        "agent_payload": {
            "fix_attempts": [
                {
                    "failure_kind": "patch_apply_failed",
                    "error_log": "search block was not found in a.py",
                },
                {"failure_kind": "test_failed", "error_log": "FAILED test_x"},
            ]
        },
    }
    c = classify_sample(sample)
    assert c["decisive"] == "test_failed"  # last attempt wins
    assert c["attempts"] == ["search_not_found", "test_failed"]


def test_sample_resolved():
    assert (
        classify_sample({"id": "a", "success": True, **verified_coverage()})["decisive"]
        == "agent_success"
    )


def test_sample_success_without_strict_coverage_is_not_agent_success():
    classified = classify_sample({"id": "a", "success": True})

    assert classified["decisive"] != "agent_success"


def test_official_resolved_is_not_inferred_from_agent_success():
    classified = classify_sample(
        {
            "id": "a",
            "success": True,
            "official_resolved": None,
            **verified_coverage(),
        }
    )

    assert classified["decisive"] == "agent_success"
    assert classified["official_resolved"] is None


@pytest.mark.parametrize(
    ("sample", "expected"),
    [
        ({"error": "opus_no_progress_limit"}, "opus_no_progress_limit"),
        ({"error": "PatchGate rejected unsafe scope"}, "patch_gate_rejected"),
        ({"error": "model gateway returned 503"}, "model_gateway_infra"),
        ({"coverage_failure_reason": "coverage_infra"}, "coverage_infra"),
        (
            {"coverage_failure_reason": "test_generation_failed"},
            "test_generation_failed",
        ),
    ],
)
def test_success_first_terminal_failure_labels(sample, expected):
    sample = {"id": "a", "success": False, **sample}

    assert classify_sample(sample)["decisive"] == expected


def test_real_generate_plan_http_status_error_is_model_gateway_infra():
    sample = {
        "id": "a",
        "success": False,
        "error": "Failed to generate fix plan: HTTPStatusError",
    }

    assert classify_sample(sample)["decisive"] == "model_gateway_infra"


def test_structured_model_invocation_error_is_model_gateway_infra():
    sample = {
        "id": "a",
        "success": False,
        "model_invocations": [
            {
                "node": "plan_fix",
                "status": "error",
                "error_class": "HTTPStatusError",
            }
        ],
    }

    assert classify_sample(sample)["decisive"] == "model_gateway_infra"


@pytest.mark.parametrize(
    ("status", "is_gateway_failure"),
    [("cancelled", False), ("error", True)],
)
def test_cancelled_invocation_is_not_structured_gateway_failure(
    status,
    is_gateway_failure,
):
    sample = {
        "id": "a",
        "success": False,
        "model_invocations": [
            {
                "node": "test_generation",
                "status": status,
                "error_class": "APITimeoutError",
            }
        ],
    }

    assert has_structured_model_gateway_failure(sample) is is_gateway_failure
    assert (
        classify_sample(sample)["decisive"] == "model_gateway_infra"
    ) is is_gateway_failure


def test_safe_model_failure_code_is_model_gateway_infra():
    sample = {
        "id": "a",
        "success": False,
        "failure_code": "model_gateway_infra",
    }

    assert classify_sample(sample)["decisive"] == "model_gateway_infra"


def test_coverage_model_gateway_code_is_terminal_without_attempts():
    sample = {
        "id": "a",
        "success": False,
        "coverage_failure_reason": "model_gateway_infra",
    }

    assert classify_sample(sample)["decisive"] == "model_gateway_infra"


@pytest.mark.parametrize(
    "terminal_evidence",
    [
        {"failure_code": "model_gateway_infra"},
        {"coverage_failure_reason": "model_gateway_infra"},
        {"error": "Failed to generate fix plan: HTTPStatusError"},
    ],
)
def test_explicit_terminal_gateway_evidence_overrides_prior_attempt(
    terminal_evidence,
):
    sample = {
        "id": "a",
        "success": False,
        "failure_class": "other",
        "agent_payload": {
            "fix_attempts": [
                {"failure_kind": "test_failed", "error_log": "FAILED prior attempt"}
            ]
        },
        **terminal_evidence,
    }

    assert classify_sample(sample)["decisive"] == "model_gateway_infra"


def test_final_attempt_overrides_historical_model_gateway_invocation():
    sample = {
        "id": "a",
        "success": False,
        "model_invocations": [
            {
                "node": "plan_fix",
                "status": "error",
                "error_class": "HTTPStatusError",
            }
        ],
        "agent_payload": {
            "fix_attempts": [
                {"failure_kind": "test_failed", "error_log": "FAILED test_widget"}
            ]
        },
    }

    classified = classify_sample(sample)

    assert classified["decisive"] == "test_failed"
    assert classified["attempts"] == ["test_failed"]


def test_terminal_error_overrides_historical_model_gateway_invocation():
    sample = {
        "id": "a",
        "success": False,
        "error": "No relevant files could be located or read.",
        "model_invocations": [
            {
                "node": "plan_fix",
                "status": "error",
                "error_class": "HTTPStatusError",
            }
        ],
    }

    assert classify_sample(sample)["decisive"] == "other"


@pytest.mark.parametrize(
    "invocation",
    [
        {"node": "commit", "status": "error", "error_class": "HTTPStatusError"},
        {"node": "plan_fix", "status": "ok", "error_class": "HTTPStatusError"},
        {"node": "plan_fix", "status": "error", "error_class": "ValidationError"},
    ],
)
def test_structured_model_gateway_detection_is_allowlisted(invocation):
    classified = classify_sample(
        {
            "id": "a",
            "success": False,
            "error": "Failed to generate fix plan: ValidationError",
            "model_invocations": [invocation],
        }
    )

    assert classified["decisive"] != "model_gateway_infra"


@pytest.mark.parametrize("value", ["true", "false", 1, 0, [], {}])
def test_taxonomy_does_not_score_non_boolean_official_result(value):
    classified = classify_sample(
        {"id": "a", "success": False, "official_resolved": value}
    )

    assert classified["official_resolved"] is None


def test_sample_no_attempts_prepatch_locate_failure():
    sample = {
        "id": "a",
        "success": False,
        "error": "No relevant files could be located or read.",
        "agent_payload": {"fix_attempts": []},
    }
    # Died before any patch — not a patch-stage failure.
    assert classify_sample(sample)["decisive"] == "other"


def test_sample_no_attempts_hallucination_gate_is_search_not_found():
    # The gate clears the patch in PLAN (no fix_attempt recorded), but the
    # failure_reason names it — must classify as search_not_found, not "other".
    sample = {
        "id": "a",
        "success": False,
        "error": "Planner kept emitting search blocks that do not exist in the target files.",
        "agent_payload": {"fix_attempts": []},
    }
    assert classify_sample(sample)["decisive"] == "search_not_found"


def test_summarize_distribution():
    results = [
        {
            "id": "1",
            "success": True,
            "agent_payload": {"fix_attempts": []},
            **verified_coverage(),
        },
        {
            "id": "2",
            "success": False,
            "agent_payload": {
                "fix_attempts": [{"failure_kind": "test_failed", "error_log": "FAILED"}]
            },
        },
        {
            "id": "3",
            "success": False,
            "agent_payload": {
                "fix_attempts": [
                    {
                        "failure_kind": "patch_apply_failed",
                        "error_log": "target file was not found: x.py",
                    }
                ]
            },
        },
    ]
    s = summarize(results)
    assert s["n_samples"] == 3
    assert s["agent_success"] == 1
    assert abs(s["agent_success_rate"] - 1 / 3) < 1e-6
    assert s["decisive"]["test_failed"] == 1
    assert s["decisive"]["wrong_file_path"] == 1
    assert s["decisive"]["agent_success"] == 1


async def test_verify_infrastructure_error_does_not_consume_retry_budget():
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        retry_count=2,
        max_retries=3,
        current_phase=new_agent.Phase.VERIFY,
        fix_attempts=[
            new_agent.FixAttempt(
                failure_kind="infra_error",
                error_log="sentinel runner unavailable",
            )
        ],
    )

    result = await new_agent.verify_fix(state)

    assert result.retry_count == 2
    assert result.current_phase == new_agent.Phase.FAILURE


async def test_verify_loaded_counted_attempt_is_idempotent(exact_repair_state):
    state = exact_repair_state
    attempt = _append_authorized_failure(
        state,
        failure_kind="test_failed",
        error_log="FAILED tests/test_widget.py::test_widget - assert False",
    )

    state = await new_agent.verify_fix(state)
    first = (
        state.retry_count,
        state.primary_failed_repair_rounds,
        state.last_counted_repair_round_id,
        state.current_phase,
        len(state.node_diagnostics),
    )
    loaded = new_agent.AgentState.model_validate(state.model_dump(mode="json"))

    loaded = await new_agent.verify_fix(loaded)

    assert attempt.repair_round_id > 0
    assert (
        loaded.retry_count,
        loaded.primary_failed_repair_rounds,
        loaded.last_counted_repair_round_id,
        loaded.current_phase,
        len(loaded.node_diagnostics),
    ) == first


async def test_verify_missing_historical_attribution_is_infrastructure():
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        current_phase=new_agent.Phase.VERIFY,
        fix_attempts=[
            new_agent.FixAttempt(
                patch_content="historical unbound patch",
                failure_kind="test_failed",
                error_log="assert False",
            )
        ],
    )

    result = await new_agent.verify_fix(state)

    assert result.current_phase == new_agent.Phase.FAILURE
    assert result.retry_count == 0
    assert result.primary_failed_repair_rounds == 0
    assert "state-integrity" in result.failure_reason.lower()


async def test_verify_migrates_historical_attempt_from_matching_authorization(
    exact_repair_state,
):
    state = exact_repair_state
    attempt = _append_authorized_failure(
        state,
        failure_kind="test_failed",
        error_log="FAILED tests/test_widget.py::test_widget - assert False",
        legacy_attribution=True,
    )
    expected = (
        state.authorized_repair_round_id,
        state.authorized_repair_provider,
        state.authorized_repair_model,
    )
    state.current_repair_round_id = 0
    state.current_repair_provider = None
    state.current_repair_model = ""

    result = await new_agent.verify_fix(state)

    assert (
        attempt.repair_round_id,
        attempt.repair_provider,
        attempt.repair_model,
    ) == expected
    assert result.retry_count == 1
    assert result.current_phase == new_agent.Phase.REFLECT


async def test_verify_rejects_mismatched_orphaned_current_author_on_restore(
    exact_repair_state,
):
    state = exact_repair_state
    _append_authorized_failure(
        state,
        failure_kind="test_failed",
        error_log="FAILED tests/test_widget.py::test_widget - assert False",
        legacy_attribution=True,
    )
    state.current_repair_round_id = 0
    state.current_repair_provider = "escalation"
    state.current_repair_model = "claude-opus-4-8:stable"

    result = await new_agent.verify_fix(state)

    assert result.current_phase == new_agent.Phase.FAILURE
    assert result.retry_count == 0
    assert result.primary_failed_repair_rounds == 0
    assert "state-integrity" in result.failure_reason.lower()
    assert result.current_repair_round_id == 0
    assert result.current_repair_provider == "escalation"
    assert result.current_repair_model == "claude-opus-4-8:stable"


async def test_invalid_then_failed_primary_patch_switches_before_next_plan(
    exact_repair_state, monkeypatch
):
    monkeypatch.setattr(model_policy, "escalation_is_configured", lambda: True)
    monkeypatch.setattr(
        model_policy,
        "get_model_config",
        lambda _provider: SimpleNamespace(model="claude-opus-4-8:stable"),
    )
    state = exact_repair_state
    first_round = begin_repair_round(state)
    bind_repair_round_author(state)
    record_failed_repair_round(
        state,
        round_id=first_round,
        provider="primary",
        model="gemini-3.5-flash:stable",
        failure_reason="invalid_edit",
        retry_phase=new_agent.Phase.PLAN,
    )
    _append_authorized_failure(
        state,
        failure_kind="test_failed",
        error_log="FAILED tests/test_widget.py::test_widget - assert False",
    )

    result = await new_agent.verify_fix(state)

    assert result.retry_count == 2
    assert result.primary_failed_repair_rounds == 2
    assert result.last_counted_repair_round_id == 2
    assert result.active_provider == "escalation"
    assert result.active_model == "claude-opus-4-8:stable"
    assert result.escalation_reason == "primary_repair_round_limit"


async def test_failed_escalation_attempt_spends_global_retry_not_primary_counter(
    exact_repair_state,
):
    state = exact_repair_state
    state.active_provider = "escalation"
    state.active_model = "claude-opus-4-8:stable"
    state.escalated = True
    _append_authorized_failure(
        state,
        failure_kind="test_failed",
        error_log="FAILED tests/test_widget.py::test_widget - assert False",
    )

    result = await new_agent.verify_fix(state)

    assert result.retry_count == 1
    assert result.primary_failed_repair_rounds == 0
    assert result.current_phase == new_agent.Phase.REFLECT


@pytest.mark.parametrize(
    ("failure_kind", "error_log", "expected_phase"),
    [
        ("test_failed", "SyntaxError: invalid syntax", new_agent.Phase.PLAN),
        ("test_failed", "ImportError: cannot import Widget", new_agent.Phase.PLAN),
        (
            "assertion_failure",
            "AssertionError: expected new-sentinel",
            new_agent.Phase.REFLECT,
        ),
        ("test_failed", "FAILED tests/test_widget.py", new_agent.Phase.REFLECT),
    ],
)
async def test_verify_noninfra_failure_classes_share_repair_ledger(
    exact_repair_state,
    failure_kind,
    error_log,
    expected_phase,
):
    state = exact_repair_state
    attempt = _append_authorized_failure(
        state,
        failure_kind=failure_kind,
        error_log=error_log,
    )

    result = await new_agent.verify_fix(state)

    assert result.current_phase == expected_phase
    assert result.retry_count == 1
    assert result.primary_failed_repair_rounds == 1
    assert result.last_counted_repair_round_id == attempt.repair_round_id


async def test_repeated_assertion_requires_diversity_without_early_opus_terminal(
    exact_repair_state,
):
    state = exact_repair_state
    state.max_retries = 4
    state.active_provider = "escalation"
    state.active_model = "claude-opus-4-8:stable"
    state.escalated = True

    for _ in range(3):
        _append_authorized_failure(
            state,
            failure_kind="assertion_failure",
            error_log="AssertionError: expected new-sentinel",
        )
        state = await new_agent.verify_fix(state)

    assert state.current_phase == new_agent.Phase.REFLECT
    assert state.retry_count == 3
    assert state.primary_failed_repair_rounds == 0
    assert state.assertion_diversity_required is True
    assert state.failure_reason != "repeated_assertion_no_progress"


async def test_verify_syntax_and_import_failures_route_directly_to_plan(
    exact_repair_state,
):
    state = exact_repair_state
    for expected_retry, error_log in enumerate(
        (
            "SyntaxError: invalid syntax at src/widget.py:4",
            "ImportError: cannot import name Widget",
            "ModuleNotFoundError: No module named widget",
        ),
        start=1,
    ):
        _append_authorized_failure(
            state,
            failure_kind="test_failed",
            error_log=error_log,
        )

        state = await new_agent.verify_fix(state)

        assert state.current_phase == new_agent.Phase.PLAN
        assert state.retry_count == expected_retry
        assert any(
            item.get("event") == "direct_patch_correction"
            for item in state.node_diagnostics
        )


async def test_repeated_unchanged_assertion_diversifies_without_early_terminal(
    exact_repair_state,
):
    state = exact_repair_state
    state.max_retries = 4

    for _ in range(3):
        _append_authorized_failure(
            state,
            failure_kind="assertion_failure",
            error_log="AssertionError: expected new-sentinel",
        )
        state = await new_agent.verify_fix(state)

    assert state.current_phase == new_agent.Phase.REFLECT
    assert state.retry_count == 3
    assert state.assertion_diversity_required is True
    assert state.failure_reason != "repeated_assertion_no_progress"


async def test_assertion_streak_survives_intervening_plan_progress(
    exact_repair_state,
):
    state = exact_repair_state
    state.max_retries = 4

    for _ in range(3):
        _append_authorized_failure(
            state,
            failure_kind="assertion_failure",
            error_log="AssertionError: expected new-sentinel",
        )
        state = await new_agent.verify_fix(state)
        record_progress(state)

    assert state.current_phase == new_agent.Phase.REFLECT
    assert state.assertion_diversity_required is True


async def test_repeated_syntax_failure_routes_to_plan_before_generic_replay_brake(
    exact_repair_state,
):
    state = exact_repair_state
    _append_authorized_failure(
        state,
        failure_kind="syntax_error",
        error_log="SyntaxError: invalid syntax",
    )

    state = await new_agent.verify_fix(state)

    assert state.current_phase == new_agent.Phase.PLAN
    assert state.failure_reason != "Same patch produced the same failure twice."


async def test_repeated_infra_failure_keeps_infra_semantics_without_retry():
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        retry_count=2,
        current_phase=new_agent.Phase.VERIFY,
        fix_attempts=[
            new_agent.FixAttempt(
                failure_kind="infra_error",
                error_log="worker unavailable",
            ),
            new_agent.FixAttempt(
                failure_kind="infra_error",
                error_log="worker unavailable",
            ),
        ],
    )

    state = await new_agent.verify_fix(state)

    assert state.current_phase == new_agent.Phase.FAILURE
    assert state.failure_reason.startswith("Infrastructure error during execution")
    assert state.retry_count == 2


async def test_assertion_signature_ignores_volatile_paths_lines_times_and_summary(
    exact_repair_state,
):
    state = exact_repair_state
    logs = [
        (
            "/private/tmp/run-a/repo/tests/test_widget.py:41: AssertionError\n"
            "FAILED tests/test_widget.py::test_value - AssertionError: expected 2 actual 1\n"
            "1 failed in 0.21s at 2026-07-21T10:11:12Z"
        ),
        (
            "/tmp/run-b/repo/tests/test_widget.py:99: AssertionError\n"
            "FAILED tests/test_widget.py::test_value - AssertionError: expected 2 actual 1\n"
            "=== short test summary info ===\n1 failed in 9.87s at 2027-01-02T03:04:05Z"
        ),
    ]

    for log in logs:
        _append_authorized_failure(
            state,
            failure_kind="assertion_failure",
            error_log=log,
        )
        state = await new_agent.verify_fix(state)
        record_progress(state)

    assert state.assertion_no_progress_rounds == 1
    assert state.assertion_diversity_required is True


async def test_assertion_signature_resets_when_failing_test_identity_changes(
    exact_repair_state,
):
    state = exact_repair_state
    for test_id in ("test_value", "test_other_value"):
        _append_authorized_failure(
            state,
            failure_kind="assertion_failure",
            error_log=(
                f"FAILED tests/test_widget.py::{test_id} - "
                "AssertionError: expected 2 actual 1"
            ),
        )
        state = await new_agent.verify_fix(state)
        record_progress(state)

    assert state.assertion_no_progress_rounds == 0
    assert state.assertion_diversity_required is False


async def test_assertion_signature_ignores_object_memory_addresses(
    exact_repair_state,
):
    state = exact_repair_state
    for address in ("0x7ffdeadbeef", "0x1a2b3c4d"):
        _append_authorized_failure(
            state,
            failure_kind="assertion_failure",
            error_log=(
                "FAILED tests/test_widget.py::test_identity - AssertionError: "
                f"expected <Widget object at {address}> actual ready"
            ),
        )
        state = await new_agent.verify_fix(state)
        record_progress(state)

    assert state.assertion_no_progress_rounds == 1
    assert state.assertion_diversity_required is True
