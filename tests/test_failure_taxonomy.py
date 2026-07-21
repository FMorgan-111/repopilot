"""Failure taxonomy classifier — the fine-grained mapping is the whole value,
so pin each category to a representative error log."""

import pytest

from eval.failure_taxonomy import (
    classify_attempt,
    classify_sample,
    summarize,
)
from src import new_agent
from src.model_policy import record_progress
from src.state import PatchEdit


def test_wrong_file_path():
    assert classify_attempt(
        "patch_apply_failed",
        "Search/replace edit failed: edit 1 target file was not found: lib/x.py.",
    ) == "wrong_file_path"


def test_empty_patch_not_invalid_diff():
    # "No valid patches in input" is git apply on an EMPTY patch (a gate cleared
    # the edits), NOT the model emitting a bad diff. Must not inflate invalid_diff.
    assert classify_attempt(
        "patch_apply_failed",
        "Patch preflight check failed:\nerror: No valid patches in input",
    ) == "empty_patch"


def test_invalid_diff_real_hunks():
    assert classify_attempt(
        "patch_apply_failed",
        "Patch preflight check failed:\ndiff --git a/x b/x\n@@ -1,3 +1,4 @@\ncorrupt",
    ) == "invalid_diff"


def test_search_not_found():
    assert classify_attempt(
        "patch_apply_failed",
        "Search/replace edit failed: edit 1 search block was not found in a.py.",
    ) == "search_not_found"


def test_test_failed():
    assert classify_attempt(
        "test_failed", "===== test session starts =====\nFAILED tests/test_x.py"
    ) == "test_failed"


def test_infra_timeout():
    assert classify_attempt("", "httpx.ReadTimeout") == "infra"
    assert classify_attempt("infra_error", "Infrastructure error during execution") == "infra"


def test_budget():
    assert classify_attempt("", "Token budget exceeded during verification.") == "budget"


def test_sample_decisive_is_last_attempt():
    sample = {
        "id": "x/y#1",
        "success": False,
        "agent_payload": {
            "fix_attempts": [
                {"failure_kind": "patch_apply_failed",
                 "error_log": "search block was not found in a.py"},
                {"failure_kind": "test_failed", "error_log": "FAILED test_x"},
            ]
        },
    }
    c = classify_sample(sample)
    assert c["decisive"] == "test_failed"                 # last attempt wins
    assert c["attempts"] == ["search_not_found", "test_failed"]


def test_sample_resolved():
    assert classify_sample({"id": "a", "success": True})["decisive"] == "agent_success"


def test_official_resolved_is_not_inferred_from_agent_success():
    classified = classify_sample(
        {"id": "a", "success": True, "official_resolved": None}
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
        ({"coverage_failure_reason": "test_generation_failed"}, "test_generation_failed"),
    ],
)
def test_success_first_terminal_failure_labels(sample, expected):
    sample = {"id": "a", "success": False, **sample}

    assert classify_sample(sample)["decisive"] == expected


def test_sample_no_attempts_prepatch_locate_failure():
    sample = {
        "id": "a", "success": False,
        "error": "No relevant files could be located or read.",
        "agent_payload": {"fix_attempts": []},
    }
    # Died before any patch — not a patch-stage failure.
    assert classify_sample(sample)["decisive"] == "other"


def test_sample_no_attempts_hallucination_gate_is_search_not_found():
    # The gate clears the patch in PLAN (no fix_attempt recorded), but the
    # failure_reason names it — must classify as search_not_found, not "other".
    sample = {
        "id": "a", "success": False,
        "error": "Planner kept emitting search blocks that do not exist in the target files.",
        "agent_payload": {"fix_attempts": []},
    }
    assert classify_sample(sample)["decisive"] == "search_not_found"


def test_summarize_distribution():
    results = [
        {"id": "1", "success": True, "agent_payload": {"fix_attempts": []}},
        {"id": "2", "success": False, "agent_payload": {"fix_attempts": [
            {"failure_kind": "test_failed", "error_log": "FAILED"}]}},
        {"id": "3", "success": False, "agent_payload": {"fix_attempts": [
            {"failure_kind": "patch_apply_failed",
             "error_log": "target file was not found: x.py"}]}},
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


async def test_verify_syntax_and_import_failures_route_directly_to_plan():
    for error_log in (
        "SyntaxError: invalid syntax at src/widget.py:4",
        "ImportError: cannot import name Widget",
        "ModuleNotFoundError: No module named widget",
    ):
        state = new_agent.AgentState(
            issue_url="https://github.com/acme/widget/issues/7",
            retry_count=0,
            max_retries=3,
            current_phase=new_agent.Phase.VERIFY,
            fix_attempts=[
                new_agent.FixAttempt(
                    patch_edits=[
                        PatchEdit(
                            file_path="src/widget.py",
                            search="old-sentinel",
                            replace="new-sentinel",
                        )
                    ],
                    failure_kind="test_failed",
                    error_log=error_log,
                )
            ],
        )

        result = await new_agent.verify_fix(state)

        assert result.current_phase == new_agent.Phase.PLAN
        assert result.retry_count == 1
        assert result.node_diagnostics[-1]["event"] == "direct_patch_correction"


async def test_repeated_unchanged_assertion_diversifies_once_then_terminates():
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        retry_count=0,
        max_retries=5,
        current_phase=new_agent.Phase.VERIFY,
    )

    def assertion_attempt(symbol: str):
        return new_agent.FixAttempt(
            patch_edits=[
                PatchEdit(
                    file_path="src/widget.py",
                    node_target=symbol,
                    replace=f"def {symbol}():\n    return 'wrong'\n",
                )
            ],
            failure_kind="assertion_failure",
            error_log="AssertionError: expected new-sentinel",
        )

    state.fix_attempts.append(assertion_attempt("first_target"))
    state = await new_agent.verify_fix(state)
    assert state.current_phase == new_agent.Phase.REFLECT
    assert state.no_progress_rounds == 0

    state.current_phase = new_agent.Phase.VERIFY
    state.fix_attempts.append(assertion_attempt("second_target"))
    state = await new_agent.verify_fix(state)
    assert state.current_phase == new_agent.Phase.REFLECT
    assert state.no_progress_rounds == 1
    assert state.node_diagnostics[-1]["event"] == "assertion_diversity_required"

    state.current_phase = new_agent.Phase.VERIFY
    state.fix_attempts.append(assertion_attempt("third_target"))
    state = await new_agent.verify_fix(state)
    assert state.current_phase == new_agent.Phase.FAILURE
    assert state.failure_reason == "repeated_assertion_no_progress"


async def test_assertion_streak_survives_intervening_plan_progress():
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        max_retries=5,
        current_phase=new_agent.Phase.VERIFY,
    )

    def attempt(symbol: str):
        return new_agent.FixAttempt(
            patch_edits=[
                PatchEdit(
                    file_path="src/widget.py",
                    node_target=symbol,
                    replace=f"def {symbol}():\n    return 'wrong'\n",
                )
            ],
            failure_kind="assertion_failure",
            error_log="AssertionError: expected new-sentinel",
        )

    for symbol in ("first_target", "second_target", "third_target"):
        state.current_phase = new_agent.Phase.VERIFY
        state.fix_attempts.append(attempt(symbol))
        state = await new_agent.verify_fix(state)
        if symbol != "third_target":
            record_progress(state)  # successful PLAN result before EXECUTE/VERIFY

    assert state.current_phase == new_agent.Phase.FAILURE
    assert state.failure_reason == "repeated_assertion_no_progress"


async def test_repeated_syntax_failure_routes_to_plan_before_generic_replay_brake():
    edit = PatchEdit(
        file_path="src/widget.py",
        search="return old",
        replace="return new",
    )
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        max_retries=5,
        current_phase=new_agent.Phase.VERIFY,
        fix_attempts=[
            new_agent.FixAttempt(
                patch_edits=[edit],
                failure_kind="syntax_error",
                error_log="SyntaxError: invalid syntax",
            ),
            new_agent.FixAttempt(
                patch_edits=[edit],
                failure_kind="syntax_error",
                error_log="SyntaxError: invalid syntax",
            ),
        ],
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


async def test_assertion_signature_ignores_volatile_paths_lines_times_and_summary():
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        max_retries=5,
        current_phase=new_agent.Phase.VERIFY,
    )
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
        state.current_phase = new_agent.Phase.VERIFY
        state.fix_attempts.append(
            new_agent.FixAttempt(
                patch_edits=[
                    PatchEdit(
                        file_path="src/widget.py",
                        node_target="widget",
                        replace="def widget():\n    return 1\n",
                    )
                ],
                failure_kind="assertion_failure",
                error_log=log,
            )
        )
        state = await new_agent.verify_fix(state)
        record_progress(state)

    assert state.assertion_no_progress_rounds == 1
    assert state.assertion_diversity_required is True


async def test_assertion_signature_resets_when_failing_test_identity_changes():
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        max_retries=5,
        current_phase=new_agent.Phase.VERIFY,
    )
    for test_id in ("test_value", "test_other_value"):
        state.current_phase = new_agent.Phase.VERIFY
        state.fix_attempts.append(
            new_agent.FixAttempt(
                failure_kind="assertion_failure",
                error_log=(
                    f"FAILED tests/test_widget.py::{test_id} - "
                    "AssertionError: expected 2 actual 1"
                ),
            )
        )
        state = await new_agent.verify_fix(state)
        record_progress(state)

    assert state.assertion_no_progress_rounds == 0
    assert state.assertion_diversity_required is False


async def test_assertion_signature_ignores_object_memory_addresses():
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        max_retries=5,
        current_phase=new_agent.Phase.VERIFY,
    )
    for address in ("0x7ffdeadbeef", "0x1a2b3c4d"):
        state.current_phase = new_agent.Phase.VERIFY
        state.fix_attempts.append(
            new_agent.FixAttempt(
                failure_kind="assertion_failure",
                error_log=(
                    "FAILED tests/test_widget.py::test_identity - AssertionError: "
                    f"expected <Widget object at {address}> actual ready"
                ),
            )
        )
        state = await new_agent.verify_fix(state)
        record_progress(state)

    assert state.assertion_no_progress_rounds == 1
    assert state.assertion_diversity_required is True
