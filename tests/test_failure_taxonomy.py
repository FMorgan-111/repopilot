"""Failure taxonomy classifier — the fine-grained mapping is the whole value,
so pin each category to a representative error log."""

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
    assert classify_sample({"id": "a", "success": True})["decisive"] == "resolved"


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
    assert s["resolved"] == 1
    assert abs(s["resolve_rate"] - 1 / 3) < 1e-6
    assert s["decisive"]["test_failed"] == 1
    assert s["decisive"]["wrong_file_path"] == 1
    assert s["decisive"]["resolved"] == 1


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
