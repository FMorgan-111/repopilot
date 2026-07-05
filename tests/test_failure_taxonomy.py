"""Failure taxonomy classifier — the fine-grained mapping is the whole value,
so pin each category to a representative error log."""

from eval.failure_taxonomy import (
    classify_attempt,
    classify_sample,
    summarize,
)


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
