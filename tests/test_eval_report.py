import json
import subprocess
import sys
from pathlib import Path

from eval import report

ROOT = Path(__file__).resolve().parents[1]


def agent_v2_result():
    return {
        "id": "acme/widget#7:8",
        "mode": "agent_v2",
        "model": "claude-sonnet-5:stable",
        "commit_sha": "deadbeef",
        "repo": "acme/widget",
        "issue_url": "https://github.com/acme/widget/issues/7",
        "issue_title": "Login crash",
        "actual_files": ["src/auth.py"],
        "success": False,
        "waiting_for_user": False,
        "final_phase": "FAILED",
        "run_id": "abc123def456",
        "trace_id": "abc123def456",
        "turns_taken": 4,
        "token_used": 1234,
        "error": "Patch failed tests.",
        "replay": {
            "run_id": "abc123def456",
            "issue_url": "https://github.com/acme/widget/issues/7",
            "current_phase": "FAILED",
            "timeline": [
                {
                    "index": 1,
                    "type": "decision_frame",
                    "frame_id": "df_0001",
                    "stage": "plan",
                    "summary": "Patch auth submit handling.",
                    "selected_hypothesis_id": "H1",
                    "selected_hypothesis": {
                        "id": "H1",
                        "claim": "The crash is caused by missing auth validation.",
                    },
                    "recommended_action": "execute",
                    "risk": "medium",
                    "confidence": 0.7,
                    "route": {"route": "execute_fix"},
                    "warnings": [],
                    "next_checks": ["Run auth regression tests."],
                    "trace_notes": "",
                },
                {
                    "index": 2,
                    "type": "decision_frame",
                    "frame_id": "df_0002",
                    "stage": "reflect",
                    "summary": "Patch failed because the true root cause was session expiry.",
                    "selected_hypothesis_id": "H2",
                    "selected_hypothesis": {
                        "id": "H2",
                        "claim": "The root cause is stale session handling.",
                    },
                    "recommended_action": "plan",
                    "risk": "high",
                    "confidence": 0.61,
                    "route": {"route": "plan_fix"},
                    "warnings": [
                        {
                            "frame_id": "df_0002",
                            "expected_phase": "PLAN",
                            "actual_phase": "REFLECT",
                        }
                    ],
                    "next_checks": ["Inspect session refresh middleware."],
                    "trace_notes": "",
                },
            ],
        },
        "replay_error": None,
    }


def success_first_result():
    result = agent_v2_result()
    result.update(
        {
            "id": "acme/widget#11:12",
            "success": True,
            "agent_success": True,
            "official_resolved": None,
            "models_used": [
                "gemini-3.5-flash:stable",
                "claude-opus-4-8:stable",
            ],
            "escalated": True,
            "escalation_reason": "repeated_edit",
            "model_invocations": [
                {"model": "gemini-3.5-flash:stable", "status": "ok"},
                {"model": "claude-opus-4-8:stable", "status": "ok"},
            ],
            "tool_invocations": [
                {"action": "read_symbol", "status": "ok"},
                {"action": "read_symbol", "status": "duplicate"},
                {"action": "run_targeted_test", "status": "rejected"},
            ],
            "unique_evidence_count": 2,
            "max_consecutive_no_progress": 2,
            "attempt_outcome_summary": "Verified a focused regression.",
            "coverage_status": "generated_verified",
            "coverage_test_files": ["tests/test_auth.py"],
            "coverage_test_command": "pytest tests/test_auth.py",
            "coverage_proof": {
                "source": "generated",
                "status": "generated_verified",
                "test_files": ["tests/test_auth.py"],
                "fixed_runs": [
                    {"outcome": "pass", "failing_test_ids": [], "assertion_fingerprint": ""},
                    {"outcome": "pass", "failing_test_ids": [], "assertion_fingerprint": ""},
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
            "coverage_failure_reason": "",
            "test_generation_attempts": 1,
        }
    )
    return result


def test_report_imports_when_run_from_eval_directory():
    subprocess.run(
        [sys.executable, "-c", "import report"],
        cwd=ROOT / "eval",
        check=True,
        capture_output=True,
        text=True,
    )


def test_generate_markdown_includes_agent_v2_replay_diagnostics():
    results = [agent_v2_result()]

    metrics = report.compute_metrics(results)
    markdown = report.generate_markdown(results, metrics)

    assert "| agent_v2_samples | 1 |" in markdown
    assert "| agent_v2_combined_all_modes_agent_success_rate | 0.000 |" in markdown
    assert "| official_resolved_rate | N/A (not scored) |" in markdown
    assert "| agent_v2_waiting_for_user | 0 |" in markdown
    assert "**Models**: claude-sonnet-5:stable" in markdown
    assert "**Commits**: deadbeef" in markdown
    assert "## Agent V2 Results" in markdown
    assert "| `acme/widget#7:8` | end_to_end | `abc123def456` | FAILED | no | 4 | 1234 | Patch failed tests. |" in markdown
    assert "## Replay Diagnostics" in markdown
    assert "### acme/widget#7:8 (`abc123def456`)" in markdown
    assert "- Final phase: FAILED" in markdown
    assert "- Latest frame: reflect `df_0002`" in markdown
    assert "- Selected hypothesis: H2" in markdown
    assert "- Hypothesis claim: The root cause is stale session handling." in markdown
    assert "- Recommended action: plan" in markdown
    assert "- Actual route: plan_fix" in markdown
    assert "- Warning: expected PLAN but actual REFLECT" in markdown
    assert "- Next check: Inspect session refresh middleware." in markdown


def test_generate_markdown_includes_agent_v2_node_diagnostics_without_decision_frame():
    result = agent_v2_result()
    result["id"] = "acme/widget#9:10"
    result["run_id"] = "node123"
    result["trace_id"] = "node123"
    result["final_phase"] = "FAILED"
    result["error"] = "Replay failed during validation."
    result["replay"] = {
        "run_id": "node123",
        "issue_url": "https://github.com/acme/widget/issues/9",
        "current_phase": "FAILED",
        "timeline": [
            {
                "index": 1,
                "type": "node_diagnostic",
                "diagnostic": {
                    "node": "plan_fix",
                    "event": "llm_call",
                    "status": "success",
                    "elapsed_seconds": 88.647,
                    "prompt_tokens_estimate": 2286,
                    "response_tokens_estimate": 1226,
                },
            },
            {
                "index": 2,
                "type": "node_diagnostic",
                "diagnostic": {
                    "node": "phase",
                    "event": "advance",
                    "status": "error",
                    "error_type": "ValidationError",
                    "error": "Invalid phase transition.",
                },
            },
        ],
    }

    metrics = report.compute_metrics([result])
    markdown = report.generate_markdown([result], metrics)

    assert "### acme/widget#9:10 (`node123`)" in markdown
    assert "- Latest frame: none" in markdown
    assert "#### Node Diagnostics" in markdown
    assert "| Node | Event | Status | Error Type | Error |" in markdown
    assert "| `plan_fix` | llm_call | success |  |  |" in markdown
    assert "| `phase` | advance | error | ValidationError | Invalid phase transition. |" in markdown


def test_generate_markdown_surfaces_plan_fix_phase_timeout():
    result = agent_v2_result()
    result["id"] = "tox-dev/tox#3075:3748"
    result["run_id"] = "abc123"
    result["trace_id"] = "abc123"
    result["error"] = "Phase plan_fix timed out after 150.0s"
    result["token_used"] = 5601
    result["turns_taken"] = 14
    result["replay"] = {
        "current_phase": "FAILED",
        "timeline": [
            {
                "type": "node_diagnostic",
                "diagnostic": {
                    "node": "plan_fix",
                    "event": "phase",
                    "status": "timeout",
                    "error_type": "TimeoutError",
                    "error": "TimeoutError",
                    "phase_timeout_seconds": 150.0,
                },
            }
        ],
    }

    markdown = report.generate_markdown(
        [result],
        report.compute_metrics([result]),
    )

    assert "Planner timeout" in markdown
    assert "plan_fix exceeded 150.0s" in markdown


def test_success_first_metrics_aggregate_models_tools_escalation_and_coverage():
    result = success_first_result()

    metrics = report.compute_metrics([result])

    agent = metrics["agent_v2"]
    assert agent["agent_successes"] == 1
    assert agent["official_scored_samples"] == 0
    assert agent["official_resolved"] == 0
    assert agent["by_model"] == {
        "claude-opus-4-8:stable": {
            "samples": 1,
            "invocations": 1,
            "agent_successes": 1,
        },
        "gemini-3.5-flash:stable": {
            "samples": 1,
            "invocations": 1,
            "agent_successes": 1,
        },
    }
    assert agent["escalation_reasons"] == {"repeated_edit": 1}
    assert agent["tool_statuses"] == {"duplicate": 1, "ok": 1, "rejected": 1}
    assert agent["unique_evidence_count"] == 2
    assert agent["max_consecutive_no_progress"] == 2
    assert agent["coverage"] == {
        "statuses": {"generated_verified": 1},
        "proofs": 1,
        "test_generation_attempts": 1,
    }


def test_report_loads_legacy_result_with_safe_defaults(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps([{"id": "legacy", "mode": "agent_v2", "success": False}]),
        encoding="utf-8",
    )

    [loaded] = report.load_results(path)

    assert loaded["models_used"] == []
    assert loaded["model_invocations"] == []
    assert loaded["tool_invocations"] == []
    assert loaded["unique_evidence_count"] == 0
    assert loaded["max_consecutive_no_progress"] == 0
    assert loaded["attempt_outcome_summary"] == ""
    assert loaded["coverage_status"] == "pending"
    assert loaded["coverage_test_files"] == []
    assert loaded["coverage_test_command"] == ""
    assert loaded["coverage_proof"] is None
    assert loaded["coverage_failure_reason"] == ""
    assert loaded["test_generation_attempts"] == 0
    assert loaded["official_resolved"] is None


def test_report_cli_forwards_result_and_summary_paths(tmp_path):
    results_path = tmp_path / "results.json"
    summary_path = tmp_path / "summary.md"
    results_path.write_text(json.dumps([success_first_result()]), encoding="utf-8")

    report.main(
        [
            "--results-file",
            str(results_path),
            "--summary-file",
            str(summary_path),
        ]
    )

    assert summary_path.exists()
    assert "RepoPilot Eval Summary" in summary_path.read_text(encoding="utf-8")


def test_markdown_renders_only_safe_coverage_proof_fields():
    result = success_first_result()
    result["coverage_proof"]["raw_logs"] = "secret full raw test logs"

    markdown = report.generate_markdown([result], report.compute_metrics([result]))

    assert "tests/test_auth.py" in markdown
    assert "assertion_failure" in markdown
    assert "tests/test_auth.py::test_login" in markdown
    assert "a" * 64 in markdown
    assert "secret full raw test logs" not in markdown


def test_markdown_redacts_legacy_credentials_evaluator_and_archive_payloads():
    result = agent_v2_result()
    result["model"] = "sk-test-secret-123456789"
    result["models_used"] = ["sk-test-secret-123456789"]
    result["commit_sha"] = "Bearer sk-test-secret-123456789"
    result["error"] = (
        "Bearer sk-test-secret-123456789 FAIL_TO_PASS "
        "setuptools-33.1.1.zip PK\\x03\\x04 generated bytes"
    )
    result["replay"]["timeline"][0]["summary"] = result["error"]

    markdown = report.generate_markdown([result], report.compute_metrics([result]))

    assert "sk-test-secret-123456789" not in markdown
    assert "FAIL_TO_PASS" not in markdown
    assert "PK\\x03\\x04 generated bytes" not in markdown


def test_agent_v2_metrics_and_reports_separate_evaluation_modes(capsys):
    legacy = agent_v2_result()
    legacy.update(
        {
            "model": "z-model",
            "commit_sha": "commit-2",
            "agent_payload": {
                "fix_attempts": [
                    {
                        "failure_kind": "test_failed",
                        "error_log": "assertion failed",
                    }
                ]
            },
        }
    )
    explicit_end_to_end = success_first_result()
    explicit_end_to_end.update(
        {
            "id": "acme/widget#8:9",
            "evaluation_mode": "end_to_end",
            "success": True,
            "model": "a-model",
            "commit_sha": "commit-1",
        }
    )
    oracle = agent_v2_result()
    oracle.update(
        {
            "id": "acme/widget#10:11",
            "evaluation_mode": "oracle_files",
            "model": "oracle-b",
            "commit_sha": "oracle-2",
            "agent_payload": {
                "fix_attempts": [
                    {
                        "failure_kind": "patch_apply_failed",
                        "error_log": "target file was not found",
                    }
                ]
            },
        }
    )
    oracle_success = success_first_result()
    oracle_success.update(
        {
            "id": "acme/widget#12:13",
            "evaluation_mode": "oracle_files",
            "success": True,
            "model": "oracle-a",
            "commit_sha": "oracle-1",
        }
    )
    results = [legacy, explicit_end_to_end, oracle, oracle_success]

    metrics = report.compute_metrics(results)
    markdown = report.generate_markdown(results, metrics)
    report.print_summary(metrics)
    terminal = capsys.readouterr().out

    by_mode = metrics["agent_v2"]["by_evaluation_mode"]
    assert by_mode["end_to_end"] == {
        "samples": 2,
        "agent_successes": 1,
        "agent_success_rate": 0.5,
        "official_scored_samples": 0,
        "official_resolved": 0,
        "official_resolved_rate": None,
        "successes": 1,
        "success_rate": 0.5,
        "failure_taxonomy": {"agent_success": 1, "test_failed": 1},
        "models": ["a-model", "z-model"],
        "commits": ["commit-1", "commit-2"],
    }
    assert by_mode["oracle_files"] == {
        "samples": 2,
        "agent_successes": 1,
        "agent_success_rate": 0.5,
        "official_scored_samples": 0,
        "official_resolved": 0,
        "official_resolved_rate": None,
        "successes": 1,
        "success_rate": 0.5,
        "failure_taxonomy": {"agent_success": 1, "wrong_file_path": 1},
        "models": ["oracle-a", "oracle-b"],
        "commits": ["oracle-1", "oracle-2"],
    }
    assert "## Agent V2 Evaluation Modes" in markdown
    assert "| end_to_end | 2 | a-model, z-model | commit-1, commit-2 | 1/2 (50.0%) | not scored | agent_success: 1; test_failed: 1 |" in markdown
    assert "| oracle_files | 2 | oracle-a, oracle-b | oracle-1, oracle-2 | 1/2 (50.0%) | not scored | agent_success: 1; wrong_file_path: 1 |" in markdown
    assert "agent_v2_combined_all_modes_agent_success_rate" in markdown
    assert "official_resolved_rate | N/A (not scored)" in markdown
    assert "| Sample ID | Evaluation Mode | Run ID |" in markdown
    assert "agent_v2 all modes combined: 2/4 agent_success (50.0%)" in terminal
    assert "official resolved: not scored" in terminal
    assert (
        "agent_v2 end_to_end: samples=2 | models=a-model, z-model | "
        "commits=commit-1, commit-2 | agent_success=1/2 (50.0%) | "
        "decisive taxonomy=agent_success: 1; test_failed: 1"
    ) in terminal
    assert (
        "agent_v2 oracle_files: samples=2 | models=oracle-a, oracle-b | "
        "commits=oracle-1, oracle-2 | agent_success=1/2 (50.0%) | "
        "decisive taxonomy=agent_success: 1; wrong_file_path: 1"
    ) in terminal
    assert "agent_v2:     2/4 success" not in terminal
