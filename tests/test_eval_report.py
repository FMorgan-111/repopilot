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
    assert "| agent_v2_combined_all_modes_resolved_rate | 0.000 |" in markdown
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
    explicit_end_to_end = agent_v2_result()
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
    oracle_success = agent_v2_result()
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
        "successes": 1,
        "success_rate": 0.5,
        "resolved_rate": 0.5,
        "failure_taxonomy": {"resolved": 1, "test_failed": 1},
        "models": ["a-model", "z-model"],
        "commits": ["commit-1", "commit-2"],
    }
    assert by_mode["oracle_files"] == {
        "samples": 2,
        "successes": 1,
        "success_rate": 0.5,
        "resolved_rate": 0.5,
        "failure_taxonomy": {"resolved": 1, "wrong_file_path": 1},
        "models": ["oracle-a", "oracle-b"],
        "commits": ["oracle-1", "oracle-2"],
    }
    assert "## Agent V2 Evaluation Modes" in markdown
    assert "| end_to_end | 2 | a-model, z-model | commit-1, commit-2 | 1/2 (50.0%) | resolved: 1; test_failed: 1 |" in markdown
    assert "| oracle_files | 2 | oracle-a, oracle-b | oracle-1, oracle-2 | 1/2 (50.0%) | resolved: 1; wrong_file_path: 1 |" in markdown
    assert "agent_v2_success_rate" not in markdown
    assert "agent_v2_combined_all_modes_resolved_rate" in markdown
    assert "| Sample ID | Evaluation Mode | Run ID |" in markdown
    assert "agent_v2 all modes combined: 2/4 resolved (50.0%)" in terminal
    assert (
        "agent_v2 end_to_end: samples=2 | models=a-model, z-model | "
        "commits=commit-1, commit-2 | resolved=1/2 (50.0%) | "
        "decisive taxonomy=resolved: 1; test_failed: 1"
    ) in terminal
    assert (
        "agent_v2 oracle_files: samples=2 | models=oracle-a, oracle-b | "
        "commits=oracle-1, oracle-2 | resolved=1/2 (50.0%) | "
        "decisive taxonomy=resolved: 1; wrong_file_path: 1"
    ) in terminal
    assert "agent_v2:     2/4 success" not in terminal
