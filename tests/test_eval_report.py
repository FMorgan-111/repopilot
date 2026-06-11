from eval import report


def agent_v2_result():
    return {
        "id": "acme/widget#7:8",
        "mode": "agent_v2",
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


def test_generate_markdown_includes_agent_v2_replay_diagnostics():
    results = [agent_v2_result()]

    metrics = report.compute_metrics(results)
    markdown = report.generate_markdown(results, metrics)

    assert "| agent_v2_samples | 1 |" in markdown
    assert "| agent_v2_success_rate | 0.000 |" in markdown
    assert "| agent_v2_waiting_for_user | 0 |" in markdown
    assert "## Agent V2 Results" in markdown
    assert "| `acme/widget#7:8` | `abc123def456` | FAILED | no | 4 | 1234 | Patch failed tests. |" in markdown
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
