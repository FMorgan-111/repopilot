import json

from eval import agent_v2_harness


def sample_record():
    return {
        "id": "acme/widget#7:8",
        "repo": {"owner": "acme", "name": "widget"},
        "issue": {
            "url": "https://github.com/acme/widget/issues/7",
            "title": "Login crash",
            "body": "The login endpoint crashes.",
        },
        "patch": {
            "files": [{"path": "src/auth.py"}],
        },
        "signals": {"has_tests_changed": True},
    }


async def test_evaluate_agent_v2_sample_saves_run_and_attaches_replay(monkeypatch):
    calls = []

    async def fake_agent_v2(issue_url, max_retries=3, token_budget=50000, save_final_run=False):
        calls.append(
            {
                "issue_url": issue_url,
                "max_retries": max_retries,
                "token_budget": token_budget,
                "save_final_run": save_final_run,
            }
        )
        return {
            "success": False,
            "waiting_for_user": False,
            "final_phase": "FAILED",
            "run_id": "abc123def456",
            "trace_id": "abc123def456",
            "error": "Patch failed tests.",
            "turns_taken": 4,
            "token_used": 1234,
            "decision_warnings": [{"frame_id": "df_0001"}],
        }

    def fake_replay_run(run_id):
        return {
            "run_id": run_id,
            "issue_url": "https://github.com/acme/widget/issues/7",
            "current_phase": "FAILED",
            "timeline": [
                {
                    "index": 1,
                    "type": "decision_frame",
                    "frame_id": "df_0001",
                    "stage": "reflect",
                    "summary": "Patch failed because the root cause was wrong.",
                    "recommended_action": "plan",
                    "route": {"route": "plan_fix"},
                    "warnings": [{"frame_id": "df_0001"}],
                }
            ],
        }

    monkeypatch.setattr(agent_v2_harness, "agent_v2", fake_agent_v2)
    monkeypatch.setattr(agent_v2_harness, "replay_run", fake_replay_run)

    result = await agent_v2_harness.evaluate_agent_v2_sample(
        sample_record(),
        idx=0,
        max_retries=1,
        token_budget=1000,
    )

    assert calls == [
        {
            "issue_url": "https://github.com/acme/widget/issues/7",
            "max_retries": 1,
            "token_budget": 1000,
            "save_final_run": True,
        }
    ]
    assert result == {
        "id": "acme/widget#7:8",
        "mode": "agent_v2",
        "repo": "acme/widget",
        "issue_url": "https://github.com/acme/widget/issues/7",
        "issue_title": "Login crash",
        "actual_files": ["src/auth.py"],
        "has_tests_changed": True,
        "success": False,
        "waiting_for_user": False,
        "final_phase": "FAILED",
        "run_id": "abc123def456",
        "trace_id": "abc123def456",
        "turns_taken": 4,
        "token_used": 1234,
        "error": "Patch failed tests.",
        "agent_payload": {
            "success": False,
            "waiting_for_user": False,
            "final_phase": "FAILED",
            "run_id": "abc123def456",
            "trace_id": "abc123def456",
            "error": "Patch failed tests.",
            "turns_taken": 4,
            "token_used": 1234,
            "decision_warnings": [{"frame_id": "df_0001"}],
        },
        "replay": {
            "run_id": "abc123def456",
            "issue_url": "https://github.com/acme/widget/issues/7",
            "current_phase": "FAILED",
            "timeline": [
                {
                    "index": 1,
                    "type": "decision_frame",
                    "frame_id": "df_0001",
                    "stage": "reflect",
                    "summary": "Patch failed because the root cause was wrong.",
                    "recommended_action": "plan",
                    "route": {"route": "plan_fix"},
                    "warnings": [{"frame_id": "df_0001"}],
                }
            ],
        },
        "replay_error": None,
    }


async def test_run_agent_v2_eval_writes_results(monkeypatch, tmp_path):
    samples = [sample_record()]

    async def fake_evaluate(sample, idx, max_retries=3, token_budget=50000):
        return {
            "id": sample["id"],
            "mode": "agent_v2",
            "run_id": "abc123def456",
            "success": True,
        }

    monkeypatch.setattr(agent_v2_harness, "load_samples", lambda n: samples[:n])
    monkeypatch.setattr(agent_v2_harness, "evaluate_agent_v2_sample", fake_evaluate)

    results_path = tmp_path / "agent_v2_results.json"
    results = await agent_v2_harness.run_agent_v2_eval(
        n_samples=1,
        max_retries=2,
        token_budget=2000,
        results_path=results_path,
    )

    assert results == [
        {
            "id": "acme/widget#7:8",
            "mode": "agent_v2",
            "run_id": "abc123def456",
            "success": True,
        }
    ]
    assert json.loads(results_path.read_text(encoding="utf-8")) == results


def test_harness_main_dispatches_agent_v2_mode(monkeypatch):
    from eval import harness

    calls = []

    async def fake_run_agent_v2_eval(
        n_samples=5,
        max_retries=3,
        token_budget=50000,
    ):
        calls.append(
            {
                "n_samples": n_samples,
                "max_retries": max_retries,
                "token_budget": token_budget,
            }
        )
        return []

    monkeypatch.setattr(harness, "run_agent_v2_eval", fake_run_agent_v2_eval)

    harness.main(
        [
            "--agent-v2",
            "--samples",
            "2",
            "--max-retries",
            "1",
            "--token-budget",
            "1000",
        ]
    )

    assert calls == [
        {
            "n_samples": 2,
            "max_retries": 1,
            "token_budget": 1000,
        }
    ]
