import json

import pytest

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

    async def fake_agent_v2(
        issue_url,
        max_retries=3,
        token_budget=50000,
        save_final_run=False,
        skip_commit=False,
        seed=None,
    ):
        calls.append(
            {
                "issue_url": issue_url,
                "max_retries": max_retries,
                "token_budget": token_budget,
                "save_final_run": save_final_run,
                "skip_commit": skip_commit,
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
            "skip_commit": True,
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

    async def fake_evaluate(sample, idx, max_retries=3, token_budget=50000,
                            seed_gold_files=False):
        return {
            "id": sample["id"],
            "mode": "agent_v2",
            "run_id": "abc123def456",
            "success": True,
        }

    monkeypatch.setattr(agent_v2_harness, "load_samples", lambda n, sample_id=None, **kw: samples[:n])
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


async def test_run_agent_v2_eval_closes_memory_store_after_success(
    monkeypatch, tmp_path
):
    calls = []

    async def fake_evaluate(sample, idx, max_retries=3, token_budget=50000,
                            seed_gold_files=False):
        return {
            "id": sample["id"],
            "mode": "agent_v2",
            "run_id": "abc123def456",
            "success": True,
        }

    async def fake_close_llm_client():
        calls.append("llm")

    async def fake_close_store():
        calls.append("memory")

    monkeypatch.setattr(agent_v2_harness, "load_samples", lambda n, sample_id=None, **kw: [sample_record()])
    monkeypatch.setattr(agent_v2_harness, "evaluate_agent_v2_sample", fake_evaluate)
    monkeypatch.setattr(agent_v2_harness, "close_llm_client", fake_close_llm_client)
    monkeypatch.setattr(agent_v2_harness, "close_store", fake_close_store, raising=False)

    results = await agent_v2_harness.run_agent_v2_eval(
        n_samples=1,
        results_path=tmp_path / "agent_v2_results.json",
    )

    assert results == [
        {
            "id": "acme/widget#7:8",
            "mode": "agent_v2",
            "run_id": "abc123def456",
            "success": True,
        }
    ]
    assert calls == ["llm", "memory"]


async def test_run_agent_v2_eval_falls_back_when_results_path_write_fails(
    monkeypatch, tmp_path
):
    samples = [sample_record()]

    async def fake_evaluate(sample, idx, max_retries=3, token_budget=50000,
                            seed_gold_files=False):
        return {
            "id": sample["id"],
            "mode": "agent_v2",
            "run_id": "abc123def456",
            "success": True,
        }

    async def fake_close_llm_client():
        return None

    monkeypatch.setattr(agent_v2_harness, "load_samples", lambda n, sample_id=None, **kw: samples[:n])
    monkeypatch.setattr(agent_v2_harness, "evaluate_agent_v2_sample", fake_evaluate)
    monkeypatch.setattr(agent_v2_harness, "close_llm_client", fake_close_llm_client)
    repopilot_home = tmp_path / "home"
    monkeypatch.setenv("REPOPILOT_HOME", str(repopilot_home))

    requested_path = tmp_path / "readonly" / "eval_results.json"
    original_write_text = agent_v2_harness.Path.write_text

    def fail_requested_path(self, data, *args, **kwargs):
        if self == requested_path:
            raise OSError("read-only eval results")
        return original_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(agent_v2_harness.Path, "write_text", fail_requested_path)

    results = await agent_v2_harness.run_agent_v2_eval(
        n_samples=1,
        results_path=requested_path,
    )

    assert results == [
        {
            "id": "acme/widget#7:8",
            "mode": "agent_v2",
            "run_id": "abc123def456",
            "success": True,
        }
    ]
    fallback_path = repopilot_home / "eval" / "eval_results.json"
    assert json.loads(fallback_path.read_text(encoding="utf-8")) == results
    assert not requested_path.exists()


async def test_run_agent_v2_eval_closes_shared_resources_when_sample_raises(
    monkeypatch, tmp_path
):
    calls = []

    async def fake_evaluate(sample, idx, max_retries=3, token_budget=50000,
                            seed_gold_files=False):
        raise RuntimeError("sample failed after partial work")

    async def fake_close_llm_client():
        calls.append("llm")

    async def fake_close_store():
        calls.append("memory")

    monkeypatch.setattr(agent_v2_harness, "load_samples", lambda n, sample_id=None, **kw: [sample_record()])
    monkeypatch.setattr(agent_v2_harness, "evaluate_agent_v2_sample", fake_evaluate)
    monkeypatch.setattr(agent_v2_harness, "close_llm_client", fake_close_llm_client)
    monkeypatch.setattr(agent_v2_harness, "close_store", fake_close_store, raising=False)

    with pytest.raises(RuntimeError, match="sample failed after partial work"):
        await agent_v2_harness.run_agent_v2_eval(
            n_samples=1,
            results_path=tmp_path / "agent_v2_results.json",
        )

    assert calls == ["llm", "memory"]


async def test_run_agent_v2_eval_does_not_mask_results_when_cleanup_raises(
    monkeypatch, tmp_path
):
    async def fake_evaluate(sample, idx, max_retries=3, token_budget=50000,
                            seed_gold_files=False):
        return {
            "id": sample["id"],
            "mode": "agent_v2",
            "run_id": "abc123def456",
            "success": True,
        }

    async def fake_close_llm_client():
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(agent_v2_harness, "load_samples", lambda n, sample_id=None, **kw: [sample_record()])
    monkeypatch.setattr(agent_v2_harness, "evaluate_agent_v2_sample", fake_evaluate)
    monkeypatch.setattr(agent_v2_harness, "close_llm_client", fake_close_llm_client)

    results_path = tmp_path / "agent_v2_results.json"
    results = await agent_v2_harness.run_agent_v2_eval(
        n_samples=1,
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


async def test_run_agent_v2_eval_does_not_mask_results_when_memory_cleanup_raises(
    monkeypatch, tmp_path, capsys
):
    async def fake_evaluate(sample, idx, max_retries=3, token_budget=50000,
                            seed_gold_files=False):
        return {
            "id": sample["id"],
            "mode": "agent_v2",
            "run_id": "abc123def456",
            "success": True,
        }

    async def fake_close_llm_client():
        return None

    async def fake_close_store():
        raise RuntimeError("memory cleanup failed")

    monkeypatch.setattr(agent_v2_harness, "load_samples", lambda n, sample_id=None, **kw: [sample_record()])
    monkeypatch.setattr(agent_v2_harness, "evaluate_agent_v2_sample", fake_evaluate)
    monkeypatch.setattr(agent_v2_harness, "close_llm_client", fake_close_llm_client)
    monkeypatch.setattr(agent_v2_harness, "close_store", fake_close_store, raising=False)

    results_path = tmp_path / "agent_v2_results.json"
    results = await agent_v2_harness.run_agent_v2_eval(
        n_samples=1,
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
    captured = capsys.readouterr()
    assert "Warning: failed to close shared memory store: RuntimeError: memory cleanup failed" in captured.err


def test_harness_main_dispatches_agent_v2_mode(monkeypatch):
    from eval import harness

    calls = []

    async def fake_run_agent_v2_eval(
        n_samples=5,
        max_retries=3,
        token_budget=50000,
        sample_id=None,
        seed_gold_files=False,
    ):
        calls.append(
            {
                "n_samples": n_samples,
                "max_retries": max_retries,
                "token_budget": token_budget,
                "sample_id": sample_id,
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
            "sample_id": None,
        }
    ]


async def test_harness_run_agent_v2_eval_forwards_sample_id(monkeypatch):
    from eval import harness

    calls = []

    class FakeAgentV2Harness:
        async def run_agent_v2_eval(
            self,
            n_samples=5,
            max_retries=3,
            token_budget=50000,
            sample_id=None,
            seed_gold_files=False,
        ):
            calls.append(
                {
                    "n_samples": n_samples,
                    "max_retries": max_retries,
                    "token_budget": token_budget,
                    "sample_id": sample_id,
                }
            )
            return [{"id": sample_id}]

    monkeypatch.setattr(
        harness.importlib,
        "import_module",
        lambda name: FakeAgentV2Harness(),
    )

    results = await harness.run_agent_v2_eval(
        n_samples=2,
        max_retries=1,
        token_budget=1000,
        sample_id="scrapy/scrapy#6195:7095",
    )

    assert results == [{"id": "scrapy/scrapy#6195:7095"}]
    assert calls == [
        {
            "n_samples": 2,
            "max_retries": 1,
            "token_budget": 1000,
            "sample_id": "scrapy/scrapy#6195:7095",
        }
    ]


def test_load_samples_filters_by_sample_id(monkeypatch, tmp_path):
    # Build a 3-record dataset; sample_id must select the right one regardless
    # of position, ignoring the positional n.
    dataset = tmp_path / "issues_fixes.jsonl"
    records = [
        {"id": "a/a#1:1", "issue": {"title": "first"}},
        {"id": "b/b#2:2", "issue": {"title": "second"}},
        {"id": "c/c#3:3", "issue": {"title": "third"}},
    ]
    dataset.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    monkeypatch.setattr(agent_v2_harness, "SAMPLES_PATH", dataset)

    got = agent_v2_harness.load_samples(1, sample_id="c/c#3:3")

    assert len(got) == 1
    assert got[0]["id"] == "c/c#3:3"


def test_load_samples_raises_for_unknown_sample_id(monkeypatch, tmp_path):
    dataset = tmp_path / "issues_fixes.jsonl"
    dataset.write_text(json.dumps({"id": "a/a#1:1"}) + "\n")
    monkeypatch.setattr(agent_v2_harness, "SAMPLES_PATH", dataset)

    with pytest.raises(ValueError, match="sample_id not found"):
        agent_v2_harness.load_samples(5, sample_id="missing/x#9:9")


def test_load_samples_positional_when_no_sample_id(monkeypatch, tmp_path):
    dataset = tmp_path / "issues_fixes.jsonl"
    records = [{"id": f"r/r#{i}:{i}"} for i in range(5)]
    dataset.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    monkeypatch.setattr(agent_v2_harness, "SAMPLES_PATH", dataset)

    got = agent_v2_harness.load_samples(2)

    assert [r["id"] for r in got] == ["r/r#0:0", "r/r#1:1"]
