import json

import pytest

from eval import agent_v2_harness, swe_bench


def test_legacy_harness_default_model_is_gemini_flash():
    from eval import harness

    assert harness.DEFAULT_MODEL == "gemini-3.5-flash:stable"


def test_legacy_harness_base_url_falls_back_to_linoapi(monkeypatch):
    from eval import harness

    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    assert harness._get_llm_base_url() == "https://linoapi.com.cn/v1"


def test_legacy_harness_base_url_honors_normalized_override(monkeypatch):
    from eval import harness

    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1/")

    assert harness._get_llm_base_url() == "https://example.test/v1"


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("LINOAPI_API_KEY", "lino-key"),
        ("LLM_API_KEY", "llm-key"),
        ("DEEPSEEK_API_KEY", "deepseek-key"),
    ],
)
def test_legacy_harness_resolves_each_supported_api_key(
    monkeypatch, variable, value
):
    from eval import harness

    for name in ("LINOAPI_API_KEY", "LLM_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(variable, value)

    assert harness._get_llm_api_key() == value


def test_legacy_harness_api_key_precedence_matches_production(monkeypatch):
    from eval import harness

    monkeypatch.setenv("LINOAPI_API_KEY", "lino-key")
    monkeypatch.setenv("LLM_API_KEY", "llm-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")

    assert harness._get_llm_api_key() == "lino-key"

    monkeypatch.delenv("LINOAPI_API_KEY")
    assert harness._get_llm_api_key() == "llm-key"


def test_legacy_harness_api_key_falls_back_to_empty(monkeypatch):
    from eval import harness

    for name in ("LINOAPI_API_KEY", "LLM_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    assert harness._get_llm_api_key() == ""


async def test_legacy_harness_llm_request_resolves_api_key_at_request_time(
    monkeypatch,
):
    from eval import harness

    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": []}

    class FakeClient:
        async def post(self, url, *, json, headers):
            captured.update(headers)
            return FakeResponse()

    monkeypatch.setattr(harness, "_get_client", lambda: FakeClient())
    monkeypatch.setenv("LINOAPI_API_KEY", "request-time-key")

    await harness.llm_request([{"role": "user", "content": "hi"}])

    assert captured["Authorization"] == "Bearer request-time-key"


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


def swe_bench_sample():
    return swe_bench.normalize_verified_row(
        {
            "instance_id": "acme__widget-8",
            "repo": "acme/widget",
            "issue_id": "7",
            "issue_url": "https://github.com/acme/widget/issues/7",
            "problem_statement": "Login crash\n\nThe endpoint crashes.",
            "base_commit": "a" * 40,
            "patch": "gold patch must remain hidden",
            "test_patch": "test patch must remain hidden",
            "FAIL_TO_PASS": "[]",
            "PASS_TO_PASS": "[]",
            "version": "1.0",
            "created_at": "2026-01-01",
            "difficulty": "medium",
        }
    )


async def test_swe_bench_eval_prepares_exact_checkout_and_passes_safe_seed(
    monkeypatch,
):
    captured = {}

    async def fake_clone(state):
        captured["ref"] = state.repo_ref
        return "/tmp/exact-work"

    async def fake_agent(issue_url, **kwargs):
        captured["seed"] = kwargs["seed"]
        return {
            "success": False,
            "final_phase": "FAILED",
            "trace_id": "trace",
            "run_id": "trace",
            "model_patch": "diff --git",
        }

    monkeypatch.setattr(agent_v2_harness, "git_clone", fake_clone, raising=False)
    monkeypatch.setattr(agent_v2_harness, "agent_v2", fake_agent)
    monkeypatch.setattr(agent_v2_harness, "replay_run", lambda run_id: {})

    result = await agent_v2_harness.evaluate_agent_v2_sample(
        swe_bench_sample(),
        idx=0,
        seed_gold_files=True,
    )

    assert captured["ref"] == "a" * 40
    assert captured["seed"]["repo_path"] == "/tmp/exact-work"
    assert "evaluation" not in captured["seed"]
    assert "gold patch must remain hidden" not in json.dumps(captured["seed"])
    assert result["instance_id"] == "acme__widget-8"
    assert result["base_commit"] == "a" * 40
    assert result["model_patch"] == "diff --git"
    assert result["evaluation_mode"] == "end_to_end"


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
    monkeypatch.setattr(
        agent_v2_harness, "_configured_model", lambda: "test-model", raising=False
    )
    monkeypatch.setattr(
        agent_v2_harness, "_current_commit_sha", lambda: "deadbeef", raising=False
    )

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
        "evaluation_mode": "end_to_end",
        "model": "test-model",
        "commit_sha": "deadbeef",
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


async def test_seeded_eval_is_labeled_oracle_even_when_gold_seed_is_unavailable(
    monkeypatch,
):
    captured = {}

    async def fake_agent_v2(issue_url, **kwargs):
        captured.update(kwargs)
        return {
            "success": False,
            "final_phase": "FAILED",
            "trace_id": "trace-1",
        }

    monkeypatch.setattr(agent_v2_harness, "agent_v2", fake_agent_v2)
    monkeypatch.setattr(agent_v2_harness, "_build_gold_seed", lambda sample: None)
    monkeypatch.setattr(
        agent_v2_harness, "replay_run", lambda run_id: {"run_id": run_id}
    )

    result = await agent_v2_harness.evaluate_agent_v2_sample(
        sample_record(), idx=0, seed_gold_files=True
    )

    assert result["evaluation_mode"] == "oracle_files"
    assert captured["seed"] is None


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


async def test_run_swe_bench_eval_selects_dataset_and_writes_predictions(
    monkeypatch, tmp_path
):
    sample = swe_bench_sample()
    calls = []

    def fake_load(count, seed):
        calls.append((count, seed))
        return [sample]

    async def fake_evaluate(
        selected,
        idx,
        max_retries=3,
        token_budget=50000,
        seed_gold_files=False,
    ):
        return {
            "id": selected["id"],
            "instance_id": selected["instance_id"],
            "model": "gemini-3.5-flash:stable",
            "model_patch": "diff --git",
            "success": False,
        }

    monkeypatch.setattr(
        agent_v2_harness,
        "load_verified_samples",
        fake_load,
        raising=False,
    )
    monkeypatch.setattr(
        agent_v2_harness,
        "evaluate_agent_v2_sample",
        fake_evaluate,
    )
    predictions_path = tmp_path / "predictions.jsonl"

    results = await agent_v2_harness.run_agent_v2_eval(
        n_samples=1,
        dataset="swe-bench-verified",
        dataset_seed=23,
        predictions_path=predictions_path,
        results_path=tmp_path / "results.json",
    )

    assert calls == [(1, 23)]
    assert results[0]["instance_id"] == "acme__widget-8"
    assert json.loads(predictions_path.read_text().strip()) == {
        "instance_id": "acme__widget-8",
        "model_name_or_path": "gemini-3.5-flash:stable",
        "model_patch": "diff --git",
    }


async def test_predictions_file_requires_swe_bench_dataset(tmp_path):
    with pytest.raises(
        ValueError,
        match="predictions_path requires dataset='swe-bench-verified'",
    ):
        await agent_v2_harness.run_agent_v2_eval(
            dataset="custom",
            predictions_path=tmp_path / "predictions.jsonl",
            results_path=tmp_path / "results.json",
        )


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
    monkeypatch.setattr(agent_v2_harness, "repopilot_home", lambda: repopilot_home)

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
