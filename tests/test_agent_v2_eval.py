import asyncio
import inspect
import json
from types import SimpleNamespace

import pytest

from eval import agent_v2_harness, swe_bench


def coverage_proof():
    return {
        "source": "generated",
        "status": "generated_verified",
        "test_files": ["tests/test_auth.py"],
        "argv": ["pytest", "tests/test_auth.py"],
        "fixed_runs": [
            {
                "exit_code": 0,
                "outcome": "pass",
                "failing_test_ids": [],
                "assertion_fingerprint": "",
                "summary": "1 passed in 0.01s raw log must not be reported",
            },
            {
                "exit_code": 0,
                "outcome": "pass",
                "failing_test_ids": [],
                "assertion_fingerprint": "",
                "summary": "1 passed in 0.02s raw log must not be reported",
            },
        ],
        "base_runs": [
            {
                "exit_code": 1,
                "outcome": "assertion_failure",
                "failing_test_ids": ["tests/test_auth.py::test_login"],
                "assertion_fingerprint": "a" * 64,
                "summary": "full assertion output must not be reported",
            },
            {
                "exit_code": 1,
                "outcome": "assertion_failure",
                "failing_test_ids": ["tests/test_auth.py::test_login"],
                "assertion_fingerprint": "a" * 64,
                "summary": "full assertion output must not be reported",
            },
        ],
        "base_ref": "b" * 40,
        "patch_sha256": "c" * 64,
        "patch_gate_fingerprint": "d" * 64,
        "manifest_fingerprint": "e" * 64,
        "test_content_digests": {"tests/test_auth.py": "f" * 64},
    }


def test_agent_v2_eval_public_defaults_are_100000():
    from eval import harness

    assert (
        inspect.signature(agent_v2_harness.evaluate_agent_v2_sample)
        .parameters["token_budget"]
        .default
        == 100_000
    )
    assert (
        inspect.signature(agent_v2_harness.run_agent_v2_eval)
        .parameters["token_budget"]
        .default
        == 100_000
    )
    assert (
        inspect.signature(harness.run_agent_v2_eval)
        .parameters["token_budget"]
        .default
        == 100_000
    )


def test_agent_v2_eval_cli_uses_100000_default_budget(monkeypatch):
    calls = []

    async def fake_run_agent_v2_eval(**kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(
        agent_v2_harness,
        "run_agent_v2_eval",
        fake_run_agent_v2_eval,
    )

    agent_v2_harness.main([])

    assert calls[0]["token_budget"] == 100_000


def test_legacy_eval_cli_uses_100000_agent_v2_default_budget(monkeypatch):
    from eval import harness

    calls = []

    async def fake_run_agent_v2_eval(**kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(harness, "run_agent_v2_eval", fake_run_agent_v2_eval)

    harness.main(["--agent-v2"])

    assert calls[0]["token_budget"] == 100_000


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


@pytest.mark.parametrize(
    ("enabled", "key", "expected"),
    [
        ("0", "escalation-secret", False),
        ("1", "", False),
        ("1", "escalation-secret", True),
    ],
)
def test_eval_escalation_requires_explicit_flag_and_key(
    monkeypatch, enabled, key, expected
):
    monkeypatch.setenv("REPOPILOT_ESCALATION_ENABLED", enabled)
    monkeypatch.setenv("LLM_ESCALATION_API_KEY", key)

    assert agent_v2_harness.escalation_is_configured() is expected


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


def swe_bench_sample_with_id(instance_id):
    sample = swe_bench_sample()
    sample["id"] = instance_id
    sample["instance_id"] = instance_id
    return sample


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
    assert result["id"] == "acme/widget#7:8"
    assert result["evaluation_mode"] == "end_to_end"
    assert result["model"] == "test-model"
    assert result["commit_sha"] == "deadbeef"
    assert result["actual_files"] == ["src/auth.py"]
    assert result["success"] is False
    assert result["agent_success"] is False
    assert result["coverage_status"] == "pending"
    assert result["coverage_proof"] is None
    assert result["models_used"] == ["test-model"]
    assert result["error"] == "Patch failed tests."
    assert result["replay"]["timeline"][0]["stage"] == "reflect"
    assert "agent_payload" not in result


async def test_eval_result_exposes_safe_success_first_fields(monkeypatch):
    proof = coverage_proof()
    long_summary = "Generated a stable login regression. " + ("x" * 240)
    payload = {
        "success": True,
        "waiting_for_user": False,
        "final_phase": "DONE",
        "run_id": "safe-run",
        "trace_id": "safe-run",
        "turns_taken": 7,
        "token_used": 1234,
        "model_patch": "diff --git a/src/auth.py b/src/auth.py\n",
        "escalated": True,
        "escalation_reason": "repeated_edit",
        "coverage_status": "generated_verified",
        "coverage_test_files": ["tests/test_auth.py"],
        "coverage_test_command": "pytest tests/test_auth.py",
        "coverage_proof": proof,
        "coverage_failure_reason": "",
        "test_generation_attempts": 1,
        "attempt_outcome_summary": long_summary,
        "model_history": [
            {
                "model": "gemini-3.5-flash:stable",
                "provider": "primary",
                "node": "plan",
                "elapsed_seconds": 1.5,
                "input_tokens": 100,
                "output_tokens": 20,
                "status": "ok",
                "error_class": "",
            },
            {
                "model": "claude-opus-4-8:stable",
                "provider": "escalation",
                "node": "test_generation",
                "elapsed_seconds": 2.5,
                "input_tokens": 120,
                "output_tokens": 40,
                "status": "ok",
                "error_class": "",
            },
        ],
        "tool_history": [
            {
                "action": "read_symbol",
                "args_fingerprint": "1" * 64,
                "status": "ok",
                "evidence_id": "ev_1",
                "error_class": "",
            },
            {
                "action": "read_symbol",
                "args_fingerprint": "1" * 64,
                "status": "duplicate",
                "evidence_id": None,
                "error_class": "",
            },
        ],
        "evidence": [
            {"evidence_id": "ev_1", "fingerprint": "9" * 64},
            {"evidence_id": "ev_2", "fingerprint": "9" * 64},
        ],
        "no_progress_history": [
            {"kind": "repeated_edit", "fingerprint": "8" * 64, "node": "plan"},
            {"kind": "repeated_edit", "fingerprint": "8" * 64, "node": "plan"},
        ],
    }

    async def fake_agent(*args, **kwargs):
        return payload

    monkeypatch.setattr(agent_v2_harness, "agent_v2", fake_agent)
    monkeypatch.setattr(agent_v2_harness, "replay_run", lambda run_id: {})
    monkeypatch.setattr(
        agent_v2_harness,
        "load_run",
        lambda run_id: SimpleNamespace(
            model_history=[],
            tool_history=[],
            evidence=[],
            no_progress_history=[],
            no_progress_rounds=0,
            attempt_outcome_summary="",
            coverage_status="pending",
            coverage_test_files=[],
            coverage_test_command="",
            coverage_proof=None,
            coverage_failure_reason="",
            test_generation_attempts=0,
        ),
    )

    result = await agent_v2_harness.evaluate_agent_v2_sample(sample_record(), 0)

    assert result["success"] is True
    assert result["agent_success"] is True
    assert result["official_resolved"] is None
    assert result["models_used"] == [
        "gemini-3.5-flash:stable",
        "claude-opus-4-8:stable",
    ]
    assert len(result["model_invocations"]) == 2
    assert len(result["tool_invocations"]) == 2
    assert result["unique_evidence_count"] == 1
    assert result["max_consecutive_no_progress"] == 2
    assert result["attempt_outcome_summary"].startswith(
        "Generated a stable login regression."
    )
    assert len(result["attempt_outcome_summary"]) == 200
    assert result["coverage_status"] == "generated_verified"
    assert result["coverage_test_files"] == ["tests/test_auth.py"]
    assert result["coverage_test_command"] == "pytest tests/test_auth.py"
    assert result["coverage_proof"]["fixed_runs"] == [
        {"outcome": "pass", "failing_test_ids": [], "assertion_fingerprint": ""},
        {"outcome": "pass", "failing_test_ids": [], "assertion_fingerprint": ""},
    ]
    assert "summary" not in json.dumps(result["coverage_proof"])
    assert result["coverage_failure_reason"] == ""
    assert result["test_generation_attempts"] == 1


@pytest.mark.parametrize(
    ("status", "proof", "expected"),
    [
        ("pending", None, False),
        ("existing_verified", None, False),
        ("failed", coverage_proof(), False),
        (
            "generated_verified",
            {
                "source": "generated",
                "status": "generated_verified",
                "test_files": ["tests/test_auth.py"],
            },
            False,
        ),
        ("generated_verified", coverage_proof(), True),
    ],
)
async def test_eval_serializes_success_only_with_verified_coverage(
    monkeypatch, status, proof, expected
):
    async def fake_agent(*args, **kwargs):
        return {
            "success": True,
            "final_phase": "DONE",
            "run_id": "coverage-run",
            "trace_id": "coverage-run",
            "coverage_status": status,
            "coverage_proof": proof,
        }

    monkeypatch.setattr(agent_v2_harness, "agent_v2", fake_agent)
    monkeypatch.setattr(agent_v2_harness, "replay_run", lambda run_id: {})
    monkeypatch.setattr(agent_v2_harness, "load_run", lambda run_id: None)

    result = await agent_v2_harness.evaluate_agent_v2_sample(sample_record(), 0)

    assert result["success"] is expected
    assert result["agent_success"] is expected


async def test_eval_does_not_coerce_string_success_to_true(monkeypatch):
    async def fake_agent(*args, **kwargs):
        return {
            "success": "false",
            "final_phase": "DONE",
            "run_id": "coverage-run",
            "trace_id": "coverage-run",
            "coverage_status": "generated_verified",
            "coverage_proof": coverage_proof(),
        }

    monkeypatch.setattr(agent_v2_harness, "agent_v2", fake_agent)
    monkeypatch.setattr(agent_v2_harness, "replay_run", lambda run_id: {})
    monkeypatch.setattr(agent_v2_harness, "load_run", lambda run_id: None)

    result = await agent_v2_harness.evaluate_agent_v2_sample(sample_record(), 0)

    assert result["success"] is False
    assert result["agent_success"] is False


async def test_eval_artifacts_drop_credentials_evaluator_fields_and_archive_payloads(
    monkeypatch, tmp_path
):
    secret = "sk-test-secret-123456789"
    archive_payload = "setuptools-33.1.1.zip PK\\x03\\x04 generated bytes"

    async def fake_agent(*args, **kwargs):
        return {
            "success": False,
            "final_phase": "FAILED",
            "run_id": "unsafe-run",
            "trace_id": "unsafe-run",
            "turns_taken": secret,
            "token_used": secret,
            "error": f"Bearer {secret} FAIL_TO_PASS {archive_payload}",
            "model_history": [
                {
                    "model": "safe-model",
                    "provider": "primary",
                    "node": "plan",
                    "elapsed_seconds": secret,
                    "input_tokens": secret,
                    "output_tokens": secret,
                    "status": "error",
                    "error_class": "GatewayError",
                }
            ],
            "model_patch": (
                "diff --git a/gold_patch.zip b/gold_patch.zip\n"
                f"+Authorization: Bearer {secret}\n+{archive_payload}\n"
            ),
            "generated_zip_raw_payload": archive_payload,
        }

    monkeypatch.setattr(agent_v2_harness, "agent_v2", fake_agent)
    monkeypatch.setattr(agent_v2_harness, "replay_run", lambda run_id: {})
    monkeypatch.setattr(agent_v2_harness, "load_run", lambda run_id: None)
    monkeypatch.setattr(agent_v2_harness, "_configured_model", lambda: secret)

    async def fake_seed(*args, **kwargs):
        return {"repo_path": "/tmp/safe-eval"}

    monkeypatch.setattr(agent_v2_harness, "_build_eval_seed", fake_seed)

    result = await agent_v2_harness.evaluate_agent_v2_sample(
        swe_bench_sample(), 0
    )
    results_path = tmp_path / "results.json"
    predictions_path = tmp_path / "predictions.jsonl"
    agent_v2_harness._write_results_with_fallback([result], results_path)
    swe_bench.write_predictions([result], predictions_path)
    rendered = results_path.read_text() + predictions_path.read_text()

    assert secret not in rendered
    assert "FAIL_TO_PASS" not in rendered
    assert archive_payload not in rendered
    assert "generated_zip_raw_payload" not in rendered


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

    async def fake_evaluate(sample, idx, max_retries=3, token_budget=100000,
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
    stored = json.loads(results_path.read_text(encoding="utf-8"))
    assert stored[0]["success"] is False
    assert stored[0]["agent_success"] is False


def test_result_writer_downgrades_unproven_agent_success(tmp_path):
    path = tmp_path / "results.json"
    results = [
        {
            "id": "unsafe-success",
            "mode": "agent_v2",
            "success": True,
            "agent_success": True,
            "coverage_status": "pending",
            "coverage_proof": None,
        }
    ]

    agent_v2_harness._write_results_with_fallback(results, path)

    [stored] = json.loads(path.read_text(encoding="utf-8"))
    assert stored["success"] is False
    assert stored["agent_success"] is False


def test_result_writer_downgrades_structurally_incomplete_coverage(tmp_path):
    path = tmp_path / "results.json"
    proof = coverage_proof()
    proof["test_files"] = ["vendor/tests/test_auth.py"]
    proof["argv"] = ["pytest", "vendor/tests/test_auth.py"]
    proof["test_content_digests"] = {
        "vendor/tests/test_auth.py": proof["test_content_digests"].pop(
            "tests/test_auth.py"
        )
    }
    for run in proof["base_runs"]:
        run["failing_test_ids"] = ["vendor/tests/test_auth.py::test_login"]
    results = [
        {
            "id": "unsafe-success",
            "mode": "agent_v2",
            "success": True,
            "agent_success": True,
            "coverage_status": "generated_verified",
            "coverage_proof": proof,
        }
    ]

    agent_v2_harness._write_results_with_fallback(results, path)

    [stored] = json.loads(path.read_text(encoding="utf-8"))
    assert stored["success"] is False
    assert stored["agent_success"] is False
    assert stored["coverage_proof"] is None


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


async def test_run_swe_bench_eval_checkpoints_and_continues_after_sample_error(
    monkeypatch, tmp_path, capsys
):
    samples = [
        swe_bench_sample_with_id("acme__widget-1"),
        swe_bench_sample_with_id("acme__widget-2"),
        swe_bench_sample_with_id("acme__widget-3"),
    ]
    secret = "sample-secret-sentinel"
    evaluator = "FAIL_TO_PASS evaluator-sentinel"
    archive = "archive-payload-sentinel"
    raw_log = "raw-log-sentinel"
    result_snapshots = []
    prediction_snapshots = []
    original_result_writer = agent_v2_harness._write_results_with_fallback
    original_prediction_writer = agent_v2_harness.write_predictions

    async def fake_evaluate(selected, idx, **kwargs):
        if idx == 1:
            raise RuntimeError(
                "LibreSSL SSL_connect: SSL_ERROR_SYSCALL "
                f"{secret} {evaluator} {archive} {raw_log}"
            )
        return {
            "id": selected["id"],
            "mode": "agent_v2",
            "instance_id": selected["instance_id"],
            "base_commit": selected["base_commit"],
            "model": "test-model",
            "model_patch": f"diff --git sample-{idx}",
            "success": False,
            "agent_success": False,
            "official_resolved": None,
        }

    def recording_result_writer(results, path):
        result_snapshots.append([item["instance_id"] for item in results])
        return original_result_writer(results, path)

    def recording_prediction_writer(results, path):
        prediction_snapshots.append([item["instance_id"] for item in results])
        return original_prediction_writer(results, path)

    monkeypatch.setattr(
        agent_v2_harness,
        "load_verified_samples",
        lambda count, seed: samples,
    )
    monkeypatch.setattr(
        agent_v2_harness, "evaluate_agent_v2_sample", fake_evaluate
    )
    monkeypatch.setattr(
        agent_v2_harness,
        "_write_results_with_fallback",
        recording_result_writer,
    )
    monkeypatch.setattr(
        agent_v2_harness, "write_predictions", recording_prediction_writer
    )
    monkeypatch.setattr(agent_v2_harness, "_configured_model", lambda: "test-model")

    results_path = tmp_path / "results.json"
    predictions_path = tmp_path / "predictions.jsonl"
    results = await agent_v2_harness.run_agent_v2_eval(
        n_samples=3,
        dataset="swe-bench-verified",
        results_path=results_path,
        predictions_path=predictions_path,
    )

    assert [item["instance_id"] for item in results] == [
        "acme__widget-1",
        "acme__widget-2",
        "acme__widget-3",
    ]
    failed = results[1]
    assert failed["id"] == "acme__widget-2"
    assert failed["base_commit"] == "a" * 40
    assert failed["model"] == "test-model"
    assert failed["model_patch"] == ""
    assert failed["success"] is False
    assert failed["agent_success"] is False
    assert failed["official_resolved"] is None
    assert failed["failure_class"] == "infra"
    assert "evaluation" not in failed
    assert result_snapshots == [
        ["acme__widget-1"],
        ["acme__widget-1", "acme__widget-2"],
        ["acme__widget-1", "acme__widget-2", "acme__widget-3"],
    ]
    assert prediction_snapshots == result_snapshots

    stored_results = json.loads(results_path.read_text(encoding="utf-8"))
    predictions = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [item["instance_id"] for item in stored_results] == [
        "acme__widget-1",
        "acme__widget-2",
        "acme__widget-3",
    ]
    assert all(
        set(item) == {"instance_id", "model_name_or_path", "model_patch"}
        for item in predictions
    )
    assert predictions[1] == {
        "instance_id": "acme__widget-2",
        "model_name_or_path": "test-model",
        "model_patch": "",
    }

    serialized = json.dumps(results) + results_path.read_text(encoding="utf-8")
    serialized += predictions_path.read_text(encoding="utf-8")
    captured = capsys.readouterr()
    diagnostics = captured.out + captured.err
    for sentinel in (secret, evaluator, archive, raw_log):
        assert sentinel not in serialized
        assert sentinel not in diagnostics
    assert "gold patch must remain hidden" not in serialized
    assert "test patch must remain hidden" not in serialized


async def test_run_swe_bench_eval_selects_requested_ids_in_caller_order(
    monkeypatch, tmp_path
):
    available = [
        swe_bench_sample_with_id("acme__widget-1"),
        swe_bench_sample_with_id("acme__widget-2"),
        swe_bench_sample_with_id("acme__widget-3"),
    ]
    load_calls = []

    def fake_load(count, seed):
        load_calls.append((count, seed))
        return available[:count]

    async def fake_evaluate(selected, idx, **kwargs):
        return {
            "id": selected["id"],
            "instance_id": selected["instance_id"],
            "success": False,
        }

    monkeypatch.setattr(agent_v2_harness, "load_verified_samples", fake_load)
    monkeypatch.setattr(
        agent_v2_harness, "evaluate_agent_v2_sample", fake_evaluate
    )

    results = await agent_v2_harness.run_agent_v2_eval(
        n_samples=1,
        dataset="swe-bench-verified",
        dataset_seed=29,
        sample_ids=["acme__widget-3", "acme__widget-1"],
        results_path=tmp_path / "results.json",
    )

    assert [result["instance_id"] for result in results] == [
        "acme__widget-3",
        "acme__widget-1",
    ]
    assert load_calls[0][0] >= len(available)
    assert load_calls[0][1] == 29


async def test_run_swe_bench_eval_rejects_unknown_requested_id(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        agent_v2_harness,
        "load_verified_samples",
        lambda count, seed: [swe_bench_sample_with_id("acme__widget-1")],
    )

    with pytest.raises(ValueError, match="unknown SWE-bench instance ID"):
        await agent_v2_harness.run_agent_v2_eval(
            dataset="swe-bench-verified",
            sample_ids=["acme__missing-9"],
            results_path=tmp_path / "results.json",
        )


async def test_run_swe_bench_eval_rejects_duplicate_requested_ids(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        agent_v2_harness,
        "load_verified_samples",
        lambda count, seed: [swe_bench_sample_with_id("acme__widget-1")],
    )

    with pytest.raises(ValueError, match="duplicate SWE-bench instance ID"):
        await agent_v2_harness.run_agent_v2_eval(
            dataset="swe-bench-verified",
            sample_ids=["acme__widget-1", "acme__widget-1"],
            results_path=tmp_path / "results.json",
        )


async def test_run_swe_bench_eval_without_requested_ids_keeps_seed_selection(
    monkeypatch, tmp_path
):
    sample = swe_bench_sample_with_id("acme__widget-seeded")
    load_calls = []

    def fake_load(count, seed):
        load_calls.append((count, seed))
        return [sample]

    async def fake_evaluate(selected, idx, **kwargs):
        return {
            "id": selected["id"],
            "instance_id": selected["instance_id"],
            "success": False,
        }

    monkeypatch.setattr(agent_v2_harness, "load_verified_samples", fake_load)
    monkeypatch.setattr(
        agent_v2_harness, "evaluate_agent_v2_sample", fake_evaluate
    )

    results = await agent_v2_harness.run_agent_v2_eval(
        n_samples=2,
        dataset="swe-bench-verified",
        dataset_seed=31,
        results_path=tmp_path / "results.json",
    )

    assert load_calls == [(2, 31)]
    assert [result["instance_id"] for result in results] == [
        "acme__widget-seeded"
    ]


def test_agent_v2_eval_cli_forwards_sample_ids_and_results_path(
    monkeypatch, tmp_path
):
    calls = []
    ids_path = tmp_path / "sample_ids.txt"
    ids_path.write_text(
        "acme__widget-3\n\n  acme__widget-1  \n", encoding="utf-8"
    )
    results_path = tmp_path / "selected_results.json"

    async def fake_run_agent_v2_eval(**kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(
        agent_v2_harness, "run_agent_v2_eval", fake_run_agent_v2_eval
    )

    agent_v2_harness.main(
        [
            "--dataset",
            "swe-bench-verified",
            "--sample-ids-file",
            str(ids_path),
            "--results-file",
            str(results_path),
        ]
    )

    assert calls[0]["sample_ids"] == [
        "acme__widget-3",
        "acme__widget-1",
    ]
    assert calls[0]["results_path"] == results_path


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

    async def fake_evaluate(sample, idx, max_retries=3, token_budget=100000,
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

    async def fake_evaluate(sample, idx, max_retries=3, token_budget=100000,
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
    original_replace = swe_bench.os.replace

    def fail_requested_path(source, destination):
        if str(destination) == str(requested_path):
            raise OSError("read-only eval results")
        return original_replace(source, destination)

    monkeypatch.setattr(swe_bench.os, "replace", fail_requested_path)

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
    stored = json.loads(fallback_path.read_text(encoding="utf-8"))
    assert stored[0]["success"] is False
    assert stored[0]["agent_success"] is False
    assert not requested_path.exists()


def test_result_writer_preserves_requested_and_fallback_checkpoints_when_replace_fails(
    monkeypatch, tmp_path
):
    requested_path = tmp_path / "requested" / "results.json"
    fallback_path = tmp_path / "home" / "eval" / "eval_results.json"
    requested_path.parent.mkdir(parents=True)
    fallback_path.parent.mkdir(parents=True)
    requested_previous = b'[{"id":"requested-previous"}]'
    fallback_previous = b'[{"id":"fallback-previous"}]'
    requested_path.write_bytes(requested_previous)
    fallback_path.write_bytes(fallback_previous)
    monkeypatch.setattr(
        agent_v2_harness, "repopilot_home", lambda: tmp_path / "home"
    )

    def fail_replace(source, destination):
        raise OSError(f"replace interrupted for {destination}")

    monkeypatch.setattr(swe_bench.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace interrupted"):
        agent_v2_harness._write_results_with_fallback(
            [{"id": "new-result"}], requested_path
        )

    assert requested_path.read_bytes() == requested_previous
    assert fallback_path.read_bytes() == fallback_previous
    assert list(requested_path.parent.glob(".*.tmp")) == []
    assert list(fallback_path.parent.glob(".*.tmp")) == []


async def test_run_agent_v2_eval_records_sample_failure_and_preserves_cleanup_warning(
    monkeypatch, tmp_path, capsys
):
    calls = []

    async def fake_evaluate(sample, idx, max_retries=3, token_budget=100000,
                            seed_gold_files=False):
        raise RuntimeError("sample-secret-sentinel")

    async def fake_close_llm_client():
        calls.append("llm")

    async def fake_close_store():
        calls.append("memory")
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(agent_v2_harness, "load_samples", lambda n, sample_id=None, **kw: [sample_record()])
    monkeypatch.setattr(agent_v2_harness, "evaluate_agent_v2_sample", fake_evaluate)
    monkeypatch.setattr(agent_v2_harness, "close_llm_client", fake_close_llm_client)
    monkeypatch.setattr(agent_v2_harness, "close_store", fake_close_store, raising=False)

    results = await agent_v2_harness.run_agent_v2_eval(
        n_samples=1,
        results_path=tmp_path / "agent_v2_results.json",
    )

    assert results[0]["id"] == "acme/widget#7:8"
    assert results[0]["success"] is False
    assert results[0]["agent_success"] is False
    assert results[0]["official_resolved"] is None
    assert calls == ["llm", "memory"]
    captured = capsys.readouterr()
    assert "cleanup failed" in captured.err
    assert "sample-secret-sentinel" not in captured.out + captured.err


@pytest.mark.parametrize(
    "error_type",
    [asyncio.CancelledError, KeyboardInterrupt, SystemExit],
)
async def test_run_agent_v2_eval_does_not_catch_base_exceptions(
    monkeypatch, tmp_path, error_type
):
    async def fake_evaluate(sample, idx, **kwargs):
        raise error_type()

    monkeypatch.setattr(
        agent_v2_harness,
        "load_samples",
        lambda n, sample_id=None, **kwargs: [sample_record()],
    )
    monkeypatch.setattr(
        agent_v2_harness, "evaluate_agent_v2_sample", fake_evaluate
    )

    with pytest.raises(error_type):
        await agent_v2_harness.run_agent_v2_eval(
            n_samples=1,
            results_path=tmp_path / "agent_v2_results.json",
        )


async def test_run_agent_v2_eval_does_not_mask_results_when_cleanup_raises(
    monkeypatch, tmp_path
):
    async def fake_evaluate(sample, idx, max_retries=3, token_budget=100000,
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
    stored = json.loads(results_path.read_text(encoding="utf-8"))
    assert stored[0]["success"] is False
    assert stored[0]["agent_success"] is False


async def test_run_agent_v2_eval_does_not_mask_results_when_memory_cleanup_raises(
    monkeypatch, tmp_path, capsys
):
    async def fake_evaluate(sample, idx, max_retries=3, token_budget=100000,
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
    stored = json.loads(results_path.read_text(encoding="utf-8"))
    assert stored[0]["success"] is False
    assert stored[0]["agent_success"] is False
    captured = capsys.readouterr()
    assert "Warning: failed to close shared memory store: RuntimeError: memory cleanup failed" in captured.err


def test_harness_main_dispatches_agent_v2_mode(monkeypatch):
    from eval import harness

    calls = []

    async def fake_run_agent_v2_eval(
        n_samples=5,
        max_retries=3,
        token_budget=100000,
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
            token_budget=100000,
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
