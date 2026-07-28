from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from pathlib import Path

import pytest

from eval import oci_aggregate, oci_runner
from eval.oci_aggregate import ArtifactContractError, aggregate_artifacts
from eval.oci_contract import OfficialResult, RuntimeRecord, sha256_file, write_model
from eval.oci_runner import (
    _write_generation_failure,
    generate_instance,
    package_instance,
    score_instance,
)
from eval.swe_bench import verified_row_sha256, write_predictions
from src import test_generator
from src.async_safety import wait_for_phase
from src.safe_subprocess import BoundedProcessResult
from src.state import AgentState, ModelInvocation

COMMIT_SHA = "a" * 40
IMAGE_SHA = "sha256:" + "b" * 64
PRIMARY_MODEL = "gemini-3.5-flash:stable"
ESCALATION_MODEL = "claude-opus-4-8:stable"
_DEFAULT_MODEL_INVOCATIONS = object()
_MISSING_AGENT_VERDICT = object()
_DEFAULT_MODEL_PATCH = object()
_DEFAULT_COVERAGE_PROOF = object()


def _artifact_row(instance_id: str) -> dict[str, str]:
    return {
        "repo": "owner/repo",
        "instance_id": instance_id,
        "base_commit": "d" * 40,
        "patch": "",
        "test_patch": "",
        "problem_statement": "A pinned row",
        "hints_text": "",
        "created_at": "2026-01-01",
        "version": "1.0",
        "FAIL_TO_PASS": "[]",
        "PASS_TO_PASS": "[]",
        "environment_setup_commit": "e" * 40,
        "difficulty": "medium",
    }


ROW_SHA = verified_row_sha256(_artifact_row("owner__repo-1"))


def _repo_root(tmp_path: Path, instance_ids: list[str]) -> Path:
    root = tmp_path / "repo"
    eval_dir = root / "eval"
    eval_dir.mkdir(parents=True)
    (eval_dir / "checkpoint_5_ids.txt").write_text(
        "\n".join(instance_ids) + "\n", encoding="utf-8"
    )
    return root


def test_aggregate_cli_accepts_baseline_50(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake_aggregate(mode, artifacts_dir, output_dir, *, expected_commit):
        seen.update(
            mode=mode,
            artifacts_dir=artifacts_dir,
            output_dir=output_dir,
            expected_commit=expected_commit,
        )

    monkeypatch.setattr(oci_aggregate, "aggregate_artifacts", fake_aggregate)

    assert (
        oci_aggregate.main(
            [
                "--mode",
                "baseline_50",
                "--artifacts-dir",
                str(tmp_path / "artifacts"),
                "--output-dir",
                str(tmp_path / "combined"),
                "--expected-commit",
                COMMIT_SHA,
            ]
        )
        == 0
    )
    assert seen["mode"] == "baseline_50"


def test_aggregate_cli_rejects_retired_baseline_10(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        oci_aggregate.main(
            [
                "--mode",
                "baseline_10",
                "--artifacts-dir",
                str(tmp_path / "artifacts"),
                "--output-dir",
                str(tmp_path / "combined"),
                "--expected-commit",
                COMMIT_SHA,
            ]
        )


def _completed_output(
    root: Path,
    instance_id: str,
    *,
    resolved: bool = False,
    official_status: str | None = None,
    agent_success: object = None,
    input_tokens: object = 10,
    output_tokens: object = 5,
    elapsed_seconds: object = 1.0,
    model_invocations: object = _DEFAULT_MODEL_INVOCATIONS,
    failure_class: str | None = None,
    model_patch: object = _DEFAULT_MODEL_PATCH,
    test_generation_attempts: int = 0,
    coverage_status: str | None = None,
    coverage_proof: object = _DEFAULT_COVERAGE_PROOF,
    runtime_status: str = "ready",
) -> tuple[Path, Path]:
    output_dir = root / "output"
    artifact_dir = root / "artifact"
    output_dir.mkdir(parents=True)
    runtime_payload = {
        "mode": "checkpoint_5",
        "instance_id": instance_id,
        "commit_sha": COMMIT_SHA,
        "status": runtime_status,
        "error_class": "" if runtime_status == "ready" else "RuntimeError",
    }
    if runtime_status == "ready":
        runtime_payload.update(
            row_sha256=verified_row_sha256(_artifact_row(instance_id)),
            remote_image="swebench/sweb.eval.x86_64.owner_repo-1:latest",
            image_sha=IMAGE_SHA,
        )
    runtime = RuntimeRecord.model_validate(runtime_payload)
    runtime_path = write_model(output_dir / "runtime.json", runtime)
    status = official_status or ("resolved" if resolved else "unresolved")
    if runtime_status != "ready":
        _write_generation_failure(
            runtime,
            output_dir,
            RuntimeError(runtime.error_class),
        )
        write_model(
            output_dir / "official_result.json",
            OfficialResult(
                instance_id=instance_id,
                status="scorer_infra",
                submitted=False,
                completed=False,
                resolved=False,
                error_class="RuntimeError",
            ),
        )
        return runtime_path, artifact_dir
    internal_success = (
        status in {"resolved", "unresolved"}
        if agent_success in {None, _MISSING_AGENT_VERDICT}
        else bool(agent_success)
    )
    invocations = (
        [
            {
                "model": PRIMARY_MODEL,
                "provider": "primary",
                "node": "plan",
                "elapsed_seconds": elapsed_seconds,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "status": "ok",
                "error_class": "",
            }
        ]
        if model_invocations is _DEFAULT_MODEL_INVOCATIONS
        else model_invocations
    )
    patch = (
        ("diff --git a/a.py b/a.py\n" if internal_success else "")
        if model_patch is _DEFAULT_MODEL_PATCH
        else model_patch
    )
    result_coverage_status = coverage_status or (
        "existing_verified" if internal_success else "failed"
    )
    result_coverage_proof = (
        _coverage_proof(source="existing")
        if internal_success and coverage_proof is _DEFAULT_COVERAGE_PROOF
        else (
            None
            if coverage_proof is _DEFAULT_COVERAGE_PROOF
            else coverage_proof
        )
    )
    serialized_token_total = _serialized_token_total(invocations)
    result = {
        "id": instance_id,
        "mode": "agent_v2",
        "evaluation_mode": "end_to_end",
        "instance_id": instance_id,
        "commit_sha": COMMIT_SHA,
        "model": PRIMARY_MODEL,
        "repo": "owner/repo",
        "issue_url": "https://github.com/owner/repo/issues/1",
        "issue_title": "A regression",
        "model_patch": patch,
        "patch_generated": bool(patch),
        "tests_passed": None,
        "success": internal_success,
        "official_resolved": None,
        "waiting_for_user": False,
        "final_phase": "DONE" if internal_success else "FAILED",
        "run_id": "run-1",
        "trace_id": "trace-1",
        "turns_taken": 2,
        "token_used": (
            serialized_token_total
            if serialized_token_total is not None
            else 15
        ),
        "error": None,
        "replay": None,
        "replay_error": None,
        "models_used": (
            list(dict.fromkeys(item["model"] for item in invocations))
            or [PRIMARY_MODEL]
        ),
        "escalated": False,
        "escalation_reason": "",
        "model_invocations": invocations,
        "tool_invocations": [],
        "unique_evidence_count": 0,
        "max_consecutive_no_progress": 0,
        "attempt_outcome_summary": "complete",
        "coverage_status": result_coverage_status,
        "coverage_test_files": [],
        "coverage_test_command": "",
        "coverage_proof": result_coverage_proof,
        "coverage_failure_reason": "" if internal_success else "test_failed",
        "test_generation_attempts": test_generation_attempts,
        "failure_class": failure_class
        or ("agent_success" if internal_success else "test_failed"),
        "base_commit": "d" * 40 if runtime_status == "ready" else "",
    }
    if agent_success is not _MISSING_AGENT_VERDICT:
        result["agent_success"] = (
            internal_success if agent_success is None else agent_success
        )
    (output_dir / "result.json").write_text(
        json.dumps([result]) + "\n", encoding="utf-8"
    )
    write_predictions([result], output_dir / "prediction.jsonl")
    official = OfficialResult(
        instance_id=instance_id,
        status=status,
        submitted=status != "scorer_infra",
        completed=status in {"resolved", "unresolved"},
        resolved=status == "resolved",
        error_class="RuntimeError" if status == "scorer_infra" else "",
    )
    write_model(output_dir / "official_result.json", official)
    return runtime_path, artifact_dir


def _package(
    artifacts_root: Path,
    instance_id: str,
    *,
    resolved: bool = False,
    official_status: str | None = None,
    agent_success: object = None,
    input_tokens: object = 10,
    output_tokens: object = 5,
    elapsed_seconds: object = 1.0,
    model_invocations: object = _DEFAULT_MODEL_INVOCATIONS,
    failure_class: str | None = None,
    model_patch: object = _DEFAULT_MODEL_PATCH,
    test_generation_attempts: int = 0,
    coverage_status: str | None = None,
    coverage_proof: object = _DEFAULT_COVERAGE_PROOF,
    runtime_status: str = "ready",
) -> Path:
    work = artifacts_root.parent / f"work-{instance_id}"
    runtime_path, artifact_dir = _completed_output(
        work,
        instance_id,
        resolved=resolved,
        official_status=official_status,
        agent_success=agent_success,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        elapsed_seconds=elapsed_seconds,
        model_invocations=model_invocations,
        failure_class=failure_class,
        model_patch=model_patch,
        test_generation_attempts=test_generation_attempts,
        coverage_status=coverage_status,
        coverage_proof=coverage_proof,
        runtime_status=runtime_status,
    )
    package_instance(
        runtime_path,
        runtime_path.parent,
        artifact_dir,
        row_loader=_artifact_row,
    )
    destination = artifacts_root / f"bundle-{instance_id}"
    artifact_dir.rename(destination)
    return destination


def _rewrite_bundle_json(bundle: Path, filename: str, payload: object) -> None:
    path = bundle / filename
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][filename] = sha256_file(path)
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")


def _rewrite_bundle_bytes(bundle: Path, filename: str, payload: bytes) -> None:
    path = bundle / filename
    path.write_bytes(payload)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][filename] = sha256_file(path)
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")


def _invocation(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": PRIMARY_MODEL,
        "provider": "primary",
        "node": "plan",
        "elapsed_seconds": 1.0,
        "input_tokens": 10,
        "output_tokens": 5,
        "status": "ok",
        "error_class": "",
    }
    payload.update(overrides)
    return payload


def _serialized_token_total(invocations: object) -> int | None:
    if not isinstance(invocations, list):
        return None
    total = 0
    for invocation in invocations:
        if not isinstance(invocation, dict):
            return None
        input_tokens = invocation.get("input_tokens")
        output_tokens = invocation.get("output_tokens")
        if (
            type(input_tokens) is not int
            or type(output_tokens) is not int
            or input_tokens < 0
            or output_tokens < 0
        ):
            return None
        total += input_tokens + output_tokens
    return total


def _coverage_proof(*, source: str = "generated") -> dict[str, object]:
    passing_run = {
        "outcome": "pass",
        "failing_test_ids": [],
        "assertion_fingerprint": "",
    }
    failing_run = {
        "outcome": "assertion_failure",
        "failing_test_ids": ["tests/test_auth.py::test_login"],
        "assertion_fingerprint": "f" * 64,
    }
    return {
        "source": source,
        "status": f"{source}_verified",
        "test_files": ["tests/test_auth.py"],
        "fixed_runs": [dict(passing_run), dict(passing_run)],
        "base_runs": [dict(failing_run), dict(failing_run)],
    }


def test_package_copies_only_safe_hash_bound_files(tmp_path: Path) -> None:
    runtime_path, artifact_dir = _completed_output(tmp_path / "work", "owner__repo-1")
    (runtime_path.parent / "raw.log").write_text(
        "raw evaluator output", encoding="utf-8"
    )

    manifest = package_instance(
        runtime_path,
        runtime_path.parent,
        artifact_dir,
        row_loader=_artifact_row,
    )

    assert sorted(path.name for path in artifact_dir.iterdir()) == [
        "manifest.json",
        "official_result.json",
        "prediction.jsonl",
        "result.json",
    ]
    assert manifest.dataset_name == "SWE-bench/SWE-bench_Verified"
    assert manifest.dataset_revision == (
        "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
    )
    assert manifest.row_sha256 == ROW_SHA
    for filename, digest in manifest.files.items():
        assert digest == sha256_file(artifact_dir / filename)
    assert "raw evaluator output" not in "".join(
        path.read_text(encoding="utf-8") for path in artifact_dir.iterdir()
    )


def test_package_rejects_oversized_regular_payload_before_copying(
    tmp_path: Path,
) -> None:
    runtime_path, artifact_dir = _completed_output(
        tmp_path / "work", "owner__repo-1"
    )
    (runtime_path.parent / "result.json").write_bytes(
        b"x" * (oci_aggregate.ARTIFACT_FILE_BYTE_LIMIT + 1)
    )

    with pytest.raises(ArtifactContractError, match="byte limit"):
        package_instance(
            runtime_path,
            runtime_path.parent,
            artifact_dir,
            row_loader=_artifact_row,
        )

    assert not artifact_dir.exists() or not any(artifact_dir.iterdir())


def test_package_rejects_runtime_dataset_row_digest_mismatch(
    tmp_path: Path,
) -> None:
    runtime_path, artifact_dir = _completed_output(
        tmp_path / "work", "owner__repo-1"
    )
    different_row = {
        "repo": "owner/repo",
        "instance_id": "owner__repo-1",
        "base_commit": "d" * 40,
        "patch": "",
        "test_patch": "",
        "problem_statement": "Different pinned row",
        "hints_text": "",
        "created_at": "2026-01-01",
        "version": "1.0",
        "FAIL_TO_PASS": "[]",
        "PASS_TO_PASS": "[]",
        "environment_setup_commit": "e" * 40,
        "difficulty": "medium",
    }

    with pytest.raises(ValueError, match="dataset row digest mismatch"):
        package_instance(
            runtime_path,
            runtime_path.parent,
            artifact_dir,
            row_loader=lambda _instance_id: different_row,
        )

    assert not artifact_dir.exists() or not any(artifact_dir.iterdir())


@pytest.mark.parametrize("value", [1, "true"])
def test_package_rejects_coerced_official_boolean(
    tmp_path: Path,
    value: object,
) -> None:
    runtime_path, artifact_dir = _completed_output(
        tmp_path / "work", "owner__repo-1"
    )
    official_path = runtime_path.parent / "official_result.json"
    official = json.loads(official_path.read_text(encoding="utf-8"))
    official["completed"] = value
    official_path.write_text(json.dumps(official) + "\n", encoding="utf-8")

    with pytest.raises(ArtifactContractError, match="invalid safe artifact payload"):
        package_instance(
            runtime_path,
            runtime_path.parent,
            artifact_dir,
            row_loader=_artifact_row,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_package_rejects_nonfinite_number_in_nested_result_telemetry(
    tmp_path: Path,
    value: float,
) -> None:
    runtime_path, artifact_dir = _completed_output(
        tmp_path / "work", "owner__repo-1"
    )
    result_path = runtime_path.parent / "result.json"
    results = json.loads(result_path.read_text(encoding="utf-8"))
    results[0]["tool_invocations"] = [{"unsafe_number": value}]
    result_path.write_text(json.dumps(results) + "\n", encoding="utf-8")

    with pytest.raises(ArtifactContractError, match="invalid safe artifact payload"):
        package_instance(
            runtime_path,
            runtime_path.parent,
            artifact_dir,
            row_loader=_artifact_row,
        )


def test_package_rejects_exponent_overflow_in_nested_result_telemetry(
    tmp_path: Path,
) -> None:
    runtime_path, artifact_dir = _completed_output(
        tmp_path / "work", "owner__repo-1"
    )
    result_path = runtime_path.parent / "result.json"
    results = json.loads(result_path.read_text(encoding="utf-8"))
    results[0]["tool_invocations"] = [{"unsafe_number": 0}]
    raw = json.dumps(results).replace('"unsafe_number": 0', '"unsafe_number": 1e309')
    result_path.write_text(raw + "\n", encoding="utf-8")

    with pytest.raises(ArtifactContractError, match="invalid safe artifact payload"):
        package_instance(
            runtime_path,
            runtime_path.parent,
            artifact_dir,
            row_loader=_artifact_row,
        )


def test_package_rejects_completed_patch_without_model_history(
    tmp_path: Path,
) -> None:
    runtime_path, artifact_dir = _completed_output(
        tmp_path / "work",
        "owner__repo-1",
        model_invocations=[],
    )

    with pytest.raises(ArtifactContractError, match="invalid safe artifact payload"):
        package_instance(
            runtime_path,
            runtime_path.parent,
            artifact_dir,
            row_loader=_artifact_row,
        )


def test_package_rejects_ready_non_synthetic_result_without_base_commit(
    tmp_path: Path,
) -> None:
    runtime_path, artifact_dir = _completed_output(
        tmp_path / "work", "owner__repo-1"
    )
    result_path = runtime_path.parent / "result.json"
    results = json.loads(result_path.read_text(encoding="utf-8"))
    results[0]["base_commit"] = ""
    result_path.write_text(json.dumps(results) + "\n", encoding="utf-8")

    with pytest.raises(ArtifactContractError, match="inconsistent artifact bundle"):
        package_instance(
            runtime_path,
            runtime_path.parent,
            artifact_dir,
            row_loader=_artifact_row,
        )


def test_aggregate_rejects_oversized_regular_payload_before_parsing(
    tmp_path: Path,
) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    bundle = _package(artifacts, instance_id)
    (bundle / "result.json").write_bytes(
        b"x" * (oci_aggregate.ARTIFACT_FILE_BYTE_LIMIT + 1)
    )

    with pytest.raises(ArtifactContractError, match="byte limit"):
        aggregate_artifacts(
            "checkpoint_5",
            artifacts,
            tmp_path / "combined",
            expected_commit=COMMIT_SHA,
            repo_root=repo_root,
        )


def test_aggregate_enforces_cumulative_bundle_byte_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    bundle = _package(artifacts, instance_id)
    total_bytes = sum(path.stat().st_size for path in bundle.iterdir())
    monkeypatch.setattr(
        oci_aggregate, "ARTIFACT_TOTAL_BYTE_LIMIT", total_bytes - 1, raising=False
    )

    with pytest.raises(ArtifactContractError, match="total byte limit"):
        aggregate_artifacts(
            "checkpoint_5",
            artifacts,
            tmp_path / "combined",
            expected_commit=COMMIT_SHA,
            repo_root=repo_root,
        )


def test_aggregate_preserves_tracked_order_and_writes_separate_metrics(
    tmp_path: Path,
) -> None:
    instance_ids = ["owner__repo-2", "owner__repo-1"]
    repo_root = _repo_root(tmp_path, instance_ids)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _package(artifacts, "owner__repo-1", resolved=True)
    _package(artifacts, "owner__repo-2", resolved=False)

    summary_path = aggregate_artifacts(
        "checkpoint_5",
        artifacts,
        tmp_path / "combined",
        expected_commit=COMMIT_SHA,
        repo_root=repo_root,
    )

    results = json.loads(
        (summary_path.parent / "results.json").read_text(encoding="utf-8")
    )
    official = json.loads(
        (summary_path.parent / "official_results.json").read_text(encoding="utf-8")
    )
    assert [item["instance_id"] for item in results] == instance_ids
    assert [item["instance_id"] for item in official] == instance_ids
    summary = summary_path.read_text(encoding="utf-8")
    assert "Requested | 2" in summary
    assert "Internal success | 2" in summary
    assert "Official resolved | 1" in summary
    assert "Infrastructure failure | 0" in summary
    assert "| Official score | 50.00/100 |" in summary
    assert "| Official terminal coverage | 2/2 |" in summary
    assert "| Internal/official agreement | 1/2 |" in summary
    assert "| Model tokens | 30 |" in summary
    assert "| Model elapsed seconds | 2.000 |" in summary
    assert "| Engineering score | 57.50/100 |" in summary
    assert "| Official resolution | 1/2 | 40.00/80 |" in summary
    assert "| Non-infrastructure | 2/2 | 10.00/10 |" in summary
    assert "| Explicit internal/official agreement | 1/2 | 2.50/5 |" in summary
    assert "| Completed within time/token budget | 2/2 | 5.00/5 |" in summary
    assert (
        "| owner__repo-2 | ready | success | unresolved | 15 | 1.000 |"
        in summary
    )
    assert summary.index("| Official score |") < summary.index("| Engineering score |")


def test_aggregate_accepts_patch_only_unresolved_result(tmp_path: Path) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _package(
        artifacts,
        instance_id,
        agent_success=False,
        model_patch="diff --git a/a.py b/a.py\n",
    )

    summary_path = aggregate_artifacts(
        "checkpoint_5",
        artifacts,
        tmp_path / "combined",
        expected_commit=COMMIT_SHA,
        repo_root=repo_root,
    )

    [result] = json.loads(
        (summary_path.parent / "results.json").read_text(encoding="utf-8")
    )
    assert result["model_patch"] == "diff --git a/a.py b/a.py\n"
    assert result["patch_generated"] is True
    assert result["tests_passed"] is None
    assert result["agent_success"] is False


def test_aggregate_keeps_scorer_infrastructure_in_requested_denominator(
    tmp_path: Path,
) -> None:
    instance_ids = ["owner__repo-1", "owner__repo-2"]
    repo_root = _repo_root(tmp_path, instance_ids)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _package(artifacts, instance_ids[0], resolved=True)
    _package(
        artifacts,
        instance_ids[1],
        official_status="scorer_infra",
        agent_success=False,
    )

    summary_path = aggregate_artifacts(
        "checkpoint_5",
        artifacts,
        tmp_path / "combined",
        expected_commit=COMMIT_SHA,
        repo_root=repo_root,
    )

    summary = summary_path.read_text(encoding="utf-8")
    assert "| Official score | 50.00/100 |" in summary
    assert "| Official terminal coverage | 1/2 |" in summary
    assert "| Internal/official agreement | 1/1 |" in summary
    assert "| Infrastructure failure | 1 |" in summary
    assert "| Engineering score | 52.50/100 |" in summary


def test_aggregate_rejects_mixed_valid_and_malformed_invocations(
    tmp_path: Path,
) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    bundle = _package(artifacts, instance_id)
    results = json.loads((bundle / "result.json").read_text(encoding="utf-8"))
    results[0]["model_invocations"] = [
        _invocation(),
        {"model": PRIMARY_MODEL},
    ]
    _rewrite_bundle_json(bundle, "result.json", results)

    with pytest.raises(ArtifactContractError, match="invalid safe artifact payload"):
        aggregate_artifacts(
            "checkpoint_5",
            artifacts,
            tmp_path / "combined",
            expected_commit=COMMIT_SHA,
            repo_root=repo_root,
        )


def test_aggregate_rejects_garbage_invocation_status(tmp_path: Path) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    bundle = _package(artifacts, instance_id)
    results = json.loads((bundle / "result.json").read_text(encoding="utf-8"))
    results[0]["model_invocations"] = [_invocation(status="garbage")]
    _rewrite_bundle_json(bundle, "result.json", results)

    with pytest.raises(ArtifactContractError, match="invalid safe artifact payload"):
        aggregate_artifacts(
            "checkpoint_5",
            artifacts,
            tmp_path / "combined",
            expected_commit=COMMIT_SHA,
            repo_root=repo_root,
        )


def test_aggregate_rejects_non_list_model_invocations(tmp_path: Path) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    bundle = _package(artifacts, instance_id)
    results = json.loads((bundle / "result.json").read_text(encoding="utf-8"))
    results[0]["model_invocations"] = None
    _rewrite_bundle_json(bundle, "result.json", results)

    with pytest.raises(ArtifactContractError, match="invalid safe artifact payload"):
        aggregate_artifacts(
            "checkpoint_5",
            artifacts,
            tmp_path / "combined",
            expected_commit=COMMIT_SHA,
            repo_root=repo_root,
        )


def test_aggregate_rejects_forged_model_invocation_node(tmp_path: Path) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    bundle = _package(artifacts, instance_id)
    results = json.loads((bundle / "result.json").read_text(encoding="utf-8"))
    results[0]["model_invocations"][0]["node"] = "forged-node"
    _rewrite_bundle_json(bundle, "result.json", results)

    with pytest.raises(ArtifactContractError, match="invalid safe artifact payload"):
        aggregate_artifacts(
            "checkpoint_5",
            artifacts,
            tmp_path / "combined",
            expected_commit=COMMIT_SHA,
            repo_root=repo_root,
        )


def test_package_rejects_generation_attempt_history_mismatch(
    tmp_path: Path,
) -> None:
    runtime_path, artifact_dir = _completed_output(
        tmp_path / "work",
        "owner__repo-1",
        test_generation_attempts=1,
    )

    with pytest.raises(ArtifactContractError, match="invalid safe artifact payload"):
        package_instance(
            runtime_path,
            runtime_path.parent,
            artifact_dir,
            row_loader=_artifact_row,
        )


@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra", "failed_only", "legacy_alias_only"],
)
def test_aggregate_rejects_contradictory_generation_telemetry(
    tmp_path: Path,
    mutation: str,
) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    bundle = _package(artifacts, instance_id)
    results = json.loads((bundle / "result.json").read_text(encoding="utf-8"))
    result = results[0]
    generation = _invocation(node="test_generation")
    if mutation == "missing":
        result["test_generation_attempts"] = 1
    elif mutation == "extra":
        result["model_invocations"] = [generation]
        result["test_generation_attempts"] = 0
    elif mutation == "failed_only":
        result.update(
            coverage_status="generated_verified",
            coverage_proof=_coverage_proof(),
            model_invocations=[
                _invocation(
                    node="test_generation",
                    status="error",
                    error_class="RuntimeError",
                )
            ],
            test_generation_attempts=1,
        )
    else:
        result["model_invocations"] = [_invocation(node="test_generator")]
        result["test_generation_attempts"] = 1
    _rewrite_bundle_json(bundle, "result.json", results)

    with pytest.raises(ArtifactContractError, match="invalid safe artifact payload"):
        aggregate_artifacts(
            "checkpoint_5",
            artifacts,
            tmp_path / "combined",
            expected_commit=COMMIT_SHA,
            repo_root=repo_root,
        )


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("node", "plan"),
        ("node", "test_generator"),
        ("error_class", "RuntimeError"),
        ("output_tokens", 1),
        ("input_tokens", 0),
    ],
)
def test_aggregate_rejects_noncanonical_cancelled_invocation(
    tmp_path: Path,
    mutation: str,
    value: object,
) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    cancelled = _invocation(
        node="test_generation",
        status="cancelled",
        error_class="CancelledError",
        output_tokens=0,
    )
    bundle = _package(
        artifacts,
        instance_id,
        official_status="empty_patch",
        agent_success=False,
        model_patch="",
        model_invocations=[cancelled],
        failure_class="other",
        test_generation_attempts=1,
    )
    results = json.loads((bundle / "result.json").read_text(encoding="utf-8"))
    invocation = results[0]["model_invocations"][0]
    invocation[mutation] = value
    results[0]["test_generation_attempts"] = (
        1 if invocation["node"] == "test_generation" else 0
    )
    token_total = _serialized_token_total(results[0]["model_invocations"])
    assert token_total is not None
    results[0]["token_used"] = token_total
    _rewrite_bundle_json(bundle, "result.json", results)

    with pytest.raises(ArtifactContractError, match="invalid safe artifact payload"):
        aggregate_artifacts(
            "checkpoint_5",
            artifacts,
            tmp_path / "combined",
            expected_commit=COMMIT_SHA,
            repo_root=repo_root,
        )


def test_aggregate_rejects_token_used_below_complete_invocation_total(
    tmp_path: Path,
) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    bundle = _package(
        artifacts,
        instance_id,
        model_invocations=[
            _invocation(input_tokens=10, output_tokens=5),
            _invocation(node="reflect", input_tokens=7, output_tokens=3),
        ],
    )
    results = json.loads((bundle / "result.json").read_text(encoding="utf-8"))
    results[0]["token_used"] = 24
    _rewrite_bundle_json(bundle, "result.json", results)

    with pytest.raises(ArtifactContractError, match="invalid safe artifact payload"):
        aggregate_artifacts(
            "checkpoint_5",
            artifacts,
            tmp_path / "combined",
            expected_commit=COMMIT_SHA,
            repo_root=repo_root,
        )


def test_aggregate_counts_canonical_generation_telemetry(
    tmp_path: Path,
) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _package(
        artifacts,
        instance_id,
        model_invocations=[
            _invocation(input_tokens=10, output_tokens=5, elapsed_seconds=1.0),
            _invocation(
                node="test_generation",
                input_tokens=20,
                output_tokens=10,
                elapsed_seconds=2.0,
            ),
        ],
        test_generation_attempts=1,
    )

    summary_path = aggregate_artifacts(
        "checkpoint_5",
        artifacts,
        tmp_path / "combined",
        expected_commit=COMMIT_SHA,
        repo_root=repo_root,
    )

    summary = summary_path.read_text(encoding="utf-8")
    assert "| Model tokens | 45 |" in summary
    assert "| Model elapsed seconds | 3.000 |" in summary


@pytest.mark.parametrize(
    ("status", "error_class"),
    [("ok", "RuntimeError"), ("error", "")],
)
def test_aggregate_rejects_inconsistent_invocation_error_class(
    tmp_path: Path,
    status: str,
    error_class: str,
) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    bundle = _package(artifacts, instance_id)
    results = json.loads((bundle / "result.json").read_text(encoding="utf-8"))
    results[0]["model_invocations"] = [
        _invocation(status=status, error_class=error_class)
    ]
    _rewrite_bundle_json(bundle, "result.json", results)

    with pytest.raises(ArtifactContractError, match="invalid safe artifact payload"):
        aggregate_artifacts(
            "checkpoint_5",
            artifacts,
            tmp_path / "combined",
            expected_commit=COMMIT_SHA,
            repo_root=repo_root,
        )


@pytest.mark.parametrize(
    "error_class",
    ["ReadTimeout", "ConnectTimeout", "PoolTimeout", "WriteTimeout"],
)
def test_package_and_aggregate_accept_gateway_timeout_invocation(
    tmp_path: Path,
    error_class: str,
) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    producer_invocation = ModelInvocation(
        model=PRIMARY_MODEL,
        provider="primary",
        node="plan",
        elapsed_seconds=1.0,
        input_tokens=10,
        output_tokens=0,
        status="error",
        error_class=type(error_class, (TimeoutError,), {})(),
    )

    _package(
        artifacts,
        instance_id,
        official_status="empty_patch",
        agent_success=False,
        model_patch="",
        model_invocations=[producer_invocation.model_dump(mode="json")],
        failure_class="model_gateway_infra",
    )

    summary_path = aggregate_artifacts(
        "checkpoint_5",
        artifacts,
        tmp_path / "combined",
        expected_commit=COMMIT_SHA,
        repo_root=repo_root,
    )

    assert summary_path.is_file()


@pytest.mark.asyncio
async def test_internal_generation_timeout_packages_and_aggregates_cancelled_invocation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    state = AgentState(
        issue_url="https://github.com/owner/repo/issues/1",
        active_model=PRIMARY_MODEL,
        active_provider="primary",
    )

    async def in_flight_request(*_args, **_kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(
        test_generator,
        "_generation_prompt",
        lambda *_args: "bounded generation prompt",
    )
    monkeypatch.setattr(test_generator, "llm_call", in_flight_request)

    with pytest.raises(asyncio.TimeoutError):
        await wait_for_phase(
            test_generator.request_test_batch(state, "no_candidate"),
            timeout=0.01,
        )

    assert state.test_generation_attempts == 1
    assert len(state.model_history) == 1
    invocation = state.model_history[0]
    assert invocation.status == "cancelled"
    assert invocation.error_class == "CancelledError"
    assert invocation.output_tokens == 0
    assert invocation.input_tokens > 0
    assert state.token_usage == invocation.input_tokens

    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _package(
        artifacts,
        instance_id,
        official_status="empty_patch",
        agent_success=False,
        model_patch="",
        model_invocations=[invocation.model_dump(mode="json")],
        failure_class="other",
        test_generation_attempts=state.test_generation_attempts,
    )

    summary_path = aggregate_artifacts(
        "checkpoint_5",
        artifacts,
        tmp_path / "combined",
        expected_commit=COMMIT_SHA,
        repo_root=repo_root,
    )

    summary = summary_path.read_text(encoding="utf-8")
    assert f"| Model tokens | {invocation.input_tokens} |" in summary
    assert f"| Model elapsed seconds | {invocation.elapsed_seconds:.3f} |" in summary


def test_aggregate_rejects_completed_bundle_with_only_error_invocation(
    tmp_path: Path,
) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    bundle = _package(artifacts, instance_id)
    results = json.loads((bundle / "result.json").read_text(encoding="utf-8"))
    results[0]["model_invocations"] = [
        _invocation(status="error", error_class="RuntimeError")
    ]
    _rewrite_bundle_json(bundle, "result.json", results)

    with pytest.raises(ArtifactContractError, match="invalid safe artifact payload"):
        aggregate_artifacts(
            "checkpoint_5",
            artifacts,
            tmp_path / "combined",
            expected_commit=COMMIT_SHA,
            repo_root=repo_root,
        )


@pytest.mark.parametrize("value", [1, "true"])
def test_aggregate_rejects_coerced_official_boolean(
    tmp_path: Path,
    value: object,
) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    bundle = _package(artifacts, instance_id)
    official = json.loads(
        (bundle / "official_result.json").read_text(encoding="utf-8")
    )
    official["completed"] = value
    _rewrite_bundle_json(bundle, "official_result.json", official)

    with pytest.raises(ArtifactContractError, match="invalid safe artifact payload"):
        aggregate_artifacts(
            "checkpoint_5",
            artifacts,
            tmp_path / "combined",
            expected_commit=COMMIT_SHA,
            repo_root=repo_root,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_aggregate_rejects_nonfinite_nested_result_number(
    tmp_path: Path,
    value: float,
) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    bundle = _package(artifacts, instance_id)
    results = json.loads((bundle / "result.json").read_text(encoding="utf-8"))
    results[0]["replay"] = {"unsafe_number": value}
    _rewrite_bundle_json(bundle, "result.json", results)

    with pytest.raises(ArtifactContractError, match="invalid safe artifact payload"):
        aggregate_artifacts(
            "checkpoint_5",
            artifacts,
            tmp_path / "combined",
            expected_commit=COMMIT_SHA,
            repo_root=repo_root,
        )


def test_aggregate_rejects_exponent_overflow_in_nested_result_number(
    tmp_path: Path,
) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    bundle = _package(artifacts, instance_id)
    results = json.loads((bundle / "result.json").read_text(encoding="utf-8"))
    results[0]["replay"] = {"unsafe_number": 0}
    raw = json.dumps(results).replace('"unsafe_number": 0', '"unsafe_number": 1e309')
    _rewrite_bundle_bytes(bundle, "result.json", (raw + "\n").encode())

    with pytest.raises(ArtifactContractError, match="invalid safe artifact payload"):
        aggregate_artifacts(
            "checkpoint_5",
            artifacts,
            tmp_path / "combined",
            expected_commit=COMMIT_SHA,
            repo_root=repo_root,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_aggregate_strictly_rejects_nonfinite_prediction_number(
    tmp_path: Path,
    value: float,
) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    bundle = _package(artifacts, instance_id)
    prediction = json.loads(
        (bundle / "prediction.jsonl").read_text(encoding="utf-8")
    )
    prediction["model_patch"] = value
    _rewrite_bundle_json(bundle, "prediction.jsonl", prediction)

    with pytest.raises(ArtifactContractError, match="invalid prediction row"):
        aggregate_artifacts(
            "checkpoint_5",
            artifacts,
            tmp_path / "combined",
            expected_commit=COMMIT_SHA,
            repo_root=repo_root,
        )


def test_aggregate_rejects_missing_internal_verdict(
    tmp_path: Path,
) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    bundle = _package(artifacts, instance_id)
    results = json.loads((bundle / "result.json").read_text(encoding="utf-8"))
    del results[0]["agent_success"]
    _rewrite_bundle_json(bundle, "result.json", results)

    with pytest.raises(ArtifactContractError, match="invalid safe artifact payload"):
        aggregate_artifacts(
            "checkpoint_5",
            artifacts,
            tmp_path / "combined",
            expected_commit=COMMIT_SHA,
            repo_root=repo_root,
        )


@pytest.mark.parametrize("failure_class", [None, "unknown_failure"])
def test_aggregate_rejects_missing_or_unknown_failure_class(
    tmp_path: Path,
    failure_class: str | None,
) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    bundle = _package(artifacts, instance_id)
    results = json.loads((bundle / "result.json").read_text(encoding="utf-8"))
    if failure_class is None:
        del results[0]["failure_class"]
    else:
        results[0]["failure_class"] = failure_class
    _rewrite_bundle_json(bundle, "result.json", results)

    with pytest.raises(ArtifactContractError, match="invalid safe artifact payload"):
        aggregate_artifacts(
            "checkpoint_5",
            artifacts,
            tmp_path / "combined",
            expected_commit=COMMIT_SHA,
            repo_root=repo_root,
        )


@pytest.mark.parametrize(
    "failure_class",
    ["infra", "model_gateway_infra", "coverage_infra"],
)
def test_every_infrastructure_class_receives_zero_noninfra_and_budget_credit(
    tmp_path: Path,
    failure_class: str,
) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _package(
        artifacts,
        instance_id,
        official_status="scorer_infra",
        agent_success=False,
        failure_class=failure_class,
    )

    summary_path = aggregate_artifacts(
        "checkpoint_5",
        artifacts,
        tmp_path / "combined",
        expected_commit=COMMIT_SHA,
        repo_root=repo_root,
    )

    summary = summary_path.read_text(encoding="utf-8")
    assert "| Non-infrastructure | 0/1 | 0.00/10 |" in summary
    assert "| Completed within time/token budget | 0/1 | 0.00/5 |" in summary


@pytest.mark.parametrize(
    ("field", "value"),
    [("id", "other-id"), ("coverage_proof", None)],
)
def test_aggregate_rejects_tampered_success_identity_or_coverage(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    bundle = _package(artifacts, instance_id, resolved=True)
    results = json.loads((bundle / "result.json").read_text(encoding="utf-8"))
    results[0][field] = value
    _rewrite_bundle_json(bundle, "result.json", results)

    with pytest.raises(ArtifactContractError, match="invalid safe artifact payload"):
        aggregate_artifacts(
            "checkpoint_5",
            artifacts,
            tmp_path / "combined",
            expected_commit=COMMIT_SHA,
            repo_root=repo_root,
        )


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("primary", ESCALATION_MODEL),
        ("escalation", PRIMARY_MODEL),
    ],
)
def test_aggregate_rejects_provider_model_mismatch(
    tmp_path: Path,
    provider: str,
    model: str,
) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    bundle = _package(artifacts, instance_id)
    results = json.loads((bundle / "result.json").read_text(encoding="utf-8"))
    results[0]["model_invocations"] = [
        _invocation(provider=provider, model=model)
    ]
    _rewrite_bundle_json(bundle, "result.json", results)

    with pytest.raises(ArtifactContractError, match="invalid safe artifact payload"):
        aggregate_artifacts(
            "checkpoint_5",
            artifacts,
            tmp_path / "combined",
            expected_commit=COMMIT_SHA,
            repo_root=repo_root,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("elapsed_seconds", float("nan")),
        ("elapsed_seconds", float("inf")),
        ("elapsed_seconds", -1.0),
        ("input_tokens", -1),
        ("output_tokens", -1),
    ],
)
def test_aggregate_rejects_invalid_invocation_numbers(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    bundle = _package(artifacts, instance_id)
    results = json.loads((bundle / "result.json").read_text(encoding="utf-8"))
    results[0]["model_invocations"] = [_invocation(**{field: value})]
    _rewrite_bundle_json(bundle, "result.json", results)

    with pytest.raises(ArtifactContractError, match="invalid safe artifact payload"):
        aggregate_artifacts(
            "checkpoint_5",
            artifacts,
            tmp_path / "combined",
            expected_commit=COMMIT_SHA,
            repo_root=repo_root,
        )


@pytest.mark.parametrize("scope", ["result", "invocation"])
def test_aggregate_rejects_extra_result_or_invocation_fields(
    tmp_path: Path,
    scope: str,
) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    bundle = _package(artifacts, instance_id)
    results = json.loads((bundle / "result.json").read_text(encoding="utf-8"))
    target = results[0] if scope == "result" else results[0]["model_invocations"][0]
    target["unexpected"] = "unsafe"
    _rewrite_bundle_json(bundle, "result.json", results)

    with pytest.raises(ArtifactContractError, match="invalid safe artifact payload"):
        aggregate_artifacts(
            "checkpoint_5",
            artifacts,
            tmp_path / "combined",
            expected_commit=COMMIT_SHA,
            repo_root=repo_root,
        )


@pytest.mark.parametrize(
    "mutation",
    ["terminal_official", "model_patch", "model_invocations"],
)
def test_nonready_runtime_rejects_terminal_patch_or_invocations(
    tmp_path: Path,
    mutation: str,
) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    bundle = _package(
        artifacts,
        instance_id,
        official_status="scorer_infra",
        model_patch="",
        model_invocations=[],
        failure_class="infra",
        runtime_status="dataset_infra",
    )
    if mutation == "terminal_official":
        official = json.loads(
            (bundle / "official_result.json").read_text(encoding="utf-8")
        )
        official.update(
            status="unresolved",
            submitted=True,
            completed=True,
            error_class="",
        )
        _rewrite_bundle_json(bundle, "official_result.json", official)
    else:
        results = json.loads(
            (bundle / "result.json").read_text(encoding="utf-8")
        )
        if mutation == "model_patch":
            results[0].update(
                success=True,
                agent_success=True,
                final_phase="DONE",
                model_patch="diff --git a/a.py b/a.py\n",
                patch_generated=True,
                model_invocations=[_invocation(node="test_generation")],
                coverage_status="generated_verified",
                coverage_proof=_coverage_proof(),
                coverage_failure_reason="",
                test_generation_attempts=1,
                failure_class="agent_success",
            )
            prediction = json.loads(
                (bundle / "prediction.jsonl").read_text(encoding="utf-8")
            )
            prediction["model_patch"] = results[0]["model_patch"]
            _rewrite_bundle_json(bundle, "prediction.jsonl", prediction)
        else:
            results[0]["model_invocations"] = [_invocation()]
        token_total = _serialized_token_total(
            results[0]["model_invocations"]
        )
        assert token_total is not None
        results[0]["token_used"] = token_total
        _rewrite_bundle_json(bundle, "result.json", results)

    with pytest.raises(ArtifactContractError, match="inconsistent artifact bundle"):
        aggregate_artifacts(
            "checkpoint_5",
            artifacts,
            tmp_path / "combined",
            expected_commit=COMMIT_SHA,
            repo_root=repo_root,
        )


@pytest.mark.parametrize(
    ("scope", "field", "value"),
    [
        ("official", "schema_version", 2),
        ("official", "instance_id", "other__repo-2"),
        ("official", "status", "empty_patch"),
        ("official", "submitted", True),
        ("official", "completed", True),
        ("official", "resolved", True),
        ("result", "id", "other__repo-2"),
        ("result", "mode", "legacy"),
        ("result", "evaluation_mode", "oracle_files"),
        ("result", "model", ESCALATION_MODEL),
        ("result", "commit_sha", "b" * 40),
        ("result", "repo", "other/repo"),
        ("result", "issue_url", "https://example.test/issues/1"),
        ("result", "issue_title", "Different title"),
        ("result", "success", True),
        ("result", "agent_success", True),
        ("result", "official_resolved", False),
        ("result", "waiting_for_user", True),
        ("result", "final_phase", "DONE"),
        ("result", "run_id", "run-1"),
        ("result", "trace_id", "trace-1"),
        ("result", "turns_taken", 1),
        ("result", "token_used", 1),
        ("result", "error", None),
        ("result", "replay", {}),
        ("result", "replay_error", "ValueError"),
        ("result", "models_used", [ESCALATION_MODEL]),
        ("result", "escalated", True),
        ("result", "escalation_reason", "repeated_no_progress"),
        ("result", "model_invocations", [_invocation()]),
        ("result", "tool_invocations", [{"tool_name": "search"}]),
        ("result", "unique_evidence_count", 1),
        ("result", "max_consecutive_no_progress", 1),
        ("result", "attempt_outcome_summary", "attempted"),
        ("result", "base_commit", "d" * 40),
        ("result", "model_patch", "diff --git a/a.py b/a.py\n"),
        ("result", "tests_passed", True),
        ("result", "coverage_status", "pending"),
        ("result", "coverage_test_files", ["tests/test_auth.py"]),
        ("result", "coverage_test_command", "pytest tests/test_auth.py"),
        ("result", "coverage_proof", _coverage_proof()),
        ("result", "coverage_failure_reason", "other"),
        ("result", "test_generation_attempts", 1),
        ("result", "failure_class", "other"),
        ("result", "instance_id", "other__repo-2"),
    ],
)
@pytest.mark.parametrize("boundary", ["package", "aggregate"])
def test_nonready_runtime_rejects_every_fixed_nonproducer_field_at_each_boundary(
    tmp_path: Path,
    scope: str,
    field: str,
    value: object,
    boundary: str,
) -> None:
    instance_id = "owner__repo-1"
    work = tmp_path / "work"
    runtime_path, artifact_dir = _completed_output(
        work,
        instance_id,
        official_status="scorer_infra",
        model_patch="",
        model_invocations=[],
        failure_class="infra",
        runtime_status="dataset_infra",
    )
    filename = (
        "official_result.json" if scope == "official" else "result.json"
    )
    source_path = runtime_path.parent / filename
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    target = payload if scope == "official" else payload[0]
    target[field] = value
    if boundary == "package":
        source_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        with pytest.raises(
            ArtifactContractError,
            match=(
                "invalid safe artifact payload|inconsistent artifact bundle|"
                "cross-file artifact identity mismatch"
            ),
        ):
            package_instance(
                runtime_path,
                runtime_path.parent,
                artifact_dir,
                row_loader=_artifact_row,
            )
        return

    package_instance(
        runtime_path,
        runtime_path.parent,
        artifact_dir,
        row_loader=_artifact_row,
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    bundle = artifacts / f"bundle-{instance_id}"
    artifact_dir.rename(bundle)
    _rewrite_bundle_json(bundle, filename, payload)
    repo_root = _repo_root(tmp_path, [instance_id])

    with pytest.raises(
        ArtifactContractError,
        match=(
            "invalid safe artifact payload|inconsistent artifact bundle|"
            "cross-file artifact identity mismatch"
        ),
    ):
        aggregate_artifacts(
            "checkpoint_5",
            artifacts,
            tmp_path / "combined",
            expected_commit=COMMIT_SHA,
            repo_root=repo_root,
        )


def test_ready_runtime_rejects_empty_base_commit(tmp_path: Path) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    bundle = _package(artifacts, instance_id)
    results = json.loads((bundle / "result.json").read_text(encoding="utf-8"))
    results[0]["base_commit"] = ""
    _rewrite_bundle_json(bundle, "result.json", results)

    with pytest.raises(ArtifactContractError, match="inconsistent artifact bundle"):
        aggregate_artifacts(
            "checkpoint_5",
            artifacts,
            tmp_path / "combined",
            expected_commit=COMMIT_SHA,
            repo_root=repo_root,
        )


def test_aggregate_rejects_models_used_history_mismatch(tmp_path: Path) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    bundle = _package(artifacts, instance_id)
    results = json.loads((bundle / "result.json").read_text(encoding="utf-8"))
    results[0]["models_used"] = [ESCALATION_MODEL]
    _rewrite_bundle_json(bundle, "result.json", results)

    with pytest.raises(ArtifactContractError, match="invalid safe artifact payload"):
        aggregate_artifacts(
            "checkpoint_5",
            artifacts,
            tmp_path / "combined",
            expected_commit=COMMIT_SHA,
            repo_root=repo_root,
        )


def test_aggregate_rejects_ready_completed_result_without_invocations(
    tmp_path: Path,
) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    bundle = _package(artifacts, instance_id)
    results = json.loads((bundle / "result.json").read_text(encoding="utf-8"))
    results[0]["model_invocations"] = []
    _rewrite_bundle_json(bundle, "result.json", results)

    with pytest.raises(ArtifactContractError, match="invalid safe artifact payload"):
        aggregate_artifacts(
            "checkpoint_5",
            artifacts,
            tmp_path / "combined",
            expected_commit=COMMIT_SHA,
            repo_root=repo_root,
        )


def test_valid_nonready_infrastructure_bundle_scores_zero(
    tmp_path: Path,
) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _package(
        artifacts,
        instance_id,
        official_status="scorer_infra",
        model_patch="",
        model_invocations=[],
        failure_class="infra",
        runtime_status="dataset_infra",
    )

    summary_path = aggregate_artifacts(
        "checkpoint_5",
        artifacts,
        tmp_path / "combined",
        expected_commit=COMMIT_SHA,
        repo_root=repo_root,
    )

    summary = summary_path.read_text(encoding="utf-8")
    assert "| Non-infrastructure | 0/1 | 0.00/10 |" in summary
    assert "| Completed within time/token budget | 0/1 | 0.00/5 |" in summary
    assert "| Engineering score | 0.00/100 |" in summary


async def test_ready_generation_failure_packages_and_aggregates_real_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    runtime = RuntimeRecord(
        mode="checkpoint_5",
        instance_id=instance_id,
        commit_sha=COMMIT_SHA,
        row_sha256=ROW_SHA,
        status="ready",
        remote_image="swebench/sweb.eval.x86_64.owner_repo-1:latest",
        image_sha=IMAGE_SHA,
    )
    runtime_path = write_model(output_dir / "runtime.json", runtime)
    for name in oci_runner._SCORER_FORBIDDEN_ENV:
        monkeypatch.delenv(name, raising=False)

    async def fail_generation(*args, **kwargs):
        raise RuntimeError("generation failed")

    await generate_instance(
        runtime_path,
        output_dir,
        agent_runner=fail_generation,
    )

    tags = {runtime.image_sha: runtime.image_sha}

    def command_runner(argv, **kwargs):
        command = list(argv)
        if command[1:3] == ["image", "tag"]:
            tags[command[4]] = tags[command[3]]
            return BoundedProcessResult(command, 0, "", "")
        if command[1:3] == ["image", "inspect"]:
            digest = tags.get(command[-1], "")
            return BoundedProcessResult(
                command,
                0 if digest else 1,
                digest,
                "",
            )
        if command[1:3] == ["image", "rm"]:
            tags.pop(command[-1], None)
            return BoundedProcessResult(command, 0, "", "")
        raise AssertionError(f"unexpected command: {command}")

    def empty_patch_scorer(**kwargs):
        report_path = Path.cwd() / "official-report.json"
        report_path.write_text(
            json.dumps(
                {
                    "submitted_ids": [instance_id],
                    "completed_ids": [],
                    "resolved_ids": [],
                    "unresolved_ids": [],
                    "empty_patch_ids": [instance_id],
                    "error_ids": [],
                }
            ),
            encoding="utf-8",
        )
        return report_path

    official = score_instance(
        runtime_path,
        output_dir,
        scorer=empty_patch_scorer,
        command_runner=command_runner,
        row_loader=_artifact_row,
    )
    assert official.status == "empty_patch"

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    package_instance(
        runtime_path,
        output_dir,
        artifacts / f"bundle-{instance_id}",
        row_loader=_artifact_row,
    )
    summary_path = aggregate_artifacts(
        "checkpoint_5",
        artifacts,
        tmp_path / "combined",
        expected_commit=COMMIT_SHA,
        repo_root=repo_root,
    )

    [combined] = json.loads(
        (summary_path.parent / "results.json").read_text(encoding="utf-8")
    )
    assert combined["base_commit"] == ""
    assert combined["failure_class"] == "infra"
    summary = summary_path.read_text(encoding="utf-8")
    assert "| Internal/official agreement | 0/1 |" in summary
    assert "| Explicit internal/official agreement | 0/1 | 0.00/5 |" in summary
    assert "| Engineering score | 0.00/100 |" in summary


def test_incomplete_official_result_receives_no_budget_credit(
    tmp_path: Path,
) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _package(
        artifacts,
        instance_id,
        official_status="empty_patch",
        agent_success=False,
        model_patch="",
    )

    summary_path = aggregate_artifacts(
        "checkpoint_5",
        artifacts,
        tmp_path / "combined",
        expected_commit=COMMIT_SHA,
        repo_root=repo_root,
    )

    summary = summary_path.read_text(encoding="utf-8")
    assert "| Non-infrastructure | 0/1 | 0.00/10 |" in summary
    assert "| Completed within time/token budget | 0/1 | 0.00/5 |" in summary


@pytest.mark.parametrize(
    ("input_tokens", "expected_score"),
    [(100_000, "15.00"), (100_001, "10.00")],
)
def test_engineering_score_has_exact_token_budget_threshold(
    tmp_path: Path,
    input_tokens: int,
    expected_score: str,
) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _package(
        artifacts,
        instance_id,
        input_tokens=input_tokens,
        output_tokens=0,
    )

    summary_path = aggregate_artifacts(
        "checkpoint_5",
        artifacts,
        tmp_path / "combined",
        expected_commit=COMMIT_SHA,
        repo_root=repo_root,
    )

    summary = summary_path.read_text(encoding="utf-8")
    assert f"| Model tokens | {input_tokens} |" in summary
    assert f"| Engineering score | {expected_score}/100 |" in summary


def test_aggregate_rejects_bundle_replaced_during_bound_check(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    probe = tmp_path / "symlink-probe"
    try:
        probe.symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pytest.skip("operating system denied symlink creation")
    probe.unlink()

    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    bundle = _package(artifacts, instance_id)

    external_root = tmp_path / "external"
    external_artifacts = external_root / "artifacts"
    external_artifacts.mkdir(parents=True)
    external_bundle = _package(external_artifacts, instance_id, resolved=True)
    moved_bundle = tmp_path / "moved-bundle"
    real_stat = os.stat
    raced = False

    def racing_stat(path, *args, **kwargs):
        nonlocal raced
        if not raced and (
            (path == bundle and kwargs.get("follow_symlinks") is not False)
            or (path == bundle.name and kwargs.get("dir_fd") is not None)
        ):
            raced = True
            bundle.rename(moved_bundle)
            bundle.symlink_to(external_bundle, target_is_directory=True)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", racing_stat)

    with pytest.raises(
        ArtifactContractError, match="artifact bundle changed during snapshot"
    ):
        aggregate_artifacts(
            "checkpoint_5",
            artifacts,
            tmp_path / "combined",
            expected_commit=COMMIT_SHA,
            repo_root=repo_root,
        )
    assert raced is True


def test_snapshot_file_open_uses_nonblocking_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _package(artifacts, instance_id)
    real_open = os.open
    observed_flags: list[int] = []

    def checking_open(path, flags, *args, **kwargs):
        if path == "result.json":
            observed_flags.append(flags)
            raise OSError("stop before opening payload")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", checking_open)

    with pytest.raises(ArtifactContractError):
        aggregate_artifacts(
            "checkpoint_5",
            artifacts,
            tmp_path / "combined",
            expected_commit=COMMIT_SHA,
            repo_root=repo_root,
        )
    assert observed_flags
    assert observed_flags[0] & os.O_NONBLOCK


def test_aggregate_rejects_fifo_payload_without_blocking(tmp_path: Path) -> None:
    mkfifo = getattr(os, "mkfifo", None)
    if mkfifo is None:
        pytest.skip("operating system does not provide mkfifo")
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    bundle = _package(artifacts, instance_id)
    (bundle / "result.json").unlink()
    mkfifo(bundle / "result.json")

    started = time.monotonic()
    with pytest.raises(ArtifactContractError, match="artifact file set mismatch"):
        aggregate_artifacts(
            "checkpoint_5",
            artifacts,
            tmp_path / "combined",
            expected_commit=COMMIT_SHA,
            repo_root=repo_root,
        )
    assert time.monotonic() - started < 1.0


def test_aggregate_rejects_root_symlink_to_external_valid_bundle(
    tmp_path: Path,
) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    external = tmp_path / "external"
    external.mkdir()
    bundle = _package(external, instance_id)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    try:
        (artifacts / "linked-bundle").symlink_to(bundle, target_is_directory=True)
    except OSError:
        pytest.skip("operating system denied symlink creation")

    with pytest.raises(ArtifactContractError):
        aggregate_artifacts(
            "checkpoint_5",
            artifacts,
            tmp_path / "combined",
            expected_commit=COMMIT_SHA,
            repo_root=repo_root,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "extra",
        "commit",
        "hash",
        "unsafe_file",
        "unsafe_dir",
        "root_file",
        "root_dir_without_manifest",
        "root_symlink",
    ],
)
def test_aggregate_rejects_invalid_artifact_sets(tmp_path: Path, mutation: str) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    bundle = _package(artifacts, instance_id)

    if mutation == "missing":
        (bundle / "official_result.json").unlink()
    elif mutation == "duplicate":
        shutil.copytree(bundle, artifacts / "duplicate-bundle")
    elif mutation == "extra":
        extra = _package(artifacts, "owner__other-2")
        assert extra.exists()
    elif mutation == "commit":
        manifest = json.loads((bundle / "manifest.json").read_text())
        manifest["commit_sha"] = "c" * 40
        (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    elif mutation == "hash":
        (bundle / "prediction.jsonl").write_text("{}\n", encoding="utf-8")
    elif mutation == "unsafe_file":
        (bundle / "raw.log").write_text("unsafe", encoding="utf-8")
    elif mutation == "unsafe_dir":
        (bundle / "raw").mkdir()
    elif mutation == "root_file":
        (artifacts / "unexpected.txt").write_text("unsafe", encoding="utf-8")
    elif mutation == "root_dir_without_manifest":
        (artifacts / "unmanifested").mkdir()
    else:
        try:
            (artifacts / "linked-bundle").symlink_to(bundle, target_is_directory=True)
        except OSError:
            pytest.skip("operating system denied symlink creation")

    with pytest.raises(ArtifactContractError):
        aggregate_artifacts(
            "checkpoint_5",
            artifacts,
            tmp_path / "combined",
            expected_commit=COMMIT_SHA,
            repo_root=repo_root,
        )
