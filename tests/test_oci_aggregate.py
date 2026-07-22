from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from eval import oci_aggregate
from eval.oci_aggregate import ArtifactContractError, aggregate_artifacts
from eval.oci_contract import OfficialResult, RuntimeRecord, sha256_file, write_model
from eval.oci_runner import package_instance
from eval.swe_bench import write_predictions

COMMIT_SHA = "a" * 40
IMAGE_SHA = "sha256:" + "b" * 64
PRIMARY_MODEL = "gemini-3.5-flash:stable"
_DEFAULT_MODEL_INVOCATIONS = object()


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
    agent_success: bool | None = None,
    input_tokens: object = 10,
    output_tokens: object = 5,
    elapsed_seconds: object = 1.0,
    model_invocations: object = _DEFAULT_MODEL_INVOCATIONS,
) -> tuple[Path, Path]:
    output_dir = root / "output"
    artifact_dir = root / "artifact"
    output_dir.mkdir(parents=True)
    runtime = RuntimeRecord(
        mode="checkpoint_5",
        instance_id=instance_id,
        commit_sha=COMMIT_SHA,
        status="ready",
        remote_image="swebench/sweb.eval.x86_64.owner_repo-1:latest",
        image_sha=IMAGE_SHA,
    )
    runtime_path = write_model(output_dir / "runtime.json", runtime)
    result = {
        "id": instance_id,
        "mode": "agent_v2",
        "instance_id": instance_id,
        "commit_sha": COMMIT_SHA,
        "model": PRIMARY_MODEL,
        "model_patch": "diff --git a/a.py b/a.py\n",
        "success": resolved,
        "agent_success": resolved if agent_success is None else agent_success,
        "failure_class": "success" if resolved else "tests",
        "model_invocations": (
            [
                {
                    "model": PRIMARY_MODEL,
                    "provider": "openai-compatible",
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
        ),
    }
    (output_dir / "result.json").write_text(
        json.dumps([result]) + "\n", encoding="utf-8"
    )
    write_predictions([result], output_dir / "prediction.jsonl")
    status = official_status or ("resolved" if resolved else "unresolved")
    official = OfficialResult(
        instance_id=instance_id,
        status=status,
        submitted=status != "scorer_infra",
        completed=status in {"resolved", "unresolved"},
        resolved=status == "resolved",
        error_class="DockerUnavailable" if status == "scorer_infra" else "",
    )
    write_model(output_dir / "official_result.json", official)
    return runtime_path, artifact_dir


def _package(
    artifacts_root: Path,
    instance_id: str,
    *,
    resolved: bool = False,
    official_status: str | None = None,
    agent_success: bool | None = None,
    input_tokens: object = 10,
    output_tokens: object = 5,
    elapsed_seconds: object = 1.0,
    model_invocations: object = _DEFAULT_MODEL_INVOCATIONS,
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
    )
    package_instance(runtime_path, runtime_path.parent, artifact_dir)
    destination = artifacts_root / f"bundle-{instance_id}"
    artifact_dir.rename(destination)
    return destination


def test_package_copies_only_safe_hash_bound_files(tmp_path: Path) -> None:
    runtime_path, artifact_dir = _completed_output(tmp_path / "work", "owner__repo-1")
    (runtime_path.parent / "raw.log").write_text(
        "raw evaluator output", encoding="utf-8"
    )

    manifest = package_instance(runtime_path, runtime_path.parent, artifact_dir)

    assert sorted(path.name for path in artifact_dir.iterdir()) == [
        "manifest.json",
        "official_result.json",
        "prediction.jsonl",
        "result.json",
    ]
    assert manifest.dataset_name == "SWE-bench/SWE-bench_Verified"
    assert manifest.dataset_revision == "main"
    for filename, digest in manifest.files.items():
        assert digest == sha256_file(artifact_dir / filename)
    assert "raw evaluator output" not in "".join(
        path.read_text(encoding="utf-8") for path in artifact_dir.iterdir()
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
    assert "Internal success | 1" in summary
    assert "Official resolved | 1" in summary
    assert "Infrastructure failure | 0" in summary
    assert "| Official score | 50.00/100 |" in summary
    assert "| Official terminal coverage | 2/2 |" in summary
    assert "| Internal/official agreement | 2/2 |" in summary
    assert "| Model tokens | 30 |" in summary
    assert "| Model elapsed seconds | 2.000 |" in summary
    assert "| Engineering score | 60.00/100 |" in summary
    assert summary.index("| Official score |") < summary.index("| Engineering score |")


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
    assert "| Engineering score | 55.00/100 |" in summary


def test_model_usage_ignores_invalid_numeric_projections() -> None:
    result = {
        "model_invocations": [
            {
                "input_tokens": True,
                "output_tokens": -3,
                "elapsed_seconds": float("nan"),
            },
            {
                "input_tokens": 4,
                "output_tokens": 2.5,
                "elapsed_seconds": float("inf"),
            },
            {
                "input_tokens": 6,
                "output_tokens": 7,
                "elapsed_seconds": 1.25,
            },
        ]
    }

    assert oci_aggregate._model_usage(result) == (17, 1.25)
    assert oci_aggregate._model_usage({"model_invocations": {}}) == (0, 0.0)


def test_model_usage_ignores_overflowing_duration() -> None:
    result = {
        "model_invocations": [
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "elapsed_seconds": 10**400,
            }
        ]
    }

    assert oci_aggregate._model_usage(result) == (0, 0.0)


def test_aggregate_ignores_non_list_model_invocations(tmp_path: Path) -> None:
    instance_id = "owner__repo-1"
    repo_root = _repo_root(tmp_path, [instance_id])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _package(artifacts, instance_id, model_invocations=None)

    summary_path = aggregate_artifacts(
        "checkpoint_5",
        artifacts,
        tmp_path / "combined",
        expected_commit=COMMIT_SHA,
        repo_root=repo_root,
    )

    summary = summary_path.read_text(encoding="utf-8")
    assert "| Model tokens | 0 |" in summary
    assert "| Model elapsed seconds | 0.000 |" in summary
    assert "| Primary model invocations | 0 |" in summary


@pytest.mark.parametrize(
    ("input_tokens", "expected_score"),
    [(100_000, "20.00"), (100_001, "15.00")],
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
