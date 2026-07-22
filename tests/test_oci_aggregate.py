from __future__ import annotations

import json
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

    assert oci_aggregate.main([
        "--mode", "baseline_50",
        "--artifacts-dir", str(tmp_path / "artifacts"),
        "--output-dir", str(tmp_path / "combined"),
        "--expected-commit", COMMIT_SHA,
    ]) == 0
    assert seen["mode"] == "baseline_50"


def test_aggregate_cli_rejects_retired_baseline_10(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        oci_aggregate.main([
            "--mode", "baseline_10",
            "--artifacts-dir", str(tmp_path / "artifacts"),
            "--output-dir", str(tmp_path / "combined"),
            "--expected-commit", COMMIT_SHA,
        ])


def _completed_output(
    root: Path,
    instance_id: str,
    *,
    resolved: bool = False,
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
        "agent_success": resolved,
        "failure_class": "success" if resolved else "tests",
        "model_invocations": [
            {
                "model": PRIMARY_MODEL,
                "provider": "openai-compatible",
                "node": "plan",
                "elapsed_seconds": 1.0,
                "input_tokens": 10,
                "output_tokens": 5,
                "status": "ok",
                "error_class": "",
            }
        ],
    }
    (output_dir / "result.json").write_text(
        json.dumps([result]) + "\n", encoding="utf-8"
    )
    write_predictions([result], output_dir / "prediction.jsonl")
    official = OfficialResult(
        instance_id=instance_id,
        status="resolved" if resolved else "unresolved",
        submitted=True,
        completed=True,
        resolved=resolved,
    )
    write_model(output_dir / "official_result.json", official)
    return runtime_path, artifact_dir


def _package(
    artifacts_root: Path,
    instance_id: str,
    *,
    resolved: bool = False,
) -> Path:
    work = artifacts_root / f"work-{instance_id}"
    runtime_path, artifact_dir = _completed_output(
        work, instance_id, resolved=resolved
    )
    package_instance(runtime_path, runtime_path.parent, artifact_dir)
    destination = artifacts_root / f"bundle-{instance_id}"
    artifact_dir.rename(destination)
    return destination


def test_package_copies_only_safe_hash_bound_files(tmp_path: Path) -> None:
    runtime_path, artifact_dir = _completed_output(
        tmp_path / "work", "owner__repo-1"
    )
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
        (summary_path.parent / "official_results.json").read_text(
            encoding="utf-8"
        )
    )
    assert [item["instance_id"] for item in results] == instance_ids
    assert [item["instance_id"] for item in official] == instance_ids
    summary = summary_path.read_text(encoding="utf-8")
    assert "Requested | 2" in summary
    assert "Internal success | 1" in summary
    assert "Official resolved | 1" in summary
    assert "Infrastructure failure | 0" in summary


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
    ],
)
def test_aggregate_rejects_invalid_artifact_sets(
    tmp_path: Path, mutation: str
) -> None:
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
    else:
        (bundle / "raw").mkdir()

    with pytest.raises(ArtifactContractError):
        aggregate_artifacts(
            "checkpoint_5",
            artifacts,
            tmp_path / "combined",
            expected_commit=COMMIT_SHA,
            repo_root=repo_root,
        )
