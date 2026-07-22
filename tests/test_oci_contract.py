from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval.oci_contract import (
    REPO_ROOT,
    InstanceManifest,
    OfficialResult,
    RuntimeRecord,
    load_mode_instance_ids,
    require_mode_instance,
    sha256_file,
    write_model,
)

COMMIT_SHA = "a" * 40
IMAGE_SHA = "sha256:" + "b" * 64
ROW_SHA = "c" * 64
INSTANCE_ID = "pytest-dev__pytest-10081"


def _manifest_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "mode": "checkpoint_5",
        "instance_id": INSTANCE_ID,
        "commit_sha": COMMIT_SHA,
        "row_sha256": ROW_SHA,
        "runtime_status": "ready",
        "image_sha": IMAGE_SHA,
        "primary_model": "gemini-3.5-flash:stable",
        "escalation_model": "claude-opus-4-8:stable",
        "files": {
            "result.json": "1" * 64,
            "prediction.jsonl": "2" * 64,
            "official_result.json": "3" * 64,
        },
    }
    payload.update(overrides)
    return payload


def test_mode_ids_preserve_tracked_order(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "checkpoint_5_ids.txt").write_text("b\na\n", encoding="utf-8")

    assert load_mode_instance_ids("checkpoint_5", tmp_path) == ("b", "a")


def test_baseline_50_is_fixed_unique_and_preserves_historical_sets() -> None:
    baseline_50 = load_mode_instance_ids("baseline_50")
    baseline_10 = tuple(
        line.strip()
        for line in (REPO_ROOT / "eval" / "baseline_10_ids.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    )
    checkpoint_5 = load_mode_instance_ids("checkpoint_5")

    assert len(baseline_50) == 50
    assert len(set(baseline_50)) == 50
    assert baseline_50[:10] == baseline_10
    assert set(checkpoint_5) <= set(baseline_50)


def test_retired_baseline_10_is_not_a_public_mode() -> None:
    with pytest.raises(ValueError, match="unsupported evaluation mode"):
        load_mode_instance_ids("baseline_10")


def test_mode_ids_reject_duplicates(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "baseline_50_ids.txt").write_text("a\na\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        load_mode_instance_ids("baseline_50", tmp_path)


def test_require_mode_instance_rejects_untracked_id(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "checkpoint_5_ids.txt").write_text("allowed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not tracked"):
        require_mode_instance("checkpoint_5", "other", tmp_path)


def test_runtime_record_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="gold_patch"):
        RuntimeRecord(
            mode="checkpoint_5",
            instance_id=INSTANCE_ID,
            commit_sha=COMMIT_SHA,
            status="dataset_infra",
            error_class="DatasetUnavailable",
            gold_patch="must not persist",
        )


def test_ready_runtime_and_manifest_bind_valid_dataset_row_digest() -> None:
    runtime = RuntimeRecord(
        mode="checkpoint_5",
        instance_id=INSTANCE_ID,
        commit_sha=COMMIT_SHA,
        status="ready",
        remote_image="swebench/sweb.eval.x86_64.owner_repo-1:latest",
        image_sha=IMAGE_SHA,
        row_sha256=ROW_SHA,
    )
    manifest = InstanceManifest.model_validate(
        _manifest_payload(row_sha256=ROW_SHA)
    )

    assert runtime.row_sha256 == ROW_SHA
    assert manifest.row_sha256 == ROW_SHA

    with pytest.raises(ValidationError, match="row_sha256"):
        RuntimeRecord(
            mode="checkpoint_5",
            instance_id=INSTANCE_ID,
            commit_sha=COMMIT_SHA,
            status="ready",
            remote_image="swebench/sweb.eval.x86_64.owner_repo-1:latest",
            image_sha=IMAGE_SHA,
        )
    with pytest.raises(ValidationError, match="row_sha256"):
        InstanceManifest.model_validate(_manifest_payload(row_sha256="d" * 63))


def test_manifest_requires_image_sha_only_for_ready_runtime() -> None:
    with pytest.raises(ValidationError, match="image_sha"):
        InstanceManifest.model_validate(_manifest_payload(image_sha=""))

    manifest = InstanceManifest.model_validate(
        _manifest_payload(runtime_status="oci_image_infra", image_sha="")
    )

    assert manifest.image_sha == ""


def test_manifest_rejects_image_claim_for_failed_runtime() -> None:
    with pytest.raises(ValidationError, match="image_sha"):
        InstanceManifest.model_validate(
            _manifest_payload(runtime_status="oci_boundary_infra")
        )


def test_manifest_requires_exact_safe_file_set() -> None:
    files = dict(_manifest_payload()["files"])
    files["raw.log"] = "4" * 64

    with pytest.raises(ValidationError, match="files"):
        InstanceManifest.model_validate(_manifest_payload(files=files))


def test_official_result_distinguishes_unresolved_from_infrastructure() -> None:
    unresolved = OfficialResult(
        instance_id=INSTANCE_ID,
        status="unresolved",
        submitted=True,
        completed=True,
        resolved=False,
    )
    infrastructure = OfficialResult(
        instance_id=INSTANCE_ID,
        status="scorer_infra",
        submitted=False,
        completed=False,
        resolved=False,
        error_class="DockerUnavailable",
    )

    assert unresolved.status == "unresolved"
    assert infrastructure.status == "scorer_infra"


def test_official_empty_patch_is_submitted_but_not_completed() -> None:
    result = OfficialResult(
        instance_id=INSTANCE_ID,
        status="empty_patch",
        submitted=True,
        completed=False,
        resolved=False,
    )

    assert result.status == "empty_patch"


def test_sha256_file_hashes_exact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "payload"
    path.write_bytes(b"exact\x00bytes")

    assert sha256_file(path) == hashlib.sha256(b"exact\x00bytes").hexdigest()


def test_write_model_uses_strict_json_schema(tmp_path: Path) -> None:
    path = tmp_path / "official_result.json"
    model = OfficialResult(
        instance_id=INSTANCE_ID,
        status="resolved",
        submitted=True,
        completed=True,
        resolved=True,
    )

    written = write_model(path, model)

    assert written == path
    assert json.loads(path.read_text(encoding="utf-8")) == model.model_dump(
        mode="json"
    )
