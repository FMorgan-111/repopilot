from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from eval.oci_contract import TESTBED_PYTHON
from eval import oci_runner
from eval.oci_runner import prepare_instance
from src.safe_subprocess import BoundedProcessResult

INSTANCE_ID = "pytest-dev__pytest-10081"
COMMIT_SHA = "a" * 40
IMAGE_SHA = "sha256:" + "b" * 64
OFFICIAL_IMAGE = (
    "swebench/sweb.eval.x86_64.pytest-dev__pytest-10081:latest"
)


def _row_loader(instance_id: str) -> dict[str, str]:
    assert instance_id == INSTANCE_ID
    return {"instance_id": instance_id}


def _test_spec_factory(row, *, namespace):
    assert row == {"instance_id": INSTANCE_ID}
    assert namespace == "swebench"
    return SimpleNamespace(instance_image_key=OFFICIAL_IMAGE)


def test_prepare_pulls_official_image_and_preflights_digest(tmp_path: Path) -> None:
    commands: list[list[str]] = []
    preflights: list[tuple[list[str], object]] = []

    def command_runner(argv, **kwargs):
        commands.append(list(argv))
        if argv[1:3] == ["image", "inspect"]:
            return BoundedProcessResult(list(argv), 0, IMAGE_SHA + "\n", "")
        return BoundedProcessResult(list(argv), 0, "pulled", "")

    def oci_runner(command, *, sandbox, config, **kwargs):
        preflights.append((list(command), config))
        assert sandbox.workspace.is_dir()
        return BoundedProcessResult(list(command), 0, "pytest import ok", "")

    record = prepare_instance(
        "checkpoint_5",
        INSTANCE_ID,
        tmp_path,
        row_loader=_row_loader,
        test_spec_factory=_test_spec_factory,
        command_runner=command_runner,
        oci_runner=oci_runner,
        commit_loader=lambda: COMMIT_SHA,
    )

    assert record.status == "ready"
    assert record.remote_image == OFFICIAL_IMAGE
    assert record.image_sha == IMAGE_SHA
    assert commands == [
        ["docker", "pull", "--platform=linux/amd64", OFFICIAL_IMAGE],
        ["docker", "image", "inspect", "--format={{.Id}}", OFFICIAL_IMAGE],
    ]
    command, config = preflights[0]
    assert command == [TESTBED_PYTHON, "-P", "-c", "import pytest"]
    assert config.image == IMAGE_SHA
    assert config.user == "65532:65532"
    assert config.memory == "4g"
    assert config.cpus == 2.0
    assert config.pids_limit == 256
    persisted = json.loads((tmp_path / "runtime.json").read_text(encoding="utf-8"))
    assert persisted == record.model_dump(mode="json")


def test_prepare_records_dataset_infrastructure_without_docker(
    tmp_path: Path,
) -> None:
    docker_called = False

    def fail_dataset(instance_id: str):
        raise ConnectionError("dataset unavailable")

    def command_runner(argv, **kwargs):
        nonlocal docker_called
        docker_called = True
        raise AssertionError("docker must not run")

    record = prepare_instance(
        "checkpoint_5",
        INSTANCE_ID,
        tmp_path,
        row_loader=fail_dataset,
        test_spec_factory=_test_spec_factory,
        command_runner=command_runner,
        commit_loader=lambda: COMMIT_SHA,
    )

    assert record.status == "dataset_infra"
    assert record.error_class == "ConnectionError"
    assert record.image_sha == ""
    assert docker_called is False


def test_prepare_rejects_non_official_image_before_pull(tmp_path: Path) -> None:
    commands: list[list[str]] = []

    record = prepare_instance(
        "checkpoint_5",
        INSTANCE_ID,
        tmp_path,
        row_loader=_row_loader,
        test_spec_factory=lambda row, namespace: SimpleNamespace(
            instance_image_key="attacker.example/replacement:latest"
        ),
        command_runner=lambda argv, **kwargs: commands.append(list(argv)),
        commit_loader=lambda: COMMIT_SHA,
    )

    assert record.status == "oci_image_infra"
    assert record.error_class == "ValueError"
    assert commands == []


def test_prepare_records_pull_or_digest_failure_as_image_infrastructure(
    tmp_path: Path,
) -> None:
    def command_runner(argv, **kwargs):
        if argv[1] == "pull":
            return BoundedProcessResult(list(argv), 1, "", "registry unavailable")
        raise AssertionError("inspect must not run after failed pull")

    record = prepare_instance(
        "checkpoint_5",
        INSTANCE_ID,
        tmp_path,
        row_loader=_row_loader,
        test_spec_factory=_test_spec_factory,
        command_runner=command_runner,
        commit_loader=lambda: COMMIT_SHA,
    )

    assert record.status == "oci_image_infra"
    assert record.error_class == "OciImageInfrastructureError"
    assert record.image_sha == ""


def test_prepare_records_failed_locked_boundary_without_host_fallback(
    tmp_path: Path,
) -> None:
    def command_runner(argv, **kwargs):
        if argv[1:3] == ["image", "inspect"]:
            return BoundedProcessResult(list(argv), 0, IMAGE_SHA, "")
        return BoundedProcessResult(list(argv), 0, "pulled", "")

    def oci_runner(command, **kwargs):
        raise RuntimeError("locked boundary unavailable")

    record = prepare_instance(
        "checkpoint_5",
        INSTANCE_ID,
        tmp_path,
        row_loader=_row_loader,
        test_spec_factory=_test_spec_factory,
        command_runner=command_runner,
        oci_runner=oci_runner,
        commit_loader=lambda: COMMIT_SHA,
    )

    assert record.status == "oci_boundary_infra"
    assert record.error_class == "RuntimeError"
    assert record.image_sha == ""


def test_prepare_cli_accepts_only_mode_instance_and_output(
    monkeypatch, tmp_path: Path
) -> None:
    seen: dict[str, object] = {}

    def fake_prepare(mode, instance_id, output_dir):
        seen.update(
            mode=mode,
            instance_id=instance_id,
            output_dir=output_dir,
        )

    monkeypatch.setattr(oci_runner, "prepare_instance", fake_prepare)

    exit_code = oci_runner.main(
        [
            "prepare",
            "--mode",
            "checkpoint_5",
            "--instance-id",
            INSTANCE_ID,
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert seen == {
        "mode": "checkpoint_5",
        "instance_id": INSTANCE_ID,
        "output_dir": tmp_path,
    }
