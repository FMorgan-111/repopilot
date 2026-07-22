from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval import oci_runner
from eval.oci_contract import TESTBED_PYTHON, RuntimeRecord, write_model
from eval.oci_runner import (
    CredentialIsolationError,
    OciImageInfrastructureError,
    _pull_and_pin_image,
    generate_instance,
    prepare_instance,
    score_instance,
)
from src.safe_subprocess import BoundedProcessResult

INSTANCE_ID = "pytest-dev__pytest-10081"
COMMIT_SHA = "a" * 40
IMAGE_SHA = "sha256:" + "b" * 64
OFFICIAL_IMAGE = (
    "swebench/sweb.eval.x86_64.pytest-dev__pytest-10081:latest"
)
MODEL_CREDENTIAL_NAMES = (
    "LLM_API_KEY",
    "LLM_ESCALATION_API_KEY",
    "LINOAPI_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)


@pytest.fixture(autouse=True)
def _credential_free_test_step(monkeypatch):
    for name in MODEL_CREDENTIAL_NAMES:
        monkeypatch.delenv(name, raising=False)


def _write_runtime(tmp_path: Path, *, status: str = "ready") -> Path:
    runtime = RuntimeRecord(
        mode="checkpoint_5",
        instance_id=INSTANCE_ID,
        commit_sha=COMMIT_SHA,
        status=status,
        remote_image=OFFICIAL_IMAGE if status == "ready" else "",
        image_sha=IMAGE_SHA if status == "ready" else "",
        error_class="" if status == "ready" else "DockerUnavailable",
    )
    return write_model(tmp_path / "runtime.json", runtime)


def _row_loader(instance_id: str) -> dict[str, str]:
    assert instance_id == INSTANCE_ID
    return {"instance_id": instance_id}


def _test_spec_factory(row, *, namespace):
    assert row == {"instance_id": INSTANCE_ID}
    assert namespace == "swebench"
    return SimpleNamespace(instance_image_key=OFFICIAL_IMAGE)


def test_pull_retries_transient_failures_with_bounded_schedule(
    monkeypatch,
) -> None:
    pulls = 0
    sleeps: list[float] = []

    def command_runner(argv, **kwargs):
        nonlocal pulls
        if argv[1] == "pull":
            pulls += 1
            if pulls < 3:
                return BoundedProcessResult(
                    list(argv), 1, "", "HTTP 503 unavailable"
                )
            return BoundedProcessResult(list(argv), 0, "pulled", "")
        return BoundedProcessResult(list(argv), 0, IMAGE_SHA, "")

    monkeypatch.setattr(oci_runner.time, "sleep", sleeps.append)

    assert _pull_and_pin_image(OFFICIAL_IMAGE, command_runner) == IMAGE_SHA
    assert pulls == 3
    assert sleeps == [5.0, 20.0]


def test_pull_retries_network_unreachable_without_is(monkeypatch) -> None:
    pulls = 0
    sleeps: list[float] = []

    def command_runner(argv, **kwargs):
        nonlocal pulls
        if argv[1] == "pull":
            pulls += 1
            if pulls < 3:
                return BoundedProcessResult(
                    list(argv), 1, "", "network unreachable"
                )
            return BoundedProcessResult(list(argv), 0, "pulled", "")
        return BoundedProcessResult(list(argv), 0, IMAGE_SHA, "")

    monkeypatch.setattr(oci_runner.time, "sleep", sleeps.append)

    assert _pull_and_pin_image(OFFICIAL_IMAGE, command_runner) == IMAGE_SHA
    assert pulls == 3
    assert sleeps == [5.0, 20.0]


def test_pull_does_not_retry_permanent_authentication_failure(monkeypatch) -> None:
    pulls = 0
    sleeps: list[float] = []

    def command_runner(argv, **kwargs):
        nonlocal pulls
        if argv[1] == "pull":
            pulls += 1
            return BoundedProcessResult(
                list(argv), 1, "", "unauthorized: authentication required"
            )
        raise AssertionError("inspect must not run after failed pull")

    monkeypatch.setattr(oci_runner.time, "sleep", sleeps.append)

    with pytest.raises(OciImageInfrastructureError, match="official image pull failed"):
        _pull_and_pin_image(OFFICIAL_IMAGE, command_runner)

    assert pulls == 1
    assert sleeps == []


def test_pull_raises_after_bounded_transient_retries(monkeypatch) -> None:
    pulls = 0
    inspected = False
    sleeps: list[float] = []

    def command_runner(argv, **kwargs):
        nonlocal pulls, inspected
        if argv[1] == "pull":
            pulls += 1
            return BoundedProcessResult(
                list(argv), 1, "", "429 Too Many Requests"
            )
        inspected = True
        raise AssertionError("inspect must not run after failed pull")

    monkeypatch.setattr(oci_runner.time, "sleep", sleeps.append)

    with pytest.raises(OciImageInfrastructureError, match="official image pull failed"):
        _pull_and_pin_image(OFFICIAL_IMAGE, command_runner)

    assert pulls == 3
    assert sleeps == [5.0, 20.0]
    assert inspected is False


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


@pytest.mark.parametrize("mode", ["checkpoint_5", "baseline_50"])
def test_prepare_cli_accepts_public_modes(
    monkeypatch, tmp_path: Path, mode: str
) -> None:
    seen: dict[str, object] = {}

    def fake_prepare(selected_mode, instance_id, output_dir):
        seen.update(mode=selected_mode, instance_id=instance_id, output_dir=output_dir)

    monkeypatch.setattr(oci_runner, "prepare_instance", fake_prepare)

    assert oci_runner.main([
        "prepare", "--mode", mode, "--instance-id", INSTANCE_ID,
        "--output-dir", str(tmp_path),
    ]) == 0
    assert seen["mode"] == mode


def test_prepare_cli_rejects_retired_baseline_10(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        oci_runner.main([
            "prepare", "--mode", "baseline_10", "--instance-id", INSTANCE_ID,
            "--output-dir", str(tmp_path),
        ])


async def test_generate_uses_exact_limits_and_temporary_oci_environment(
    monkeypatch, tmp_path: Path
) -> None:
    runtime_path = _write_runtime(tmp_path)
    seen: dict[str, object] = {}
    managed_names = (
        "REPOPILOT_TOOL_OCI_BACKEND",
        "REPOPILOT_TOOL_OCI_IMAGE",
        "REPOPILOT_TOOL_PYTHON_EXECUTABLE",
        "REPOPILOT_TOOL_PROJECT_EXECUTABLES",
        "REPOPILOT_TOOL_MEMORY",
        "REPOPILOT_TOOL_CPUS",
        "REPOPILOT_TOOL_PIDS_LIMIT",
    )
    for name in managed_names:
        monkeypatch.delenv(name, raising=False)

    async def fake_agent_runner(instance_id, **kwargs):
        seen.update(
            instance_id=instance_id,
            kwargs=kwargs,
            environment={name: os.environ.get(name) for name in managed_names},
        )
        (tmp_path / "result.json").write_text("[]\n", encoding="utf-8")
        (tmp_path / "prediction.jsonl").write_text("{}\n", encoding="utf-8")
        return {}

    await generate_instance(
        runtime_path,
        tmp_path,
        agent_runner=fake_agent_runner,
    )

    assert seen["instance_id"] == INSTANCE_ID
    assert seen["kwargs"] == {
        "output_dir": tmp_path,
        "max_retries": 3,
        "token_budget": 100_000,
    }
    assert seen["environment"] == {
        "REPOPILOT_TOOL_OCI_BACKEND": "docker",
        "REPOPILOT_TOOL_OCI_IMAGE": IMAGE_SHA,
        "REPOPILOT_TOOL_PYTHON_EXECUTABLE": TESTBED_PYTHON,
        "REPOPILOT_TOOL_PROJECT_EXECUTABLES": "{}",
        "REPOPILOT_TOOL_MEMORY": "4g",
        "REPOPILOT_TOOL_CPUS": "2.0",
        "REPOPILOT_TOOL_PIDS_LIMIT": "256",
    }
    assert all(name not in os.environ for name in managed_names)


async def test_generate_infrastructure_runtime_writes_empty_prediction_without_agent(
    tmp_path: Path,
) -> None:
    runtime_path = _write_runtime(tmp_path, status="oci_image_infra")

    async def forbidden_agent(*args, **kwargs):
        raise AssertionError("agent must not run")

    await generate_instance(
        runtime_path,
        tmp_path,
        agent_runner=forbidden_agent,
    )

    [result] = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    prediction = json.loads(
        (tmp_path / "prediction.jsonl").read_text(encoding="utf-8")
    )
    assert result["failure_class"] == "infra"
    assert result["commit_sha"] == COMMIT_SHA
    assert prediction["instance_id"] == INSTANCE_ID
    assert prediction["model_patch"] == ""


@pytest.mark.parametrize(
    "credential_name",
    MODEL_CREDENTIAL_NAMES,
)
def test_scorer_rejects_model_credentials(
    monkeypatch, tmp_path: Path, credential_name: str
) -> None:
    runtime_path = _write_runtime(tmp_path)
    monkeypatch.setenv(credential_name, "must-not-cross-boundary")

    with pytest.raises(CredentialIsolationError):
        score_instance(runtime_path, tmp_path, scorer=lambda **kwargs: None)


def test_score_invokes_official_harness_and_projects_resolved_result(
    tmp_path: Path,
) -> None:
    runtime_path = _write_runtime(tmp_path)
    (tmp_path / "prediction.jsonl").write_text(
        json.dumps(
            {
                "instance_id": INSTANCE_ID,
                "model_name_or_path": "gemini-3.5-flash:stable",
                "model_patch": "diff --git a/a.py b/a.py\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    seen: dict[str, object] = {}

    def fake_scorer(**kwargs):
        seen.update(kwargs)
        report_path = Path.cwd() / "official-report.json"
        report_path.write_text(
            json.dumps(
                {
                    "submitted_ids": [INSTANCE_ID],
                    "completed_ids": [INSTANCE_ID],
                    "resolved_ids": [INSTANCE_ID],
                    "unresolved_ids": [],
                    "empty_patch_ids": [],
                    "error_ids": [],
                }
            ),
            encoding="utf-8",
        )
        return report_path

    result = score_instance(runtime_path, tmp_path, scorer=fake_scorer)

    assert result.status == "resolved"
    assert result.submitted is True
    assert result.completed is True
    assert result.resolved is True
    assert seen["dataset_name"] == "SWE-bench/SWE-bench_Verified"
    assert seen["instance_ids"] == [INSTANCE_ID]
    assert seen["max_workers"] == 1
    assert seen["force_rebuild"] is False
    assert seen["cache_level"] == "none"
    assert seen["clean"] is True
    assert seen["open_file_limit"] == 4096
    assert seen["timeout"] == 1800
    assert seen["namespace"] == "swebench"
    assert seen["modal"] is False
    stored = json.loads(
        (tmp_path / "official_result.json").read_text(encoding="utf-8")
    )
    assert stored == result.model_dump(mode="json")


def test_infrastructure_runtime_skips_official_scorer(tmp_path: Path) -> None:
    runtime_path = _write_runtime(tmp_path, status="oci_boundary_infra")

    def forbidden_scorer(**kwargs):
        raise AssertionError("official scorer must not run")

    result = score_instance(runtime_path, tmp_path, scorer=forbidden_scorer)

    assert result.status == "scorer_infra"
    assert result.resolved is False
    assert result.error_class == "DockerUnavailable"


def test_generate_cli_forwards_runtime_and_output(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Path] = {}
    runtime_path = tmp_path / "runtime.json"

    async def fake_generate(runtime, output_dir):
        seen.update(runtime=runtime, output_dir=output_dir)

    monkeypatch.setattr(oci_runner, "generate_instance", fake_generate)

    exit_code = oci_runner.main(
        [
            "generate",
            "--runtime",
            str(runtime_path),
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert seen == {"runtime": runtime_path, "output_dir": tmp_path}


def test_score_cli_forwards_runtime_and_output(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Path] = {}
    runtime_path = tmp_path / "runtime.json"

    def fake_score(runtime, output_dir):
        seen.update(runtime=runtime, output_dir=output_dir)

    monkeypatch.setattr(oci_runner, "score_instance", fake_score)

    exit_code = oci_runner.main(
        [
            "score",
            "--runtime",
            str(runtime_path),
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert seen == {"runtime": runtime_path, "output_dir": tmp_path}


def test_package_cli_forwards_only_explicit_directories(
    monkeypatch, tmp_path: Path
) -> None:
    seen: dict[str, Path] = {}
    runtime_path = tmp_path / "runtime.json"
    artifact_dir = tmp_path / "upload"

    def fake_package(runtime, output_dir, artifact):
        seen.update(
            runtime=runtime,
            output_dir=output_dir,
            artifact_dir=artifact,
        )

    monkeypatch.setattr(oci_runner, "package_instance", fake_package)

    exit_code = oci_runner.main(
        [
            "package",
            "--runtime",
            str(runtime_path),
            "--output-dir",
            str(tmp_path),
            "--artifact-dir",
            str(artifact_dir),
        ]
    )

    assert exit_code == 0
    assert seen == {
        "runtime": runtime_path,
        "output_dir": tmp_path,
        "artifact_dir": artifact_dir,
    }
