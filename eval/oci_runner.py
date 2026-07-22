"""Prepare one immutable official SWE-bench OCI runtime."""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from eval.oci_contract import (
    REPO_ROOT,
    TESTBED_PYTHON,
    EvalMode,
    RuntimeRecord,
    RuntimeStatus,
    require_mode_instance,
    write_model,
)
from eval.swe_bench import load_verified_instance
from src.safe_subprocess import (
    BoundedProcessResult,
    SandboxPaths,
    run_bounded_process,
    run_oci_process,
)
from src.state import ToolSandboxConfig

RUNTIME_PATH = "runtime.json"
_OFFICIAL_IMAGE_RE = re.compile(
    r"swebench/sweb\.eval\.x86_64\.[a-z0-9][a-z0-9_.-]*:latest"
)
_IMAGE_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}")


class CommandRunner(Protocol):
    def __call__(
        self,
        argv: list[str],
        **kwargs: Any,
    ) -> BoundedProcessResult: ...


class OciImageInfrastructureError(RuntimeError):
    """The official image could not be pulled or pinned locally."""


class OciBoundaryInfrastructureError(RuntimeError):
    """The pinned image could not satisfy RepoPilot's locked boundary."""


def _default_test_spec_factory(
    row: Mapping[str, Any], *, namespace: str
) -> Any:
    from swebench.harness.test_spec.test_spec import make_test_spec

    return make_test_spec(row, namespace=namespace)


def _current_commit_sha() -> str:
    result = run_bounded_process(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        timeout=30,
        max_output_bytes=1_000,
        decode_errors="strict",
    )
    commit_sha = result.stdout.strip()
    if result.returncode or not re.fullmatch(r"[0-9a-f]{40}", commit_sha):
        raise RuntimeError("repository commit identity unavailable")
    return commit_sha


def _runtime_record(
    *,
    mode: EvalMode,
    instance_id: str,
    commit_sha: str,
    status: RuntimeStatus,
    remote_image: str = "",
    image_sha: str = "",
    error: BaseException | None = None,
) -> RuntimeRecord:
    return RuntimeRecord(
        mode=mode,
        instance_id=instance_id,
        commit_sha=commit_sha,
        status=status,
        remote_image=remote_image,
        image_sha=image_sha,
        error_class=type(error).__name__ if error is not None else "",
    )


def _persist_runtime(output_dir: Path, record: RuntimeRecord) -> RuntimeRecord:
    write_model(Path(output_dir) / RUNTIME_PATH, record)
    return record


def _official_image(
    row: Mapping[str, Any],
    test_spec_factory: Callable[..., Any],
) -> str:
    image = str(
        test_spec_factory(row, namespace="swebench").instance_image_key
    )
    if not _OFFICIAL_IMAGE_RE.fullmatch(image):
        raise ValueError("official SWE-bench x86_64 image required")
    return image


def _pull_and_pin_image(
    image: str,
    command_runner: CommandRunner,
) -> str:
    pulled = command_runner(
        ["docker", "pull", "--platform=linux/amd64", image],
        cwd=REPO_ROOT,
        timeout=1_800,
        max_output_bytes=32_000,
        decode_errors="strict",
    )
    if pulled.returncode:
        raise OciImageInfrastructureError("official image pull failed")
    inspected = command_runner(
        ["docker", "image", "inspect", "--format={{.Id}}", image],
        cwd=REPO_ROOT,
        timeout=60,
        max_output_bytes=1_000,
        decode_errors="strict",
    )
    image_sha = inspected.stdout.strip()
    if inspected.returncode or not _IMAGE_SHA_RE.fullmatch(image_sha):
        raise OciImageInfrastructureError("official image digest unavailable")
    return image_sha


def _preflight_image(
    image_sha: str,
    output_dir: Path,
    oci_runner: Callable[..., BoundedProcessResult],
) -> None:
    config = ToolSandboxConfig(
        backend="docker",
        image=image_sha,
        python_executable=TESTBED_PYTHON,
        user="65532:65532",
        memory="4g",
        cpus=2.0,
        pids_limit=256,
    )
    sandbox = SandboxPaths.create(Path(output_dir) / "preflight")
    result = oci_runner(
        [TESTBED_PYTHON, "-P", "-c", "import pytest"],
        sandbox=sandbox,
        config=config,
        timeout=60,
        max_output_bytes=8_000,
    )
    if result.returncode:
        raise OciBoundaryInfrastructureError("OCI capability preflight failed")


def prepare_instance(
    mode: EvalMode,
    instance_id: str,
    output_dir: Path,
    *,
    row_loader: Callable[[str], Mapping[str, Any]] = load_verified_instance,
    test_spec_factory: Callable[..., Any] | None = None,
    command_runner: CommandRunner = run_bounded_process,
    oci_runner: Callable[..., BoundedProcessResult] = run_oci_process,
    commit_loader: Callable[[], str] = _current_commit_sha,
) -> RuntimeRecord:
    """Prepare one allowlisted instance and always persist a safe runtime record."""
    require_mode_instance(mode, instance_id)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    commit_sha = commit_loader()
    try:
        row = row_loader(instance_id)
    except Exception as exc:
        return _persist_runtime(
            output_dir,
            _runtime_record(
                mode=mode,
                instance_id=instance_id,
                commit_sha=commit_sha,
                status="dataset_infra",
                error=exc,
            ),
        )

    remote_image = ""
    try:
        remote_image = _official_image(
            row,
            test_spec_factory or _default_test_spec_factory,
        )
        image_sha = _pull_and_pin_image(remote_image, command_runner)
    except Exception as exc:
        return _persist_runtime(
            output_dir,
            _runtime_record(
                mode=mode,
                instance_id=instance_id,
                commit_sha=commit_sha,
                status="oci_image_infra",
                remote_image=remote_image,
                error=exc,
            ),
        )

    try:
        _preflight_image(image_sha, output_dir, oci_runner)
    except Exception as exc:
        return _persist_runtime(
            output_dir,
            _runtime_record(
                mode=mode,
                instance_id=instance_id,
                commit_sha=commit_sha,
                status="oci_boundary_infra",
                remote_image=remote_image,
                error=exc,
            ),
        )

    return _persist_runtime(
        output_dir,
        _runtime_record(
            mode=mode,
            instance_id=instance_id,
            commit_sha=commit_sha,
            status="ready",
            remote_image=remote_image,
            image_sha=image_sha,
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare", help="prepare one immutable OCI runtime"
    )
    prepare.add_argument(
        "--mode", choices=("checkpoint_5", "baseline_10"), required=True
    )
    prepare.add_argument("--instance-id", required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the single-instance OCI evaluation command line."""
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        prepare_instance(args.mode, args.instance_id, args.output_dir)
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
