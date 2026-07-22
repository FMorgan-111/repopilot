"""Prepare one immutable official SWE-bench OCI runtime."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Protocol

from eval.oci_contract import (
    REPO_ROOT,
    TESTBED_PYTHON,
    EvalMode,
    InstanceManifest,
    OfficialResult,
    RuntimeRecord,
    RuntimeStatus,
    require_mode_instance,
    write_model,
    sha256_file,
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
_PULL_RETRY_DELAYS = (5.0, 20.0)
_TRANSIENT_PULL_RE = re.compile(
    r"(?:\b(?:429|500|502|503|504)\b|"
    r"timed?\s*out|timeout|connection reset|temporary failure|"
    r"tls handshake timeout|unexpected eof|network (?:is )?unreachable|i/o timeout)",
    re.IGNORECASE,
)


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


class CredentialIsolationError(RuntimeError):
    """The scorer process contains a model credential and must not run."""


class OfficialReportContractError(RuntimeError):
    """The official harness returned an ambiguous or malformed report."""


_SCORER_FORBIDDEN_ENV = frozenset(
    {
        "LLM_API_KEY",
        "LLM_ESCALATION_API_KEY",
        "LINOAPI_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    }
)
_TOOL_ENV = {
    "REPOPILOT_TOOL_OCI_BACKEND": "docker",
    "REPOPILOT_TOOL_PYTHON_EXECUTABLE": TESTBED_PYTHON,
    "REPOPILOT_TOOL_PROJECT_EXECUTABLES": "{}",
    "REPOPILOT_TOOL_MEMORY": "4g",
    "REPOPILOT_TOOL_CPUS": "2.0",
    "REPOPILOT_TOOL_PIDS_LIMIT": "256",
}


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


def _is_transient_pull_failure(result: BoundedProcessResult) -> bool:
    diagnostic = f"{result.stdout}\n{result.stderr}"
    return bool(_TRANSIENT_PULL_RE.search(diagnostic))


def _pull_and_pin_image(
    image: str,
    command_runner: CommandRunner,
) -> str:
    for attempt in range(len(_PULL_RETRY_DELAYS) + 1):
        pulled = command_runner(
            ["docker", "pull", "--platform=linux/amd64", image],
            cwd=REPO_ROOT,
            timeout=1_800,
            max_output_bytes=32_000,
            decode_errors="strict",
        )
        if not pulled.returncode:
            break
        if (
            attempt == len(_PULL_RETRY_DELAYS)
            or not _is_transient_pull_failure(pulled)
        ):
            raise OciImageInfrastructureError("official image pull failed")
        time.sleep(_PULL_RETRY_DELAYS[attempt])
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


def _load_runtime(runtime_path: Path) -> RuntimeRecord:
    return RuntimeRecord.model_validate_json(
        Path(runtime_path).read_text(encoding="utf-8")
    )


@contextmanager
def _operator_tool_environment(runtime: RuntimeRecord) -> Iterator[None]:
    values = {**_TOOL_ENV, "REPOPILOT_TOOL_OCI_IMAGE": runtime.image_sha}
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _synthetic_verified_sample(runtime: RuntimeRecord) -> dict[str, Any]:
    owner, separator, repo_and_number = runtime.instance_id.partition("__")
    repo, number_separator, _number = repo_and_number.rpartition("-")
    if not separator or not number_separator:
        owner, repo = "unknown", "unknown"
    return {
        "id": runtime.instance_id,
        "instance_id": runtime.instance_id,
        "source": "swe-bench-verified",
        "repo": {"owner": owner, "name": repo},
        "issue": {
            "number": 0,
            "url": "",
            "title": runtime.instance_id,
            "body": "",
        },
        "base_commit": "",
    }


def _write_generation_failure(
    runtime: RuntimeRecord,
    output_dir: Path,
    error: Exception,
) -> None:
    from eval import agent_v2_harness

    result = agent_v2_harness.safe_failed_sample_result(
        _synthetic_verified_sample(runtime),
        error,
        failure_class="infra",
        commit_sha=runtime.commit_sha,
    )
    agent_v2_harness._write_results_with_fallback(
        [result], Path(output_dir) / "result.json"
    )
    agent_v2_harness.write_predictions(
        [result], Path(output_dir) / "prediction.jsonl"
    )


async def generate_instance(
    runtime_path: Path,
    output_dir: Path,
    *,
    agent_runner: Callable[..., Any] | None = None,
) -> None:
    """Generate one prediction with temporary digest-pinned tool settings."""
    runtime = _load_runtime(runtime_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if runtime.status != "ready":
        _write_generation_failure(
            runtime,
            output_dir,
            RuntimeError(runtime.error_class or runtime.status),
        )
        return
    if agent_runner is None:
        from eval.agent_v2_harness import run_exact_verified_instance

        agent_runner = run_exact_verified_instance
    try:
        with _operator_tool_environment(runtime):
            await agent_runner(
                runtime.instance_id,
                output_dir=output_dir,
                max_retries=3,
                token_budget=100_000,
            )
    except Exception as exc:
        _write_generation_failure(runtime, output_dir, exc)


def _require_credential_free_scorer_env() -> None:
    if any(os.environ.get(name) for name in _SCORER_FORBIDDEN_ENV):
        raise CredentialIsolationError(
            "model credentials present in scorer environment"
        )


def _official_scorer(**kwargs: Any) -> Path:
    from swebench.harness.run_evaluation import main

    result = main(**kwargs)
    if not isinstance(result, Path):
        raise OfficialReportContractError("official scorer did not return a report")
    return result


def _report_ids(report: Mapping[str, Any], key: str) -> set[str]:
    value = report.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise OfficialReportContractError(f"invalid official report field: {key}")
    return set(value)


def _project_official_report(
    report: Mapping[str, Any], instance_id: str
) -> OfficialResult:
    keys = (
        "submitted_ids",
        "completed_ids",
        "resolved_ids",
        "unresolved_ids",
        "empty_patch_ids",
        "error_ids",
    )
    groups = {key: _report_ids(report, key) for key in keys}
    if any(group - {instance_id} for group in groups.values()):
        raise OfficialReportContractError("official report contains another instance")
    submitted = instance_id in groups["submitted_ids"]
    completed = instance_id in groups["completed_ids"]
    resolved = instance_id in groups["resolved_ids"]
    unresolved = instance_id in groups["unresolved_ids"]
    empty_patch = instance_id in groups["empty_patch_ids"]
    errored = instance_id in groups["error_ids"]
    if not submitted:
        raise OfficialReportContractError("official report omitted submitted instance")
    if sum((resolved, unresolved, empty_patch, errored)) != 1:
        raise OfficialReportContractError("official report has ambiguous terminal status")
    if empty_patch:
        return OfficialResult(
            instance_id=instance_id,
            status="empty_patch",
            submitted=True,
            completed=False,
            resolved=False,
        )
    if resolved:
        return OfficialResult(
            instance_id=instance_id,
            status="resolved",
            submitted=True,
            completed=completed,
            resolved=True,
        )
    if unresolved:
        return OfficialResult(
            instance_id=instance_id,
            status="unresolved",
            submitted=True,
            completed=completed,
            resolved=False,
        )
    return OfficialResult(
        instance_id=instance_id,
        status="scorer_infra",
        submitted=True,
        completed=completed,
        resolved=False,
        error_class="OfficialHarnessError",
    )


def _scorer_infrastructure_result(
    runtime: RuntimeRecord,
    error_class: str,
) -> OfficialResult:
    return OfficialResult(
        instance_id=runtime.instance_id,
        status="scorer_infra",
        submitted=False,
        completed=False,
        resolved=False,
        error_class=error_class or "ScorerInfrastructureError",
    )


def score_instance(
    runtime_path: Path,
    output_dir: Path,
    *,
    scorer: Callable[..., Path] = _official_scorer,
) -> OfficialResult:
    """Run the official scorer only in a model-credential-free environment."""
    _require_credential_free_scorer_env()
    runtime = _load_runtime(runtime_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if runtime.status != "ready":
        result = _scorer_infrastructure_result(runtime, runtime.error_class)
        write_model(output_dir / "official_result.json", result)
        return result

    prediction_path = (output_dir / "prediction.jsonl").resolve()
    scorer_dir = output_dir / "official-scorer"
    scorer_dir.mkdir(parents=True, exist_ok=True)
    run_id = (
        f"repopilot-{runtime.commit_sha[:12]}-"
        f"{hashlib.sha256(runtime.instance_id.encode()).hexdigest()[:12]}"
    )
    original_cwd = Path.cwd()
    try:
        os.chdir(scorer_dir)
        report_path = scorer(
            dataset_name="SWE-bench/SWE-bench_Verified",
            split="test",
            instance_ids=[runtime.instance_id],
            predictions_path=str(prediction_path),
            max_workers=1,
            force_rebuild=False,
            cache_level="none",
            clean=True,
            open_file_limit=4096,
            run_id=run_id,
            timeout=1800,
            namespace="swebench",
            rewrite_reports=False,
            modal=False,
            instance_image_tag="latest",
            env_image_tag="latest",
            report_dir=str(scorer_dir),
        )
        report_path = Path(report_path)
        if not report_path.is_absolute():
            report_path = scorer_dir / report_path
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise OfficialReportContractError("official report must be an object")
        result = _project_official_report(report, runtime.instance_id)
    except Exception as exc:
        result = _scorer_infrastructure_result(runtime, type(exc).__name__)
    finally:
        os.chdir(original_cwd)
    write_model(output_dir / "official_result.json", result)
    return result


def package_instance(
    runtime_path: Path,
    output_dir: Path,
    artifact_dir: Path,
) -> InstanceManifest:
    """Copy only validated safe payloads and bind their exact bytes."""
    from eval.oci_aggregate import SAFE_PAYLOAD_FILES, parse_safe_payloads

    runtime = _load_runtime(runtime_path)
    output_dir = Path(output_dir)
    artifact_dir = Path(artifact_dir)
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise ValueError("artifact directory must be empty")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for filename in SAFE_PAYLOAD_FILES:
        source = output_dir / filename
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"safe artifact source unavailable: {filename}")
        shutil.copyfile(source, artifact_dir / filename)
    parse_safe_payloads(
        artifact_dir,
        expected_instance_id=runtime.instance_id,
        expected_commit=runtime.commit_sha,
    )
    manifest = InstanceManifest(
        mode=runtime.mode,
        instance_id=runtime.instance_id,
        commit_sha=runtime.commit_sha,
        runtime_status=runtime.status,
        image_sha=runtime.image_sha,
        files={
            filename: sha256_file(artifact_dir / filename)
            for filename in SAFE_PAYLOAD_FILES
        },
    )
    write_model(artifact_dir / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare", help="prepare one immutable OCI runtime"
    )
    prepare.add_argument(
        "--mode", choices=("checkpoint_5", "baseline_50"), required=True
    )
    prepare.add_argument("--instance-id", required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    for command, help_text in (
        ("generate", "generate one RepoPilot prediction"),
        ("score", "score one prediction with the official harness"),
    ):
        subparser = subparsers.add_parser(command, help=help_text)
        subparser.add_argument("--runtime", type=Path, required=True)
        subparser.add_argument("--output-dir", type=Path, required=True)
    package = subparsers.add_parser(
        "package", help="package one sanitized hash-bound artifact"
    )
    package.add_argument("--runtime", type=Path, required=True)
    package.add_argument("--output-dir", type=Path, required=True)
    package.add_argument("--artifact-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the single-instance OCI evaluation command line."""
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        prepare_instance(args.mode, args.instance_id, args.output_dir)
        return 0
    if args.command == "generate":
        asyncio.run(generate_instance(args.runtime, args.output_dir))
        return 0
    if args.command == "score":
        score_instance(args.runtime, args.output_dir)
        return 0
    if args.command == "package":
        package_instance(args.runtime, args.output_dir, args.artifact_dir)
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
