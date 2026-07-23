"""Verify and deterministically aggregate per-instance OCI artifacts."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import stat
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.oci_contract import (
    ESCALATION_MODEL,
    INFRASTRUCTURE_FAILURE_CLASSES,
    PRIMARY_MODEL,
    REPO_ROOT,
    EvalMode,
    InstanceManifest,
    ModelInvocationRecord,
    OfficialResult,
    ResultRecord,
    load_mode_instance_ids,
)
from eval.safe_contracts import sanitize_output_text
from eval.swe_bench import atomic_write_text

SAFE_PAYLOAD_FILES = (
    "result.json",
    "prediction.jsonl",
    "official_result.json",
)
SAFE_ARTIFACT_FILES = frozenset((*SAFE_PAYLOAD_FILES, "manifest.json"))
ARTIFACT_FILE_BYTE_LIMIT = 8 * 1024 * 1024
ARTIFACT_TOTAL_BYTE_LIMIT = 16 * 1024 * 1024
_EVALUATOR_KEYS = frozenset(
    {"gold_patch", "test_patch", "fail_to_pass", "pass_to_pass"}
)


class ArtifactContractError(RuntimeError):
    """A matrix artifact is incomplete, ambiguous, or unsafe."""


@dataclass(frozen=True)
class VerifiedPayload:
    result: ResultRecord
    prediction: dict[str, str]
    official: OfficialResult


@dataclass(frozen=True)
class _BundleSnapshot:
    bundle_name: str
    manifest: bytes
    result: bytes
    prediction: bytes
    official: bytes

    def file_bytes(self, filename: str) -> bytes:
        try:
            return {
                "manifest.json": self.manifest,
                "result.json": self.result,
                "prediction.jsonl": self.prediction,
                "official_result.json": self.official,
            }[filename]
        except KeyError as exc:  # pragma: no cover - manifest contract is strict
            raise ArtifactContractError("unknown artifact snapshot file") from exc


@dataclass
class _ArtifactByteBudget:
    total_read: int = 0

    def check_before_read(self, size: int) -> None:
        if size > ARTIFACT_FILE_BYTE_LIMIT:
            raise ArtifactContractError("artifact file byte limit exceeded")
        if self.total_read + size > ARTIFACT_TOTAL_BYTE_LIMIT:
            raise ArtifactContractError("artifact total byte limit exceeded")

    def consume(self, chunk_size: int, file_read: int) -> None:
        if file_read > ARTIFACT_FILE_BYTE_LIMIT:
            raise ArtifactContractError("artifact file byte limit exceeded")
        self.total_read += chunk_size
        if self.total_read > ARTIFACT_TOTAL_BYTE_LIMIT:
            raise ArtifactContractError("artifact total byte limit exceeded")


def _discover_manifest_paths(artifacts_dir: Path) -> list[Path]:
    root = Path(artifacts_dir)
    return [
        root / snapshot.bundle_name / "manifest.json"
        for snapshot in _snapshot_artifacts(root)
    ]


def _directory_open_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise ArtifactContractError("no-follow artifact traversal unavailable")
    return os.O_RDONLY | nofollow | directory


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _assert_bound_directory(
    parent_fd: int,
    entry_name: str,
    opened: os.stat_result,
) -> None:
    try:
        current = os.stat(entry_name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise ArtifactContractError("artifact bundle changed during snapshot") from exc
    if not stat.S_ISDIR(current.st_mode) or not _same_identity(opened, current):
        raise ArtifactContractError("artifact bundle changed during snapshot")


def _read_snapshot_file(
    bundle_fd: int,
    filename: str,
    budget: _ArtifactByteBudget,
) -> bytes:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    nonblock = getattr(os, "O_NONBLOCK", None)
    if nofollow is None or nonblock is None:
        raise ArtifactContractError("no-follow artifact traversal unavailable")
    try:
        file_fd = os.open(
            filename,
            os.O_RDONLY | nofollow | nonblock,
            dir_fd=bundle_fd,
        )
    except OSError as exc:
        raise ArtifactContractError("artifact file unavailable") from exc
    try:
        opened = os.fstat(file_fd)
        if not stat.S_ISREG(opened.st_mode):
            raise ArtifactContractError("artifact file set mismatch")
        budget.check_before_read(opened.st_size)
        chunks: list[bytes] = []
        file_read = 0
        while chunk := os.read(file_fd, 1024 * 1024):
            file_read += len(chunk)
            budget.consume(len(chunk), file_read)
            chunks.append(chunk)
        if file_read != opened.st_size:
            raise ArtifactContractError("artifact file changed during snapshot")
        return b"".join(chunks)
    except OSError as exc:
        raise ArtifactContractError("artifact file unavailable") from exc
    finally:
        os.close(file_fd)


def _snapshot_bundle(root_fd: int, bundle_name: str) -> _BundleSnapshot:
    try:
        bundle_fd = os.open(
            bundle_name,
            _directory_open_flags(),
            dir_fd=root_fd,
        )
    except OSError as exc:
        raise ArtifactContractError("unexpected artifact root entry") from exc
    try:
        opened = os.fstat(bundle_fd)
        if not stat.S_ISDIR(opened.st_mode):
            raise ArtifactContractError("unexpected artifact root entry")
        _assert_bound_directory(root_fd, bundle_name, opened)
        try:
            names = frozenset(os.listdir(bundle_fd))
        except OSError as exc:
            raise ArtifactContractError("artifact bundle unavailable") from exc
        if names != SAFE_ARTIFACT_FILES:
            raise ArtifactContractError("artifact file set mismatch")
        budget = _ArtifactByteBudget()
        files = {
            filename: _read_snapshot_file(bundle_fd, filename, budget)
            for filename in SAFE_ARTIFACT_FILES
        }
        try:
            final_names = frozenset(os.listdir(bundle_fd))
        except OSError as exc:
            raise ArtifactContractError("artifact bundle unavailable") from exc
        if final_names != SAFE_ARTIFACT_FILES:
            raise ArtifactContractError("artifact file set mismatch")
        _assert_bound_directory(root_fd, bundle_name, opened)
        return _BundleSnapshot(
            bundle_name=bundle_name,
            manifest=files["manifest.json"],
            result=files["result.json"],
            prediction=files["prediction.jsonl"],
            official=files["official_result.json"],
        )
    finally:
        os.close(bundle_fd)


def _snapshot_artifacts(artifacts_dir: Path) -> list[_BundleSnapshot]:
    root = Path(artifacts_dir)
    try:
        root_fd = os.open(root, _directory_open_flags())
    except (ArtifactContractError, OSError) as exc:
        if isinstance(exc, ArtifactContractError):
            raise
        raise ArtifactContractError("artifact root unavailable") from exc
    try:
        opened = os.fstat(root_fd)
        if not stat.S_ISDIR(opened.st_mode):
            raise ArtifactContractError("artifact root unavailable")
        try:
            bundle_names = os.listdir(root_fd)
        except OSError as exc:
            raise ArtifactContractError("artifact root unavailable") from exc
        snapshots = [
            _snapshot_bundle(root_fd, bundle_name) for bundle_name in bundle_names
        ]
        try:
            final_names = os.listdir(root_fd)
            current = os.stat(root, follow_symlinks=False)
        except OSError as exc:
            raise ArtifactContractError(
                "artifact root changed during snapshot"
            ) from exc
        if (
            frozenset(final_names) != frozenset(bundle_names)
            or not stat.S_ISDIR(current.st_mode)
            or not _same_identity(opened, current)
        ):
            raise ArtifactContractError("artifact root changed during snapshot")
        return snapshots
    finally:
        os.close(root_fd)


def _complete_model_usage(
    invocations: list[ModelInvocationRecord],
) -> tuple[int, float] | None:
    if not invocations:
        return None
    tokens = sum(
        invocation.input_tokens + invocation.output_tokens
        for invocation in invocations
    )
    try:
        elapsed = math.fsum(
            invocation.elapsed_seconds for invocation in invocations
        )
    except OverflowError:
        raise ArtifactContractError("model elapsed total is not finite") from None
    if not math.isfinite(elapsed):
        raise ArtifactContractError("model elapsed total is not finite")
    return tokens, elapsed


def _assert_safe_tree(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_").replace(" ", "_")
            if normalized in _EVALUATOR_KEYS:
                raise ArtifactContractError("evaluator-only field in artifact")
            _assert_safe_tree(item)
        return
    if isinstance(value, list):
        for item in value:
            _assert_safe_tree(item)
        return
    if isinstance(value, str):
        sanitized = sanitize_output_text(value, len(value) + 1)
        if sanitized != value.strip():
            raise ArtifactContractError("unsafe string in artifact")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ArtifactContractError("non-finite number in artifact")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _strict_json_loads(value: str | bytes) -> Any:
    return json.loads(value, parse_constant=_reject_json_constant)


def _parse_safe_payload_bytes(
    result_bytes: bytes,
    prediction_bytes: bytes,
    official_bytes: bytes,
    *,
    expected_instance_id: str,
    expected_commit: str,
) -> VerifiedPayload:
    try:
        result_payload = _strict_json_loads(result_bytes)
        prediction_lines = prediction_bytes.decode("utf-8").splitlines()
        official_payload = _strict_json_loads(official_bytes)
        official = OfficialResult.model_validate(official_payload)
    except (UnicodeError, ValueError, TypeError) as exc:
        raise ArtifactContractError("invalid safe artifact payload") from exc
    if (
        not isinstance(result_payload, list)
        or len(result_payload) != 1
        or not isinstance(result_payload[0], dict)
    ):
        raise ArtifactContractError("result.json must contain exactly one result")
    try:
        result = ResultRecord.model_validate(result_payload[0])
    except (ValueError, TypeError) as exc:
        raise ArtifactContractError("invalid safe artifact payload") from exc
    if len(prediction_lines) != 1:
        raise ArtifactContractError("prediction.jsonl must contain exactly one row")
    try:
        prediction = _strict_json_loads(prediction_lines[0])
    except (ValueError, TypeError) as exc:
        raise ArtifactContractError("invalid prediction row") from exc
    if not isinstance(prediction, dict) or set(prediction) != {
        "instance_id",
        "model_name_or_path",
        "model_patch",
    }:
        raise ArtifactContractError("prediction row has an unsafe schema")
    if any(not isinstance(value, str) for value in prediction.values()):
        raise ArtifactContractError("prediction fields must be strings")

    if (
        result.instance_id != expected_instance_id
        or result.commit_sha != expected_commit
        or result.model != PRIMARY_MODEL
        or prediction["instance_id"] != expected_instance_id
        or prediction["model_name_or_path"] != PRIMARY_MODEL
        or prediction["model_patch"] != result.model_patch
        or official.instance_id != expected_instance_id
    ):
        raise ArtifactContractError("cross-file artifact identity mismatch")
    _assert_safe_tree(result.model_dump(mode="json"))
    _assert_safe_tree(prediction)
    _assert_safe_tree(official.model_dump(mode="json"))
    return VerifiedPayload(result=result, prediction=prediction, official=official)


def parse_safe_payloads(
    directory: Path,
    *,
    expected_instance_id: str,
    expected_commit: str,
) -> VerifiedPayload:
    """Parse the three uploadable payloads and enforce cross-file identity."""
    payload, _files = snapshot_safe_payloads(
        directory,
        expected_instance_id=expected_instance_id,
        expected_commit=expected_commit,
    )
    return payload


def snapshot_safe_payloads(
    directory: Path,
    *,
    expected_instance_id: str,
    expected_commit: str,
) -> tuple[VerifiedPayload, dict[str, bytes]]:
    """Read a bounded regular-file snapshot and validate its safe payloads."""
    directory = Path(directory)
    try:
        directory_fd = os.open(directory, _directory_open_flags())
    except (ArtifactContractError, OSError) as exc:
        if isinstance(exc, ArtifactContractError):
            raise
        raise ArtifactContractError("invalid safe artifact payload") from exc
    try:
        budget = _ArtifactByteBudget()
        files = {
            filename: _read_snapshot_file(directory_fd, filename, budget)
            for filename in SAFE_PAYLOAD_FILES
        }
    finally:
        os.close(directory_fd)
    payload = _parse_safe_payload_bytes(
        files["result.json"],
        files["prediction.jsonl"],
        files["official_result.json"],
        expected_instance_id=expected_instance_id,
        expected_commit=expected_commit,
    )
    return payload, files


def _validate_bundle_consistency(
    manifest: InstanceManifest,
    payload: VerifiedPayload,
) -> None:
    result = payload.result
    official = payload.official
    if manifest.runtime_status != "ready" and (
        official.status != "scorer_infra"
        or official.submitted
        or official.completed
        or official.resolved
        or not official.error_class
        or result.failure_class != "infra"
        or result.success
        or result.agent_success
        or result.final_phase != "FAILED"
        or bool(result.base_commit)
        or bool(result.model_patch)
        or bool(result.model_invocations)
        or result.models_used != [PRIMARY_MODEL]
        or result.escalated
        or bool(result.escalation_reason)
        or result.coverage_status != "failed"
        or result.coverage_proof is not None
    ):
        raise ArtifactContractError("inconsistent artifact bundle")
    if manifest.runtime_status == "ready" and not result.base_commit:
        raise ArtifactContractError("inconsistent artifact bundle")
    if official.status == "empty_patch" and result.model_patch:
        raise ArtifactContractError("inconsistent artifact bundle")
    if official.completed and not result.model_patch:
        raise ArtifactContractError("inconsistent artifact bundle")
    if result.model_patch and (
        not result.model_invocations
        or not any(
            invocation.status == "ok"
            for invocation in result.model_invocations
        )
    ):
        raise ArtifactContractError("inconsistent artifact bundle")


def _verify_bundle(
    snapshot: _BundleSnapshot,
    *,
    expected_instance_id: str,
    expected_commit: str,
    mode: EvalMode,
) -> tuple[InstanceManifest, VerifiedPayload]:
    try:
        manifest = InstanceManifest.model_validate_json(snapshot.manifest)
    except ValueError as exc:
        raise ArtifactContractError("invalid artifact manifest") from exc
    if (
        manifest.mode != mode
        or manifest.instance_id != expected_instance_id
        or manifest.commit_sha != expected_commit
        or manifest.primary_model != PRIMARY_MODEL
        or manifest.escalation_model != ESCALATION_MODEL
    ):
        raise ArtifactContractError("artifact manifest identity mismatch")
    for filename, expected_hash in manifest.files.items():
        actual_hash = hashlib.sha256(snapshot.file_bytes(filename)).hexdigest()
        if not hmac.compare_digest(actual_hash, expected_hash):
            raise ArtifactContractError(f"artifact hash mismatch: {filename}")
    payload = _parse_safe_payload_bytes(
        snapshot.result,
        snapshot.prediction,
        snapshot.official,
        expected_instance_id=expected_instance_id,
        expected_commit=expected_commit,
    )
    _validate_bundle_consistency(manifest, payload)
    return manifest, payload


def _summary(
    mode: EvalMode,
    ordered: list[tuple[InstanceManifest, VerifiedPayload]],
    expected_commit: str,
) -> str:
    requested = len(ordered)
    completed = sum(item.official.completed for _manifest, item in ordered)
    internal_verdicts = [
        item.result.agent_success for _manifest, item in ordered
    ]
    internal_success = sum(internal_verdicts)
    official_resolved = sum(item.official.resolved for _manifest, item in ordered)
    official_terminal = sum(
        item.official.status != "scorer_infra" for _manifest, item in ordered
    )
    invocation_groups = [
        item.result.model_invocations for _manifest, item in ordered
    ]
    usage = [_complete_model_usage(invocations) for invocations in invocation_groups]
    non_infrastructure = sum(
        manifest.runtime_status == "ready"
        and payload.official.completed
        and payload.official.status != "scorer_infra"
        and usage_item is not None
        and payload.result.failure_class not in INFRASTRUCTURE_FAILURE_CLASSES
        for (manifest, payload), usage_item in zip(ordered, usage)
    )
    infrastructure = sum(
        manifest.runtime_status != "ready"
        or payload.official.status == "scorer_infra"
        or payload.result.failure_class in INFRASTRUCTURE_FAILURE_CLASSES
        for manifest, payload in ordered
    )
    agreements = sum(
        item.official.status != "scorer_infra"
        and verdict == item.official.resolved
        for (_manifest, item), verdict in zip(ordered, internal_verdicts)
    )
    model_tokens = sum(item[0] for item in usage if item is not None)
    try:
        model_elapsed = math.fsum(
            item[1] for item in usage if item is not None
        )
    except OverflowError:
        raise ArtifactContractError("model elapsed total is not finite") from None
    if not math.isfinite(model_elapsed):
        raise ArtifactContractError("model elapsed total is not finite")
    within_budget = sum(
        manifest.runtime_status == "ready"
        and payload.official.completed
        and payload.official.status != "scorer_infra"
        and item is not None
        and payload.result.failure_class not in INFRASTRUCTURE_FAILURE_CLASSES
        and item[0] <= 100_000
        and item[1] <= 360 * 60
        for (manifest, payload), item in zip(ordered, usage)
    )
    official_score = 100.0 * official_resolved / requested
    resolution_component = 80.0 * official_resolved / requested
    infrastructure_component = 10.0 * non_infrastructure / requested
    agreement_component = (
        5.0 * agreements / official_terminal if official_terminal else 0.0
    )
    budget_component = 5.0 * within_budget / requested
    engineering_score = (
        resolution_component
        + infrastructure_component
        + agreement_component
        + budget_component
    )
    failure_counts = Counter(
        item.result.failure_class for _manifest, item in ordered
    )
    model_counts = Counter(
        invocation.model
        for invocations in invocation_groups
        for invocation in invocations
    )
    lines = [
        f"# SWE-bench OCI evaluation: {mode}",
        "",
        f"Commit: `{expected_commit}`",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
        f"| Requested | {requested} |",
        f"| Completed | {completed} |",
        f"| Internal success | {internal_success} |",
        f"| Official resolved | {official_resolved} |",
        f"| Official score | {official_score:.2f}/100 |",
        f"| Official terminal coverage | {official_terminal}/{requested} |",
        f"| Internal/official agreement | {agreements}/{official_terminal} |",
        f"| Infrastructure failure | {infrastructure} |",
        f"| Model tokens | {model_tokens} |",
        f"| Model elapsed seconds | {model_elapsed:.3f} |",
        f"| Primary model invocations | {model_counts[PRIMARY_MODEL]} |",
        f"| Escalation model invocations | {model_counts[ESCALATION_MODEL]} |",
        f"| Engineering score | {engineering_score:.2f}/100 |",
        "",
        "## Engineering score components",
        "",
        "| Component | Fraction | Points |",
        "| --- | ---: | ---: |",
        f"| Official resolution | {official_resolved}/{requested} | {resolution_component:.2f}/80 |",
        f"| Non-infrastructure | {non_infrastructure}/{requested} | {infrastructure_component:.2f}/10 |",
        f"| Explicit internal/official agreement | {agreements}/{official_terminal} | {agreement_component:.2f}/5 |",
        f"| Completed within time/token budget | {within_budget}/{requested} | {budget_component:.2f}/5 |",
        "",
        "## Failure taxonomy",
        "",
    ]
    lines.extend(
        f"- `{name}`: {count}" for name, count in sorted(failure_counts.items())
    )
    lines.extend(
        [
            "",
            "## Instances",
            "",
            "| Instance | Runtime | Internal | Official | Model tokens | Model elapsed seconds |",
            "| --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    lines.extend(
        "| {instance} | {runtime} | {internal} | {official} | {tokens} | {elapsed} |".format(
            instance=manifest.instance_id,
            runtime=manifest.runtime_status,
            internal=(
                "success" if verdict else "failed"
            ),
            official=payload.official.status,
            tokens="unavailable" if item is None else str(item[0]),
            elapsed="unavailable" if item is None else f"{item[1]:.3f}",
        )
        for (manifest, payload), verdict, item in zip(
            ordered, internal_verdicts, usage
        )
    )
    return "\n".join(lines) + "\n"


def aggregate_artifacts(
    mode: EvalMode,
    artifacts_dir: Path,
    output_dir: Path,
    *,
    expected_commit: str,
    repo_root: Path = REPO_ROOT,
) -> Path:
    """Verify all matrix bundles and emit tracked-order combined outputs."""
    expected_ids = load_mode_instance_ids(mode, repo_root)
    snapshots = _snapshot_artifacts(artifacts_dir)
    bundles_by_id: dict[str, _BundleSnapshot] = {}
    for snapshot in snapshots:
        try:
            manifest = InstanceManifest.model_validate_json(snapshot.manifest)
        except ValueError as exc:
            raise ArtifactContractError("invalid discovered manifest") from exc
        if manifest.instance_id in bundles_by_id:
            raise ArtifactContractError("duplicate instance artifact")
        bundles_by_id[manifest.instance_id] = snapshot
    if set(bundles_by_id) != set(expected_ids):
        raise ArtifactContractError("missing or extra instance artifact")

    ordered = [
        _verify_bundle(
            bundles_by_id[instance_id],
            expected_instance_id=instance_id,
            expected_commit=expected_commit,
            mode=mode,
        )
        for instance_id in expected_ids
    ]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        payload.result.model_dump(mode="json")
        for _manifest, payload in ordered
    ]
    predictions = [payload.prediction for _manifest, payload in ordered]
    official = [
        payload.official.model_dump(mode="json") for _manifest, payload in ordered
    ]
    atomic_write_text(
        output_dir / "results.json",
        json.dumps(results, ensure_ascii=False, sort_keys=True) + "\n",
    )
    atomic_write_text(
        output_dir / "predictions.jsonl",
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in predictions
        ),
    )
    atomic_write_text(
        output_dir / "official_results.json",
        json.dumps(official, ensure_ascii=False, sort_keys=True) + "\n",
    )
    summary_path = output_dir / "summary.md"
    atomic_write_text(summary_path, _summary(mode, ordered, expected_commit))
    return summary_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("checkpoint_5", "baseline_50"), required=True
    )
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args(argv)
    aggregate_artifacts(
        args.mode,
        args.artifacts_dir,
        args.output_dir,
        expected_commit=args.expected_commit,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
