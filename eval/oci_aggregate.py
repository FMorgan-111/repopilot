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
    PRIMARY_MODEL,
    REPO_ROOT,
    EvalMode,
    InstanceManifest,
    OfficialResult,
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
_EVALUATOR_KEYS = frozenset(
    {"gold_patch", "test_patch", "fail_to_pass", "pass_to_pass"}
)


class ArtifactContractError(RuntimeError):
    """A matrix artifact is incomplete, ambiguous, or unsafe."""


@dataclass(frozen=True)
class VerifiedPayload:
    result: dict[str, Any]
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


def _read_snapshot_file(bundle_fd: int, filename: str) -> bytes:
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
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            raise ArtifactContractError("artifact file set mismatch")
        chunks: list[bytes] = []
        while chunk := os.read(file_fd, 1024 * 1024):
            chunks.append(chunk)
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
        files = {
            filename: _read_snapshot_file(bundle_fd, filename)
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


def _model_invocations(result: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    invocations = result.get("model_invocations", [])
    if not isinstance(invocations, list):
        return ()
    return tuple(
        invocation for invocation in invocations if isinstance(invocation, dict)
    )


def _model_usage(
    source: dict[str, Any] | tuple[dict[str, Any], ...],
) -> tuple[int, float]:
    tokens = 0
    elapsed = 0.0
    invocations = _model_invocations(source) if isinstance(source, dict) else source
    for invocation in invocations:
        for key in ("input_tokens", "output_tokens"):
            value = invocation.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                tokens += value
        duration = invocation.get("elapsed_seconds")
        if not isinstance(duration, (int, float)) or isinstance(duration, bool):
            continue
        try:
            projected_duration = float(duration)
        except (OverflowError, ValueError):
            continue
        if math.isfinite(projected_duration) and projected_duration > 0:
            elapsed += projected_duration
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


def _parse_safe_payload_bytes(
    result_bytes: bytes,
    prediction_bytes: bytes,
    official_bytes: bytes,
    *,
    expected_instance_id: str,
    expected_commit: str,
) -> VerifiedPayload:
    try:
        result_payload = json.loads(result_bytes)
        prediction_lines = prediction_bytes.decode("utf-8").splitlines()
        official = OfficialResult.model_validate_json(official_bytes)
    except (UnicodeError, ValueError, TypeError) as exc:
        raise ArtifactContractError("invalid safe artifact payload") from exc
    if (
        not isinstance(result_payload, list)
        or len(result_payload) != 1
        or not isinstance(result_payload[0], dict)
    ):
        raise ArtifactContractError("result.json must contain exactly one result")
    if len(prediction_lines) != 1:
        raise ArtifactContractError("prediction.jsonl must contain exactly one row")
    try:
        prediction = json.loads(prediction_lines[0])
    except (json.JSONDecodeError, TypeError) as exc:
        raise ArtifactContractError("invalid prediction row") from exc
    if not isinstance(prediction, dict) or set(prediction) != {
        "instance_id",
        "model_name_or_path",
        "model_patch",
    }:
        raise ArtifactContractError("prediction row has an unsafe schema")
    if any(not isinstance(value, str) for value in prediction.values()):
        raise ArtifactContractError("prediction fields must be strings")

    result = result_payload[0]
    if (
        result.get("instance_id") != expected_instance_id
        or result.get("commit_sha") != expected_commit
        or result.get("model") != PRIMARY_MODEL
        or prediction["instance_id"] != expected_instance_id
        or prediction["model_name_or_path"] != PRIMARY_MODEL
        or prediction["model_patch"] != result.get("model_patch")
        or official.instance_id != expected_instance_id
    ):
        raise ArtifactContractError("cross-file artifact identity mismatch")
    _assert_safe_tree(result)
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
    directory = Path(directory)
    try:
        result_bytes = (directory / "result.json").read_bytes()
        prediction_bytes = (directory / "prediction.jsonl").read_bytes()
        official_bytes = (directory / "official_result.json").read_bytes()
    except OSError as exc:
        raise ArtifactContractError("invalid safe artifact payload") from exc
    return _parse_safe_payload_bytes(
        result_bytes,
        prediction_bytes,
        official_bytes,
        expected_instance_id=expected_instance_id,
        expected_commit=expected_commit,
    )


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
    return manifest, payload


def _summary(
    mode: EvalMode,
    ordered: list[tuple[InstanceManifest, VerifiedPayload]],
    expected_commit: str,
) -> str:
    requested = len(ordered)
    completed = sum(item.official.completed for _manifest, item in ordered)
    internal_success = sum(
        item.result.get("agent_success") is True for _manifest, item in ordered
    )
    official_resolved = sum(item.official.resolved for _manifest, item in ordered)
    official_terminal = sum(
        item.official.status != "scorer_infra" for _manifest, item in ordered
    )
    non_infrastructure = sum(
        manifest.runtime_status == "ready"
        and item.official.status != "scorer_infra"
        and item.result.get("failure_class") != "infra"
        for manifest, item in ordered
    )
    infrastructure = requested - non_infrastructure
    agreements = sum(
        item.official.status != "scorer_infra"
        and (item.result.get("agent_success") is True) == item.official.resolved
        for _manifest, item in ordered
    )
    invocation_groups = [_model_invocations(item.result) for _manifest, item in ordered]
    usage = [_model_usage(invocations) for invocations in invocation_groups]
    model_tokens = sum(tokens for tokens, _elapsed in usage)
    model_elapsed = sum(elapsed for _tokens, elapsed in usage)
    within_budget = sum(tokens <= 100_000 for tokens, _elapsed in usage)
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
        str(item.result.get("failure_class") or "unknown")
        for _manifest, item in ordered
    )
    model_counts = Counter(
        str(invocation.get("model"))
        for invocations in invocation_groups
        for invocation in invocations
        if invocation.get("model")
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
            "| Instance | Runtime | Internal | Official |",
            "| --- | --- | --- | --- |",
        ]
    )
    lines.extend(
        "| {instance} | {runtime} | {internal} | {official} |".format(
            instance=manifest.instance_id,
            runtime=manifest.runtime_status,
            internal=(
                "success" if payload.result.get("agent_success") is True else "failed"
            ),
            official=payload.official.status,
        )
        for manifest, payload in ordered
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
    results = [payload.result for _manifest, payload in ordered]
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
