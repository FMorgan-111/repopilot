"""Verify and deterministically aggregate per-instance OCI artifacts."""

from __future__ import annotations

import argparse
import hmac
import json
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
    sha256_file,
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


def parse_safe_payloads(
    directory: Path,
    *,
    expected_instance_id: str,
    expected_commit: str,
) -> VerifiedPayload:
    """Parse the three uploadable payloads and enforce cross-file identity."""
    directory = Path(directory)
    try:
        result_payload = json.loads(
            (directory / "result.json").read_text(encoding="utf-8")
        )
        prediction_lines = (
            (directory / "prediction.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        official = OfficialResult.model_validate_json(
            (directory / "official_result.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError, TypeError) as exc:
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


def _verify_bundle(
    bundle: Path,
    *,
    expected_instance_id: str,
    expected_commit: str,
    mode: EvalMode,
) -> tuple[InstanceManifest, VerifiedPayload]:
    entries = list(bundle.iterdir())
    names = frozenset(path.name for path in entries)
    if names != SAFE_ARTIFACT_FILES or any(
        path.is_symlink() or not path.is_file() for path in entries
    ):
        raise ArtifactContractError("artifact file set mismatch")
    try:
        manifest = InstanceManifest.model_validate_json(
            (bundle / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
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
        actual_hash = sha256_file(bundle / filename)
        if not hmac.compare_digest(actual_hash, expected_hash):
            raise ArtifactContractError(f"artifact hash mismatch: {filename}")
    payload = parse_safe_payloads(
        bundle,
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
    infrastructure = sum(
        manifest.runtime_status != "ready"
        or item.official.status == "scorer_infra"
        or item.result.get("failure_class") == "infra"
        for manifest, item in ordered
    )
    failure_counts = Counter(
        str(item.result.get("failure_class") or "unknown")
        for _manifest, item in ordered
    )
    model_counts = Counter(
        str(invocation.get("model"))
        for _manifest, item in ordered
        for invocation in item.result.get("model_invocations", [])
        if isinstance(invocation, dict) and invocation.get("model")
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
        f"| Infrastructure failure | {infrastructure} |",
        f"| Primary model invocations | {model_counts[PRIMARY_MODEL]} |",
        f"| Escalation model invocations | {model_counts[ESCALATION_MODEL]} |",
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
    manifests = list(Path(artifacts_dir).glob("*/manifest.json"))
    bundles_by_id: dict[str, Path] = {}
    for manifest_path in manifests:
        try:
            manifest = InstanceManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise ArtifactContractError("invalid discovered manifest") from exc
        if manifest.instance_id in bundles_by_id:
            raise ArtifactContractError("duplicate instance artifact")
        bundles_by_id[manifest.instance_id] = manifest_path.parent
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
