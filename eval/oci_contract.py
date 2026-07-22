"""Strict contracts shared by the per-instance OCI evaluation pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from eval.swe_bench import atomic_write_text

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "SWE-bench/SWE-bench_Verified"
DATASET_REVISION = "main"
PRIMARY_MODEL = "gemini-3.5-flash:stable"
ESCALATION_MODEL = "claude-opus-4-8:stable"
TESTBED_PYTHON = "/opt/miniconda3/envs/testbed/bin/python"

EvalMode = Literal["checkpoint_5", "baseline_10"]
RuntimeStatus = Literal[
    "ready", "dataset_infra", "oci_image_infra", "oci_boundary_infra"
]
OfficialStatus = Literal[
    "resolved", "unresolved", "empty_patch", "scorer_infra"
]

_MODE_FILES: dict[str, str] = {
    "checkpoint_5": "checkpoint_5_ids.txt",
    "baseline_10": "baseline_10_ids.txt",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_IMAGE_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}")
_COMMIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
_SAFE_FILES = frozenset(
    {"result.json", "prediction.jsonl", "official_result.json"}
)


def load_mode_instance_ids(
    mode: EvalMode, repo_root: Path = REPO_ROOT
) -> tuple[str, ...]:
    """Load an allowlisted mode in its tracked order."""
    try:
        filename = _MODE_FILES[mode]
    except KeyError as exc:
        raise ValueError(f"unsupported evaluation mode: {mode}") from exc
    path = Path(repo_root) / "eval" / filename
    instance_ids = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not instance_ids:
        raise ValueError(f"evaluation mode has no instances: {mode}")
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError(f"duplicate instance ID in evaluation mode: {mode}")
    return instance_ids


def require_mode_instance(
    mode: EvalMode,
    instance_id: str,
    repo_root: Path = REPO_ROOT,
) -> None:
    """Reject arbitrary workflow-provided IDs outside the tracked mode."""
    if instance_id not in load_mode_instance_ids(mode, repo_root):
        raise ValueError(f"instance ID is not tracked for {mode}: {instance_id}")


class RuntimeRecord(BaseModel):
    """Safe output of dataset/image/boundary preparation."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    mode: EvalMode
    instance_id: str
    dataset_name: Literal["SWE-bench/SWE-bench_Verified"] = DATASET_NAME
    dataset_revision: Literal["main"] = DATASET_REVISION
    commit_sha: str
    status: RuntimeStatus
    remote_image: str = ""
    image_sha: str = ""
    python_executable: Literal[
        "/opt/miniconda3/envs/testbed/bin/python"
    ] = TESTBED_PYTHON
    error_class: str = ""

    @field_validator("commit_sha")
    @classmethod
    def validate_commit_sha(cls, value: str) -> str:
        if not _COMMIT_SHA_RE.fullmatch(value):
            raise ValueError("commit_sha must be a 40-character lowercase hex SHA")
        return value

    @model_validator(mode="after")
    def validate_runtime_identity(self) -> RuntimeRecord:
        if self.status == "ready":
            if not self.remote_image:
                raise ValueError("ready runtime requires remote_image")
            if not _IMAGE_SHA_RE.fullmatch(self.image_sha):
                raise ValueError("ready runtime requires immutable image_sha")
            if self.error_class:
                raise ValueError("ready runtime cannot have error_class")
        elif self.image_sha:
            raise ValueError("infrastructure runtime cannot claim image_sha")
        return self


class OfficialResult(BaseModel):
    """Sanitized projection of the official SWE-bench report."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    instance_id: str
    status: OfficialStatus
    submitted: bool
    completed: bool
    resolved: bool
    error_class: str = ""

    @model_validator(mode="after")
    def validate_status_flags(self) -> OfficialResult:
        if self.status == "resolved" and not (
            self.submitted and self.completed and self.resolved
        ):
            raise ValueError("resolved status requires all result flags")
        if self.status == "unresolved" and (
            not self.submitted or not self.completed or self.resolved
        ):
            raise ValueError(f"{self.status} status has inconsistent result flags")
        if self.status == "empty_patch" and (
            not self.submitted or self.completed or self.resolved
        ):
            raise ValueError("empty_patch status has inconsistent result flags")
        if self.status == "scorer_infra" and self.resolved:
            raise ValueError("scorer infrastructure failure cannot be resolved")
        if self.status != "scorer_infra" and self.error_class:
            raise ValueError("terminal scorer result cannot have error_class")
        return self


class InstanceManifest(BaseModel):
    """Hash-bound identity for one sanitized matrix artifact."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    mode: EvalMode
    instance_id: str
    commit_sha: str
    runtime_status: RuntimeStatus
    image_sha: str
    primary_model: Literal["gemini-3.5-flash:stable"] = PRIMARY_MODEL
    escalation_model: Literal["claude-opus-4-8:stable"] = ESCALATION_MODEL
    files: dict[str, str]

    @field_validator("commit_sha")
    @classmethod
    def validate_commit_sha(cls, value: str) -> str:
        if not _COMMIT_SHA_RE.fullmatch(value):
            raise ValueError("commit_sha must be a 40-character lowercase hex SHA")
        return value

    @field_validator("files")
    @classmethod
    def validate_files(cls, value: dict[str, str]) -> dict[str, str]:
        if frozenset(value) != _SAFE_FILES:
            raise ValueError("files must contain exactly the safe artifact set")
        if any(not _SHA256_RE.fullmatch(digest) for digest in value.values()):
            raise ValueError("files must contain lowercase SHA-256 digests")
        return value

    @model_validator(mode="after")
    def validate_image_identity(self) -> InstanceManifest:
        if self.runtime_status == "ready" and not _IMAGE_SHA_RE.fullmatch(
            self.image_sha
        ):
            raise ValueError("ready runtime requires immutable image_sha")
        if self.runtime_status != "ready" and self.image_sha:
            raise ValueError("infrastructure runtime cannot claim image_sha")
        return self


def sha256_file(path: Path) -> str:
    """Hash exact file bytes without loading a potentially large file at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_model(path: Path, model: BaseModel) -> Path:
    """Atomically persist one strict model using deterministic JSON."""
    payload = json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return atomic_write_text(Path(path), payload + "\n")
