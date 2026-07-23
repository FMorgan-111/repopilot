"""Strict contracts shared by the per-instance OCI evaluation pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)

from eval.safe_contracts import has_verified_coverage_proof
from eval.swe_bench import (
    DATASET_NAME,
    DATASET_REVISION,
    atomic_write_text,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_MODEL = "gemini-3.5-flash:stable"
ESCALATION_MODEL = "claude-opus-4-8:stable"
TESTBED_PYTHON = "/opt/miniconda3/envs/testbed/bin/python"

EvalMode = Literal["checkpoint_5", "baseline_50"]
RuntimeStatus = Literal[
    "ready", "dataset_infra", "oci_image_infra", "oci_boundary_infra"
]
OfficialStatus = Literal[
    "resolved", "unresolved", "empty_patch", "scorer_infra"
]
FailureClass = Literal[
    "agent_success",
    "resolved",
    "opus_no_progress_limit",
    "patch_gate_rejected",
    "model_gateway_infra",
    "coverage_infra",
    "test_generation_failed",
    "wrong_file_path",
    "invalid_diff",
    "empty_patch",
    "search_not_found",
    "test_failed",
    "infra",
    "budget",
    "other",
]
ModelProvider = Literal["primary", "escalation"]
ModelName = Literal[
    "gemini-3.5-flash:stable", "claude-opus-4-8:stable"
]
InvocationStatus = Literal["ok", "invalid_response", "error"]
CoverageStatus = Literal[
    "pending", "existing_verified", "generated_verified", "failed"
]
INFRASTRUCTURE_FAILURE_CLASSES = frozenset(
    {"infra", "model_gateway_infra", "coverage_infra"}
)

_NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
_FiniteNonNegativeFloat = Annotated[
    float, Field(strict=True, ge=0, allow_inf_nan=False)
]
_NonEmptyString = Annotated[str, Field(min_length=1)]
_NonEmptyModels = Annotated[list[ModelName], Field(min_length=1)]

_MODE_FILES: dict[str, str] = {
    "checkpoint_5": "checkpoint_5_ids.txt",
    "baseline_50": "baseline_50_ids.txt",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_IMAGE_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}")
_COMMIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
_EXCEPTION_CLASS_RE = re.compile(
    r"(?:[A-Z][A-Za-z0-9_]*(?:Error|Exception)|Exception|BaseException)"
)
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
    dataset_revision: Literal[
        "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
    ] = DATASET_REVISION
    row_sha256: str = ""
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

    @field_validator("row_sha256")
    @classmethod
    def validate_row_sha256(cls, value: str) -> str:
        if value and not _SHA256_RE.fullmatch(value):
            raise ValueError("row_sha256 must be a lowercase SHA-256 digest")
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
            if not self.row_sha256:
                raise ValueError("ready runtime requires row_sha256")
        elif self.image_sha:
            raise ValueError("infrastructure runtime cannot claim image_sha")
        return self


class OfficialResult(BaseModel):
    """Sanitized projection of the official SWE-bench report."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    instance_id: str
    status: OfficialStatus
    submitted: StrictBool
    completed: StrictBool
    resolved: StrictBool
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
        if self.status == "scorer_infra" and not self.error_class:
            raise ValueError(
                "scorer infrastructure failure requires error_class"
            )
        if self.status != "scorer_infra" and self.error_class:
            raise ValueError("terminal scorer result cannot have error_class")
        return self


class ModelInvocationRecord(BaseModel):
    """One complete, configured model invocation."""

    model_config = ConfigDict(extra="forbid")

    model: ModelName
    provider: ModelProvider
    node: _NonEmptyString
    elapsed_seconds: _FiniteNonNegativeFloat
    input_tokens: _NonNegativeInt
    output_tokens: _NonNegativeInt
    status: InvocationStatus
    error_class: str

    @field_validator("error_class")
    @classmethod
    def validate_error_class(cls, value: str) -> str:
        if value and not _EXCEPTION_CLASS_RE.fullmatch(value):
            raise ValueError("error_class must be a sanitized exception class")
        return value

    @model_validator(mode="after")
    def validate_provider_model_pair(self) -> ModelInvocationRecord:
        configured_model = {
            "primary": PRIMARY_MODEL,
            "escalation": ESCALATION_MODEL,
        }[self.provider]
        if self.model != configured_model:
            raise ValueError("provider does not match configured model")
        if self.status == "ok" and self.error_class:
            raise ValueError("ok invocation cannot have error_class")
        if self.status == "error" and not self.error_class:
            raise ValueError("error invocation requires error_class")
        return self


class ResultRecord(BaseModel):
    """Complete pre-official Agent V2 result accepted for OCI scoring."""

    model_config = ConfigDict(extra="forbid")

    id: str
    mode: Literal["agent_v2"]
    evaluation_mode: Literal["oracle_files", "end_to_end"]
    model: Literal["gemini-3.5-flash:stable"]
    commit_sha: str
    repo: str
    issue_url: str
    issue_title: str
    success: StrictBool
    agent_success: StrictBool
    official_resolved: None
    waiting_for_user: StrictBool
    final_phase: str
    run_id: str
    trace_id: str
    turns_taken: _NonNegativeInt
    token_used: _NonNegativeInt
    error: str | None
    replay: dict[str, Any] | None
    replay_error: str | None
    models_used: _NonEmptyModels
    escalated: StrictBool
    escalation_reason: str
    model_invocations: list[ModelInvocationRecord]
    tool_invocations: list[dict[str, Any]]
    unique_evidence_count: _NonNegativeInt
    max_consecutive_no_progress: _NonNegativeInt
    attempt_outcome_summary: str
    coverage_status: CoverageStatus
    coverage_test_files: list[str]
    coverage_test_command: str
    coverage_proof: dict[str, Any] | None
    coverage_failure_reason: str
    test_generation_attempts: _NonNegativeInt
    failure_class: FailureClass
    instance_id: str
    base_commit: str
    model_patch: str

    @field_validator("commit_sha")
    @classmethod
    def validate_commit_sha(cls, value: str) -> str:
        if not _COMMIT_SHA_RE.fullmatch(value):
            raise ValueError("commit_sha must be a 40-character lowercase hex SHA")
        return value

    @field_validator("base_commit")
    @classmethod
    def validate_base_commit(cls, value: str) -> str:
        if value and not _COMMIT_SHA_RE.fullmatch(value):
            raise ValueError(
                "base_commit must be empty or a 40-character lowercase hex SHA"
            )
        return value

    @model_validator(mode="after")
    def validate_internal_verdict(self) -> ResultRecord:
        if self.id != self.instance_id:
            raise ValueError("id must match instance_id")
        if self.success is not self.agent_success:
            raise ValueError("success and agent_success must match")
        if self.agent_success and self.failure_class != "agent_success":
            raise ValueError("successful result requires agent_success failure class")
        if self.agent_success and not has_verified_coverage_proof(
            self.coverage_status,
            self.coverage_proof,
        ):
            raise ValueError("agent_success requires verified coverage proof")
        if not self.agent_success and self.failure_class in {
            "agent_success",
            "resolved",
        }:
            raise ValueError("failed result cannot claim a success failure class")
        invocation_models = list(
            dict.fromkeys(
                invocation.model for invocation in self.model_invocations
            )
        )
        expected_models = invocation_models or [PRIMARY_MODEL]
        if self.models_used != expected_models:
            raise ValueError(
                "models_used must match ordered unique invocation history"
            )
        if (
            any(
                invocation.provider == "escalation"
                for invocation in self.model_invocations
            )
            and not self.escalated
        ):
            raise ValueError("escalation invocation requires escalated")
        if self.escalated and not self.escalation_reason:
            raise ValueError("escalated result requires escalation_reason")
        if not self.escalated and self.escalation_reason:
            raise ValueError(
                "non-escalated result cannot have escalation_reason"
            )
        if self.model_patch and not self.agent_success:
            raise ValueError("model_patch requires agent_success")
        if self.model_patch and not self.model_invocations:
            raise ValueError("model_patch requires model invocation history")
        if self.model_patch and not any(
            invocation.status == "ok"
            for invocation in self.model_invocations
        ):
            raise ValueError("model_patch requires a successful model invocation")
        return self


class InstanceManifest(BaseModel):
    """Hash-bound identity for one sanitized matrix artifact."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    mode: EvalMode
    instance_id: str
    commit_sha: str
    dataset_name: Literal["SWE-bench/SWE-bench_Verified"] = DATASET_NAME
    dataset_revision: Literal[
        "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
    ] = DATASET_REVISION
    row_sha256: str = ""
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

    @field_validator("row_sha256")
    @classmethod
    def validate_row_sha256(cls, value: str) -> str:
        if value and not _SHA256_RE.fullmatch(value):
            raise ValueError("row_sha256 must be a lowercase SHA-256 digest")
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
        if self.runtime_status == "ready" and not self.row_sha256:
            raise ValueError("ready runtime requires row_sha256")
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
