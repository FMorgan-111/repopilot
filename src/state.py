"""RepoPilot v2 state models and helper functions."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_serializer,
    field_validator,
    model_validator,
)

from .evaluator_safety import contains_evaluator_only, sanitize_evaluator_text
from .exception_safety import normalize_exception_class
from .repo_paths import canonical_repo_path
from .summary_safety import sanitize_summary_text

DEFAULT_AGENT_V2_MAX_RETRIES = 4
MAX_AGENT_V2_MAX_RETRIES = 4
DEFAULT_AGENT_V2_TOKEN_BUDGET = 100_000

_REPAIR_SMUGGLING_RE = re.compile(
    r"(?im)(?:"
    r"^\s*(?:diff --git\b|@@\s|---\s+(?:a/|/dev/null)|"
    r"\+\+\+\s+(?:b/|/dev/null)|\*\*\*\s+(?:begin|end)\s+patch\b)"
    r"|[\"'](?:patch|patch_edits|edits|file_path|search|replace)[\"']\s*:"
    r"|(?:^|[{,])\s*(?:patch|patch_edits|edits|file_path|search|replace)\s*:"
    r"|^\s*(?:-\s*)?(?:patch|patch_edits|edits|file_path|search|replace)\s*[:=]"
    r")"
)

APPROVED_NO_PROGRESS_KINDS = frozenset(
    {
        "nonexistent_search_block",
        "nonexistent_search_blocks",
        "repeated_patch_signature",
        "repeated_edit",
        "repeated_unlocatable_edit",
        "unchanged_context",
        "unchanged_hypothesis",
        "unchanged_plan",
        "unchanged_test_failure",
        "no_evidence_or_applicable_patch",
        "plan",
        "context",
        "edit",
        "test_failure",
        "repeated_plan",
        "repeated_context",
        "repeated_test",
        "repeated_test_failure",
    }
)

APPROVED_ESCALATION_REASONS = frozenset(
    {
        "empty_completion_after_retries",
        "invalid_structured_response_after_retries",
        "primary_gateway_unavailable_after_retries",
        "primary_repair_round_limit",
        "primary_budget_reserve",
        "repeated_no_progress",
        "test_generation_retry",
        *APPROVED_NO_PROGRESS_KINDS,
    }
)

APPROVED_ROUTING_NODES = frozenset(
    {
        "understand",
        "locate",
        "plan",
        "reflect",
        "execute",
        "verify",
        "commit",
        "failure",
        "coverage",
        "test_generation",
        "outcome_summary",
        "understand_issue",
        "locate_code",
        "plan_fix",
        "reflect_on_failure",
        "execute_fix",
        "verify_fix",
        "commit_fix",
        "handle_failure",
        "coverage_gate",
        "test_generator",
    }
)


def sanitize_escalation_reason(value: Any) -> str:
    candidate = str(value or "").strip()
    return candidate if candidate in APPROVED_ESCALATION_REASONS else ""


def sanitize_no_progress_kind(value: Any) -> str:
    candidate = str(value or "").strip()
    return candidate if candidate in APPROVED_NO_PROGRESS_KINDS else ""


def sanitize_routing_node(value: Any) -> str:
    candidate = str(value or "").strip()
    return candidate if candidate in APPROVED_ROUTING_NODES else ""


def sanitize_node_diagnostics(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    sanitized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        diagnostic = dict(item)
        if diagnostic.get("event") == "model_escalated":
            diagnostic["reason"] = sanitize_escalation_reason(
                diagnostic.get("reason")
            )
        sanitized.append(diagnostic)
    return sanitized


class Phase(str, Enum):
    UNDERSTAND = "UNDERSTAND"
    LOCATE = "LOCATE"
    PLAN = "PLAN"
    REFLECT = "REFLECT"
    EXECUTE = "EXECUTE"
    VERIFY = "VERIFY"
    COVERAGE = "COVERAGE"
    COMMIT = "COMMIT"
    WAITING_FOR_USER = "WAITING_FOR_USER"
    FAILURE = "FAILURE"
    DONE = "DONE"
    FAILED = "FAILED"


class ConversationTurn(BaseModel):
    role: str
    content: str


class FileInfo(BaseModel):
    path: str
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    content: str = ""
    sha: str = ""


class PatchEdit(BaseModel):
    file_path: str = Field(min_length=1)
    search: str = ""
    replace: str = ""
    replace_all: bool = False
    # Alternative to search/replace: dotted name of a function/method/class
    # (e.g. "MyClass.method") whose ENTIRE definition is replaced by `replace`.
    # The executor locates the node via AST — no verbatim text anchoring, no
    # line drift. When set, `search` must be empty.
    node_target: str = ""
    # Safe identity resolved from the exact pre-apply source for search-only edits.
    # This is metadata only; the executor continues to apply `search` verbatim.
    resolved_target_symbol: str = Field(default="", max_length=300)
    # Verified escalation edits are bound to this exact whole-file preimage.
    # Task 8's PatchGate consumes the digest; `exact_only` already disables all
    # normalized/fuzzy fallbacks in the legacy executor.
    expected_content_sha256: str = Field(default="", max_length=64)
    exact_only: bool = False

    @field_validator("expected_content_sha256")
    @classmethod
    def _validate_expected_digest(cls, value: str) -> str:
        if value and not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("expected_content_sha256 must be lowercase SHA-256")
        return value

    @field_validator("resolved_target_symbol")
    @classmethod
    def _validate_resolved_target_symbol(cls, value: str) -> str:
        symbol = str(value or "").strip()
        if symbol and not re.fullmatch(
            r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*",
            symbol,
        ):
            raise ValueError("resolved_target_symbol must be a dotted code symbol")
        return symbol

    @model_validator(mode="before")
    @classmethod
    def _normalize_file_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if "file_path" not in normalized:
            for alias in ["file", "path"]:
                if alias in normalized:
                    normalized["file_path"] = normalized[alias]
                    break
        # Tolerate the model supplying BOTH search and node_target: prefer the
        # exact search path (node_target is only a rescue). Never raise here — a
        # malformed edit must not crash the whole plan phase.
        if normalized.get("search") and normalized.get("node_target"):
            normalized["node_target"] = ""
        return normalized

    @model_validator(mode="after")
    def _require_anchor(self) -> "PatchEdit":
        intentional_new = (
            self.exact_only
            and self.expected_content_sha256 == hashlib.sha256(b"").hexdigest()
            and bool(self.replace)
        )
        if not self.search and not self.node_target and not intentional_new:
            raise ValueError(
                "PatchEdit requires either `search` or `node_target`."
            )
        return self


class FixAttempt(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    patch_content: str = ""
    patch_edits: list[PatchEdit] = Field(default_factory=list)
    file_path: str = ""
    test_result: str = ""
    failure_kind: str = ""
    error_log: str = ""
    success: bool = False
    repair_provider: Literal["primary", "escalation"] | None = None
    repair_model: str = Field(default="", max_length=200)
    repair_round_id: int = Field(default=0, ge=0)

    @field_validator("repair_provider")
    @classmethod
    def _prevent_orphaned_repair_provider(cls, value: Any, info: Any) -> Any:
        if value is None and info.data.get("repair_round_id", 0) > 0:
            raise ValueError("positive repair round requires runtime repair attribution")
        return value

    @field_validator("repair_model")
    @classmethod
    def _prevent_orphaned_repair_model(cls, value: str, info: Any) -> str:
        if not value and info.data.get("repair_round_id", 0) > 0:
            raise ValueError("positive repair round requires runtime repair attribution")
        return value

    @field_validator("repair_round_id")
    @classmethod
    def _require_bound_repair_author(cls, value: int, info: Any) -> int:
        if value > 0 and (
            info.data.get("repair_provider") is None
            or not info.data.get("repair_model")
        ):
            raise ValueError("positive repair round requires runtime repair attribution")
        return value

    @model_validator(mode="after")
    def _require_runtime_repair_attribution(self) -> "FixAttempt":
        if self.repair_round_id > 0 and (
            self.repair_provider is None or not self.repair_model
        ):
            raise ValueError(
                "positive repair round requires runtime repair attribution"
            )
        return self


def _normalize_string_list(value: Any) -> Any:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("Expected a sequence of strings")
            normalized.append(item)
        return normalized
    raise ValueError("Expected None, a string, or a sequence of strings")


class Hypothesis(BaseModel):
    id: str
    claim: str
    evidence: list[str] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    why_selected: str = ""
    why_not_selected: str = ""

    @field_validator("id", mode="before")
    @classmethod
    def _coerce_id_to_str(cls, value: Any) -> Any:
        if isinstance(value, (int, float)):
            return str(value)
        return value

    @field_validator("evidence", mode="before")
    @classmethod
    def _normalize_evidence(cls, value: Any) -> Any:
        return _normalize_string_list(value)

    @field_validator("score", mode="before")
    @classmethod
    def _normalize_score(cls, value: Any) -> Any:
        if isinstance(value, (int, float)) and 1.0 < float(value) <= 10.0:
            return float(value) / 10.0
        return value


class DecisionFrame(BaseModel):
    frame_id: str = ""
    stage: Literal["diagnose", "plan", "reflect"]
    summary: str = ""
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    selected_hypothesis_id: str | None = None
    evidence: list[str] = Field(default_factory=list)
    next_checks: list[str] = Field(default_factory=list)
    recommended_action: Literal[
        "collect_more_context",
        "plan",
        "execute",
        "reflect",
        "stop",
        "ask_user",
    ] = "stop"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    risk: Literal["low", "medium", "high", "unknown"] = "unknown"
    parent_frame_id: str | None = None
    trace_notes: str = ""

    @field_validator("selected_hypothesis_id", mode="before")
    @classmethod
    def _coerce_selected_hypothesis_id_to_str(cls, value: Any) -> Any:
        if isinstance(value, (int, float)):
            return str(value)
        return value

    @field_validator("evidence", "next_checks", mode="before")
    @classmethod
    def _normalize_string_lists(cls, value: Any) -> Any:
        return _normalize_string_list(value)


class ToolCall(BaseModel):
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    error: str | None = None


class FinalReport(BaseModel):
    issue_url: str
    fix_applied: bool = False
    pr_url: str | None = None
    test_results: str = ""
    turns_taken: int = 0
    token_used: int = 0


class ModelInvocation(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    model: str
    provider: Literal["primary", "escalation"]
    node: str
    elapsed_seconds: float
    input_tokens: int
    output_tokens: int
    status: Literal["ok", "invalid_response", "error", "cancelled"]
    error_class: str = ""

    @field_validator("error_class", mode="before")
    @classmethod
    def _keep_exception_class_only(cls, value: Any) -> str:
        return normalize_exception_class(value)

    @field_validator("node", mode="before")
    @classmethod
    def _keep_approved_node(cls, value: Any) -> str:
        return sanitize_routing_node(value)


class NoProgressEvent(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    kind: str
    fingerprint: str
    node: str

    @field_validator("kind", mode="before")
    @classmethod
    def _keep_approved_kind(cls, value: Any) -> str:
        return sanitize_no_progress_kind(value)

    @field_validator("node", mode="before")
    @classmethod
    def _keep_approved_node(cls, value: Any) -> str:
        return sanitize_routing_node(value)


class Evidence(BaseModel):
    evidence_id: str
    tool: str
    file_path: str | None = None
    symbol: str | None = None
    summary: str
    content: str
    fingerprint: str


class RepairPlan(BaseModel):
    """Patch-free repair intent produced by the escalation model."""

    model_config = ConfigDict(extra="forbid")

    root_cause: str = Field(min_length=1, max_length=4_000)
    target_files: list[str] = Field(min_length=1, max_length=8)
    target_symbols: list[str] = Field(default_factory=list, max_length=16)
    required_behavior: str = Field(min_length=1, max_length=4_000)
    regression_test_strategy: str = Field(min_length=1, max_length=4_000)
    rejected_approaches: list[str] = Field(default_factory=list, max_length=16)

    @staticmethod
    def _reject_smuggling(value: str) -> str:
        if contains_evaluator_only(value):
            raise ValueError("RepairPlan fields cannot contain evaluator metadata")
        if _REPAIR_SMUGGLING_RE.search(value):
            raise ValueError("RepairPlan fields cannot contain patch or edit payloads")
        return value

    @field_validator(
        "root_cause",
        "required_behavior",
        "regression_test_strategy",
        mode="before",
    )
    @classmethod
    def _validate_narrative(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("RepairPlan narrative fields must be strings")
        normalized = value.strip()
        if not normalized:
            raise ValueError("RepairPlan narrative fields cannot be blank")
        return cls._reject_smuggling(normalized)

    @field_validator("target_files", mode="before")
    @classmethod
    def _validate_target_files(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            raise ValueError("RepairPlan target_files must be an array")
        if not all(isinstance(item, str) for item in value):
            raise ValueError("RepairPlan target_files must contain strings")
        files = [item.strip() for item in value]
        if any(not item for item in files):
            raise ValueError("RepairPlan target_files cannot contain blank paths")
        canonical = [canonical_repo_path(cls._reject_smuggling(path)) for path in files]
        if any(len(path) > 500 for path in canonical):
            raise ValueError("RepairPlan target file is too long")
        if len(canonical) != len(set(canonical)):
            raise ValueError("RepairPlan target_files must be unique")
        return canonical

    @field_validator("target_symbols", "rejected_approaches", mode="before")
    @classmethod
    def _validate_unique_strings(cls, value: Any, info: Any) -> list[str]:
        if not isinstance(value, list):
            raise ValueError(f"RepairPlan {info.field_name} must be an array")
        if not all(isinstance(item, str) for item in value):
            raise ValueError(f"RepairPlan {info.field_name} must contain strings")
        items = [cls._reject_smuggling(item.strip()) for item in value]
        if any(not item for item in items):
            raise ValueError("RepairPlan string lists cannot contain empty values")
        maximum = 300 if info.field_name == "target_symbols" else 1_000
        if any(len(item) > maximum for item in items):
            raise ValueError("RepairPlan list item is too long")
        if len(items) != len(set(items)):
            raise ValueError("RepairPlan string lists must contain unique values")
        return items


class VerifiedEdit(BaseModel):
    """One model edit whose anchors still require checkout validation."""

    model_config = ConfigDict(extra="forbid")

    file_path: str = Field(max_length=500)
    node_target: str | None = Field(default=None, max_length=300)
    search: str = Field(max_length=8_000)
    replace: str = Field(max_length=100_000)
    intent: str = Field(min_length=1, max_length=2_000)
    _expected_content_sha256: str = PrivateAttr(default="")
    _exact_only: bool = PrivateAttr(default=False)

    @model_validator(mode="before")
    @classmethod
    def _reject_evaluator_metadata(cls, value: Any) -> Any:
        if isinstance(value, dict) and any(
            contains_evaluator_only(item)
            for item in value.values()
            if item is not None
        ):
            raise ValueError("VerifiedEdit cannot contain evaluator metadata")
        return value

    @field_validator("file_path", mode="before")
    @classmethod
    def _validate_file_path(cls, value: Any) -> str:
        return canonical_repo_path(value)

    @field_validator("node_target", mode="before")
    @classmethod
    def _normalize_node_target(cls, value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("VerifiedEdit node_target must be a string or null")
        target = value.strip()
        return target or None

    @field_validator("intent", mode="before")
    @classmethod
    def _validate_intent(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("VerifiedEdit intent must be a string")
        normalized = value.strip()
        if not normalized:
            raise ValueError("VerifiedEdit intent cannot be blank")
        return normalized

    @model_validator(mode="after")
    def _use_one_anchor(self) -> "VerifiedEdit":
        if self.node_target and self.search:
            raise ValueError("VerifiedEdit must use either node_target or search")
        if not self.replace:
            raise ValueError("VerifiedEdit replace text cannot be empty")
        return self


class VerifiedEditBatch(BaseModel):
    """A bounded batch returned by the second escalation-model call."""

    model_config = ConfigDict(extra="forbid")

    edits: list[VerifiedEdit] = Field(min_length=1, max_length=16)


CoverageStatus = Literal[
    "pending", "existing_verified", "generated_verified", "failed"
]


class TestRunFingerprint(BaseModel):
    """Bounded, non-sensitive evidence from one targeted test execution."""

    model_config = ConfigDict(extra="forbid")

    exit_code: int
    outcome: Literal["pass", "assertion_failure", "infra"]
    failing_test_ids: list[str] = Field(default_factory=list, max_length=100)
    assertion_fingerprint: str = Field(
        default="", pattern=r"^(?:|[0-9a-f]{64})$"
    )
    summary: str = Field(default="", max_length=500)

    @field_validator("failing_test_ids", mode="before")
    @classmethod
    def _sanitize_failing_test_ids(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            raise ValueError("test run IDs must be an array")
        return [
            sanitized
            for item in value
            if (sanitized := sanitize_evaluator_text(item))
        ]

    @field_validator("summary", mode="before")
    @classmethod
    def _sanitize_summary(cls, value: Any) -> str:
        return sanitize_evaluator_text(value)[:500]

    @model_validator(mode="after")
    def _require_outcome_evidence(self) -> "TestRunFingerprint":
        if any(not item or len(item) > 500 for item in self.failing_test_ids):
            raise ValueError("test run IDs must be non-empty and bounded")
        if self.outcome == "pass" and (
            self.exit_code != 0
            or self.failing_test_ids
            or self.assertion_fingerprint
        ):
            raise ValueError("passing run cannot carry failure evidence")
        if self.outcome == "assertion_failure" and (
            self.exit_code == 0
            or not self.failing_test_ids
            or not self.assertion_fingerprint
        ):
            raise ValueError("assertion run requires IDs and fingerprint")
        return self


class CoverageProof(BaseModel):
    """Persistable differential proof without raw process output."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["existing", "generated"]
    status: Literal["existing_verified", "generated_verified"]
    test_files: list[str] = Field(min_length=1, max_length=16)
    argv: list[str] = Field(min_length=2, max_length=64)
    fixed_runs: list[TestRunFingerprint] = Field(min_length=2, max_length=2)
    base_runs: list[TestRunFingerprint] = Field(min_length=2, max_length=2)
    base_ref: str = Field(pattern=r"^[0-9a-f]{40}$")
    patch_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    patch_gate_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    test_content_digests: dict[str, str] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def _require_differential_proof(self) -> "CoverageProof":
        canonical = [canonical_repo_path(path) for path in self.test_files]
        if canonical != self.test_files or len(canonical) != len(set(canonical)):
            raise ValueError("coverage test files must be canonical and unique")
        if self.source == "existing" and self.status != "existing_verified":
            raise ValueError("coverage source and status disagree")
        if self.source == "generated" and self.status != "generated_verified":
            raise ValueError("coverage source and status disagree")
        if any(run.outcome != "pass" for run in self.fixed_runs):
            raise ValueError("coverage proof requires two fixed passes")
        if any(run.outcome != "assertion_failure" for run in self.base_runs):
            raise ValueError("coverage proof requires two base assertion failures")
        first, second = self.base_runs
        if (
            first.failing_test_ids != second.failing_test_ids
            or first.assertion_fingerprint != second.assertion_fingerprint
        ):
            raise ValueError("base assertion proof must be stable")
        if set(self.test_content_digests) != set(self.test_files) or any(
            not re.fullmatch(r"[0-9a-f]{64}", digest)
            for digest in self.test_content_digests.values()
        ):
            raise ValueError("coverage test digests must bind every candidate")
        prefix = (
            1
            if self.argv[:1] == ["pytest"]
            else 3
            if self.argv[:3] == ["python", "-m", "pytest"]
            else 0
        )
        selectors = {
            token.split("::", 1)[0]
            for token in self.argv[prefix:]
            if prefix and not token.startswith("-")
        }
        if not prefix or selectors != set(self.test_files):
            raise ValueError("coverage argv must exactly select its test files")
        return self


class ToolSandboxConfig(BaseModel):
    """Persisted immutable OCI boundary for repository-controlled commands."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: Literal["docker", "podman"]
    image: str = Field(
        pattern=(
            r"^(?:[A-Za-z0-9][A-Za-z0-9._:/-]*@)?sha256:[0-9a-f]{64}$"
        ),
        max_length=512,
    )
    python_executable: str = "/usr/bin/python3"
    project_executables: tuple[tuple[str, str], ...] = ()
    user: str = Field(default="65532:65532", pattern=r"^[1-9][0-9]*:[1-9][0-9]*$")
    pids_limit: int = Field(default=128, ge=16, le=1024)
    memory: str = Field(default="1g", pattern=r"^[1-9][0-9]*(?:[kKmMgG])$")
    cpus: float = Field(default=1.0, gt=0, le=16)

    @staticmethod
    def _container_executable(value: Any) -> str:
        candidate = str(value or "")
        path = Path(candidate)
        if (
            not candidate.startswith("/")
            or "\\" in candidate
            or any(character.isspace() or ord(character) < 32 for character in candidate)
            or ".." in path.parts
            or candidate.endswith("/")
            or len(candidate) > 256
        ):
            raise ValueError("container executable must be an absolute clean path")
        return candidate

    @field_validator("python_executable", mode="before")
    @classmethod
    def _validate_python_executable(cls, value: Any) -> str:
        return cls._container_executable(value)

    @field_validator("project_executables", mode="before")
    @classmethod
    def _validate_project_executables(cls, value: Any) -> tuple[tuple[str, str], ...]:
        if value is None:
            return ()
        pairs = value.items() if isinstance(value, dict) else value
        if not isinstance(pairs, (Sequence, type({}.items()))):
            raise ValueError("invalid project executable map")
        normalized: dict[str, str] = {}
        for raw_name, raw_path in pairs:
            name = str(raw_name).lower()
            if not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", name):
                raise ValueError("invalid project executable name")
            normalized[name] = cls._container_executable(raw_path)
        if len(normalized) > 16:
            raise ValueError("invalid project executable map")
        return tuple(sorted(normalized.items()))

    def project_executable(self, name: str) -> str | None:
        return dict(self.project_executables).get(name.lower())


class SnapshotManifestEntry(BaseModel):
    """One canonical changed entry approved for a disposable snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    change: Literal["added", "modified", "deleted"]
    mode: Literal["100644", "100755"]
    content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=0, le=512_000_000)

    @field_validator("path", mode="before")
    @classmethod
    def _validate_path(cls, value: Any) -> str:
        return canonical_repo_path(value)

    @model_validator(mode="after")
    def _validate_content_hash(self) -> "SnapshotManifestEntry":
        if self.change == "deleted" and self.content_sha256 is not None:
            raise ValueError("deleted manifest entry must not carry content")
        if self.change != "deleted" and self.content_sha256 is None:
            raise ValueError("live manifest entry requires a content hash")
        return self


def tool_manifest_fingerprint(entries: Sequence[SnapshotManifestEntry]) -> str:
    """Hash a canonical ordered manifest without accepting alternate encodings."""
    payload = json.dumps(
        [entry.model_dump(mode="json") for entry in entries],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ToolPatchApproval(BaseModel):
    """PatchGate authorization for the exact patch exposed to a tool sandbox."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_ref: str = Field(pattern=r"^[0-9a-fA-F]{40}$")
    patch_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    patch_gate_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    changed_manifest: tuple[SnapshotManifestEntry, ...] = Field(max_length=20_000)
    manifest_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _require_canonical_manifest(self) -> "ToolPatchApproval":
        paths = [entry.path for entry in self.changed_manifest]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("changed manifest must be sorted and unique")
        return self


class GeneratedTestApproval(BaseModel):
    """Persisted authorization for one exact generated test payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    change: Literal["added", "modified"] = "added"
    base_content_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    patch_gate_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path", mode="before")
    @classmethod
    def _validate_path(cls, value: Any) -> str:
        return canonical_repo_path(value)

    @model_validator(mode="after")
    def _validate_base_binding(self) -> "GeneratedTestApproval":
        if self.change == "added" and self.base_content_sha256 is not None:
            raise ValueError("added generated test cannot have a base preimage")
        if self.change == "modified" and self.base_content_sha256 is None:
            raise ValueError("modified generated test requires a base preimage")
        return self


ToolAction = Literal[
    "search_symbol",
    "search_text",
    "read_symbol",
    "read_range",
    "find_references",
    "list_related_tests",
    "run_targeted_test",
    "inspect_git_diff",
    "validate_patch",
    "request_repair",
    "finish_investigation",
]


class ToolInvocation(BaseModel):
    action: ToolAction
    args_fingerprint: str
    status: Literal["approved", "rejected", "ok", "error", "duplicate"]
    evidence_id: str | None = None
    error_class: str = ""

    @field_validator("error_class", mode="before")
    @classmethod
    def _keep_exception_class_only(cls, value: Any) -> str:
        if isinstance(value, BaseException):
            return type(value).__name__
        candidate = str(value or "").strip().partition(":")[0].strip()
        if re.fullmatch(
            r"(?:[A-Z][A-Za-z0-9_]*(?:Error|Exception)|Exception|BaseException)",
            candidate,
        ):
            return candidate
        return ""


class AgentState(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    issue_url: str
    issue_title: str = ""
    issue_body: str = ""
    current_phase: Phase = Phase.UNDERSTAND
    relevant_files: list[FileInfo] = Field(default_factory=list)
    fix_attempts: list[FixAttempt] = Field(default_factory=list)
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    token_usage: int = 0
    max_retries: int = DEFAULT_AGENT_V2_MAX_RETRIES
    token_budget: int = DEFAULT_AGENT_V2_TOKEN_BUDGET
    retry_count: int = 0
    tool_calls: list[ToolCall] = Field(default_factory=list)
    owner: str = ""
    repo: str = ""
    issue_number: int = 0
    issue_type: str = "unknown"
    severity: str = "unknown"
    fix_plan: str = ""
    patch_content: str = ""
    patch_edits: list[PatchEdit] = Field(default_factory=list)
    test_command: str = ""
    coverage_status: CoverageStatus = "pending"
    coverage_test_files: list[str] = Field(default_factory=list)
    coverage_test_command: str = ""
    coverage_failure_reason: str = ""
    test_generation_attempts: int = Field(default=0, ge=0, le=2)
    coverage_proof: CoverageProof | None = None
    repo_path: str = ""
    repo_ref: str = ""
    branch_name: str = ""
    base_branch: str = "main"
    pr_url: str | None = None
    failure_reason: str = ""
    trace_id: str = ""
    reflection_notes: str = ""
    decision_frame: DecisionFrame | None = None
    frame_history: list[DecisionFrame] = Field(default_factory=list)
    context_collection_count: int = 0
    last_locate_signature: str = ""
    repeated_patch_block_count: int = 0
    hallucinated_search_block_count: int = 0
    search_correction_context: str = ""
    decision_warnings: list[dict[str, Any]] = Field(default_factory=list)
    decision_route_checked_frame_id: str = ""
    route_decisions: list[dict[str, Any]] = Field(default_factory=list)
    node_diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    pending_human_input: bool = False
    human_input_request: dict[str, Any] = Field(default_factory=dict)
    resume_in_progress: bool = False
    active_model: str = "gemini-3.5-flash:stable"
    active_provider: Literal["primary", "escalation"] = "primary"
    escalated: bool = False
    escalation_reason: str = ""
    primary_failed_repair_rounds: int = Field(default=0, ge=0)
    repair_round_sequence: int = Field(default=0, ge=0)
    current_repair_round_id: int = Field(default=0, ge=0)
    current_repair_provider: Literal["primary", "escalation"] | None = None
    current_repair_model: str = Field(default="", max_length=200)
    authorized_repair_round_id: int = Field(default=0, ge=0)
    authorized_repair_provider: Literal["primary", "escalation"] | None = None
    authorized_repair_model: str = Field(default="", max_length=200)
    last_counted_repair_round_id: int = Field(default=0, ge=0)
    repair_correction_context: str = Field(default="", max_length=8_000)
    no_progress_rounds: int = 0
    last_plan_signature: str = ""
    last_context_fingerprint: str = ""
    last_test_failure_signature: str = ""
    last_assertion_failure_signature: str = ""
    assertion_no_progress_rounds: int = Field(default=0, ge=0)
    assertion_diversity_required: bool = False
    opus_no_progress_rounds: dict[str, int] = Field(default_factory=dict)
    model_history: list[ModelInvocation] = Field(default_factory=list)
    no_progress_history: list[NoProgressEvent] = Field(default_factory=list)
    attempt_outcome_summary: str = Field(
        default="", max_length=200
    )
    summary_token_usage: int = Field(default=0, ge=0)
    evidence: list[Evidence] = Field(default_factory=list)
    tool_history: list[ToolInvocation] = Field(default_factory=list)
    tool_patch_approval: ToolPatchApproval | None = None
    active_repair_plan: RepairPlan | None = None
    patch_correction_count: int = Field(default=0, ge=0, le=2)
    generated_test_approvals: list[GeneratedTestApproval] = Field(
        default_factory=list, max_length=20_000
    )
    tool_sandbox_config: ToolSandboxConfig | None = None
    # Benchmark/eval mode skips COMMIT only after differential coverage proof.
    skip_commit: bool = False
    _reasoning_tool_counter: list[int] = PrivateAttr(default_factory=lambda: [0])

    @model_validator(mode="before")
    @classmethod
    def _reject_evaluator_patch_state(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        patch_values = [
            value.get("patch_content"),
            value.get("patch_edits"),
            value.get("active_repair_plan"),
        ]
        for attempt in value.get("fix_attempts") or []:
            if isinstance(attempt, dict):
                patch_values.extend(
                    [attempt.get("patch_content"), attempt.get("patch_edits")]
                )
            elif isinstance(attempt, FixAttempt):
                patch_values.extend([attempt.patch_content, attempt.patch_edits])
        if any(contains_evaluator_only(item) for item in patch_values if item):
            raise ValueError("AgentState cannot contain evaluator patch metadata")
        return value

    @field_validator("coverage_proof", mode="before")
    @classmethod
    def _drop_legacy_or_incomplete_coverage_proof(cls, value: Any) -> Any:
        if value is None or isinstance(value, CoverageProof):
            return value
        try:
            return CoverageProof.model_validate(value)
        except (TypeError, ValueError):
            return None

    @field_validator("escalation_reason", mode="before")
    @classmethod
    def _keep_approved_escalation_reason(cls, value: Any) -> str:
        return sanitize_escalation_reason(value)

    @field_validator("attempt_outcome_summary", mode="before")
    @classmethod
    def _bound_outcome_summary(cls, value: Any) -> str:
        return sanitize_summary_text(value)

    @model_validator(mode="after")
    def _remove_generated_paths_from_outcome_summary(self) -> "AgentState":
        safe = sanitize_summary_text(
            self.attempt_outcome_summary,
            denied_literals=(item.path for item in self.generated_test_approvals),
        )
        if safe != self.attempt_outcome_summary:
            object.__setattr__(self, "attempt_outcome_summary", safe)
        return self

    @field_serializer("attempt_outcome_summary")
    def _serialize_safe_outcome_summary(self, value: str) -> str:
        return sanitize_summary_text(
            value,
            denied_literals=(item.path for item in self.generated_test_approvals),
        )

    @field_validator("node_diagnostics", mode="before")
    @classmethod
    def _sanitize_routing_diagnostics(cls, value: Any) -> list[dict[str, Any]]:
        return sanitize_node_diagnostics(value)


NodeFn = Callable[[AgentState], Awaitable[AgentState]]


def _as_state(value: Any) -> AgentState:
    if isinstance(value, AgentState):
        return value
    if isinstance(value, dict):
        return AgentState.model_validate(value)
    return AgentState.model_validate(dict(value))


def _estimate_tokens(*parts: str) -> int:
    return max(1, sum(len(part or "") for part in parts) // 4)


def _remember(state: AgentState, role: str, content: str, max_turns: int = 12) -> None:
    state.conversation_history.append(ConversationTurn(role=role, content=content))
    if len(state.conversation_history) > max_turns:
        state.conversation_history = state.conversation_history[-max_turns:]


def _record_tool(
    state: AgentState,
    tool_name: str,
    args: dict[str, Any],
    result: Any = None,
    error: str | None = None,
) -> None:
    state.tool_calls.append(
        ToolCall(tool_name=tool_name, args=args, result=result, error=error)
    )


def _record_decision_frame(state: AgentState, frame: DecisionFrame) -> None:
    if not frame.frame_id:
        frame.frame_id = f"df_{len(state.frame_history) + 1:04d}"
    state.decision_frame = frame
    state.frame_history.append(frame)


def _record_frame_health_warning(
    state: AgentState,
    *,
    node: str,
    expected_stage: str,
    frame: DecisionFrame | None,
    reason: str,
) -> None:
    state.decision_warnings.append(
        {
            "warning_type": "frame_health",
            "node": node,
            "frame_id": frame.frame_id if frame else "",
            "expected_stage": expected_stage,
            "actual_stage": frame.stage if frame else "",
            "reason": reason,
        }
    )


def _describe_exception(exc: BaseException) -> str:
    message = str(exc).strip()
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def _record_node_diagnostic(
    state: AgentState,
    *,
    node: str,
    event: str,
    status: str,
    elapsed_seconds: float,
    error: BaseException | None = None,
    **details: Any,
) -> None:
    diagnostic: dict[str, Any] = {
        "node": node,
        "event": event,
        "status": status,
        "elapsed_seconds": round(max(elapsed_seconds, 0.0), 3),
    }
    if error is not None:
        diagnostic["error_type"] = type(error).__name__
        diagnostic["error"] = str(error).strip() or type(error).__name__
    for key, value in details.items():
        if value is not None:
            diagnostic[key] = value
    state.node_diagnostics.append(diagnostic)


def _human_answer_context(state: AgentState, *, max_answers: int = 3) -> str:
    answers = [
        turn.content.strip()
        for turn in state.conversation_history
        if turn.role == "user"
        and turn.content.strip().startswith("Human answer for paused run")
    ]
    if not answers:
        return ""

    recent_answers = answers[-max_answers:]
    return "Human answer since resume:\n" + "\n\n".join(recent_answers)


def _is_budget_exceeded(state: AgentState) -> bool:
    return state.token_usage >= state.token_budget


def _extract_json_object(data: Any) -> dict[str, Any]:
    if isinstance(data, dict):
        return data
    if not isinstance(data, str):
        return {}
    try:
        parsed = json.loads(data)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _issue_search_terms(title: str, body: str) -> list[str]:
    text = f"{title} {body[:1200]}"
    code_terms = re.findall(r"`([^`]{2,120})`", text)
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text)
    stop = {
        "the",
        "and",
        "for",
        "with",
        "when",
        "this",
        "that",
        "from",
        "issue",
        "error",
        "bug",
    }
    terms: list[str] = []
    for term in code_terms + words:
        normalized = term.strip().replace("/", " ")
        if normalized.lower() in stop:
            continue
        if normalized not in terms:
            terms.append(normalized)
        if len(terms) >= 6:
            break
    return terms or [title[:120]]


def _rank_reason(path: str, issue_title: str, issue_body: str) -> tuple[float, str]:
    haystack = f"{issue_title} {issue_body}".lower()
    path_lower = path.lower()
    filename = Path(path).name.lower()
    score = 0.35
    reasons = []
    if filename and filename.rsplit(".", 1)[0] in haystack:
        score += 0.25
        reasons.append("filename appears in issue text")
    if any(part in haystack for part in path_lower.split("/")):
        score += 0.15
        reasons.append("path components match issue terms")
    if path_lower.startswith(("src/", "lib/", "app/", "packages/")):
        score += 0.1
        reasons.append("source file")
    if path_lower.startswith("tests/") or "/tests/" in path_lower:
        score += 0.05
        reasons.append("test file")
    return min(score, 1.0), ", ".join(reasons) or "matched GitHub code search"


def _same_failure_seen_twice(state: AgentState) -> bool:
    if len(state.fix_attempts) < 2:
        return False
    last = state.fix_attempts[-1]
    for previous in state.fix_attempts[:-1]:
        if (
            previous.patch_content == last.patch_content
            and previous.patch_edits == last.patch_edits
            and previous.error_log == last.error_log
            and not previous.success
            and not last.success
        ):
            return True
    return False


def _primary_patch_file(patch_content: str) -> str:
    match = re.search(r"^\+\+\+ b/(.+)$", patch_content, re.MULTILINE)
    return match.group(1) if match else ""
