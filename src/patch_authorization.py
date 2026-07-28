"""Provider-neutral authorization boundary for raw planner edit proposals."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .evaluator_safety import contains_evaluator_only
from .model_provider import redact_secrets
from .patch_gate import RepairFailureClass, validate_patch_batch
from .repair_rounds import retire_patch_authorization
from .schemas import PlanDecision
from .state import AgentState, DecisionFrame, RepairPlan, VerifiedEdit, VerifiedEditBatch

PatchAuthorizationStatus = Literal[
    "not_requested",
    "accepted",
    "model_correctable",
    "environment",
]


class PatchAuthorizationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1, max_length=64)
    file_path: str = Field(default="", max_length=500)
    message: str = Field(min_length=1, max_length=500)
    correction_context: str = Field(default="", max_length=1_200)
    failure_class: RepairFailureClass


class PatchAuthorizationOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: PatchAuthorizationStatus
    decision: PlanDecision | None = None
    issues: tuple[PatchAuthorizationIssue, ...] = ()

    @model_validator(mode="after")
    def _require_atomic_outcome(self) -> "PatchAuthorizationOutcome":
        if self.status in {"accepted", "not_requested"}:
            if self.decision is None or self.issues:
                raise ValueError("successful authorization requires one clean decision")
        elif self.decision is not None or not self.issues:
            raise ValueError(
                "rejected authorization cannot expose an executable decision"
            )
        return self


_RESPONSE_FIELDS = frozenset(
    {
        "kind",
        "plan",
        "patch",
        "patch_edits",
        "files",
        "test_command",
        "decision_frame",
    }
)
_EDIT_FIELDS = frozenset(
    {
        "file_path",
        "file",
        "path",
        "search",
        "node_target",
        "replace",
        "replace_all",
    }
)
_DECISION_FRAME_FIELDS = frozenset(DecisionFrame.model_fields)
_TRUST_FIELDS = frozenset(
    {
        "provider",
        "provider_name",
        "model",
        "model_name",
        "round_id",
        "repair_round_id",
        "current_repair_round_id",
        "current_repair_provider",
        "current_repair_model",
        "authorized_repair_round_id",
        "authorized_repair_provider",
        "authorized_repair_model",
        "repo_ref",
        "base_ref",
        "digest",
        "content_digest",
        "patch_sha256",
        "manifest_fingerprint",
        "patch_gate_fingerprint",
        "fingerprint",
        "approval",
        "approved",
        "tool_patch_approval",
        "resolved_target_symbol",
        "expected_content_sha256",
        "exact_only",
    }
)
_DIFF_RE = re.compile(
    r"(?m)^(?:diff --git\s|---\s+(?:a/|/dev/null)|\+\+\+\s+(?:b/|/dev/null)|@@\s+-\d)"
)


def _is_trust_field(value: object) -> bool:
    if not isinstance(value, str):
        return False
    name = value.casefold()
    if name in _TRUST_FIELDS:
        return True
    if re.fullmatch(
        r"(?:(?:active|current|authorized)_)?(?:repair_)?"
        r"(?:provider|model)(?:_(?:name|id))?",
        name,
    ):
        return True
    if re.fullmatch(
        r"(?:(?:current|authorized)_)?(?:repair_)?round_id",
        name,
    ):
        return True
    return (
        name.endswith(("_digest", "_sha", "_sha256", "_fingerprint"))
        or re.fullmatch(
            r"(?:tool_patch_)?approval(?:_(?:id|token|sha|sha256|digest|fingerprint))?",
            name,
        )
        is not None
        or name in {"exact", "resolved_symbol"}
    )


def _issue(
    code: str,
    message: str,
    *,
    file_path: str = "",
    correction_context: str = "",
    failure_class: RepairFailureClass = "model_correctable",
) -> PatchAuthorizationIssue:
    return PatchAuthorizationIssue(
        code=str(code or "invalid_proposal")[:64],
        file_path=redact_secrets(str(file_path or ""))[:500],
        message=redact_secrets(str(message or "Invalid patch proposal."))[:500],
        correction_context=redact_secrets(str(correction_context or ""))[:1_200],
        failure_class=failure_class,
    )


def _reject(
    state: AgentState,
    issues: list[PatchAuthorizationIssue],
) -> PatchAuthorizationOutcome:
    bounded = tuple(issues[:16]) or (
        _issue("invalid_proposal", "The patch proposal is invalid."),
    )
    state.repair_correction_context = render_patch_correction(bounded)
    status: Literal["model_correctable", "environment"] = (
        "environment"
        if any(issue.failure_class == "environment" for issue in bounded)
        else "model_correctable"
    )
    return PatchAuthorizationOutcome(status=status, issues=bounded)


def _contains_forbidden_content(value: object) -> bool:
    text = str(value or "")
    return (
        contains_evaluator_only(text)
        or redact_secrets(text) != text
        or _DIFF_RE.search(text) is not None
    )


def _safe_envelope(
    response: Mapping[str, Any],
    decision_frame: Mapping[str, Any],
) -> tuple[PlanDecision | None, list[PatchAuthorizationIssue]]:
    unknown = {
        key
        for key in response
        if key not in _RESPONSE_FIELDS and not _is_trust_field(key)
    }
    if unknown:
        return None, [
            _issue(
                "unknown_response_field",
                "The plan response contains an unsupported field.",
            )
        ]
    if response.get("kind", "plan") != "plan":
        return None, [
            _issue(
                "invalid_response_kind", "Patch authorization requires a plan response."
            )
        ]
    clean = {
        key: value
        for key, value in response.items()
        if key in _RESPONSE_FIELDS and key != "kind"
    }
    clean["decision_frame"] = dict(decision_frame)
    clean["patch"] = ""
    clean["patch_edits"] = []
    if _contains_forbidden_content(
        {
            key: value
            for key, value in clean.items()
            if key not in {"patch", "patch_edits"}
        }
    ):
        return None, [
            _issue(
                "forbidden_content",
                "The plan response contains forbidden evaluator, credential, or diff content.",
            )
        ]
    try:
        return PlanDecision.model_validate(clean), []
    except (TypeError, ValueError, ValidationError):
        return None, [
            _issue("invalid_plan_envelope", "The plan decision envelope is invalid.")
        ]


def _raw_edits(
    value: object,
) -> tuple[list[VerifiedEdit], list[str], list[str], list[PatchAuthorizationIssue]]:
    if not isinstance(value, list):
        return (
            [],
            [],
            [],
            [_issue("invalid_edit_batch", "patch_edits must be an array.")],
        )
    if len(value) > 16:
        return (
            [],
            [],
            [],
            [_issue("too_many_edits", "A patch batch may contain at most 16 edits.")],
        )

    verified: list[VerifiedEdit] = []
    target_files: list[str] = []
    target_symbols: list[str] = []
    issues: list[PatchAuthorizationIssue] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            issues.append(_issue("invalid_edit", "Each patch edit must be an object."))
            continue
        unknown = {
            key for key in raw if key not in _EDIT_FIELDS and not _is_trust_field(key)
        }
        if unknown:
            issues.append(
                _issue(
                    "unknown_edit_field",
                    "A patch edit contains an unsupported field.",
                )
            )
            continue

        aliases = [raw[name] for name in ("file_path", "file", "path") if name in raw]
        if (
            not aliases
            or any(not isinstance(alias, str) or not alias for alias in aliases)
            or len(set(aliases)) != 1
        ):
            issues.append(
                _issue(
                    "invalid_target",
                    "Each patch edit requires one unambiguous target path.",
                )
            )
            continue
        path = aliases[0]
        search = raw.get("search", "")
        node_target = raw.get("node_target", "")
        replace = raw.get("replace", "")
        replace_all = raw.get("replace_all", False)
        if not isinstance(search, str) or not isinstance(
            node_target, (str, type(None))
        ):
            issues.append(
                _issue(
                    "invalid_anchor",
                    "Patch edit anchors must be strings.",
                    file_path=path,
                )
            )
            continue
        node_target = node_target or ""
        if bool(search) == bool(node_target):
            issues.append(
                _issue(
                    "invalid_anchor",
                    "Each patch edit must use exactly one search or node_target anchor.",
                    file_path=path,
                )
            )
            continue
        if not isinstance(replace, str) or not replace:
            issues.append(
                _issue(
                    "invalid_replacement",
                    "Patch edit replacement text cannot be empty.",
                    file_path=path,
                )
            )
            continue
        if not isinstance(replace_all, bool) or replace_all:
            issues.append(
                _issue(
                    "replace_all_forbidden",
                    "Patch authorization permits one exact target only.",
                    file_path=path,
                )
            )
            continue
        if any(
            _contains_forbidden_content(item)
            for item in (path, search, node_target, replace)
        ):
            issues.append(
                _issue(
                    "forbidden_content",
                    "The edit contains forbidden evaluator, credential, or diff content.",
                )
            )
            continue
        try:
            edit = VerifiedEdit(
                file_path=path,
                node_target=node_target or None,
                search=search,
                replace=replace,
                intent="Model-proposed structured edit.",
            )
        except (TypeError, ValueError, ValidationError):
            issues.append(
                _issue(
                    "invalid_edit",
                    "A patch edit exceeds its structural or field-size bounds.",
                )
            )
            continue
        verified.append(edit)
        if edit.file_path not in target_files:
            target_files.append(edit.file_path)
        if edit.node_target and edit.node_target not in target_symbols:
            target_symbols.append(edit.node_target)

    if len(target_files) > 8:
        issues.append(
            _issue(
                "too_many_targets", "A repair plan may target at most 8 unique paths."
            )
        )
    return verified, target_files, target_symbols, issues[:16]


def authorize_plan_patch(
    state: AgentState,
    response: Mapping[str, Any] | object,
) -> PatchAuthorizationOutcome:
    """Validate raw model mappings before any lossy PatchEdit normalization."""
    retire_patch_authorization(state)
    if not isinstance(response, Mapping):
        return _reject(
            state,
            [_issue("invalid_response", "The plan response must be an object.")],
        )

    patch_response = "patch_edits" in response
    if patch_response:
        verified, target_files, target_symbols, edit_issues = _raw_edits(
            response["patch_edits"]
        )
        if edit_issues:
            return _reject(
                state,
                edit_issues,
            )
        if not verified:
            return _reject(
                state,
                [_issue("missing_edits", "An execute decision requires patch_edits.")],
            )
    else:
        raw_frame = response.get("decision_frame")
        if isinstance(raw_frame, Mapping):
            unknown_frame_fields = {
                key
                for key in raw_frame
                if key not in _DECISION_FRAME_FIELDS and not _is_trust_field(key)
            }
            if unknown_frame_fields:
                return _reject(
                    state,
                    [
                        _issue(
                            "unknown_decision_frame_field",
                            "The decision frame contains an unsupported field.",
                        )
                    ],
                )
            clean_frame = {
                key: value
                for key, value in raw_frame.items()
                if key in _DECISION_FRAME_FIELDS
            }
        else:
            clean_frame = {}
        raw_action = (
            clean_frame.get("recommended_action", "stop")
            if isinstance(raw_frame, Mapping)
            else None
        )
        if not isinstance(raw_action, str) or raw_action not in {
            "collect_more_context",
            "plan",
            "execute",
            "reflect",
            "stop",
            "ask_user",
        }:
            return _reject(
                state,
                [
                    _issue(
                        "invalid_recommended_action",
                        "The plan response has an invalid recommended action.",
                    )
                ],
            )
        envelope, envelope_issues = _safe_envelope(response, clean_frame)
        if envelope_issues or envelope is None:
            return _reject(state, envelope_issues)

        raw_patch = response.get("patch", "")
        if not isinstance(raw_patch, str) or raw_patch:
            return _reject(
                state,
                [
                    _issue(
                        "legacy_patch_forbidden",
                        "The legacy patch field must be empty.",
                    )
                ],
            )
        action = raw_action
        if action != "execute":
            state.repair_correction_context = ""
            return PatchAuthorizationOutcome(
                status="not_requested",
                decision=envelope.model_copy(deep=True),
            )
        return _reject(
            state,
            [_issue("missing_edits", "An execute decision requires patch_edits.")],
        )

    plan = RepairPlan(
        root_cause="Runtime-authorized structured edit proposal.",
        target_files=target_files,
        target_symbols=target_symbols,
        required_behavior="Apply only the exact validated structured edits.",
        regression_test_strategy="Run the planner-selected bounded test command.",
        rejected_approaches=[],
    )
    batch = VerifiedEditBatch(edits=verified)
    state.active_repair_plan = plan
    try:
        gate = validate_patch_batch(state, plan, batch)
    except Exception:
        retire_patch_authorization(state)
        return _reject(
            state,
            [
                _issue(
                    "patch_gate_failure",
                    "PatchGate could not validate the exact checkout.",
                    failure_class="environment",
                )
            ],
        )
    if not gate.accepted:
        converted = [
            PatchAuthorizationIssue(
                code=issue.code,
                file_path=issue.file_path,
                message=issue.message,
                correction_context=issue.correction_context,
                failure_class=issue.failure_class,
            )
            for issue in gate.issues[:16]
        ]
        return _reject(state, converted)

    summary = f"Apply {len(state.patch_edits)} validated structured edit(s)."
    canonical = PlanDecision.model_validate(
        {
            "plan": summary,
            "patch": "",
            "patch_edits": [
                edit.model_dump(mode="json") for edit in state.patch_edits
            ],
            "files": list(target_files),
            "test_command": "",
            "decision_frame": DecisionFrame(
                stage="plan",
                summary=summary,
                recommended_action="execute",
                risk="unknown",
                confidence=0.0,
            ).model_dump(mode="json"),
        }
    )
    state.repair_correction_context = ""
    return PatchAuthorizationOutcome(status="accepted", decision=canonical)


def render_patch_correction(
    issues: tuple[PatchAuthorizationIssue, ...] | list[PatchAuthorizationIssue],
) -> str:
    """Serialize bounded, non-sensitive correction evidence for one PLAN turn."""
    selected = list(issues[:16])
    limits = {
        "file_path": 500,
        "message": 500,
        "correction_context": 1_200,
    }
    while True:
        payload = [
            {
                "code": issue.code,
                "file_path": issue.file_path[: limits["file_path"]],
                "message": issue.message[: limits["message"]],
                "correction_context": issue.correction_context[
                    : limits["correction_context"]
                ],
            }
            for issue in selected
        ]
        rendered = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if len(rendered) <= 8_000:
            return rendered
        if limits["correction_context"]:
            limits["correction_context"] //= 2
        elif limits["message"] > 80:
            limits["message"] //= 2
        elif limits["file_path"] > 80:
            limits["file_path"] //= 2
        else:
            return rendered[:8_000]


__all__ = [
    "PatchAuthorizationIssue",
    "PatchAuthorizationOutcome",
    "PatchAuthorizationStatus",
    "authorize_plan_patch",
    "render_patch_correction",
    "retire_patch_authorization",
]
