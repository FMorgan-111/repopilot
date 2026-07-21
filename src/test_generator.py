"""Bounded, test-only regression-test generation policy."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from pydantic import ValidationError

from .coverage_gate import (
    ChangedTarget,
    CoverageCandidate,
    CoverageDecision,
    collect_changed_targets,
    is_allowed_test_path,
    validate_differential_coverage,
)
from .llm import llm_call
from .model_policy import EscalationDecision, apply_escalation
from .model_provider import escalation_is_configured, redact_secrets
from .patch_gate import apply_approved_patch, validate_patch_batch
from .repo_paths import canonical_repo_path
from .state import (
    AgentState,
    GeneratedTestApproval,
    RepairPlan,
    VerifiedEditBatch,
)

_EVALUATOR_ONLY_RE = re.compile(
    r"(?i)\b(?:gold[_ -]?patch|test[_ -]?patch|fail[_ -]?to[_ -]?pass|"
    r"pass[_ -]?to[_ -]?pass|instance[_ -]?id|evaluator|grading|oracle)\b"
)
_SYSTEM_PROMPT = (
    "Return only one JSON object matching VerifiedEditBatch with key 'edits'. "
    "Each edit must modify or create only an established Python test path. "
    "Use an exact search anchor for an existing file and an empty search for a "
    "new file. Add one focused behavioral regression test that fails on the base "
    "commit and passes on the fixed checkout. Do not edit production, CI, config, "
    "dependencies, generated artifacts, or binary files."
)
_PROMPT_LIMIT = 32_000


def _safe_text(value: object, *, limit: int) -> str:
    text = redact_secrets(str(value or ""))
    text = _EVALUATOR_ONLY_RE.sub("[REDACTED_EVALUATOR_FIELD]", text)
    return text[:limit]


def _target_evidence(root: Path, targets: list[ChangedTarget]) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []
    for target in targets[:8]:
        try:
            relative = canonical_repo_path(target.file_path)
            path = root / relative
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 32_000:
                continue
            content = path.read_text(encoding="utf-8")[:8_000]
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        evidence.append(
            {
                "file_path": relative,
                "symbols": [_safe_text(symbol, limit=300) for symbol in target.symbols[:16]],
                "content": _safe_text(content, limit=8_000),
            }
        )
    return evidence


def _test_conventions(root: Path) -> list[dict[str, str]]:
    conventions: list[dict[str, str]] = []
    for path in sorted(root.rglob("*.py")):
        if len(conventions) >= 4:
            break
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            continue
        if not is_allowed_test_path(root, relative):
            continue
        try:
            content = path.read_text(encoding="utf-8")[:2_000]
        except (OSError, UnicodeDecodeError):
            continue
        conventions.append(
            {"file_path": relative, "content": _safe_text(content, limit=2_000)}
        )
    return conventions


def _generation_prompt(state: AgentState, rejection_reason: str) -> str:
    root = Path(state.repo_path).resolve(strict=True)
    targets = collect_changed_targets(root, state.repo_ref)
    payload = {
        "issue": {
            "title": _safe_text(state.issue_title, limit=1_000),
            "body": _safe_text(state.issue_body, limit=4_000),
        },
        "changed_targets": [
            {
                "file_path": _safe_text(target.file_path, limit=500),
                "symbols": [
                    _safe_text(symbol, limit=300) for symbol in target.symbols[:16]
                ],
            }
            for target in targets[:8]
        ],
        "target_source_evidence": _target_evidence(root, targets),
        "test_conventions": _test_conventions(root),
        "required_behavior": "prove the issue fails on base and passes on the fix",
        "prior_rejection": _safe_text(rejection_reason, limit=300),
    }
    prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    if len(prompt) > _PROMPT_LIMIT:
        raise ValueError("test generation prompt exceeds bounded context")
    return prompt


async def request_test_batch(
    state: AgentState,
    rejection_reason: str,
) -> VerifiedEditBatch:
    """Request one structured batch without including runtime or evaluator data."""
    raw = await llm_call(
        _SYSTEM_PROMPT,
        _generation_prompt(state, rejection_reason),
        model=state.active_model,
        provider=state.active_provider,
        temperature=0.0,
    )
    if not isinstance(raw, dict):
        raise ValueError("test generator response must be a JSON object")
    payload = {key: value for key, value in raw.items() if key != "kind"}
    try:
        return VerifiedEditBatch.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("test generator returned an invalid edit batch") from exc


async def generate_test_batch(
    state: AgentState,
    rejection_reason: str,
) -> VerifiedEditBatch:
    return await request_test_batch(state, rejection_reason)


def _test_plan(batch: VerifiedEditBatch) -> RepairPlan:
    return RepairPlan(
        root_cause="missing behavioral regression test",
        target_files=[edit.file_path for edit in batch.edits],
        target_symbols=[],
        required_behavior="prove the issue fails on base and passes on the fix",
        regression_test_strategy="targeted differential test",
        rejected_approaches=[],
    )


def _snapshot_test_files(root: Path, paths: list[str]) -> dict[str, bytes | None]:
    snapshot: dict[str, bytes | None] = {}
    for raw in paths:
        relative = canonical_repo_path(raw)
        if not is_allowed_test_path(root, relative):
            raise ValueError("generated batch contains a non-test path")
        path = root / relative
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError("generated test target is not a regular file")
        snapshot[relative] = path.read_bytes() if path.exists() else None
    return snapshot


def _restore_test_files(root: Path, snapshot: dict[str, bytes | None]) -> None:
    for relative, content in snapshot.items():
        path = root / relative
        if path.is_symlink():
            raise RuntimeError("generated test target changed to a symlink")
        if content is None:
            if path.exists():
                if not path.is_file():
                    raise RuntimeError("generated test rollback target is not a file")
                path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)


def _failed(reason: str) -> CoverageDecision:
    return CoverageDecision(
        verified=False,
        status="failed",
        reason=_safe_text(reason, limit=300) or "invalid_generated_test",
    )


def _refresh_diff(state: AgentState) -> None:
    # Import lazily so importing the nodes package cannot cycle through the
    # COVERAGE node back into this module.
    from .nodes.execute import git_diff

    state.patch_content = git_diff(state.repo_path)


async def run_test_generation_attempts(
    state: AgentState,
    rejection_reason: str,
) -> CoverageDecision:
    """Try at most twice; only a differential proof can return verified."""
    root = Path(state.repo_path).resolve(strict=True)
    reason = _safe_text(rejection_reason, limit=300) or "no_coverage_candidate"
    last = _failed(reason)

    while state.test_generation_attempts < 2:
        state.test_generation_attempts += 1
        snapshot: dict[str, bytes | None] = {}
        applied = False
        try:
            batch = await request_test_batch(state, reason)
            plan = _test_plan(batch)
            state.active_repair_plan = plan
            gate = validate_patch_batch(state, plan, batch, test_only=True)
            if not gate.accepted:
                reason = gate.issues[0].code if gate.issues else "invalid_generated_test"
                last = _failed(reason)
            else:
                paths = [edit.file_path for edit in batch.edits]
                snapshot = _snapshot_test_files(root, paths)
                changed = apply_approved_patch(state)
                applied = True
                approval = state.tool_patch_approval
                if approval is None:
                    raise RuntimeError("generated test lacks exact PatchGate approval")
                state.generated_test_approvals = [
                    GeneratedTestApproval(
                        path=path,
                        content_sha256=hashlib.sha256((root / path).read_bytes()).hexdigest(),
                        patch_gate_fingerprint=approval.patch_gate_fingerprint,
                    )
                    for path in changed
                ]
                candidate = CoverageCandidate(
                    test_files=changed,
                    argv=["python", "-m", "pytest", *changed, "-q"],
                    source="generated",
                )
                state.coverage_test_files = list(changed)
                state.coverage_test_command = " ".join(candidate.argv)
                _refresh_diff(state)
                decision = await validate_differential_coverage(state, candidate)
                last = decision
                if decision.verified:
                    state.coverage_status = "generated_verified"
                    state.coverage_failure_reason = ""
                    return decision
                reason = decision.reason
        except (OSError, RuntimeError, ValueError):
            reason = "invalid_generated_test"
            last = _failed(reason)
        finally:
            if applied and not last.verified:
                try:
                    _restore_test_files(root, snapshot)
                except (OSError, RuntimeError):
                    reason = "generated_test_rollback_failed"
                    last = _failed(reason)
                state.generated_test_approvals = []
                _refresh_diff(state)
                state.patch_edits = []
                state.tool_patch_approval = None

        if state.test_generation_attempts == 1 and state.active_provider == "primary":
            if escalation_is_configured():
                apply_escalation(
                    state,
                    EscalationDecision(
                        escalate=True,
                        reason="test_generation_retry",
                    ),
                )

    state.coverage_status = "failed"
    state.coverage_failure_reason = last.reason
    return last


__all__ = [
    "generate_test_batch",
    "is_allowed_test_path",
    "request_test_batch",
    "run_test_generation_attempts",
]
