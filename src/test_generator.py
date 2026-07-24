"""Bounded, test-only regression-test generation policy."""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from .async_safety import CancellationDrainError
from .coverage_gate import (
    ChangedTarget,
    CoverageCandidate,
    CoverageDecision,
    collect_changed_targets,
    validate_differential_coverage,
)
from .evaluator_safety import sanitize_evaluator_text
from .escalation import record_model_invocation
from .llm import llm_call
from .model_policy import EscalationDecision, apply_escalation
from .model_provider import escalation_is_configured, redact_secrets
from .patch_gate import apply_approved_patch, validate_patch_batch
from .repo_paths import canonical_repo_path
from .state import (
    AgentState,
    CoverageProof,
    GeneratedTestApproval,
    PatchEdit,
    RepairPlan,
    ToolPatchApproval,
    VerifiedEditBatch,
    _estimate_tokens,
    _is_budget_exceeded,
    tool_manifest_fingerprint,
)
from .test_path_policy import is_allowed_test_path

_SYSTEM_PROMPT = (
    "Return only one JSON object matching VerifiedEditBatch with key 'edits'. "
    "Each edit must modify or create only an established Python test path. "
    "Use an exact search anchor for an existing file and an empty search for a "
    "new file. Add one focused behavioral regression test that fails on the base "
    "commit and passes on the fixed checkout. Do not edit production, CI, config, "
    "dependencies, generated artifacts, or binary files."
)
_PROMPT_LIMIT = 32_000
_TEST_GENERATION_PREFLIGHT_REASON = "test_generation_preflight_failed"
_TEST_GENERATION_BUDGET_REASON = "token_budget_exceeded"


class _TestGenerationPreflightError(ValueError):
    """Prompt construction failed before a test-generation model request."""


def _safe_text(value: object, *, limit: int) -> str:
    text = redact_secrets(str(value or ""))
    return sanitize_evaluator_text(text)[:limit]


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
    try:
        prompt = _generation_prompt(state, rejection_reason)
    except (OSError, RuntimeError, ValueError) as exc:
        raise _TestGenerationPreflightError(
            _TEST_GENERATION_PREFLIGHT_REASON
        ) from exc
    state.test_generation_attempts += 1
    model, provider = state.active_model, state.active_provider
    input_tokens = _estimate_tokens(_SYSTEM_PROMPT, prompt)
    response_text = ""
    started = time.monotonic()
    try:
        raw = await llm_call(
            _SYSTEM_PROMPT,
            prompt,
            model=model,
            provider=provider,
            temperature=0.0,
        )
    except (asyncio.CancelledError, CancellationDrainError) as exc:
        elapsed = time.monotonic() - started
        record_model_invocation(
            state,
            model=model,
            provider=provider,
            node="test_generation",
            elapsed_seconds=elapsed,
            input_tokens=input_tokens,
            output_tokens=0,
            status="cancelled",
            error=exc,
        )
        state.token_usage += input_tokens
        raise
    except Exception as exc:
        elapsed = time.monotonic() - started
        record_model_invocation(
            state,
            model=model,
            provider=provider,
            node="test_generation",
            elapsed_seconds=elapsed,
            input_tokens=input_tokens,
            output_tokens=0,
            status="error",
            error=exc,
        )
        state.token_usage += input_tokens
        raise
    try:
        response_text = json.dumps(
            raw,
            ensure_ascii=False,
            sort_keys=True,
            default=lambda value: type(value).__name__,
        )
        batch = _validate_test_batch(raw)
    except (TypeError, ValueError, RecursionError) as exc:
        elapsed = time.monotonic() - started
        output_tokens = _estimate_tokens(response_text)
        record_model_invocation(
            state,
            model=model,
            provider=provider,
            node="test_generation",
            elapsed_seconds=elapsed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            status="invalid_response",
            error=exc,
        )
        state.token_usage += input_tokens + output_tokens
        raise
    elapsed = time.monotonic() - started
    output_tokens = _estimate_tokens(response_text)
    record_model_invocation(
        state,
        model=model,
        provider=provider,
        node="test_generation",
        elapsed_seconds=elapsed,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        status="ok",
    )
    state.token_usage += input_tokens + output_tokens
    return batch


def _validate_test_batch(raw: object) -> VerifiedEditBatch:
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


@dataclass(frozen=True)
class _ProductionState:
    active_repair_plan: RepairPlan | None
    patch_edits: list[PatchEdit]
    tool_patch_approval: ToolPatchApproval | None
    patch_content: str
    coverage_status: str
    coverage_test_files: list[str]
    coverage_test_command: str
    coverage_failure_reason: str
    coverage_proof: CoverageProof | None
    generated_test_approvals: list[GeneratedTestApproval]


def _capture_production_state(state: AgentState) -> _ProductionState:
    return _ProductionState(
        active_repair_plan=(
            state.active_repair_plan.model_copy(deep=True)
            if state.active_repair_plan
            else None
        ),
        patch_edits=[edit.model_copy(deep=True) for edit in state.patch_edits],
        tool_patch_approval=(
            state.tool_patch_approval.model_copy(deep=True)
            if state.tool_patch_approval
            else None
        ),
        patch_content=state.patch_content,
        coverage_status=state.coverage_status,
        coverage_test_files=list(state.coverage_test_files),
        coverage_test_command=state.coverage_test_command,
        coverage_failure_reason=state.coverage_failure_reason,
        coverage_proof=(
            state.coverage_proof.model_copy(deep=True)
            if state.coverage_proof
            else None
        ),
        generated_test_approvals=[
            item.model_copy(deep=True) for item in state.generated_test_approvals
        ],
    )


def _restore_production_state(state: AgentState, saved: _ProductionState) -> None:
    state.active_repair_plan = (
        saved.active_repair_plan.model_copy(deep=True)
        if saved.active_repair_plan
        else None
    )
    state.patch_edits = [edit.model_copy(deep=True) for edit in saved.patch_edits]
    state.tool_patch_approval = (
        saved.tool_patch_approval.model_copy(deep=True)
        if saved.tool_patch_approval
        else None
    )
    state.patch_content = saved.patch_content
    state.coverage_status = saved.coverage_status  # type: ignore[assignment]
    state.coverage_test_files = list(saved.coverage_test_files)
    state.coverage_test_command = saved.coverage_test_command
    state.coverage_failure_reason = saved.coverage_failure_reason
    state.coverage_proof = (
        saved.coverage_proof.model_copy(deep=True) if saved.coverage_proof else None
    )
    state.generated_test_approvals = [
        item.model_copy(deep=True) for item in saved.generated_test_approvals
    ]


def _combined_approval(
    state: AgentState,
    saved: _ProductionState,
    generated: ToolPatchApproval,
) -> ToolPatchApproval:
    production = saved.tool_patch_approval
    if production is None or production.base_ref != state.repo_ref:
        raise ValueError("generated tests require an existing production approval")
    full_patch = state.patch_content
    entries = {item.path: item for item in production.changed_manifest}
    entries.update({item.path: item for item in generated.changed_manifest})
    manifest = [entries[path] for path in sorted(entries)]
    patch_sha256 = hashlib.sha256(full_patch.encode("utf-8")).hexdigest()
    payload = json.dumps(
        {
            "base_ref": state.repo_ref,
            "production_gate": production.patch_gate_fingerprint,
            "generated_gate": generated.patch_gate_fingerprint,
            "patch_sha256": patch_sha256,
            "manifest": [item.model_dump(mode="json") for item in manifest],
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return ToolPatchApproval(
        base_ref=state.repo_ref,
        patch_sha256=patch_sha256,
        patch_gate_fingerprint=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        changed_manifest=manifest,
        manifest_fingerprint=tool_manifest_fingerprint(manifest),
    )


async def run_test_generation_attempts(
    state: AgentState,
    rejection_reason: str,
) -> CoverageDecision:
    """Try at most twice; only a differential proof can return verified."""
    root = Path(state.repo_path).resolve(strict=True)
    production = _capture_production_state(state)
    reason = _safe_text(rejection_reason, limit=300) or "no_coverage_candidate"
    last = _failed(reason)
    budget_terminal = False

    while state.test_generation_attempts < 2:
        _restore_production_state(state, production)
        if _is_budget_exceeded(state):
            last = _failed(_TEST_GENERATION_BUDGET_REASON)
            budget_terminal = True
            break
        snapshot: dict[str, bytes | None] = {}
        applied = False
        pending_drain: asyncio.CancelledError | CancellationDrainError | None = None
        rollback_error: BaseException | None = None
        try:
            batch = await request_test_batch(state, reason)
            if _is_budget_exceeded(state):
                last = _failed(_TEST_GENERATION_BUDGET_REASON)
                budget_terminal = True
                break
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
                generated_approval = state.tool_patch_approval
                if generated_approval is None:
                    raise RuntimeError("generated test lacks exact PatchGate approval")
                candidate = CoverageCandidate(
                    test_files=changed,
                    argv=["python", "-m", "pytest", *changed, "-q"],
                    source="generated",
                )
                state.coverage_test_files = list(changed)
                state.coverage_test_command = " ".join(candidate.argv)
                _refresh_diff(state)
                combined = _combined_approval(state, production, generated_approval)
                state.tool_patch_approval = combined
                generated_entries = {
                    entry.path: entry for entry in generated_approval.changed_manifest
                }
                markers: list[GeneratedTestApproval] = []
                for path in changed:
                    entry = generated_entries[path]
                    base = subprocess.run(
                        ["git", "show", f"{state.repo_ref}:{path}"],
                        cwd=root,
                        check=False,
                        capture_output=True,
                        timeout=30,
                    )
                    markers.append(
                        GeneratedTestApproval(
                            path=path,
                            change=entry.change,
                            base_content_sha256=(
                                hashlib.sha256(base.stdout).hexdigest()
                                if entry.change == "modified" and base.returncode == 0
                                else None
                            ),
                            content_sha256=hashlib.sha256(
                                (root / path).read_bytes()
                            ).hexdigest(),
                            patch_gate_fingerprint=combined.patch_gate_fingerprint,
                        )
                    )
                state.generated_test_approvals = markers
                state.active_repair_plan = production.active_repair_plan
                state.patch_edits = [
                    edit.model_copy(deep=True) for edit in production.patch_edits
                ]
                decision = await validate_differential_coverage(state, candidate)
                last = decision
                if decision.verified:
                    state.coverage_status = "generated_verified"
                    state.coverage_failure_reason = ""
                    return decision
                reason = decision.reason
        except (asyncio.CancelledError, CancellationDrainError) as error:
            pending_drain = error
        except _TestGenerationPreflightError:
            reason = _TEST_GENERATION_PREFLIGHT_REASON
            last = _failed(reason)
            break
        except (OSError, RuntimeError, ValueError):
            if _is_budget_exceeded(state):
                last = _failed(_TEST_GENERATION_BUDGET_REASON)
                budget_terminal = True
                break
            reason = "invalid_generated_test"
            last = _failed(reason)
        except Exception:
            if _is_budget_exceeded(state):
                last = _failed(_TEST_GENERATION_BUDGET_REASON)
                budget_terminal = True
                break
            raise
        finally:
            if applied and not last.verified:
                try:
                    _restore_test_files(root, snapshot)
                except (OSError, RuntimeError) as error:
                    rollback_error = error
                    reason = "generated_test_rollback_failed"
                    last = _failed(reason)
                    _restore_production_state(state, production)
                    state.coverage_status = "failed"
                    state.coverage_failure_reason = reason
                    state.coverage_proof = None
                    state.pr_url = None
                    if pending_drain is None:
                        return last
            if not last.verified:
                _restore_production_state(state, production)

        if pending_drain is not None:
            if rollback_error is not None:
                if isinstance(pending_drain, CancellationDrainError):
                    raise pending_drain from rollback_error
                raise CancellationDrainError(
                    "generated test rollback",
                    pending_drain,
                    rollback_error,
                ) from rollback_error
            raise pending_drain

        if state.test_generation_attempts == 1 and state.active_provider == "primary":
            if escalation_is_configured():
                apply_escalation(
                    state,
                    EscalationDecision(
                        escalate=True,
                        reason="test_generation_retry",
                    ),
                )

    _restore_production_state(state, production)
    if not budget_terminal:
        state.coverage_status = "failed"
        state.coverage_failure_reason = last.reason
    return last


__all__ = [
    "generate_test_batch",
    "is_allowed_test_path",
    "request_test_batch",
    "run_test_generation_attempts",
]
