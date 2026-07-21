"""COVERAGE phase orchestration and terminal hard gate."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..coverage_gate import (
    CoverageCandidate,
    CoverageDecision,
    collect_changed_targets,
    coverage_proof_matches_state,
    discover_coverage_candidates,
    validate_live_coverage_binding,
    validate_differential_coverage,
)
from ..state import AgentState, CoverageProof, Phase, _as_state
from ..test_generator import run_test_generation_attempts


def _persist_verified(state: AgentState, decision: CoverageDecision) -> None:
    candidate = decision.candidate
    if not decision.verified or candidate is None:
        raise ValueError("coverage proof persistence requires a verified candidate")
    validate_live_coverage_binding(state)
    state.coverage_status = decision.status
    state.coverage_test_files = list(candidate.test_files)
    state.coverage_test_command = " ".join(candidate.argv)
    state.coverage_failure_reason = ""
    approval = state.tool_patch_approval
    if approval is None:
        raise ValueError("coverage proof requires the complete approved diff")
    root = Path(state.repo_path).resolve(strict=True)
    state.coverage_proof = CoverageProof(
        source=candidate.source,
        status=decision.status,  # type: ignore[arg-type]
        test_files=list(candidate.test_files),
        argv=list(candidate.argv),
        fixed_runs=decision.fixed_runs,
        base_runs=decision.base_runs,
        base_ref=state.repo_ref,
        patch_sha256=approval.patch_sha256,
        patch_gate_fingerprint=approval.patch_gate_fingerprint,
        manifest_fingerprint=approval.manifest_fingerprint,
        test_content_digests={
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in candidate.test_files
        },
    )
    validate_live_coverage_binding(state)
    state.current_phase = Phase.DONE if state.skip_commit else Phase.COMMIT


def _try_persist_verified(state: AgentState, decision: CoverageDecision) -> bool:
    try:
        _persist_verified(state, decision)
    except (OSError, RuntimeError, ValueError):
        state.coverage_status = "failed"
        state.coverage_failure_reason = "coverage_infra"
        state.coverage_proof = None
        state.failure_reason = "coverage_infra:live_binding_mismatch"
        state.pr_url = None
        state.current_phase = Phase.FAILURE
        return False
    return True


async def ensure_coverage(state: AgentState | dict[str, Any]) -> AgentState:
    """Require a stable differential proof, generating tests only as fallback."""
    state = _as_state(state)
    replay_reason = ""
    if state.coverage_proof is not None:
        proof = state.coverage_proof
        if coverage_proof_matches_state(state, proof):
            replay_candidate = CoverageCandidate(
                test_files=proof.test_files,
                argv=proof.argv,
                source=proof.source,
            )
            replay = await validate_differential_coverage(state, replay_candidate)
            if replay.verified:
                _try_persist_verified(state, replay)
                return state
            replay_reason = replay.reason
        else:
            replay_reason = "coverage_binding_mismatch"
        state.coverage_proof = None
        state.coverage_status = "pending"

    try:
        targets = collect_changed_targets(
            Path(state.repo_path),
            state.repo_ref,
        )
        candidates = discover_coverage_candidates(state, targets)
    except (OSError, RuntimeError, ValueError):
        candidates = []

    rejection_reason = replay_reason or "no_coverage_candidate"
    for candidate in candidates:
        decision = await validate_differential_coverage(state, candidate)
        if decision.verified:
            _try_persist_verified(state, decision)
            return state
        rejection_reason = decision.reason

    generated = await run_test_generation_attempts(state, rejection_reason)
    if generated.verified:
        _try_persist_verified(state, generated)
        return state

    reason = generated.reason or "invalid_generated_test"
    state.coverage_status = "failed"
    state.coverage_failure_reason = reason[:300]
    state.coverage_proof = None
    state.failure_reason = f"test_generation_failed:{reason[:300]}"
    state.pr_url = None
    state.current_phase = Phase.FAILURE
    return state


__all__ = ["ensure_coverage"]
