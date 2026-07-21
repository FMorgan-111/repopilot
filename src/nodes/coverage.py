"""COVERAGE phase orchestration and terminal hard gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..coverage_gate import (
    CoverageDecision,
    collect_changed_targets,
    discover_coverage_candidates,
    validate_differential_coverage,
)
from ..state import AgentState, CoverageProof, Phase, _as_state
from ..test_generator import run_test_generation_attempts


def _persist_verified(state: AgentState, decision: CoverageDecision) -> None:
    candidate = decision.candidate
    if not decision.verified or candidate is None:
        raise ValueError("coverage proof persistence requires a verified candidate")
    state.coverage_status = decision.status
    state.coverage_test_files = list(candidate.test_files)
    state.coverage_test_command = " ".join(candidate.argv)
    state.coverage_failure_reason = ""
    state.coverage_proof = CoverageProof(
        source=candidate.source,
        test_files=list(candidate.test_files),
        argv=list(candidate.argv),
        fixed_runs=decision.fixed_runs,
        base_runs=decision.base_runs,
    )
    state.current_phase = Phase.DONE if state.skip_commit else Phase.COMMIT


async def ensure_coverage(state: AgentState | dict[str, Any]) -> AgentState:
    """Require a stable differential proof, generating tests only as fallback."""
    state = _as_state(state)
    if state.coverage_proof is not None and state.coverage_status in {
        "existing_verified",
        "generated_verified",
    }:
        state.current_phase = Phase.DONE if state.skip_commit else Phase.COMMIT
        return state

    try:
        targets = collect_changed_targets(
            Path(state.repo_path),
            state.repo_ref,
        )
        candidates = discover_coverage_candidates(state, targets)
    except (OSError, RuntimeError, ValueError):
        candidates = []

    rejection_reason = "no_coverage_candidate"
    for candidate in candidates:
        decision = await validate_differential_coverage(state, candidate)
        if decision.verified:
            _persist_verified(state, decision)
            return state
        rejection_reason = decision.reason

    generated = await run_test_generation_attempts(state, rejection_reason)
    if generated.verified:
        _persist_verified(state, generated)
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
