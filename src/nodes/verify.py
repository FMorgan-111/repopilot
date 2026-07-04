"""VERIFY phase: Parse test output and route to COMMIT, retry PLAN, or FAILED."""

from __future__ import annotations

from typing import Any

from ..state import (
    AgentState,
    FixAttempt,
    Phase,
    _as_state,
    _is_budget_exceeded,
    _same_failure_seen_twice,
)


def _consecutive_failure_count(attempts: list[FixAttempt], failure_kind: str) -> int:
    count = 0
    for attempt in reversed(attempts):
        if _failure_kind(attempt) != failure_kind:
            break
        count += 1
    return count


def _is_patch_preflight_failure(attempt: FixAttempt) -> bool:
    return (
        _failure_kind(attempt) == "patch_apply_failed"
        and "patch preflight check failed" in attempt.error_log.lower()
    )


def _is_patch_repair_failure(attempt: FixAttempt) -> bool:
    if _is_patch_preflight_failure(attempt):
        return True
    return (
        _failure_kind(attempt) == "patch_apply_failed"
        and "search/replace edit failed" in attempt.error_log.lower()
    )


def _consecutive_patch_repair_failure_count(attempts: list[FixAttempt]) -> int:
    count = 0
    for attempt in reversed(attempts):
        if not _is_patch_repair_failure(attempt):
            break
        count += 1
    return count


def _failure_kind(attempt: FixAttempt) -> str:
    if attempt.failure_kind:
        return attempt.failure_kind
    if attempt.test_result == "patch_apply_failed":
        return "patch_apply_failed"
    return ""


async def _record_episode_best_effort(state: AgentState, latest: FixAttempt) -> None:
    """Persist this attempt's (issue, outcome, patch) as a cross-repo episode.
    Never raises: if the episode store or embedding model is unavailable, the
    agent proceeds unaffected."""
    try:
        from ..memory.error_episode_store import get_episode_store

        store = get_episode_store()
        if store is None:
            return
        patch = latest.patch_content
        if not patch and latest.patch_edits:
            patch = "\n".join(
                f"{e.file_path}: {e.search[:80]} -> {e.replace[:80]}"
                for e in latest.patch_edits
            )
        await store.arecord(
            owner=state.owner,
            repo=state.repo,
            issue_url=state.issue_url,
            issue_title=state.issue_title,
            issue_body=state.issue_body,
            error_log=latest.error_log or "",
            patch=patch or "",
            success=bool(latest.success),
        )
    except Exception:
        return


async def verify_fix(state: AgentState | dict[str, Any]) -> AgentState:
    """Parse test output and route to COMMIT, retry PLAN, or FAILED."""
    state = _as_state(state)
    if not state.fix_attempts:
        state.failure_reason = "No fix attempt was recorded."
        state.current_phase = Phase.FAILURE
        return state

    latest = state.fix_attempts[-1]
    await _record_episode_best_effort(state, latest)
    if latest.success:
        # In benchmark/eval mode we have no write access to upstream repos, so a
        # verified test pass is the terminal success — skip the PR step.
        state.current_phase = Phase.DONE if state.skip_commit else Phase.COMMIT
        return state

    if _same_failure_seen_twice(state):
        state.failure_reason = "Same patch produced the same failure twice."
        state.current_phase = Phase.FAILURE
        return state

    if _failure_kind(latest) == "infra_error":
        message = latest.error_log.strip() or "execution infrastructure failed"
        state.failure_reason = f"Infrastructure error during execution: {message[:500]}"
        state.current_phase = Phase.FAILURE
        return state

    if _failure_kind(latest) == "patch_apply_failed":
        if _is_patch_repair_failure(latest):
            consecutive_repair_failures = _consecutive_patch_repair_failure_count(
                state.fix_attempts
            )
            repair_budget = state.max_retries + 1
            if consecutive_repair_failures <= repair_budget:
                if _is_budget_exceeded(state):
                    state.failure_reason = "Token budget exceeded during verification."
                    state.current_phase = Phase.FAILURE
                    return state
                state.current_phase = Phase.REFLECT
                return state
            state.failure_reason = (
                "Patch repair budget exhausted after "
                f"{consecutive_repair_failures} failures."
            )
            state.current_phase = Phase.FAILURE
            return state

        consecutive_patch_apply_failures = _consecutive_failure_count(
            state.fix_attempts,
            "patch_apply_failed",
        )
        if consecutive_patch_apply_failures == 1:
            state.current_phase = Phase.REFLECT
            return state
        if state.retry_count >= state.max_retries:
            state.failure_reason = f"Maximum retries reached: {state.max_retries}."
            state.current_phase = Phase.FAILURE
            return state
        if _is_budget_exceeded(state):
            state.failure_reason = "Token budget exceeded during verification."
            state.current_phase = Phase.FAILURE
            return state
        state.retry_count += 1
        state.current_phase = Phase.REFLECT
        return state

    if state.retry_count >= state.max_retries:
        state.failure_reason = f"Maximum retries reached: {state.max_retries}."
        state.current_phase = Phase.FAILURE
        return state

    if _is_budget_exceeded(state):
        state.failure_reason = "Token budget exceeded during verification."
        state.current_phase = Phase.FAILURE
        return state

    state.retry_count += 1
    state.current_phase = Phase.REFLECT
    return state
