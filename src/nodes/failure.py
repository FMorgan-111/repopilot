"""FAILURE phase: Gracefully report partial progress as an issue comment."""

from __future__ import annotations

from typing import Any

from ..memory import _fire_and_forget, get_store
from ..state import AgentState, Phase, _as_state, _record_tool
from .commit import _github_add_issue_comment


async def handle_failure(state: AgentState | dict[str, Any]) -> AgentState:
    """Gracefully report partial progress as an issue comment."""
    state = _as_state(state)
    files = "\n".join(f"- {file.path}: {file.reason}" for file in state.relevant_files[:6])
    attempts = "\n\n".join(
        f"Attempt {idx + 1}: success={attempt.success}\n"
        f"File: {attempt.file_path or 'unknown'}\n"
        f"Result: {attempt.test_result}\n"
        f"Error:\n{attempt.error_log[:1500]}"
        for idx, attempt in enumerate(state.fix_attempts)
    )
    body = (
        "RepoPilot v2 could not complete an automatic fix.\n\n"
        f"Reason: {state.failure_reason or 'unspecified failure'}\n\n"
        f"Relevant files:\n{files or 'None found'}\n\n"
        f"Attempts:\n{attempts or 'No patch attempts were made.'}\n\n"
        f"Token usage: {state.token_usage}/{state.token_budget}"
    )
    if state.owner and state.repo and state.issue_number:
        try:
            await _github_add_issue_comment(state, body)
            _record_tool(
                state,
                "add_issue_comment",
                {"issue_number": state.issue_number},
                {"ok": True},
            )
        except Exception as exc:
            _record_tool(
                state,
                "add_issue_comment",
                {"issue_number": state.issue_number},
                error=str(exc),
            )

    # ── fire-and-forget: record failure in memory ──
    store = get_store()
    _fire_and_forget(
        store.record_issue(
            state.owner, state.repo, state.issue_number, success=False
        )
    )

    state.current_phase = Phase.FAILED
    return state
