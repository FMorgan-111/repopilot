"""UNDERSTAND phase: Read and classify the GitHub issue."""

from __future__ import annotations

import json
from typing import Any

from ..state import (
    AgentState,
    Phase,
    _as_state,
    _estimate_tokens,
    _extract_json_object,
    _record_tool,
    _remember,
)
from ..tools import read_issue
from ..agent import parse_issue_url
from ..llm import llm_call


async def understand_issue(state: AgentState | dict[str, Any]) -> AgentState:
    """Read the GitHub issue, classify it, and seed conversation memory."""
    import sys
    state = _as_state(state)
    try:
        owner, repo, issue_number = parse_issue_url(state.issue_url)
    except ValueError as exc:
        state.failure_reason = str(exc)
        state.current_phase = Phase.FAILURE
        return state

    state.owner = owner
    state.repo = repo
    state.issue_number = issue_number
    print(f"  [understand] Parsed {owner}/{repo}#{issue_number}", file=sys.stderr, flush=True)

    try:
        print(f"  [understand] Fetching issue from GitHub...", file=sys.stderr, flush=True)
        issue = await read_issue(owner, repo, issue_number)
        _record_tool(
            state,
            "read_issue",
            {"owner": owner, "repo": repo, "issue_number": issue_number},
            issue,
        )
    except Exception as exc:
        _record_tool(
            state,
            "read_issue",
            {"owner": owner, "repo": repo, "issue_number": issue_number},
            error=str(exc),
        )
        state.failure_reason = f"Failed to read issue: {exc}"
        state.current_phase = Phase.FAILURE
        return state

    state.issue_title = issue.get("title", "")
    state.issue_body = issue.get("body", "")
    labels = [str(label).lower() for label in issue.get("labels", [])]
    state.issue_type = "bug" if "bug" in labels else "feature" if "feature" in labels else "unknown"
    state.severity = "high" if {"security", "critical", "regression"} & set(labels) else "medium"
    print(f"  [understand] Got issue: \"{state.issue_title[:60]}\" type={state.issue_type}", file=sys.stderr, flush=True)

    system = (
        "You classify GitHub issues for an autonomous coding agent. "
        "Return JSON with keys: type, severity, summary, likely_modules."
    )
    user = (
        f"Title: {state.issue_title}\n\nBody:\n{state.issue_body[:4000]}\n"
        f"Labels: {labels}"
    )
    try:
        analysis = _extract_json_object(await llm_call(system, user))
        state.issue_type = analysis.get("type", state.issue_type)
        state.severity = analysis.get("severity", state.severity)
        _remember(state, "assistant", json.dumps(analysis))
        state.token_usage += _estimate_tokens(system, user, json.dumps(analysis))
    except Exception as exc:
        _remember(state, "assistant", f"Issue classification skipped: {exc}")
        state.token_usage += _estimate_tokens(system, user)

    _remember(state, "user", f"{state.issue_title}\n\n{state.issue_body[:2000]}")
    state.current_phase = Phase.LOCATE
    return state
