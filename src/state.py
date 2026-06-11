"""RepoPilot v2 state models and helper functions."""

from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

from pydantic import BaseModel, Field


class Phase(str, Enum):
    UNDERSTAND = "UNDERSTAND"
    LOCATE = "LOCATE"
    PLAN = "PLAN"
    REFLECT = "REFLECT"
    EXECUTE = "EXECUTE"
    VERIFY = "VERIFY"
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


class FixAttempt(BaseModel):
    patch_content: str = ""
    file_path: str = ""
    test_result: str = ""
    error_log: str = ""
    success: bool = False


class Hypothesis(BaseModel):
    id: str
    claim: str
    evidence: list[str] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    why_selected: str = ""
    why_not_selected: str = ""


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


class AgentState(BaseModel):
    issue_url: str
    issue_title: str = ""
    issue_body: str = ""
    current_phase: Phase = Phase.UNDERSTAND
    relevant_files: list[FileInfo] = Field(default_factory=list)
    fix_attempts: list[FixAttempt] = Field(default_factory=list)
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    token_usage: int = 0
    max_retries: int = 3
    token_budget: int = 50000
    retry_count: int = 0
    tool_calls: list[ToolCall] = Field(default_factory=list)
    owner: str = ""
    repo: str = ""
    issue_number: int = 0
    issue_type: str = "unknown"
    severity: str = "unknown"
    fix_plan: str = ""
    patch_content: str = ""
    test_command: str = ""
    repo_path: str = ""
    branch_name: str = ""
    base_branch: str = "main"
    pr_url: str | None = None
    failure_reason: str = ""
    trace_id: str = ""
    reflection_notes: str = ""
    decision_frame: DecisionFrame | None = None
    frame_history: list[DecisionFrame] = Field(default_factory=list)
    decision_warnings: list[dict[str, Any]] = Field(default_factory=list)
    decision_route_checked_frame_id: str = ""
    route_decisions: list[dict[str, Any]] = Field(default_factory=list)
    pending_human_input: bool = False
    human_input_request: dict[str, Any] = Field(default_factory=dict)


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
            and previous.error_log == last.error_log
            and not previous.success
            and not last.success
        ):
            return True
    return False


def _primary_patch_file(patch_content: str) -> str:
    match = re.search(r"^\+\+\+ b/(.+)$", patch_content, re.MULTILINE)
    return match.group(1) if match else ""
