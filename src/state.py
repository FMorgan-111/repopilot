"""RepoPilot v2 state models and helper functions."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


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
        if not self.search and not self.node_target:
            raise ValueError(
                "PatchEdit requires either `search` or `node_target`."
            )
        return self


class FixAttempt(BaseModel):
    patch_content: str = ""
    patch_edits: list[PatchEdit] = Field(default_factory=list)
    file_path: str = ""
    test_result: str = ""
    failure_kind: str = ""
    error_log: str = ""
    success: bool = False


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
    patch_edits: list[PatchEdit] = Field(default_factory=list)
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
    # Benchmark/eval mode: a verified test pass routes straight to DONE instead
    # of opening a PR (we have no write access to upstream repos under eval).
    skip_commit: bool = False


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
