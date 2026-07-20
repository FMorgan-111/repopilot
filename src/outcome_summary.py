"""Bounded rolling summaries of completed repair attempts."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .escalation import record_model_invocation
from .llm import llm_call
from .state import AgentState, DecisionFrame, FixAttempt, _estimate_tokens
from .summary_safety import sanitize_summary_text

MAX_OUTCOME_SUMMARY_CHARS = 200
PRIMARY_SUMMARY_MODEL = "gemini-3.5-flash:stable"
OUTCOME_SUMMARY_SECTION = "Completed attempts (rolling summary):"

_SUMMARY_SYSTEM_PROMPT = (
    "Compress the allowlisted completed repair outcome into one factual rolling "
    "summary. Preserve useful prior facts, state what failed, and state the next "
    "action. Return ONLY JSON with exactly one key: summary. The summary must be "
    "non-empty and at most 200 Unicode characters. Do not add paths, credentials, "
    "HTTP payloads, evaluator fields, patches, or conversation history."
)


class OutcomeSummaryInput(BaseModel):
    """The complete allowlist for the auxiliary summary call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    previous_summary: str = Field(default="", max_length=MAX_OUTCOME_SUMMARY_CHARS)
    plan_signature: str = Field(max_length=80)
    patch_outcome: str = Field(max_length=120)
    test_failure_class: str = Field(max_length=80)
    reflection_action: str = Field(max_length=160)


class _OutcomeSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    summary: str


def _generated_test_paths(state: AgentState) -> tuple[str, ...]:
    return tuple(approval.path for approval in state.generated_test_approvals)


def _safe_summary_component(
    value: object,
    limit: int,
    *,
    denied_literals: Iterable[str] = (),
) -> str:
    safe = sanitize_summary_text(value, limit, denied_literals=denied_literals)
    return safe.replace(OUTCOME_SUMMARY_SECTION, "").strip()


def sanitize_outcome_summary(state: AgentState, value: object) -> str:
    """Return a prompt-safe, bounded summary for persistence or PLAN injection."""
    return _safe_summary_component(
        value,
        MAX_OUTCOME_SUMMARY_CHARS,
        denied_literals=_generated_test_paths(state),
    )


def _latest_frame(state: AgentState, stage: str) -> DecisionFrame | None:
    candidates: list[DecisionFrame] = []
    if state.decision_frame is not None:
        candidates.append(state.decision_frame)
    candidates.extend(reversed(state.frame_history))
    seen: set[str] = set()
    for frame in candidates:
        identity = frame.frame_id or str(id(frame))
        if identity in seen:
            continue
        seen.add(identity)
        if frame.stage == stage:
            return frame
    return None


def _plan_signature(frame: DecisionFrame | None, attempt: FixAttempt) -> str:
    def digest(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def canonical_digest(value: object) -> str:
        encoded_value = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return digest(encoded_value)

    material: dict[str, Any] = {
        "frame": frame.model_dump(mode="json") if frame is not None else None,
        "transaction": {
            "patch_sha256": digest(attempt.patch_content),
            "legacy_file_path_sha256": digest(attempt.file_path),
            "edits": [
                {
                    "order": index,
                    "edit_sha256": canonical_digest(edit.model_dump(mode="json")),
                    "file_path_sha256": digest(edit.file_path),
                    "search_sha256": digest(edit.search),
                    "replace_sha256": digest(edit.replace),
                    "node_target_sha256": digest(edit.node_target),
                    "replace_all": edit.replace_all,
                    "expected_content_sha256": edit.expected_content_sha256,
                    "exact_only": edit.exact_only,
                }
                for index, edit in enumerate(attempt.patch_edits)
            ],
        },
    }
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_outcome_summary_input(state: AgentState) -> OutcomeSummaryInput:
    """Copy only the latest attempt, current/history frames, and prior summary."""
    latest = state.fix_attempts[-1] if state.fix_attempts else FixAttempt()
    generated_paths = _generated_test_paths(state)
    plan_frame = _latest_frame(state, "plan")
    reflect_frame = _latest_frame(state, "reflect")

    patch_result = latest.test_result or ("passed" if latest.success else "failed")
    patch_outcome = _safe_summary_component(
        patch_result,
        120,
        denied_literals=generated_paths,
    ) or ("passed" if latest.success else "failed")
    failure_class = _safe_summary_component(
        latest.failure_kind or latest.test_result or "none",
        80,
        denied_literals=generated_paths,
    ) or "unknown"
    if reflect_frame is None:
        reflection_action = "plan"
    else:
        reflection_action = _safe_summary_component(
            f"{reflect_frame.recommended_action}: {reflect_frame.summary}",
            160,
            denied_literals=generated_paths,
        ) or reflect_frame.recommended_action

    return OutcomeSummaryInput(
        previous_summary=sanitize_outcome_summary(
            state,
            state.attempt_outcome_summary,
        ),
        plan_signature=_plan_signature(plan_frame, latest),
        patch_outcome=patch_outcome,
        test_failure_class=failure_class,
        reflection_action=reflection_action,
    )


def deterministic_outcome_summary(item: OutcomeSummaryInput) -> str:
    """Build the non-LLM fallback required at the auxiliary boundary."""
    text = (
        f"plan={item.plan_signature}; patch={item.patch_outcome}; "
        f"test={item.test_failure_class}; next={item.reflection_action}"
    )
    return sanitize_summary_text(text, MAX_OUTCOME_SUMMARY_CHARS)


async def summarize_attempt_outcome(
    state: AgentState,
    *,
    llm: Callable[..., Awaitable[dict[str, Any]]] | None = None,
) -> str:
    """Replace the rolling summary using Gemini, falling back without failing."""
    item = build_outcome_summary_input(state)
    user_prompt = item.model_dump_json()
    input_tokens = _estimate_tokens(_SUMMARY_SYSTEM_PROMPT, user_prompt)
    output_tokens = 0
    started = time.monotonic()

    try:
        call = llm or llm_call
        response = await call(
            _SUMMARY_SYSTEM_PROMPT,
            user_prompt,
            model=PRIMARY_SUMMARY_MODEL,
            provider="primary",
            temperature=0.0,
        )
        response_text = json.dumps(response, ensure_ascii=False, sort_keys=True)
        output_tokens = _estimate_tokens(response_text)
        parsed = _OutcomeSummaryResponse.model_validate(response)
        candidate = _safe_summary_component(
            parsed.summary,
            MAX_OUTCOME_SUMMARY_CHARS + 1,
            denied_literals=_generated_test_paths(state),
        )
        if not candidate or len(candidate) > MAX_OUTCOME_SUMMARY_CHARS:
            raise ValueError("Outcome summary is empty or overlong")
    except Exception as exc:
        state.summary_token_usage += input_tokens + output_tokens
        invalid = isinstance(exc, (ValidationError, ValueError, TypeError))
        record_model_invocation(
            state,
            model=PRIMARY_SUMMARY_MODEL,
            provider="primary",
            node="outcome_summary",
            elapsed_seconds=time.monotonic() - started,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            status="invalid_response" if invalid else "error",
            error=exc,
        )
        return deterministic_outcome_summary(item)

    state.summary_token_usage += input_tokens + output_tokens
    record_model_invocation(
        state,
        model=PRIMARY_SUMMARY_MODEL,
        provider="primary",
        node="outcome_summary",
        elapsed_seconds=time.monotonic() - started,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        status="ok",
    )
    return candidate
