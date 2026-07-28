"""PLAN phase: Ask the LLM for a concrete patch-oriented plan."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Sequence
from typing import Any

from ..async_safety import CancellationDrainError
from ..escalation import (
    immediate_model_policy_reason,
    record_model_invocation,
    relevance_window,
)
from ..http_client import is_retryable_llm_error
from ..llm import llm_call
from ..model_policy import (
    apply_escalation,
    primary_budget_limit,
    record_no_progress,
    record_progress,
    should_escalate,
)
from ..model_provider import escalation_is_configured
from ..outcome_summary import (
    MAX_OUTCOME_SUMMARY_CHARS,
    OUTCOME_SUMMARY_SECTION,
    sanitize_outcome_summary,
)
from ..patch_authorization import (
    PatchAuthorizationIssue,
    authorize_plan_patch,
    render_patch_correction,
    retire_patch_authorization,
)
from ..reasoning_loop import (
    ReasoningStop,
    prompt_with_new_evidence,
    route_reasoning_tool,
    validate_reasoning_response,
)
from ..repair_flow import RepairContextError, resolve_search_target_symbol_strict
from ..repair_rounds import (
    begin_repair_round,
    bind_repair_round_author,
    record_failed_repair_round,
)
from ..schemas import PlanDecision
from ..state import (
    AgentState,
    DecisionFrame,
    Hypothesis,
    Phase,
    _as_state,
    _estimate_tokens,
    _extract_json_object,
    _human_answer_context,
    _is_budget_exceeded,
    _issue_search_terms,
    _record_decision_frame,
    _record_frame_health_warning,
    _record_node_diagnostic,
    _remember,
)
from ..summary_safety import sanitize_model_context
from ..tool_router import route_tool_intent
from .verify import _test_failure_class

PLAN_ISSUE_BODY_LIMIT = 2500
# The planner needs to see real function bodies, not just import headers. At
# 1200 chars it only saw ~3% of a file (the imports) and kept asking for more
# context forever. 6000 chars surfaces the logic the fix actually touches.
PLAN_FILE_CONTENT_LIMIT = 6000
PLAN_MAX_FILES = 3
PLAN_FAILURE_LOG_LIMIT = 1000
PLAN_FAILURE_RESULT_LIMIT = 500
PLAN_PREVIOUS_FAILURES_TOTAL_LIMIT = 6_000
PLAN_FAILED_EDITS_CONTEXT_LIMIT = 4_000
PLAN_NEW_EVIDENCE_CONTEXT_LIMIT = 12_000
MODEL_CONTEXT_DENIED_LITERALS = (
    "unified diff",
    "replace_all",
    "RepairPlan",
    "VerifiedEditBatch",
)

PLAN_SYSTEM = (
    "You are RepoPilot's patch planner. Return exactly one JSON response variant. "
    "For a code change, return only {\"patch_edits\":[{\"file_path\":\"...\","
    "\"search\":\"...\",\"replace\":\"...\"}]}. Copy search text verbatim from "
    "approved file context and make it match exactly once. Use kind='tool' with one "
    "tool_intent only when one specific repository fact is missing. Use kind='stop' "
    "with stop_reason only when no safe repair is possible."
)

# Hard cap on consecutive collect_more_context rounds before we give up.
# Default 1: eval showed the planner spiralling on collect_more_context instead
# of committing a patch, so one context round is allowed then the next is forced
# to stop. Override via REPOPILOT_MAX_CONTEXT_ROUNDS.
MAX_CONTEXT_COLLECTION_ROUNDS = int(os.getenv("REPOPILOT_MAX_CONTEXT_ROUNDS", "1"))


def _is_patch_apply_failure(attempt: Any) -> bool:
    return (
        getattr(attempt, "failure_kind", "") == "patch_apply_failed"
        or getattr(attempt, "test_result", "") == "patch_apply_failed"
    )


def _selected_hypothesis(frame: DecisionFrame | None) -> Hypothesis | None:
    if frame is None or not frame.selected_hypothesis_id:
        return None
    return next(
        (
            hypothesis
            for hypothesis in frame.hypotheses
            if hypothesis.id == frame.selected_hypothesis_id
        ),
        None,
    )


def _patch_apply_hypothesis_anchor(
    state: AgentState,
) -> tuple[DecisionFrame, Hypothesis] | None:
    if not state.fix_attempts or not _is_patch_apply_failure(state.fix_attempts[-1]):
        return None

    for frame in reversed(state.frame_history):
        if frame.stage != "plan":
            continue
        selected = _selected_hypothesis(frame)
        if selected is not None:
            return frame, selected
    return None


def _truncate_prompt_text(value: str, limit: int = 500) -> str:
    return sanitize_model_context(
        value,
        limit,
        denied_literals=MODEL_CONTEXT_DENIED_LITERALS,
    )


def _normalized_edit_key(file_path: str, search: str) -> str:
    """Whitespace-insensitive identity for a search/replace edit target."""
    return f"{file_path}::{' '.join(search.split())}"


def _budget_scaled_file_limits(state: AgentState) -> tuple[int, int]:
    """Shrink the file context as the token budget depletes.

    A single global budget gates how many plan/retry attempts can run; a full-
    size prompt on every attempt can exhaust it before a late retry produces a
    patch. As the remaining balance drops we trade context breadth for more
    surviving attempts. PLAN quality is protected — the file window never
    shrinks below half, and at least one file is always shown."""
    budget = state.token_budget
    if budget <= 0:
        return PLAN_FILE_CONTENT_LIMIT, PLAN_MAX_FILES
    remaining = max(0.0, (budget - state.token_usage) / budget)
    if remaining >= 0.5:
        return PLAN_FILE_CONTENT_LIMIT, PLAN_MAX_FILES
    if remaining >= 0.25:
        return PLAN_FILE_CONTENT_LIMIT * 2 // 3, PLAN_MAX_FILES
    return PLAN_FILE_CONTENT_LIMIT // 2, max(1, PLAN_MAX_FILES - 1)


def _edit_key(edit: Any) -> str:
    """Identity of any edit — node-anchored edits key on their node_target,
    search/replace edits on their normalized search."""
    node_target = getattr(edit, "node_target", "") or ""
    if node_target:
        return f"{edit.file_path}::node::{node_target}"
    return _normalized_edit_key(edit.file_path, edit.search)


def _prior_failed_edits_context(state: AgentState) -> str:
    """List already-tried-and-failed edits so the planner is forced to diversify
    instead of re-emitting a known-failing search/replace pair."""
    seen: dict[str, Any] = {}
    for attempt in state.fix_attempts:
        if getattr(attempt, "success", False):
            continue
        for edit in getattr(attempt, "patch_edits", []) or []:
            key = _edit_key(edit)
            seen.setdefault(key, edit)
    if not seen:
        return ""
    lines = [
        "ALREADY-TRIED EDITS THAT FAILED — do NOT re-emit these exact "
        "search/replace pairs. If your best fix matches one, you MUST change the "
        "target (different file or hunk) or the root-cause approach:",
    ]
    for edit in seen.values():
        lines.append(
            f"- file: {edit.file_path}\n"
            f"  search (verbatim):\n{_truncate_prompt_text(edit.search, 300)}"
        )
    return sanitize_model_context(
        "\n".join(lines),
        PLAN_FAILED_EDITS_CONTEXT_LIMIT,
        denied_literals=MODEL_CONTEXT_DENIED_LITERALS,
    )


def _prior_assertion_symbols(state: AgentState) -> tuple[set[str], bool]:
    symbols: set[str] = set()
    unresolved = False
    for attempt in state.fix_attempts:
        if attempt.success or _test_failure_class(attempt) != "assertion_failure":
            continue
        for edit in attempt.patch_edits:
            if edit.resolved_target_symbol:
                symbols.add(edit.resolved_target_symbol)
                continue
            if edit.node_target:
                symbols.add(edit.node_target)
                continue
            symbol = resolve_search_target_symbol_strict(
                state,
                edit.file_path,
                edit.search,
            )
            if symbol is None:
                unresolved = True
            else:
                symbols.add(symbol)
    return symbols, unresolved


def _attempt_failed_to_apply(attempt: Any) -> bool:
    """The attempt's patch/edits could not be applied at all (bad anchor)."""
    kind = getattr(attempt, "failure_kind", "") or getattr(attempt, "test_result", "")
    return kind == "patch_apply_failed"


def _unappliable_edit_keys(state: AgentState) -> set[str]:
    """(file, search) anchors from attempts whose patch could not be applied —
    re-emitting the same anchor is guaranteed to fail to apply again (the
    normalized fuzzy fallback already had its chance on the same file)."""
    keys: set[str] = set()
    for attempt in state.fix_attempts:
        if getattr(attempt, "success", False) or not _attempt_failed_to_apply(attempt):
            continue
        for edit in getattr(attempt, "patch_edits", []) or []:
            keys.add(_edit_key(edit))
    return keys


def _failed_edit_signatures(state: AgentState) -> list[frozenset[tuple[str, str]]]:
    """Full (anchor, replace) fingerprints of each failed attempt's edit set."""
    sigs: list[frozenset[tuple[str, str]]] = []
    for attempt in state.fix_attempts:
        if getattr(attempt, "success", False):
            continue
        edits = getattr(attempt, "patch_edits", []) or []
        if not edits:
            continue
        sigs.append(frozenset((_edit_key(e), e.replace) for e in edits))
    return sigs


def _dead_plan_reason(state: AgentState) -> str | None:
    """Why the freshly planned edits are guaranteed to repeat a known failure —
    so we should not waste an execute+test cycle on them — or None if fresh."""
    if not state.patch_edits:
        return None
    current_sig = frozenset((_edit_key(e), e.replace) for e in state.patch_edits)
    if current_sig in _failed_edit_signatures(state):
        return "identical_to_failed_patch"
    unappliable = _unappliable_edit_keys(state)
    if unappliable and all(_edit_key(e) in unappliable for e in state.patch_edits):
        return "reuses_unappliable_anchor"
    return None


def _is_final_attempt(state: AgentState) -> bool:
    """The retry budget is spent: this plan is the last one that can execute."""
    return state.retry_count >= state.max_retries


def _final_attempt_instructions() -> str:
    return (
        " This is the FINAL planning attempt (retry budget is spent). You MUST "
        "return concrete patch_edits now. Do NOT request more context on the final "
        "attempt."
    )


def _is_first_plan(state: AgentState) -> bool:
    return not state.fix_attempts and state.context_collection_count == 0


def _force_patch_instructions(state: AgentState) -> str:
    """Push the planner to commit a patch rather than spiral on
    collect_more_context — the dominant eval failure mode."""
    text = (
        " You MUST return at least one patch_edit. If you are unsure, make your "
        "best guess at the fix rather than deferring."
    )
    if _is_first_plan(state):
        text += (
            " This is your FIRST plan: do NOT recommend collect_more_context — "
            "produce a concrete patch from the files already provided."
        )
    return text


def _format_recalled_episodes(episodes: list[Any]) -> str:
    lines = [
        "RELATED PAST FIX EPISODES (semantic recall across repositories — learn "
        "from prior outcomes; adapt to the current code, do NOT copy verbatim):",
    ]
    for ep in episodes:
        tag = "✅ SUCCESS" if ep.success else "❌ FAILURE"
        role = (
            "working approach to reuse as a template"
            if ep.success
            else "approach that FAILED here — treat as a pitfall to avoid"
        )
        lines.append(
            f"\n{tag} — {ep.owner}/{ep.repo}: "
            f"{_truncate_prompt_text(ep.issue_title, 160)}"
        )
        keyframe = _truncate_prompt_text(ep.keyframe, 300)
        if keyframe:
            lines.append(f"  error signature: {keyframe}")
        patch = _truncate_prompt_text(ep.patch, 500)
        if patch:
            lines.append(f"  {role}:\n{patch}")
    return "\n".join(lines)


async def _semantic_recall_context(state: AgentState) -> str:
    """Best-effort cross-repo recall of similar past fixes. Never raises: if the
    episode store or embedding model is unavailable, planning proceeds without
    recall."""
    import sys

    try:
        from ..memory.error_episode_store import get_episode_store

        store = get_episode_store()
        if store is None:
            return ""
        episodes = await store.arecall(
            issue_title=state.issue_title,
            issue_body=state.issue_body,
            k=3,
            exclude_issue_url=state.issue_url,
        )
    except Exception as exc:
        # Observability: recall was enabled but failed — surface it instead of
        # hiding behind a silent best-effort (the memory system had zero
        # visibility, so we couldn't tell "off" from "on-but-broken").
        print(
            f"  [recall] failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return ""
    print(
        f"  [recall] {len(episodes)} episode(s) injected", file=sys.stderr, flush=True
    )
    if not episodes:
        return ""
    return _format_recalled_episodes(episodes)


def _context_pressure_instructions(state: AgentState) -> str:
    """Escalating pressure to commit a patch as context-collection rounds mount.

    The planner tends to treat each round as a fresh investigation — reinventing
    its hypotheses and asking for yet more context while never committing. This
    nudges it to converge: mild at first, then a hard "produce patch_edits now"
    on the final round before the collect_more_context cap forces a stop.
    """
    count = state.context_collection_count
    if count <= 0:
        return ""

    is_final = count >= MAX_CONTEXT_COLLECTION_ROUNDS
    lines = [
        "Context Budget Instructions:",
        f"- You have already collected repository context {count} time(s).",
        "- Build on the strongest existing repair evidence instead of re-deriving "
        "new investigations each round.",
    ]
    if is_final:
        lines.extend(
            [
                "- This is your FINAL context round. You MUST return concrete "
                "patch_edits now using the best supported repair.",
                "- Do NOT request more context again — the run will fail without a "
                "patch.",
            ]
        )
    else:
        lines.extend(
            [
                "- You now have substantial source context. Strongly prefer producing "
                "patch_edits this round.",
                "- Use kind='tool' only if one specific repository fact is still "
                "essential. Do not keep exploring generally.",
            ]
        )
    return "\n".join(lines)


def _hypothesis_continuity_context(state: AgentState) -> str:
    anchor = _patch_apply_hypothesis_anchor(state)
    latest_attempt = state.fix_attempts[-1] if state.fix_attempts else None
    has_patch_apply_failure = latest_attempt is not None and _is_patch_apply_failure(
        latest_attempt
    )
    if anchor is None and not has_patch_apply_failure:
        return ""

    lines = [
        "Hypothesis Continuity Instructions:",
        "- Tests did not run; the proposal failed before tests and only exact target context failed.",
        "- Treat the next action as a complete structured-edit decision, not broad root-cause exploration.",
        "- Repair the previous target paths and exact anchors before changing semantics.",
        "- Search blocks must be copied exactly from the current file context and large enough to match uniquely.",
    ]

    if anchor is not None:
        anchor_frame, hypothesis = anchor
        lines.extend(
            [
                "- Preserve the prior repair focus unless the apply error proves the "
                "target file or hunk context is impossible.",
                f"- Root-cause anchor: {_truncate_prompt_text(hypothesis.claim, 500)}",
            ]
        )
        if hypothesis.evidence:
            lines.append(
                "- Anchor evidence: "
                f"{_truncate_prompt_text('; '.join(hypothesis.evidence[:3]), 500)}"
            )
    elif has_patch_apply_failure:
        lines.append(
            "- No preserved hypothesis anchor is available; keep the root-cause "
            "search constrained to repair the malformed patch before expanding scope."
        )

    latest_reflect = next(
        (frame for frame in reversed(state.frame_history) if frame.stage == "reflect"),
        None,
    )
    if latest_reflect is not None:
        lines.append(
            "- Latest reflection summary: "
            f"{_truncate_prompt_text(latest_reflect.summary, 500)}"
        )
    if has_patch_apply_failure:
        lines.append(
            "- Previous patch apply error: "
            f"{_truncate_prompt_text(latest_attempt.error_log or latest_attempt.test_result or 'patch_apply_failed', 500)}"
        )
    return "\n".join(lines)


def _preserve_patch_apply_hypothesis_anchor(
    state: AgentState,
    frame: DecisionFrame,
) -> dict[str, Any] | None:
    anchor = _patch_apply_hypothesis_anchor(state)
    if anchor is None:
        return None

    anchor_frame, hypothesis = anchor
    selected_before = frame.selected_hypothesis_id or ""
    has_anchor_hypothesis = any(
        candidate.id == hypothesis.id for candidate in frame.hypotheses
    )
    if selected_before == hypothesis.id and has_anchor_hypothesis:
        return None

    hypothesis_copy = hypothesis.model_copy(deep=True)
    for idx, candidate in enumerate(frame.hypotheses):
        if candidate.id == hypothesis_copy.id:
            frame.hypotheses[idx] = hypothesis_copy
            break
    else:
        frame.hypotheses.insert(0, hypothesis_copy)

    frame.selected_hypothesis_id = hypothesis_copy.id
    for evidence in hypothesis_copy.evidence:
        if evidence not in frame.evidence:
            frame.evidence.append(evidence)

    return {
        "warning_type": "hypothesis_consistency",
        "node": "plan_fix",
        "reason": "preserved_selected_hypothesis_after_patch_apply_failure",
        "previous_frame_id": anchor_frame.frame_id,
        "previous_selected_hypothesis_id": hypothesis_copy.id,
        "llm_selected_hypothesis_id": selected_before,
    }


def build_plan_user_prompt(
    state: AgentState,
    *,
    recall_context: str = "",
) -> str:
    """Build the provider-neutral prompt with one bounded summary section."""
    summary = sanitize_model_context(
        sanitize_outcome_summary(state, state.attempt_outcome_summary),
        MAX_OUTCOME_SUMMARY_CHARS,
        denied_literals=MODEL_CONTEXT_DENIED_LITERALS,
    )
    include_legacy_attempt_context = not summary
    previous_failure_entries: list[str] = []
    if include_legacy_attempt_context:
        for idx, attempt in enumerate(state.fix_attempts):
            safe_result = sanitize_model_context(
                attempt.test_result,
                PLAN_FAILURE_RESULT_LIMIT,
                denied_literals=MODEL_CONTEXT_DENIED_LITERALS,
            )
            safe_error = sanitize_model_context(
                attempt.error_log,
                PLAN_FAILURE_LOG_LIMIT,
                denied_literals=MODEL_CONTEXT_DENIED_LITERALS,
            )
            previous_failure_entries.append(
                f"Attempt {idx + 1}: {safe_result}\n{safe_error}"
            )
    previous_failures = sanitize_model_context(
        "\n\n".join(previous_failure_entries),
        PLAN_PREVIOUS_FAILURES_TOTAL_LIMIT,
        denied_literals=MODEL_CONTEXT_DENIED_LITERALS,
    )
    reflection_context = ""
    if include_legacy_attempt_context and state.reflection_notes:
        safe_reflection = sanitize_model_context(
            state.reflection_notes,
            PLAN_FAILURE_LOG_LIMIT,
            denied_literals=MODEL_CONTEXT_DENIED_LITERALS,
        )
        if safe_reflection:
            reflection_context = f"\n\nREFLECTION ANALYSIS:\n{safe_reflection}"
    hypothesis_continuity_context = ""
    continuity_context = (
        _hypothesis_continuity_context(state) if include_legacy_attempt_context else ""
    )
    if continuity_context:
        hypothesis_continuity_context = f"\n\n{continuity_context}"
    human_context = ""
    resumed_answer_context = _human_answer_context(state)
    if resumed_answer_context:
        safe_answer = sanitize_model_context(
            resumed_answer_context,
            PLAN_FAILURE_LOG_LIMIT,
            denied_literals=MODEL_CONTEXT_DENIED_LITERALS,
        )
        if safe_answer:
            human_context = f"\n\n{safe_answer}"
    context_pressure_context = ""
    pressure = _context_pressure_instructions(state)
    if pressure:
        context_pressure_context = f"\n\n{pressure}"
    diversity_context = ""
    prior_failed_edits = (
        _prior_failed_edits_context(state) if include_legacy_attempt_context else ""
    )
    if prior_failed_edits:
        diversity_context = f"\n\n{prior_failed_edits}"
    correction_context = ""
    if state.repair_correction_context:
        safe_correction = sanitize_model_context(
            state.repair_correction_context,
            8_000,
            denied_literals=MODEL_CONTEXT_DENIED_LITERALS,
        )
        if safe_correction:
            correction_context = (
                "\n\nCORRECTION FOR THE NEXT PATCH_EDITS RESPONSE:\n"
                f"{safe_correction}"
            )
    files_terms = _issue_search_terms(state.issue_title, state.issue_body)
    file_limit, max_files = _budget_scaled_file_limits(state)
    files_context = "\n\n".join(
        "FILE: "
        f"{sanitize_model_context(file.path, 500, denied_literals=MODEL_CONTEXT_DENIED_LITERALS)}\n"
        f"RELEVANCE: {file.relevance_score} - "
        f"{sanitize_model_context(file.reason, 500, denied_literals=MODEL_CONTEXT_DENIED_LITERALS)}\n"
        "CONTENT:\n"
        f"{sanitize_model_context(relevance_window(file.content, files_terms, file_limit), file_limit, denied_literals=MODEL_CONTEXT_DENIED_LITERALS)}"
        for file in state.relevant_files[:max_files]
    )
    completed_attempts_context = ""
    if summary:
        completed_attempts_context = f"\n\n{OUTCOME_SUMMARY_SECTION}\n{summary}"
    legacy_attempt_context = (
        f"\n\nPrevious failures:\n{previous_failures}"
        if include_legacy_attempt_context
        else ""
    )
    user = (
        "Issue URL: "
        f"{sanitize_model_context(state.issue_url, 2_048, denied_literals=MODEL_CONTEXT_DENIED_LITERALS)}\n"
        "Title: "
        f"{sanitize_model_context(state.issue_title, 500, denied_literals=MODEL_CONTEXT_DENIED_LITERALS)}\n\nBody:\n"
        f"{_truncate_prompt_text(state.issue_body, PLAN_ISSUE_BODY_LIMIT)}\n\n"
        f"Relevant files:\n{files_context}"
        f"{sanitize_model_context(recall_context, 4_000, denied_literals=MODEL_CONTEXT_DENIED_LITERALS) if include_legacy_attempt_context else ''}"
        f"{legacy_attempt_context}"
        f"{reflection_context}"
        f"{hypothesis_continuity_context}"
        f"{context_pressure_context}"
        f"{diversity_context}"
        f"{correction_context}"
        f"{human_context}"
        f"\n\n{_force_patch_instructions(state).strip()}"
        f"{_final_attempt_instructions() if _is_final_attempt(state) else ''}"
    )
    user = user.replace(OUTCOME_SUMMARY_SECTION, "")
    return f"{user}{completed_attempts_context}"


def _record_plan_frame(
    state: AgentState,
    frame: DecisionFrame,
    *,
    files: Sequence[str] = (),
    has_explicit_frame: bool = True,
) -> DecisionFrame:
    frame = frame.model_copy(deep=True)
    frame.parent_frame_id = (
        state.decision_frame.frame_id if state.decision_frame else None
    )
    if not frame.trace_notes:
        frame.trace_notes = json.dumps(
            {"files": list(files)},
            ensure_ascii=False,
            sort_keys=True,
        )
    hypothesis_warning = _preserve_patch_apply_hypothesis_anchor(state, frame)
    _record_decision_frame(state, frame)
    if hypothesis_warning is not None:
        hypothesis_warning["frame_id"] = frame.frame_id
        state.decision_warnings.append(hypothesis_warning)
    if not has_explicit_frame:
        _record_frame_health_warning(
            state,
            node="plan_fix",
            expected_stage="plan",
            frame=frame,
            reason="missing_explicit_decision_frame",
        )
    return frame


def _install_plan_decision(
    state: AgentState,
    decision: PlanDecision,
    *,
    has_explicit_frame: bool = True,
) -> DecisionFrame:
    import sys

    state.fix_plan = decision.plan
    state.test_command = decision.test_command
    print(
        "  [plan] Plan received "
        f"({len(state.fix_plan)} chars, patch={len(state.patch_content)} chars, "
        f"edits={len(state.patch_edits)})",
        file=sys.stderr,
        flush=True,
    )
    _remember(state, "assistant", state.fix_plan[:2_000])
    return _record_plan_frame(
        state,
        decision.decision_frame,
        files=decision.files,
        has_explicit_frame=has_explicit_frame,
    )


def _deterministic_plan_frame(
    summary: str,
    *,
    action: str,
) -> DecisionFrame:
    return DecisionFrame(
        stage="plan",
        summary=sanitize_model_context(
            summary,
            500,
            denied_literals=MODEL_CONTEXT_DENIED_LITERALS,
        ),
        recommended_action=action,
        risk="unknown",
        confidence=0.0,
    )


def _record_plan_invocation(
    state: AgentState,
    *,
    provider: str,
    model: str,
    elapsed_seconds: float,
    prompt_tokens: int,
    response_tokens: int,
    status: str,
    error: BaseException | None = None,
) -> None:
    record_model_invocation(
        state,
        model=model,
        provider=provider,
        node="plan_fix",
        elapsed_seconds=elapsed_seconds,
        input_tokens=prompt_tokens,
        output_tokens=response_tokens,
        status=status,
        error=error,
    )
    state.token_usage += prompt_tokens + response_tokens
    _record_node_diagnostic(
        state,
        node="plan_fix",
        event="llm_call",
        status="error" if error is not None else "success",
        elapsed_seconds=elapsed_seconds,
        error_type=type(error).__name__ if error is not None else None,
        policy_reason=(
            immediate_model_policy_reason(error) if error is not None else None
        ),
        prompt_tokens_estimate=prompt_tokens,
        response_tokens_estimate=response_tokens,
    )


def _route_plan_environment_failure(
    state: AgentState,
    *,
    reason: str,
    event: str,
    issues: Sequence[PatchAuthorizationIssue] = (),
) -> AgentState:
    retire_patch_authorization(state)
    state.repair_correction_context = ""
    state.failure_reason = sanitize_model_context(
        reason,
        500,
        denied_literals=MODEL_CONTEXT_DENIED_LITERALS,
    )
    state.current_phase = Phase.FAILURE
    frame = _deterministic_plan_frame(
        "Planning stopped because repository or provider infrastructure is unavailable.",
        action="stop",
    )
    _record_plan_frame(
        state,
        frame,
        has_explicit_frame=False,
    )
    _record_node_diagnostic(
        state,
        node="plan_fix",
        event=event,
        status="error",
        elapsed_seconds=0.0,
        issue_codes=[issue.code for issue in issues] or None,
        failure_class="environment",
    )
    return state


def _record_model_correctable_plan_failure(
    state: AgentState,
    *,
    provider: str,
    model: str,
    reason: str,
    issues: Sequence[PatchAuthorizationIssue] = (),
    frame: DecisionFrame | None = None,
    immediate_reason: str = "",
    clear_correction: bool = False,
) -> AgentState:
    retire_patch_authorization(state)
    if clear_correction:
        state.repair_correction_context = ""
    elif issues:
        state.repair_correction_context = render_patch_correction(list(issues))
    failure_reason = issues[0].code if issues else reason
    decision = record_failed_repair_round(
        state,
        round_id=state.current_repair_round_id,
        provider=provider,
        model=model,
        failure_reason=failure_reason,
        retry_phase=Phase.PLAN,
        immediate_reason=immediate_reason,
    )
    action = "plan" if decision.retry_allowed else "stop"
    next_frame = (
        frame.model_copy(deep=True)
        if frame is not None
        else _deterministic_plan_frame(
            "The previous plan decision was not executable.",
            action=action,
        )
    )
    next_frame.recommended_action = action
    _record_plan_frame(
        state,
        next_frame,
        has_explicit_frame=True,
    )
    return state


async def plan_fix(state: AgentState | dict[str, Any]) -> AgentState:
    """Run one provider-neutral full PLAN repair transaction."""
    import sys

    state = _as_state(state)
    if _is_budget_exceeded(state):
        return _route_plan_environment_failure(
            state,
            reason="Token budget exceeded before planning.",
            event="token_budget",
        )

    try:
        begin_repair_round(state)
    except ValueError:
        return _route_plan_environment_failure(
            state,
            reason="Repair transaction budget is exhausted.",
            event="repair_round",
        )

    recall = await _semantic_recall_context(state)
    recall_context = f"\n\n{recall}" if recall else ""

    # A new full decision can never inherit executable authority from an older
    # proposal. The correction suffix intentionally survives this retirement.
    retire_patch_authorization(state)

    _record_node_diagnostic(
        state,
        node="plan_fix",
        event="prompt_built",
        status="success",
        elapsed_seconds=0.0,
        prompt_tokens_estimate=_estimate_tokens(
            PLAN_SYSTEM,
            build_plan_user_prompt(state, recall_context=recall_context),
        ),
        relevant_file_count=len(state.relevant_files[:PLAN_MAX_FILES]),
        issue_body_chars=len(
            _truncate_prompt_text(state.issue_body, PLAN_ISSUE_BODY_LIMIT)
        ),
        previous_failure_count=len(state.fix_attempts),
        has_reflection_context=bool(state.reflection_notes),
        has_hypothesis_continuity_context=bool(_hypothesis_continuity_context(state)),
        context_collection_count=state.context_collection_count,
        has_context_pressure=bool(_context_pressure_instructions(state)),
    )

    calls_this_round = 0
    evidence_ids: tuple[str, ...] = ()
    while True:
        if _is_budget_exceeded(state):
            return _route_plan_environment_failure(
                state,
                reason="Token budget exceeded during planning.",
                event="token_budget",
            )

        reserve_reached = (
            state.active_provider == "primary"
            and state.token_usage >= primary_budget_limit(state)
        )
        apply_escalation(state, should_escalate(state))
        if (
            reserve_reached
            and state.active_provider == "primary"
            and not escalation_is_configured()
        ):
            return _route_plan_environment_failure(
                state,
                reason=(
                    "Primary token reserve reached while the escalation provider "
                    "is unavailable."
                ),
                event="provider_unavailable",
            )

        bind_repair_round_author(state)
        invoked_provider = state.current_repair_provider
        invoked_model = state.current_repair_model
        if invoked_provider is None or not invoked_model:
            return _route_plan_environment_failure(
                state,
                reason="Repair model attribution is unavailable.",
                event="state_integrity",
            )

        user = build_plan_user_prompt(state, recall_context=recall_context)
        if evidence_ids:
            user = prompt_with_new_evidence(
                user,
                state,
                evidence_ids,
                denied_literals=MODEL_CONTEXT_DENIED_LITERALS,
                max_evidence_chars=PLAN_NEW_EVIDENCE_CONTEXT_LIMIT,
            )
        prompt_tokens = _estimate_tokens(PLAN_SYSTEM, user)
        response_text = ""
        t0 = time.monotonic()
        try:
            print(
                "  [plan] Calling LLM for fix plan...",
                file=sys.stderr,
                flush=True,
            )
            raw_response = await llm_call(
                PLAN_SYSTEM,
                user,
                model=invoked_model,
                provider=invoked_provider,
            )
            if isinstance(raw_response, str):
                response_text = raw_response
            else:
                response_text = json.dumps(
                    raw_response,
                    ensure_ascii=False,
                    sort_keys=True,
                )
        except (CancellationDrainError, asyncio.CancelledError):
            raise
        except Exception as exc:
            elapsed = time.monotonic() - t0
            response_tokens = _estimate_tokens(response_text)
            retryable_transport = is_retryable_llm_error(exc)
            immediate_reason = immediate_model_policy_reason(exc)
            _record_plan_invocation(
                state,
                provider=invoked_provider,
                model=invoked_model,
                elapsed_seconds=elapsed,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
                status=(
                    "error"
                    if retryable_transport
                    else "invalid_response"
                    if immediate_reason
                    else "error"
                ),
                error=exc,
            )
            if retryable_transport:
                if invoked_provider == "primary":
                    previous_provider = state.active_provider
                    apply_escalation(
                        state,
                        should_escalate(
                            state,
                            immediate_reason=(
                                "primary_gateway_unavailable_after_retries"
                            ),
                        ),
                    )
                    if (
                        previous_provider == "primary"
                        and state.active_provider == "escalation"
                    ):
                        continue
                return _route_plan_environment_failure(
                    state,
                    reason="The active repair provider is unavailable.",
                    event="provider_unavailable",
                )
            if immediate_reason:
                issue = PatchAuthorizationIssue(
                    code="invalid_structured_response",
                    message="Return one valid bounded PLAN response.",
                    failure_class="model_correctable",
                )
                return _record_model_correctable_plan_failure(
                    state,
                    provider=invoked_provider,
                    model=invoked_model,
                    reason="invalid_structured_response",
                    issues=(issue,),
                    immediate_reason=immediate_reason,
                )
            if isinstance(exc, ReasoningStop):
                state.repair_correction_context = ""
                retire_patch_authorization(state)
                state.failure_reason = exc.code
                state.current_phase = Phase.FAILURE
                _record_plan_frame(
                    state,
                    _deterministic_plan_frame(
                        "Planning stopped by the bounded tool policy.",
                        action="stop",
                    ),
                    has_explicit_frame=False,
                )
                return state
            return _route_plan_environment_failure(
                state,
                reason="The planning model call failed.",
                event="provider_error",
            )

        response = _extract_json_object(raw_response)
        elapsed = time.monotonic() - t0
        response_tokens = _estimate_tokens(response_text)
        if not response:
            invalid = ValueError("Model returned an empty structured response")
            _record_plan_invocation(
                state,
                provider=invoked_provider,
                model=invoked_model,
                elapsed_seconds=elapsed,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
                status="invalid_response",
                error=invalid,
            )
            issue = PatchAuthorizationIssue(
                code="empty_structured_response",
                message="Return one valid bounded PLAN response.",
                failure_class="model_correctable",
            )
            return _record_model_correctable_plan_failure(
                state,
                provider=invoked_provider,
                model=invoked_model,
                reason="empty_structured_response",
                issues=(issue,),
                immediate_reason=immediate_model_policy_reason(invalid),
            )

        response_kind = "plan"
        if "patch_edits" not in response:
            raw_kind = str(response.get("kind") or "").strip().lower()
            is_control_variant = raw_kind in {"tool", "stop"}
        else:
            is_control_variant = False
        if is_control_variant:
            try:
                response_kind = validate_reasoning_response(
                    response,
                    outcome_kind="plan",
                )
            except (TypeError, ValueError) as exc:
                _record_plan_invocation(
                    state,
                    provider=invoked_provider,
                    model=invoked_model,
                    elapsed_seconds=elapsed,
                    prompt_tokens=prompt_tokens,
                    response_tokens=response_tokens,
                    status="invalid_response",
                    error=exc,
                )
                issue = PatchAuthorizationIssue(
                    code="invalid_control_response",
                    message="Return exactly one valid PLAN response variant.",
                    failure_class="model_correctable",
                )
                return _record_model_correctable_plan_failure(
                    state,
                    provider=invoked_provider,
                    model=invoked_model,
                    reason="invalid_control_response",
                    issues=(issue,),
                    immediate_reason=immediate_model_policy_reason(exc),
                )

        _record_plan_invocation(
            state,
            provider=invoked_provider,
            model=invoked_model,
            elapsed_seconds=elapsed,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            status="ok",
        )

        if response_kind == "tool":
            try:
                tool_step = await route_reasoning_tool(
                    state,
                    response,
                    node="plan_fix",
                    calls_this_round=calls_this_round,
                    router=route_tool_intent,
                    allow_provider_local_no_progress_stop=False,
                )
            except (CancellationDrainError, asyncio.CancelledError):
                raise
            except ReasoningStop as exc:
                state.repair_correction_context = ""
                retire_patch_authorization(state)
                state.failure_reason = exc.code
                state.current_phase = Phase.FAILURE
                _record_plan_frame(
                    state,
                    _deterministic_plan_frame(
                        "Planning stopped by the bounded tool policy.",
                        action="stop",
                    ),
                    has_explicit_frame=False,
                )
                return state
            except Exception:
                return _route_plan_environment_failure(
                    state,
                    reason="The approved planning tool failed.",
                    event="tool_error",
                )

            if tool_step.stop_reason:
                state.repair_correction_context = ""
                retire_patch_authorization(state)
                state.failure_reason = tool_step.stop_reason
                state.current_phase = Phase.FAILURE
                _record_plan_frame(
                    state,
                    _deterministic_plan_frame(
                        "Planning stopped by the bounded tool policy.",
                        action="stop",
                    ),
                    has_explicit_frame=False,
                )
                return state
            calls_this_round += 1
            evidence_ids = tool_step.evidence_ids
            continue

        if response_kind == "stop":
            state.repair_correction_context = ""
            return _record_model_correctable_plan_failure(
                state,
                provider=invoked_provider,
                model=invoked_model,
                reason="model_stop",
                clear_correction=True,
            )

        previous_correction = state.repair_correction_context
        try:
            outcome = authorize_plan_patch(state, response)
        except (CancellationDrainError, asyncio.CancelledError):
            raise
        except Exception:
            return _route_plan_environment_failure(
                state,
                reason="Patch authorization failed internally.",
                event="patch_authorization",
            )

        if outcome.status == "environment":
            return _route_plan_environment_failure(
                state,
                reason="PatchGate could not validate the exact repository state.",
                event="patch_gate",
                issues=outcome.issues,
            )
        if outcome.status == "model_correctable":
            return _record_model_correctable_plan_failure(
                state,
                provider=invoked_provider,
                model=invoked_model,
                reason="patch_authorization_rejected",
                issues=outcome.issues,
            )

        decision = outcome.decision
        if decision is None:
            return _route_plan_environment_failure(
                state,
                reason="Patch authorization returned no decision.",
                event="state_integrity",
            )
        frame = _install_plan_decision(state, decision)
        action = frame.recommended_action

        if outcome.status == "not_requested":
            if action in {"collect_more_context", "ask_user"}:
                state.repair_correction_context = previous_correction
            if action == "collect_more_context":
                if (
                    _is_final_attempt(state)
                    or state.context_collection_count >= MAX_CONTEXT_COLLECTION_ROUNDS
                ):
                    state.repair_correction_context = ""
                    frame.recommended_action = "stop"
                    state.current_phase = Phase.FAILURE
                    state.failure_reason = (
                        "Context collection made no progress within its bounded cap."
                    )
                else:
                    state.context_collection_count += 1
                    state.current_phase = Phase.LOCATE
                return state
            if action == "ask_user":
                state.current_phase = Phase.WAITING_FOR_USER
                return state
            state.repair_correction_context = ""
            return _record_model_correctable_plan_failure(
                state,
                provider=invoked_provider,
                model=invoked_model,
                reason="plan_without_executable_patch",
                frame=frame,
                clear_correction=True,
            )

        if outcome.status != "accepted":
            return _route_plan_environment_failure(
                state,
                reason="Patch authorization returned an unknown status.",
                event="state_integrity",
            )

        # PatchGate has installed and fingerprint-bound the only executable
        # patch. PLAN consumes its canonical values and never reparses raw edits.
        state.repair_correction_context = ""
        state.search_correction_context = ""
        dead_reason = _dead_plan_reason(state)
        if dead_reason is not None:
            record_no_progress(
                state,
                kind="repeated_edit",
                node="plan_fix",
                fingerprint=dead_reason,
            )
            state.decision_warnings.append(
                {
                    "node": "plan_fix",
                    "warning": "blocked_dead_patch",
                    "detail": (
                        "The authorized proposal repeats a previously failed "
                        f"transaction ({dead_reason})."
                    ),
                    "frame_id": frame.frame_id,
                }
            )
            issue = PatchAuthorizationIssue(
                code="repeated_failed_patch",
                message="Choose a different complete repair transaction.",
                failure_class="model_correctable",
            )
            return _record_model_correctable_plan_failure(
                state,
                provider=invoked_provider,
                model=invoked_model,
                reason=dead_reason,
                issues=(issue,),
                frame=frame,
            )

        try:
            prior_assertion_symbols, unresolved_assertion_target = (
                _prior_assertion_symbols(state)
            )
            current_assertion_symbols: set[str] = set()
            unresolved_current_target = False
            for edit in state.patch_edits:
                symbol = edit.resolved_target_symbol or edit.node_target
                if not symbol and edit.search:
                    symbol = resolve_search_target_symbol_strict(
                        state,
                        edit.file_path,
                        edit.search,
                    )
                if symbol:
                    current_assertion_symbols.add(symbol)
                else:
                    unresolved_current_target = True
        except (OSError, RepairContextError):
            return _route_plan_environment_failure(
                state,
                reason=(
                    "PatchGate authorization could not be checked against the "
                    "exact repository state."
                ),
                event="assertion_target_resolution",
            )
        assertion_target_not_diversified = state.assertion_diversity_required and (
            unresolved_assertion_target
            or unresolved_current_target
            or bool(current_assertion_symbols & prior_assertion_symbols)
        )
        if assertion_target_not_diversified:
            record_no_progress(
                state,
                kind="repeated_edit",
                node="plan_fix",
                fingerprint="assertion_target_not_diversified",
            )
            state.decision_warnings.append(
                {
                    "node": "plan_fix",
                    "warning": "assertion_target_not_diversified",
                    "detail": (
                        "The authorized repair reused a prior failed assertion "
                        "target; a different target is required."
                    ),
                    "frame_id": frame.frame_id,
                }
            )
            issue = PatchAuthorizationIssue(
                code="assertion_target_not_diversified",
                message="Choose a different complete assertion repair target.",
                failure_class="model_correctable",
            )
            return _record_model_correctable_plan_failure(
                state,
                provider=invoked_provider,
                model=invoked_model,
                reason="assertion_target_not_diversified",
                issues=(issue,),
                frame=frame,
            )

        record_progress(state)
        state.assertion_diversity_required = False
        frame.recommended_action = "execute"
        if state.patch_only:
            state.current_phase = Phase.DONE
            state.decision_route_checked_frame_id = frame.frame_id
        else:
            state.current_phase = Phase.EXECUTE
        return state
