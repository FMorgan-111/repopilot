"""PLAN phase: Ask the LLM for a concrete patch-oriented plan."""

from __future__ import annotations

import ast
import json
import os
import time
from collections.abc import Sequence
from typing import Any

from ..escalation import (
    build_escalation_packet,
    immediate_model_policy_reason,
    record_model_invocation,
    render_escalation_packet,
)
from ..llm import llm_call
from ..model_policy import (
    apply_escalation,
    record_no_progress,
    record_progress,
    should_escalate,
)
from ..outcome_summary import (
    OUTCOME_SUMMARY_SECTION,
    sanitize_outcome_summary,
)
from ..patch_gate import revalidate_approved_patch, validate_patch_batch
from ..patch_match import closest_region, locate_search_block
from ..repair_flow import (
    generate_opus_repair,
    request_verified_edit_correction,
)
from ..reasoning_loop import (
    ReasoningStop,
    prompt_with_new_evidence,
    record_opus_no_progress,
    route_reasoning_tool,
    validate_reasoning_response,
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
from ..tool_router import route_tool_intent

PLAN_ISSUE_BODY_LIMIT = 2500
# The planner needs to see real function bodies, not just import headers. At
# 1200 chars it only saw ~3% of a file (the imports) and kept asking for more
# context forever. 6000 chars surfaces the logic the fix actually touches.
PLAN_FILE_CONTENT_LIMIT = 6000
PLAN_MAX_FILES = 3
PLAN_FAILURE_LOG_LIMIT = 1000

ESCALATED_PLAN_SYSTEM = (
    "You are RepoPilot's escalated planning node. The user message is the complete "
    "allowlisted EscalationPacket; do not request hidden conversation or evaluator "
    "data. Return ONLY JSON with keys: plan, patch_edits, patch, files, "
    "test_command, and decision_frame. Each patch_edits item uses file, exact "
    "search, replace, and optional replace_all or node_target. Leave patch empty "
    "when patch_edits can express the edit. decision_frame must use stage='plan' "
    "and include summary, hypotheses, selected_hypothesis_id, evidence, next_checks, "
    "recommended_action, risk, and confidence. If one approved local evidence call "
    "is essential, include optional tool_intent with action, args, reason, and "
    "expected_evidence; deterministic policy decides whether it runs."
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
    value = value.strip()
    if len(value) <= limit:
        return value
    return f"{value[:limit].rstrip()}..."


def _relevance_window(content: str, terms: Sequence[str], limit: int) -> str:
    """Return up to ~`limit` chars of `content` centered on the line most
    relevant to the issue terms, instead of blindly taking the head.

    The fix site is often far below a file's imports; head-truncation hides it
    and the planner then hallucinates a search block for code it never saw.
    Centering on the best term match surfaces the actual lines to copy — at the
    same token cost. Falls back to the head when nothing matches."""
    if len(content) <= limit:
        return content
    lines = content.split("\n")
    lowered = [line.lower() for line in lines]
    lowered_terms = [t.lower() for t in terms if t.strip()]

    best_idx, best_score = -1, 0
    for i, line in enumerate(lowered):
        score = sum(1 for t in lowered_terms if t in line)
        if score > best_score:
            best_score, best_idx = score, i
    if best_idx < 0:
        return f"{content[:limit].rstrip()}..."  # no match → old head behavior

    lo = hi = best_idx
    size = len(lines[best_idx])
    while True:
        moved = False
        if lo > 0 and size + len(lines[lo - 1]) + 1 < limit:
            lo -= 1
            size += len(lines[lo]) + 1
            moved = True
        if hi < len(lines) - 1 and size + len(lines[hi + 1]) + 1 < limit:
            hi += 1
            size += len(lines[hi]) + 1
            moved = True
        if not moved:
            break

    window = "\n".join(lines[lo : hi + 1])
    if lo > 0:
        window = f"... [{lo} lines above truncated] ...\n{window}"
    if hi < len(lines) - 1:
        window = f"{window}\n... [{len(lines) - 1 - hi} lines below truncated] ..."
    return window


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


def _failed_edit_keys(state: AgentState) -> set[str]:
    """Signatures of every search/replace edit tried in a failed attempt."""
    keys: set[str] = set()
    for attempt in state.fix_attempts:
        if getattr(attempt, "success", False):
            continue
        for edit in getattr(attempt, "patch_edits", []) or []:
            keys.add(_edit_key(edit))
    return keys


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
    return "\n".join(lines)


def _planned_edits_repeat_failure(state: AgentState) -> bool:
    """True when the freshly planned edits merely repeat edits that already
    failed (no diversification happened)."""
    if not state.patch_edits:
        return False
    failed_keys = _failed_edit_keys(state)
    if not failed_keys:
        return False
    return all(
        _edit_key(edit) in failed_keys
        for edit in state.patch_edits
    )


# How many times we let the planner re-emit a known-dead patch before failing
# fast instead of burning the whole retry budget on guaranteed re-failures.
MAX_REPEATED_PATCH_BLOCKS = 1


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
        sigs.append(
            frozenset(
                (_edit_key(e), e.replace) for e in edits
            )
        )
    return sigs


def _dead_plan_reason(state: AgentState) -> str | None:
    """Why the freshly planned edits are guaranteed to repeat a known failure —
    so we should not waste an execute+test cycle on them — or None if fresh."""
    if not state.patch_edits:
        return None
    current_sig = frozenset(
        (_edit_key(e), e.replace)
        for e in state.patch_edits
    )
    if current_sig in _failed_edit_signatures(state):
        return "identical_to_failed_patch"
    unappliable = _unappliable_edit_keys(state)
    if unappliable and all(
        _edit_key(e) in unappliable
        for e in state.patch_edits
    ):
        return "reuses_unappliable_anchor"
    return None


# How many times we let the planner emit a search block that does not exist in
# the target file before failing fast (each round feeds the real lines back).
MAX_SEARCH_CORRECTIONS = 2


def _relevant_file_content(state: AgentState, file_path: str) -> str | None:
    for file in state.relevant_files:
        if file.path == file_path:
            return file.content
    return None


def _unlocatable_edits(state: AgentState) -> list[Any]:
    """Planned edits whose search block does not exist in the target file's real
    content (a hallucinated anchor). Edits whose file we don't hold are skipped
    — the executor's fuzzy apply is the backstop there."""
    missing: list[Any] = []
    for edit in state.patch_edits:
        if getattr(edit, "node_target", ""):
            continue  # node-anchored: validated by AST at EXECUTE, not by search
        content = _relevant_file_content(state, edit.file_path)
        if content is None:
            continue  # cannot validate here; let EXECUTE handle it
        if not locate_search_block(content, edit.search):
            missing.append(edit)
    return missing


def _has_valid_exact_patch_gate_approval(state: AgentState) -> bool:
    if state.tool_patch_approval is None or not state.patch_edits:
        return False
    if not all(edit.exact_only for edit in state.patch_edits):
        return False
    try:
        revalidate_approved_patch(state)
    except (OSError, ValueError):
        return False
    return True


def _clear_patch_authorization(
    state: AgentState,
    *,
    clear_active_plan: bool = True,
) -> None:
    """Atomically retire an approval whenever its bound edits are discarded."""
    state.patch_edits = []
    state.patch_content = ""
    state.tool_patch_approval = None
    state.generated_test_approvals = []
    if clear_active_plan:
        state.active_repair_plan = None


def _build_search_correction(state: AgentState, missing: list[Any]) -> str:
    """Feed the planner the ACTUAL nearby file lines so it can copy a real
    search block instead of re-hallucinating one (bypasses head-truncation)."""
    lines = [
        "YOUR PREVIOUS SEARCH BLOCK(S) DO NOT EXIST IN THE FILE — they can never "
        "apply. Copy your next search block VERBATIM from these ACTUAL file "
        "lines (do not paraphrase; keep exact indentation):",
    ]
    for edit in missing:
        content = _relevant_file_content(state, edit.file_path) or ""
        region = closest_region(content, edit.search)
        lines.append(
            f"\nfile: {edit.file_path}\n"
            f"your (nonexistent) search was:\n{_truncate_prompt_text(edit.search, 300)}\n"
            f"ACTUAL lines nearest your intent:\n{region}"
        )
    return "\n".join(lines)


def _is_final_attempt(state: AgentState) -> bool:
    """The retry budget is spent: this plan is the last one that can execute."""
    return state.retry_count >= state.max_retries


def _enclosing_node_names(content: str, search: str) -> list[str]:
    """Dotted names of the def/method/class that CONTAIN the failed search
    block's location — candidates the planner can re-target via node_target."""
    if not content or not search:
        return []
    anchor = max((ln for ln in search.split("\n") if ln.strip()), key=len, default="")
    if not anchor.strip():
        return []
    try:
        idx = content.index(anchor.strip())
    except ValueError:
        return []
    target_line = content.count("\n", 0, idx) + 1  # 1-based
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    names: list[str] = []

    def walk(node: ast.AST, stack: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                child_stack = stack + [child.name]
                start = child.lineno
                end = child.end_lineno or start
                if start <= target_line <= end:
                    names.append(".".join(child_stack))
                walk(child, child_stack)
            else:
                walk(child, stack)

    walk(tree, [])
    return names


def _final_attempt_instructions() -> str:
    return (
        " This is the FINAL planning attempt (retry budget is spent). You MUST "
        "return concrete patch_edits now and set recommended_action='execute'. "
        "Do NOT request more context: 'collect_more_context' is not allowed on "
        "the final attempt."
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
        print(f"  [recall] failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return ""
    print(f"  [recall] {len(episodes)} episode(s) injected", file=sys.stderr, flush=True)
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
        "- Build on your previous strongest hypothesis instead of re-deriving new "
        "ones each round; reuse the same hypothesis ids when the claim is unchanged.",
    ]
    if is_final:
        lines.extend(
            [
                "- This is your FINAL context round. You MUST set "
                "recommended_action='execute' and return concrete patch_edits now, "
                "using the best hypothesis supported by the files you already have.",
                "- Do NOT recommend collect_more_context again — it will be rejected "
                "and the run will fail without a patch.",
            ]
        )
    else:
        lines.extend(
            [
                "- You now have substantial source context. Strongly prefer producing "
                "patch_edits this round.",
                "- Only recommend collect_more_context if a SPECIFIC file you have not "
                "yet seen is essential — name it exactly in next_checks. Do not ask for "
                "more context to keep exploring generally.",
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
        "- Tests did not run; the patch failed before tests ran and only patch formatting/path/hunk context failed.",
        "- If the error says patch preflight check failed, git rejected the old diff during `git apply --check`; switch to exact search/replace edits before consuming semantic retries.",
        "- Treat the next LLM action as exact patch_edits repair, not root-cause exploration.",
        "- Repair the previous patch's file paths and search blocks before changing semantics.",
        "- Prefer patch_edits over unified diffs. Each edit must include file, search, replace, and optional replace_all.",
        "- Search blocks must be copied exactly from the current file context and large enough to match uniquely.",
    ]

    if anchor is not None:
        anchor_frame, hypothesis = anchor
        lines.extend(
            [
                "- Preserve "
                f"selected_hypothesis_id='{hypothesis.id}' from plan frame "
                f"{anchor_frame.frame_id or '(unrecorded)'} unless the apply error "
                "proves the target file or hunk context is impossible.",
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


def _normalize_string_list(value: Any) -> list[str]:
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
                raise TypeError("Expected a sequence of strings")
            normalized.append(item)
        return normalized
    raise TypeError("Expected None, a string, or a sequence of strings")


def _drop_invalid_edits(patch_edits: Any) -> list[Any]:
    """Keep only edit dicts that have a file and at least one anchor. A single
    malformed edit (e.g. neither search nor node_target) must not crash the
    whole plan phase — drop it and keep the rest."""
    if not isinstance(patch_edits, list):
        return patch_edits
    kept = []
    for e in patch_edits:
        if not isinstance(e, dict):
            kept.append(e)
            continue
        has_file = any(e.get(k) for k in ("file_path", "file", "path"))
        has_anchor = bool(e.get("search") or e.get("node_target"))
        if has_file and has_anchor:
            kept.append(e)
    return kept


def _normalize_plan_decision(response: dict[str, Any]) -> PlanDecision:
    files = _normalize_string_list(response.get("files"))
    patch_edits = response.get("patch_edits") or response.get("edits") or []
    patch_edits = _drop_invalid_edits(patch_edits)
    if "decision_frame" in response:
        return PlanDecision.model_validate(
            {**response, "files": files, "patch_edits": patch_edits}
        )
    has_executable_patch = bool(response.get("patch") or patch_edits)
    return PlanDecision.model_validate(
        {
            "plan": response.get("plan", ""),
            "patch": response.get("patch", ""),
            "patch_edits": patch_edits,
            "files": files,
            "test_command": response.get("test_command", ""),
            "decision_frame": {
                "stage": "plan",
                "summary": response.get("plan", ""),
                "hypotheses": response.get("hypotheses", []),
                "selected_hypothesis_id": response.get("selected_hypothesis_id"),
                "evidence": response.get("evidence", []),
                "next_checks": response.get("next_checks", []),
                "recommended_action": "execute" if has_executable_patch else "stop",
                "confidence": response.get("confidence", 0.0),
                "risk": response.get("risk", "unknown"),
                "trace_notes": json.dumps({"files": files}),
            },
        }
    )


def build_plan_user_prompt(
    state: AgentState,
    *,
    recall_context: str = "",
) -> str:
    """Build the primary planner prompt with one bounded outcome-summary section."""
    summary = sanitize_outcome_summary(state, state.attempt_outcome_summary)
    include_legacy_attempt_context = not summary
    previous_failures = (
        "\n\n".join(
            f"Attempt {idx + 1}: {attempt.test_result}\n"
            f"{_truncate_prompt_text(attempt.error_log, PLAN_FAILURE_LOG_LIMIT)}"
            for idx, attempt in enumerate(state.fix_attempts)
        )
        if include_legacy_attempt_context
        else ""
    )
    reflection_context = ""
    if include_legacy_attempt_context and state.reflection_notes:
        reflection_context = f"\n\nREFLECTION ANALYSIS:\n{state.reflection_notes}"
    hypothesis_continuity_context = ""
    continuity_context = (
        _hypothesis_continuity_context(state)
        if include_legacy_attempt_context
        else ""
    )
    if continuity_context:
        hypothesis_continuity_context = f"\n\n{continuity_context}"
    human_context = ""
    resumed_answer_context = _human_answer_context(state)
    if resumed_answer_context:
        human_context = f"\n\n{resumed_answer_context}"
    context_pressure_context = ""
    pressure = _context_pressure_instructions(state)
    if pressure:
        context_pressure_context = f"\n\n{pressure}"
    diversity_context = ""
    prior_failed_edits = (
        _prior_failed_edits_context(state)
        if include_legacy_attempt_context
        else ""
    )
    if prior_failed_edits:
        diversity_context = f"\n\n{prior_failed_edits}"
    correction_context = ""
    if include_legacy_attempt_context and state.search_correction_context:
        correction_context = f"\n\n{state.search_correction_context}"
    files_terms = _issue_search_terms(state.issue_title, state.issue_body)
    file_limit, max_files = _budget_scaled_file_limits(state)
    files_context = "\n\n".join(
        f"FILE: {file.path}\nRELEVANCE: {file.relevance_score} - {file.reason}\n"
        f"CONTENT:\n{_relevance_window(file.content, files_terms, file_limit)}"
        for file in state.relevant_files[:max_files]
    )
    completed_attempts_context = ""
    if summary:
        completed_attempts_context = (
            f"\n\n{OUTCOME_SUMMARY_SECTION}\n{summary}"
        )
    legacy_attempt_context = (
        f"\n\nPrevious failures:\n{previous_failures}"
        if include_legacy_attempt_context
        else ""
    )
    user = (
        f"Issue URL: {state.issue_url}\n"
        f"Title: {state.issue_title}\n\nBody:\n"
        f"{_truncate_prompt_text(state.issue_body, PLAN_ISSUE_BODY_LIMIT)}\n\n"
        f"Relevant files:\n{files_context}"
        f"{recall_context if include_legacy_attempt_context else ''}"
        f"{legacy_attempt_context}"
        f"{reflection_context}"
        f"{hypothesis_continuity_context}"
        f"{context_pressure_context}"
        f"{diversity_context}"
        f"{correction_context}"
        f"{human_context}"
    )
    user = user.replace(OUTCOME_SUMMARY_SECTION, "")
    return f"{user}{completed_attempts_context}"


def _build_escalated_plan_user_prompt(
    state: AgentState,
    evidence_ids: tuple[str, ...] | None = None,
) -> str:
    """Render the Task 6 packet plus one first-stage-only rolling summary."""
    packet = render_escalation_packet(
        build_escalation_packet(state, evidence_ids=evidence_ids)
    )
    packet = packet.replace(OUTCOME_SUMMARY_SECTION, "")
    summary = sanitize_outcome_summary(state, state.attempt_outcome_summary)
    if not summary:
        return packet
    return f"{packet}\n\n{OUTCOME_SUMMARY_SECTION}\n{summary}"


async def plan_fix(state: AgentState | dict[str, Any]) -> AgentState:
    """Ask the LLM for a concrete patch-oriented plan."""
    import sys
    state = _as_state(state)
    apply_escalation(state, should_escalate(state))
    if _is_budget_exceeded(state):
        state.failure_reason = "Token budget exceeded before planning."
        state.current_phase = Phase.FAILURE
        return state

    system = (
        "You are RepoPilot's planning node. Return exactly one JSON response "
        "variant: kind='tool' with one tool_intent, kind='plan' with the plan "
        "fields below, or kind='stop' with no outcome payload. The deterministic "
        "runtime alone chooses model escalation. Plan fields: "
        "plan (markdown string), patch_edits (array), patch (legacy unified diff string, usually empty), "
        "files (array of paths), test_command (string), decision_frame (object). "
        "Each patch_edits item must include file (path string), search (exact existing text), "
        "replace (replacement text), and optional replace_all (boolean, default false). "
        "You MUST express the fix as patch_edits and leave `patch` an EMPTY string. "
        "Do NOT output a unified diff / `diff --git` / `@@` hunks — the executor "
        "does exact string replacement on patch_edits, and a unified diff is rejected. "
        "Copy each search block verbatim from the file content shown; it must match "
        "uniquely unless replace_all is true. "
        "To replace an ENTIRE function, method, or class, you may instead set node_target to its dotted "
        "name (e.g. 'MyClass.my_method' or 'my_function'), leave search empty, and put the full new "
        "definition in replace — the executor locates the node by AST, so you need not copy surrounding text. "
        "decision_frame must include: "
        "stage='plan', summary, hypotheses (array of objects with id, claim, evidence, "
        "score), selected_hypothesis_id, evidence, next_checks, "
        "recommended_action='execute' when patch is present, 'collect_more_context' "
        "when more repository context is needed before deciding on a patch, "
        "'ask_user' when a human product decision, risk authorization, or external fact is required, "
        "otherwise 'stop', "
        "risk (low|medium|high|unknown), confidence (number 0.0 to 1.0). "
        "Use patch only for backward-compatible unified diff output when patch_edits cannot express the change."
    )
    if state.fix_attempts and _is_patch_apply_failure(state.fix_attempts[-1]):
        system = (
            f"{system} After a patch_apply_failed attempt, the next plan must "
            "repair the previous patch as exact patch_edits with correct file paths and search blocks "
            "before changing semantics. Do not shift the selected hypothesis "
            "unless the apply error proves the target file or hunk context is impossible."
        )
    if _is_final_attempt(state):
        system = f"{system}{_final_attempt_instructions()}"
    system = f"{system}{_force_patch_instructions(state)}"
    recall_context = ""
    if state.active_provider == "primary":
        recall = await _semantic_recall_context(state)
        if recall:
            recall_context = f"\n\n{recall}"
    primary_system = system
    primary_user = build_plan_user_prompt(state, recall_context=recall_context)
    user = primary_user
    if state.active_provider == "escalation":
        system = ESCALATED_PLAN_SYSTEM
        user = _build_escalated_plan_user_prompt(state)
    prompt_tokens_estimate = _estimate_tokens(system, user)
    _record_node_diagnostic(
        state,
        node="plan_fix",
        event="prompt_built",
        status="success",
        elapsed_seconds=0.0,
        prompt_tokens_estimate=prompt_tokens_estimate,
        relevant_file_count=len(state.relevant_files[:PLAN_MAX_FILES]),
        issue_body_chars=len(
            _truncate_prompt_text(state.issue_body, PLAN_ISSUE_BODY_LIMIT)
        ),
        previous_failure_count=len(state.fix_attempts),
        has_reflection_context=bool(state.reflection_notes),
        has_hypothesis_continuity_context=bool(
            _hypothesis_continuity_context(state)
        ),
        context_collection_count=state.context_collection_count,
        has_context_pressure=bool(_context_pressure_instructions(state)),
    )

    calls_this_round = 0
    reasoning_tool_counter = [0]
    while True:
        previous_provider = state.active_provider
        apply_escalation(state, should_escalate(state))
        if previous_provider != state.active_provider:
            system = ESCALATED_PLAN_SYSTEM
            user = _build_escalated_plan_user_prompt(state)
            prompt_tokens_estimate = _estimate_tokens(system, user)
        invoked_provider = state.active_provider
        invoked_model = state.active_model
        two_stage_repair = invoked_provider == "escalation"
        response_text = ""
        tool_step = None
        model_stopped = False
        t0 = time.monotonic()
        try:
            print("  [plan] Calling LLM for fix plan...", file=sys.stderr, flush=True)
            if two_stage_repair:
                reasoning_tool_counter[0] = calls_this_round
                repair_plan, verified_batch = await generate_opus_repair(
                    state,
                    build_escalation_packet(state),
                    first_stage_prompt=user,
                    validate_edits=False,
                    tool_counter=reasoning_tool_counter,
                )
                _clear_patch_authorization(state)
                state.active_repair_plan = repair_plan
                while True:
                    gate_result = validate_patch_batch(
                        state,
                        repair_plan,
                        verified_batch,
                    )
                    if gate_result.accepted:
                        patch_edits = gate_result.edits
                        break
                    if state.patch_correction_count >= 2:
                        _clear_patch_authorization(state)
                        state.failure_reason = (
                            "PatchGate rejected the initial batch and two local corrections."
                        )
                        state.current_phase = Phase.REFLECT
                        _record_node_diagnostic(
                            state,
                            node="plan_fix",
                            event="patch_gate",
                            status="error",
                            elapsed_seconds=0.0,
                            correction_count=state.patch_correction_count,
                            issue_codes=[issue.code for issue in gate_result.issues],
                        )
                        if record_opus_no_progress(
                            state,
                            node="plan_fix",
                            fingerprint={
                                "patch_gate_issues": [
                                    issue.code for issue in gate_result.issues
                                ]
                            },
                        ):
                            state.failure_reason = "opus_no_progress_limit"
                            state.current_phase = Phase.FAILURE
                        return state
                    state.patch_correction_count += 1
                    verified_batch = await request_verified_edit_correction(
                        state,
                        repair_plan,
                        verified_batch,
                        gate_result.issues,
                    )
                response_text = json.dumps(
                    {
                        "repair_plan": repair_plan.model_dump(mode="json"),
                        "verified_edits": verified_batch.model_dump(mode="json"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                decision = PlanDecision(
                    plan=(
                        f"Root cause: {repair_plan.root_cause}\n\n"
                        f"Required behavior: {repair_plan.required_behavior}\n\n"
                        f"Regression test strategy: "
                        f"{repair_plan.regression_test_strategy}"
                    ),
                    patch=state.patch_content,
                    patch_edits=patch_edits,
                    files=repair_plan.target_files,
                    test_command=state.test_command,
                    decision_frame=DecisionFrame(
                        stage="plan",
                        summary=repair_plan.root_cause,
                        hypotheses=[
                            Hypothesis(
                                id="H1",
                                claim=repair_plan.root_cause,
                                evidence=repair_plan.target_files,
                                score=1.0,
                                why_selected=repair_plan.required_behavior,
                            )
                        ],
                        selected_hypothesis_id="H1",
                        evidence=repair_plan.target_files,
                        next_checks=[repair_plan.regression_test_strategy],
                        recommended_action="execute",
                        confidence=1.0,
                        risk="medium",
                    ),
                )
                has_explicit_frame = True
            elif invoked_provider == "primary":
                raw_response = await llm_call(system, user)
            else:
                raw_response = await llm_call(
                    system,
                    user,
                    model=invoked_model,
                    provider="escalation",
                )
            if not two_stage_repair:
                response = _extract_json_object(raw_response)
                response_text = json.dumps(response)
                if not response:
                    raise ValueError("Model returned an empty structured response")
                response_kind = validate_reasoning_response(
                    response,
                    outcome_kind="plan",
                )
                if response_kind == "stop":
                    model_stopped = True
                else:
                    tool_step = await route_reasoning_tool(
                        state,
                        response,
                        node="plan_fix",
                        calls_this_round=calls_this_round,
                        router=route_tool_intent,
                    )
                    if not tool_step.handled:
                        has_explicit_frame = "decision_frame" in response
                        decision = _normalize_plan_decision(response)
        except Exception as exc:
            elapsed = time.monotonic() - t0
            immediate_reason = immediate_model_policy_reason(exc)
            response_tokens_estimate = _estimate_tokens(response_text)
            status = "invalid_response" if immediate_reason else "error"
            if not two_stage_repair:
                record_model_invocation(
                    state,
                    model=invoked_model,
                    provider=invoked_provider,
                    node="plan_fix",
                    elapsed_seconds=elapsed,
                    input_tokens=prompt_tokens_estimate,
                    output_tokens=response_tokens_estimate,
                    status=status,
                    error=exc,
                )
                state.token_usage += prompt_tokens_estimate + response_tokens_estimate
            _record_node_diagnostic(
                state,
                node="plan_fix",
                event="llm_call",
                status="error",
                elapsed_seconds=elapsed,
                error_type=type(exc).__name__,
                policy_reason=immediate_reason or None,
                prompt_tokens_estimate=prompt_tokens_estimate,
                response_tokens_estimate=response_tokens_estimate,
            )
            if isinstance(exc, ReasoningStop):
                state.failure_reason = exc.reason
                state.current_phase = Phase.FAILURE
                return state
            if immediate_reason and invoked_provider == "primary":
                apply_escalation(
                    state,
                    should_escalate(state, immediate_reason=immediate_reason),
                )
                if state.active_provider == "escalation":
                    system = ESCALATED_PLAN_SYSTEM
                    user = _build_escalated_plan_user_prompt(state)
                    prompt_tokens_estimate = _estimate_tokens(system, user)
                    continue
            if two_stage_repair:
                if record_opus_no_progress(
                    state,
                    node="plan_fix",
                    fingerprint={"error_class": type(exc).__name__},
                ):
                    state.failure_reason = "opus_no_progress_limit"
                    state.current_phase = Phase.FAILURE
                    return state
                system = ESCALATED_PLAN_SYSTEM
                user = _build_escalated_plan_user_prompt(state)
                prompt_tokens_estimate = _estimate_tokens(system, user)
                continue
            state.failure_reason = f"Failed to generate fix plan: {type(exc).__name__}"
            state.current_phase = Phase.FAILURE
            return state

        elapsed = time.monotonic() - t0
        response_tokens_estimate = _estimate_tokens(response_text)
        if not two_stage_repair:
            record_model_invocation(
                state,
                model=invoked_model,
                provider=invoked_provider,
                node="plan_fix",
                elapsed_seconds=elapsed,
                input_tokens=prompt_tokens_estimate,
                output_tokens=response_tokens_estimate,
                status="ok",
            )
        _record_node_diagnostic(
            state,
            node="plan_fix",
            event="llm_call",
            status="success",
            elapsed_seconds=elapsed,
            prompt_tokens_estimate=prompt_tokens_estimate,
            response_tokens_estimate=response_tokens_estimate,
        )
        if model_stopped:
            state.failure_reason = "model_stop"
            state.current_phase = Phase.FAILURE
            return state
        if tool_step is not None and tool_step.handled:
            state.token_usage += _estimate_tokens(system, user, response_text)
            if tool_step.stop_reason:
                state.failure_reason = tool_step.stop_reason
                state.current_phase = Phase.FAILURE
                return state
            calls_this_round += 1
            if state.active_provider == "escalation":
                system = ESCALATED_PLAN_SYSTEM
                user = _build_escalated_plan_user_prompt(
                    state,
                    tool_step.evidence_ids,
                )
            else:
                system = primary_system
                user = (
                    prompt_with_new_evidence(primary_user, state, tool_step.evidence_ids)
                    if tool_step.evidence_ids
                    else primary_user
                )
            prompt_tokens_estimate = _estimate_tokens(system, user)
            continue
        break

    state.fix_plan = decision.plan
    if not two_stage_repair:
        _clear_patch_authorization(state)
        state.patch_content = decision.patch
        state.patch_edits = decision.patch_edits
    else:
        # PatchGate already installed this exact approved pair; do not replace it.
        if decision.patch != state.patch_content or decision.patch_edits != state.patch_edits:
            raise ValueError("PatchGate decision diverged from its frozen approval")
    state.test_command = decision.test_command
    print(f"  [plan] Plan received ({len(state.fix_plan)} chars, patch={len(state.patch_content)} chars, edits={len(state.patch_edits)})", file=sys.stderr, flush=True)
    if not two_stage_repair:
        state.token_usage += _estimate_tokens(system, user, response_text)
    _remember(state, "assistant", state.fix_plan[:2000])
    frame = decision.decision_frame
    frame.parent_frame_id = state.decision_frame.frame_id if state.decision_frame else None
    if not frame.trace_notes:
        frame.trace_notes = json.dumps({"files": decision.files})
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
    if state.patch_content or state.patch_edits:
        dead_reason = _dead_plan_reason(state)
        if dead_reason is not None:
            record_no_progress(
                state,
                kind="repeated_edit",
                node="plan_fix",
                fingerprint=dead_reason,
            )
            apply_escalation(state, should_escalate(state))
            state.repeated_patch_block_count += 1
            state.decision_warnings.append(
                {
                    "node": "plan_fix",
                    "warning": "blocked_dead_patch",
                    "detail": (
                        "Planned patch_edits repeat a patch that already failed "
                        f"({dead_reason}); refusing to execute it again."
                    ),
                    "frame_id": frame.frame_id,
                }
            )
            _clear_patch_authorization(state)
            if (
                state.repeated_patch_block_count > MAX_REPEATED_PATCH_BLOCKS
                or _is_final_attempt(state)
            ):
                frame.recommended_action = "stop"
                state.current_phase = Phase.FAILURE
                state.failure_reason = (
                    "Planner kept re-emitting patches that already failed "
                    f"({dead_reason})."
                )
            else:
                # The router selects the phase from frame.recommended_action, not
                # current_phase — so reroute the frame too, else it stays
                # 'execute' and the (now-empty) patch leaks to EXECUTE.
                frame.recommended_action = "reflect"
                state.current_phase = Phase.REFLECT
        else:
            missing = (
                []
                if _has_valid_exact_patch_gate_approval(state)
                else _unlocatable_edits(state)
            )
            if missing:
                record_no_progress(
                    state,
                    kind="nonexistent_search_block",
                    node="plan_fix",
                    fingerprint=[
                        {
                            "file": edit.file_path,
                            "search": edit.search,
                            "replace": edit.replace,
                        }
                        for edit in missing
                    ],
                )
                apply_escalation(state, should_escalate(state))
                state.hallucinated_search_block_count += 1
                state.search_correction_context = _build_search_correction(state, missing)
                state.decision_warnings.append(
                    {
                        "node": "plan_fix",
                        "warning": "hallucinated_search_block",
                        "detail": (
                            f"{len(missing)} planned search block(s) do not exist "
                            "in the target file; feeding real lines back to replan."
                        ),
                        "frame_id": frame.frame_id,
                    }
                )
                _clear_patch_authorization(state)
                if (
                    state.hallucinated_search_block_count > MAX_SEARCH_CORRECTIONS
                    or _is_final_attempt(state)
                ):
                    frame.recommended_action = "stop"
                    state.current_phase = Phase.FAILURE
                    state.failure_reason = (
                        "Planner kept emitting search blocks that do not exist in "
                        "the target files."
                    )
                else:
                    # Reroute the frame too (router reads recommended_action, not
                    # current_phase) — otherwise it stays 'execute' and the empty
                    # patch leaks to EXECUTE, producing a misleading "No valid
                    # patches in input" diff error instead of a real replan.
                    frame.recommended_action = "plan"
                    state.current_phase = Phase.PLAN
            else:
                state.search_correction_context = ""  # resolved; stop feeding it
                prior_assertion_symbols = {
                    edit.node_target
                    for attempt in state.fix_attempts
                    if not attempt.success
                    and attempt.failure_kind == "assertion_failure"
                    for edit in attempt.patch_edits
                    if edit.node_target
                }
                assertion_target_not_diversified = (
                    state.assertion_diversity_required
                    and any(
                        not edit.node_target
                        or edit.node_target in prior_assertion_symbols
                        for edit in state.patch_edits
                    )
                )
                if assertion_target_not_diversified:
                    state.decision_warnings.append(
                        {
                            "node": "plan_fix",
                            "warning": "assertion_target_not_diversified",
                            "detail": (
                                "Repeated assertion repair reused a prior failed "
                                "target symbol; a different target is required."
                            ),
                            "frame_id": frame.frame_id,
                        }
                    )
                    record_no_progress(
                        state,
                        kind="repeated_edit",
                        node="plan_fix",
                    )
                    apply_escalation(state, should_escalate(state))
                    _clear_patch_authorization(state)
                    frame.recommended_action = "plan"
                    state.current_phase = Phase.PLAN
                else:
                    record_progress(state)
                    state.assertion_diversity_required = False
                    if _planned_edits_repeat_failure(state):
                        state.decision_warnings.append(
                            {
                                "node": "plan_fix",
                                "warning": "repeated_failed_patch",
                                "detail": (
                                    "Planned patch_edits only repeat edits that already "
                                    "failed; the planner did not diversify."
                                ),
                                "frame_id": frame.frame_id,
                            }
                        )
                    state.current_phase = Phase.EXECUTE
    elif frame.recommended_action == "collect_more_context":
        if _is_final_attempt(state):
            frame.recommended_action = "stop"
            state.current_phase = Phase.FAILURE
            state.failure_reason = (
                "Final attempt requested more context instead of producing a patch."
            )
        else:
            state.context_collection_count += 1
            if state.context_collection_count > MAX_CONTEXT_COLLECTION_ROUNDS:
                frame.recommended_action = "stop"
                state.current_phase = Phase.FAILURE
                state.failure_reason = (
                    "Context collection made no progress after "
                    f"{MAX_CONTEXT_COLLECTION_ROUNDS} attempts."
                )
            else:
                state.current_phase = Phase.PLAN
    elif frame.recommended_action == "ask_user":
        state.current_phase = Phase.PLAN
    else:
        frame.recommended_action = "stop"
        state.current_phase = Phase.FAILURE
        state.failure_reason = "Planner did not produce a patch."
    return state
