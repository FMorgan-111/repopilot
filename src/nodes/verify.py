"""VERIFY phase: Parse test output and route to COMMIT, retry PLAN, or FAILED."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from ..model_policy import record_no_progress, record_progress
from ..state import (
    AgentState,
    FixAttempt,
    Phase,
    _as_state,
    _is_budget_exceeded,
    _record_node_diagnostic,
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


def _test_failure_class(attempt: FixAttempt) -> str:
    """Classify only routing-relevant test failures from bounded local output."""
    failure_kind = _failure_kind(attempt).lower()
    error_log = attempt.error_log.lower()
    if "syntaxerror" in error_log or failure_kind == "syntax_error":
        return "syntax_error"
    if (
        "modulenotfounderror" in error_log
        or "importerror" in error_log
        or failure_kind == "import_error"
    ):
        return "import_error"
    if (
        failure_kind == "assertion_failure"
        or "assertionerror" in error_log
        or "assert " in error_log
    ):
        return "assertion_failure"
    return failure_kind or "test_failed"


def _failure_signature_payload(attempt: FixAttempt, failure_class: str) -> dict[str, str]:
    error_log = attempt.error_log.replace("\\", "/")
    test_ids = sorted(
        set(
            re.findall(
                r"(?m)^(?:FAILED|ERROR)\s+([\w./-]+(?:::[\w.\[\]-]+)+)",
                error_log,
            )
        )
    )
    assertion_lines: list[str] = []
    for line in error_log.splitlines():
        lowered = line.lower()
        if not any(
            marker in lowered
            for marker in ("assertionerror", "assert ", "expected", "actual")
        ):
            continue
        normalized = re.sub(r"(?:[A-Za-z]:)?/(?:[^\s:]+/)+", "<path>/", line)
        normalized = re.sub(r"\b0x[0-9a-fA-F]+\b", "<address>", normalized)
        normalized = re.sub(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,}\b",
            "<runtime-id>",
            normalized,
        )
        normalized = re.sub(r":\d+(?=[:\s])", ":<line>", normalized)
        normalized = re.sub(
            r"\b\d{4}-\d{2}-\d{2}[T ][0-9:.+-]+Z?\b",
            "<timestamp>",
            normalized,
        )
        normalized = re.sub(r"\b\d+(?:\.\d+)?s\b", "<duration>", normalized)
        normalized = " ".join(normalized.split())
        if normalized and normalized not in assertion_lines:
            assertion_lines.append(normalized)
    return {
        "class": failure_class,
        "test_ids": "|".join(test_ids),
        "assertion": "|".join(assertion_lines[:8]),
        "detail": (
            ""
            if failure_class == "assertion_failure"
            else " ".join(error_log.split())[:2_000]
        ),
    }


def _canonical_signature(payload: dict[str, str]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _record_test_failure_progress(
    state: AgentState,
    latest: FixAttempt,
    failure_class: str,
) -> bool:
    """Return True only when this failure repeats the prior material signature."""
    payload = _failure_signature_payload(latest, failure_class)
    signature = _canonical_signature(payload)
    assertion_failure = failure_class == "assertion_failure"
    prior_assertion_signature = state.last_assertion_failure_signature
    if assertion_failure:
        state.last_assertion_failure_signature = signature
    else:
        state.last_assertion_failure_signature = ""
        state.assertion_no_progress_rounds = 0
        state.assertion_diversity_required = False
    repeated_signature = (
        prior_assertion_signature == signature
        if assertion_failure
        else state.last_test_failure_signature == signature
    )
    if not repeated_signature:
        record_progress(state)
        state.last_test_failure_signature = signature
        state.assertion_no_progress_rounds = 0
        state.assertion_diversity_required = False
        return False
    state.last_test_failure_signature = signature
    record_no_progress(
        state,
        kind="unchanged_test_failure",
        node="verify_fix",
        fingerprint=payload,
    )
    if assertion_failure and prior_assertion_signature == signature:
        state.assertion_no_progress_rounds += 1
        state.no_progress_rounds = state.assertion_no_progress_rounds
    return True


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
        state.last_assertion_failure_signature = ""
        state.assertion_no_progress_rounds = 0
        state.assertion_diversity_required = False
        state.current_phase = Phase.DONE if state.skip_commit else Phase.COMMIT
        return state

    failure_class = _test_failure_class(latest)
    if _failure_kind(latest) == "infra_error":
        message = latest.error_log.strip() or "execution infrastructure failed"
        state.failure_reason = f"Infrastructure error during execution: {message[:500]}"
        state.current_phase = Phase.FAILURE
        return state

    if _same_failure_seen_twice(state) and failure_class not in {
        "syntax_error",
        "import_error",
        "assertion_failure",
    }:
        state.failure_reason = "Same patch produced the same failure twice."
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

    repeated_failure = _record_test_failure_progress(state, latest, failure_class)

    if failure_class in {"syntax_error", "import_error"}:
        if state.retry_count >= state.max_retries:
            state.failure_reason = f"Maximum retries reached: {state.max_retries}."
            state.current_phase = Phase.FAILURE
            return state
        if _is_budget_exceeded(state):
            state.failure_reason = "Token budget exceeded during verification."
            state.current_phase = Phase.FAILURE
            return state
        state.retry_count += 1
        state.current_phase = Phase.PLAN
        _record_node_diagnostic(
            state,
            node="verify_fix",
            event="direct_patch_correction",
            status="success",
            elapsed_seconds=0.0,
            failure_class=failure_class,
        )
        return state

    if failure_class == "assertion_failure" and repeated_failure:
        if state.assertion_no_progress_rounds >= 2:
            state.failure_reason = "repeated_assertion_no_progress"
            state.current_phase = Phase.FAILURE
            _record_node_diagnostic(
                state,
                node="verify_fix",
                event="assertion_no_progress_limit",
                status="error",
                elapsed_seconds=0.0,
                round=state.no_progress_rounds,
            )
            return state
        state.assertion_diversity_required = True
        _record_node_diagnostic(
            state,
            node="verify_fix",
            event="assertion_diversity_required",
            status="success",
            elapsed_seconds=0.0,
            round=state.no_progress_rounds,
        )

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
