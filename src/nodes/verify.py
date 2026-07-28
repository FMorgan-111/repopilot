"""VERIFY phase: Parse test output and route to COMMIT, retry PLAN, or FAILED."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from ..model_policy import record_no_progress, record_progress
from ..repair_rounds import (
    record_failed_repair_round,
    validate_repair_round_state,
)
from ..state import (
    AgentState,
    FixAttempt,
    Phase,
    _as_state,
    _record_node_diagnostic,
    tool_manifest_fingerprint,
)


def _trusted_authorized_binding(
    state: AgentState,
    latest: FixAttempt,
) -> tuple[int, str, str]:
    """Validate the persisted, fingerprint-bound author for this exact patch."""
    approval = state.tool_patch_approval
    plan = state.active_repair_plan
    binding = (
        state.authorized_repair_round_id,
        state.authorized_repair_provider or "",
        state.authorized_repair_model,
    )
    if (
        approval is None
        or plan is None
        or binding[0] <= 0
        or not binding[1]
        or not binding[2]
        or not state.patch_content
        or not state.patch_edits
    ):
        raise ValueError("trusted PatchGate authorization is unavailable")
    if approval.base_ref.lower() != state.repo_ref.lower():
        raise ValueError("PatchGate authorization base does not match state")
    patch_sha = hashlib.sha256(state.patch_content.encode("utf-8")).hexdigest()
    if patch_sha != approval.patch_sha256:
        raise ValueError("PatchGate authorization patch digest changed")
    if (
        tool_manifest_fingerprint(approval.changed_manifest)
        != approval.manifest_fingerprint
    ):
        raise ValueError("PatchGate authorization manifest changed")
    if latest.patch_content != state.patch_content:
        raise ValueError("FixAttempt patch does not match frozen authorization")
    if latest.patch_gate_fingerprint is None:
        raise ValueError("FixAttempt lacks an immutable PatchGate receipt")
    if latest.patch_gate_fingerprint != approval.patch_gate_fingerprint:
        raise ValueError("FixAttempt PatchGate receipt does not match authorization")

    from ..patch_gate import _approval_edit_payload, _fingerprint

    attempt_edits = [_approval_edit_payload(edit) for edit in latest.patch_edits]
    state_edits = [_approval_edit_payload(edit) for edit in state.patch_edits]
    if not attempt_edits or attempt_edits != state_edits:
        raise ValueError("FixAttempt edits do not match frozen authorization")
    expected_fingerprint = _fingerprint(
        state,
        plan,
        state.patch_edits,
        list(approval.changed_manifest),
        approval.patch_sha256,
        binding,
    )
    if expected_fingerprint != approval.patch_gate_fingerprint:
        raise ValueError("PatchGate authorization fingerprint changed")
    return binding


def _attempt_binding(latest: FixAttempt) -> tuple[int, str, str]:
    binding = (
        latest.repair_round_id,
        latest.repair_provider or "",
        latest.repair_model,
    )
    if binding[0] <= 0 or not binding[1] or not binding[2]:
        raise ValueError("FixAttempt lacks positive repair author attribution")
    receipt = latest.patch_gate_fingerprint
    if receipt is not None and not re.fullmatch(r"[0-9a-f]{64}", receipt):
        raise ValueError("FixAttempt PatchGate receipt is malformed")
    return binding


def _validate_counted_replay(state: AgentState, latest: FixAttempt) -> None:
    """Validate immutable attribution and the durable ledger on a candidate."""
    round_id, _provider, _model = _attempt_binding(latest)
    if round_id > state.last_counted_repair_round_id:
        raise ValueError("FixAttempt is not a counted repair round")
    candidate = state.model_copy(deep=True)
    validate_repair_round_state(candidate)


def _restore_open_binding(
    state: AgentState,
    binding: tuple[int, str, str],
) -> None:
    current = (
        state.current_repair_round_id,
        state.current_repair_provider or "",
        state.current_repair_model,
    )
    if current == binding:
        return
    if current != (0, "", ""):
        raise ValueError("open repair ledger does not match FixAttempt")
    state.current_repair_round_id = binding[0]
    state.current_repair_provider = binding[1]
    state.current_repair_model = binding[2]


def _prepare_positive_attribution(
    state: AgentState,
) -> tuple[AgentState, tuple[int, str, str]]:
    """Validate an uncounted receipt-bound attempt on an isolated candidate."""
    candidate = state.model_copy(deep=True)
    latest = candidate.fix_attempts[-1]
    validate_repair_round_state(candidate)
    attempt_binding = _attempt_binding(latest)
    binding = _trusted_authorized_binding(candidate, latest)
    if attempt_binding != binding:
        raise ValueError("FixAttempt attribution does not match authorization")
    if (
        binding[0] <= candidate.last_counted_repair_round_id
        or binding[0] != candidate.repair_round_sequence
    ):
        raise ValueError("FixAttempt does not identify the open uncounted round")
    _restore_open_binding(candidate, binding)
    validate_repair_round_state(candidate)
    return candidate, binding


def _prepare_legacy_first_round(
    state: AgentState,
) -> tuple[AgentState, tuple[int, str, str]]:
    """Migrate only a sole, pristine, exact first-round historical attempt."""
    candidate = state.model_copy(deep=True)
    latest = candidate.fix_attempts[-1]
    attempt_binding = (
        latest.repair_round_id,
        latest.repair_provider or "",
        latest.repair_model,
    )
    authorized = (
        candidate.authorized_repair_round_id,
        candidate.authorized_repair_provider or "",
        candidate.authorized_repair_model,
    )
    current = (
        candidate.current_repair_round_id,
        candidate.current_repair_provider or "",
        candidate.current_repair_model,
    )
    if (
        len(candidate.fix_attempts) != 1
        or candidate.current_phase != Phase.VERIFY
        or _failure_kind(latest) == "infra_error"
        or attempt_binding != (0, "", "")
        or latest.patch_gate_fingerprint is not None
        or candidate.repair_round_sequence != 1
        or authorized[0] != 1
        or not authorized[1]
        or not authorized[2]
        or candidate.last_counted_repair_round_id != 0
        or candidate.retry_count != 0
        or candidate.primary_failed_repair_rounds != 0
        or current not in {authorized, (0, "", "")}
    ):
        raise ValueError("historical FixAttempt is not a pristine first round")
    validate_repair_round_state(candidate)

    assert candidate.tool_patch_approval is not None
    binding = authorized
    migrated_payload = latest.model_dump(mode="python")
    migrated_payload.update(
        {
            "repair_round_id": binding[0],
            "repair_provider": binding[1],
            "repair_model": binding[2],
            "patch_gate_fingerprint": (
                candidate.tool_patch_approval.patch_gate_fingerprint
            ),
        }
    )
    migrated = FixAttempt.model_validate(migrated_payload)
    binding = _trusted_authorized_binding(candidate, migrated)
    if binding != authorized:
        raise ValueError("historical authorization does not match first binding")
    candidate.fix_attempts[-1] = migrated
    _restore_open_binding(candidate, binding)
    validate_repair_round_state(candidate)
    return candidate, binding


def _prepare_failure_attribution(
    state: AgentState,
) -> tuple[AgentState, tuple[int, str, str]]:
    latest = state.fix_attempts[-1]
    if latest.repair_round_id == 0:
        return _prepare_legacy_first_round(state)
    return _prepare_positive_attribution(state)


def _route_state_integrity_failure(state: AgentState, exc: Exception) -> AgentState:
    state.failure_reason = (
        f"Infrastructure state-integrity error during verification: {str(exc)[:500]}"
    )
    state.current_phase = Phase.FAILURE
    return state


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


def _failure_signature_payload(
    attempt: FixAttempt, failure_class: str
) -> dict[str, str]:
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
    if _failure_kind(latest) == "infra_error":
        message = latest.error_log.strip() or "execution infrastructure failed"
        state.failure_reason = f"Infrastructure error during execution: {message[:500]}"
        state.current_phase = Phase.FAILURE
        return state

    if latest.success:
        try:
            state, _binding = _prepare_positive_attribution(state)
        except (TypeError, ValueError) as exc:
            return _route_state_integrity_failure(state, exc)
        latest = state.fix_attempts[-1]
        if not latest.episode_recording_attempted:
            await _record_episode_best_effort(state, latest)
            latest.episode_recording_attempted = True
        state.last_assertion_failure_signature = ""
        state.assertion_no_progress_rounds = 0
        state.assertion_diversity_required = False
        state.current_phase = Phase.COVERAGE
        return state

    failure_class = _test_failure_class(latest)
    if latest.repair_round_id > 0 and (
        latest.repair_round_id <= state.last_counted_repair_round_id
    ):
        try:
            _validate_counted_replay(state, latest)
        except (TypeError, ValueError) as exc:
            return _route_state_integrity_failure(state, exc)
        return state

    try:
        state, binding = _prepare_failure_attribution(state)
    except (TypeError, ValueError) as exc:
        return _route_state_integrity_failure(state, exc)
    round_id, provider, model = binding
    latest = state.fix_attempts[-1]

    retry_phase = (
        Phase.PLAN
        if failure_class in {"syntax_error", "import_error"}
        else Phase.REFLECT
    )
    if not latest.episode_recording_attempted:
        await _record_episode_best_effort(state, latest)
        latest.episode_recording_attempted = True

    repeated_failure = _record_test_failure_progress(state, latest, failure_class)

    if failure_class in {"syntax_error", "import_error"}:
        _record_node_diagnostic(
            state,
            node="verify_fix",
            event="direct_patch_correction",
            status="success",
            elapsed_seconds=0.0,
            failure_class=failure_class,
        )

    if failure_class == "assertion_failure" and repeated_failure:
        state.assertion_diversity_required = True
        _record_node_diagnostic(
            state,
            node="verify_fix",
            event="assertion_diversity_required",
            status="success",
            elapsed_seconds=0.0,
            round=state.no_progress_rounds,
        )

    try:
        record_failed_repair_round(
            state,
            round_id=round_id,
            provider=provider,
            model=model,
            failure_reason=failure_class,
            retry_phase=retry_phase,
        )
    except (TypeError, ValueError) as exc:
        return _route_state_integrity_failure(state, exc)
    return state
