"""Canonical signatures for completed repair-plan transactions."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .state import AgentState, DecisionFrame, FixAttempt


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return _digest(encoded)


def _latest_plan_frame(state: AgentState) -> DecisionFrame | None:
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
        if frame.stage == "plan":
            return frame
    return None


def build_plan_transaction_signature(state: AgentState) -> str:
    """Hash the current complete plan frame and latest applied edit transaction."""
    frame = _latest_plan_frame(state)
    attempt = state.fix_attempts[-1] if state.fix_attempts else FixAttempt()
    material: dict[str, Any] = {
        "frame": (
            frame.model_dump(
                mode="json",
                exclude={"frame_id", "parent_frame_id"},
            )
            if frame is not None
            else None
        ),
        "transaction": {
            "patch_sha256": _digest(attempt.patch_content),
            "legacy_file_path_sha256": _digest(attempt.file_path),
            "edits": [
                {
                    "order": index,
                    "edit_sha256": _canonical_digest(edit.model_dump(mode="json")),
                    "file_path_sha256": _digest(edit.file_path),
                    "search_sha256": _digest(edit.search),
                    "replace_sha256": _digest(edit.replace),
                    "node_target_sha256": _digest(edit.node_target),
                    "replace_all": edit.replace_all,
                    "expected_content_sha256": edit.expected_content_sha256,
                    "exact_only": edit.exact_only,
                }
                for index, edit in enumerate(attempt.patch_edits)
            ],
        },
        "pending_transaction": {
            "patch_sha256": _digest(state.patch_content),
            "edits": [
                {
                    "order": index,
                    "edit_sha256": _canonical_digest(edit.model_dump(mode="json")),
                }
                for index, edit in enumerate(state.patch_edits)
            ],
        },
    }
    return _canonical_digest(material)
