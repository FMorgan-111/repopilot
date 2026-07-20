"""Bounded, replayable evidence captured from approved local tools."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel

from .model_provider import redact_secrets
from .state import AgentState, Evidence


class EvidenceAddResult(BaseModel):
    evidence: Evidence | None
    added: bool
    disposition: Literal["added", "duplicate", "capacity"]


class EvidenceCapacityError(RuntimeError):
    """Raised when a new evidence item cannot be persisted within the cap."""


class EvidenceStore:
    """Add safe tool results to an agent state without unbounded prompt growth."""

    def __init__(
        self,
        state: AgentState,
        *,
        max_items: int = 30,
        max_content_chars: int = 8_000,
        max_summary_chars: int = 1_000,
    ) -> None:
        self.state = state
        self.max_items = max(0, max_items)
        self.max_content_chars = max(0, max_content_chars)
        self.max_summary_chars = max(0, max_summary_chars)

    def add(
        self,
        *,
        tool: str,
        summary: str,
        content: str,
        file_path: str | None = None,
        symbol: str | None = None,
    ) -> EvidenceAddResult:
        normalized_path = _normalize_path(file_path)
        normalized_symbol = _normalize_text(symbol) if symbol is not None else None
        normalized_content = _normalize_text(content)
        fingerprint = _fingerprint(
            tool=tool,
            file_path=normalized_path,
            symbol=normalized_symbol,
            content=normalized_content,
        )

        for existing in self.state.evidence:
            if existing.fingerprint == fingerprint:
                return EvidenceAddResult(
                    evidence=existing,
                    added=False,
                    disposition="duplicate",
                )

        evidence = Evidence(
            evidence_id=f"ev_{fingerprint[:16]}",
            tool=str(tool),
            file_path=normalized_path,
            symbol=normalized_symbol,
            summary=_normalize_text(summary)[: self.max_summary_chars],
            content=normalized_content[: self.max_content_chars],
            fingerprint=fingerprint,
        )
        if len(self.state.evidence) >= self.max_items:
            return EvidenceAddResult(
                evidence=None,
                added=False,
                disposition="capacity",
            )

        self.state.evidence.append(evidence)
        return EvidenceAddResult(
            evidence=evidence,
            added=True,
            disposition="added",
        )

    def select(
        self,
        evidence_ids: list[str],
        *,
        max_total_chars: int = 24_000,
    ) -> list[Evidence]:
        limit = max(0, max_total_chars)
        by_id = {item.evidence_id: item for item in self.state.evidence}
        selected: list[Evidence] = []
        seen_ids: set[str] = set()
        for evidence_id in evidence_ids:
            if evidence_id in seen_ids:
                continue
            seen_ids.add(evidence_id)
            evidence = by_id.get(evidence_id)
            if evidence is None:
                continue
            candidate = [*selected, evidence]
            if len(self.render_for_prompt(candidate)) > limit:
                continue
            selected = candidate
        return selected

    @staticmethod
    def render_for_prompt(evidence: list[Evidence]) -> str:
        """Return the exact deterministic payload that selection budgets cover."""
        return "\n".join(
            json.dumps(
                item.model_dump(),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            for item in evidence
        )


def _normalize_text(value: object) -> str:
    return redact_secrets(str(value)).replace("\r\n", "\n").replace("\r", "\n").strip()


def _normalize_path(value: str | None) -> str | None:
    if value is None:
        return None
    parts: list[str] = []
    for part in str(value).replace("\\", "/").split("/"):
        if not part or part == ".":
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/".join(parts)


def _fingerprint(
    *, tool: str,
    file_path: str | None,
    symbol: str | None,
    content: str,
) -> str:
    payload = json.dumps(
        {
            "content": content,
            "file_path": file_path,
            "symbol": symbol,
            "tool": str(tool),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
