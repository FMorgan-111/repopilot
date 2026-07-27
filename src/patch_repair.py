"""Deterministic cleanup for model-generated unified diffs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PatchRepair:
    patch: str
    changed: bool
    reasons: list[str] = field(default_factory=list)


_HUNK_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@(?P<section>.*)$"
)
_VALID_INDEX_RE = re.compile(
    r"^index [0-9a-fA-F]+\.\.[0-9a-fA-F]+(?: \d{6})?$"
)


def repair_unified_diff(patch: str) -> PatchRepair:
    """Return a syntactically cleaner unified diff without changing intent."""
    original = patch
    reasons: list[str] = []
    lines = patch.splitlines()

    extracted = _extract_diff_lines(lines)
    if extracted != lines:
        lines = extracted
        reasons.append("extracted_diff_block")

    lines, removed_invalid_index = _remove_invalid_index_lines(lines)
    if removed_invalid_index:
        reasons.append("removed_invalid_index_line")

    lines, recounted_hunks = _recount_hunk_lengths(lines)
    if recounted_hunks:
        reasons.append("recounted_hunk_lengths")

    if not reasons:
        repaired = original
    else:
        repaired = "\n".join(lines).rstrip() + "\n" if lines else ""
    return PatchRepair(
        patch=repaired,
        changed=bool(reasons) and repaired != original,
        reasons=reasons,
    )


def _extract_diff_lines(lines: list[str]) -> list[str]:
    start = next(
        (idx for idx, line in enumerate(lines) if line.startswith("diff --git ")),
        None,
    )
    if start is None:
        return lines

    extracted: list[str] = []
    for line in lines[start:]:
        if line.startswith("```"):
            break
        extracted.append(line)
    return extracted


def _remove_invalid_index_lines(lines: list[str]) -> tuple[list[str], bool]:
    repaired: list[str] = []
    removed = False
    for line in lines:
        if line.startswith("index ") and not _VALID_INDEX_RE.match(line):
            removed = True
            continue
        repaired.append(line)
    return repaired, removed


def _recount_hunk_lengths(lines: list[str]) -> tuple[list[str], bool]:
    repaired: list[str] = []
    changed = False
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        match = _HUNK_RE.match(line)
        if match is None:
            repaired.append(line)
            idx += 1
            continue

        body: list[str] = []
        idx += 1
        while idx < len(lines) and not _starts_next_patch_section(lines[idx]):
            body.append(lines[idx])
            idx += 1

        old_count, new_count = _count_hunk_body(body)
        original_old = int(match.group("old_count") or "1")
        original_new = int(match.group("new_count") or "1")
        if old_count != original_old or new_count != original_new:
            changed = True
            line = (
                f"@@ -{match.group('old_start')},{old_count} "
                f"+{match.group('new_start')},{new_count} @@"
                f"{match.group('section')}"
            )
        repaired.append(line)
        repaired.extend(body)
    return repaired, changed


def _starts_next_patch_section(line: str) -> bool:
    return (
        line.startswith("@@ ")
        or line.startswith("diff --git ")
        or line.startswith("--- ")
    )


def _count_hunk_body(body: list[str]) -> tuple[int, int]:
    old_count = 0
    new_count = 0
    for line in body:
        if line.startswith("\\"):
            continue
        if line.startswith("-"):
            old_count += 1
        elif line.startswith("+"):
            new_count += 1
        elif line.startswith(" "):
            old_count += 1
            new_count += 1
    return old_count, new_count
