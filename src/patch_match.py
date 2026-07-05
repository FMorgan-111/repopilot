"""Locating model-authored search/replace blocks inside real file content.

Shared by the EXECUTE apply path (fuzzy application) and the PLAN validation
gate (rejecting hallucinated search blocks before they burn a retry). Pure
functions, no I/O — safe to import from any node without cycles.
"""

from __future__ import annotations

import difflib


def leading_spaces(line: str) -> int:
    """Count of leading spaces (tab-free indentation only)."""
    return len(line) - len(line.lstrip(" "))


def reindent(text: str, delta: int) -> str:
    """Shift every non-blank line of `text` by `delta` spaces.

    Guarded: a negative delta is applied only when every non-blank line has at
    least `-delta` leading spaces, so we never eat significant characters."""
    if delta == 0:
        return text
    lines = text.split("\n")
    if delta < 0:
        removable = -delta
        if any(line.strip() and leading_spaces(line) < removable for line in lines):
            return text
        return "\n".join(line[removable:] if line.strip() else line for line in lines)
    pad = " " * delta
    return "\n".join(pad + line if line.strip() else line for line in lines)


def find_normalized_span(content: str, search: str) -> tuple[int, int, int] | None:
    """Locate `search` in `content` ignoring per-line leading/trailing whitespace.

    Returns (start_offset, end_offset, indent_delta) of the ORIGINAL span in
    `content`, or None. Requires exactly one normalized match — an ambiguous
    block is treated as not found so we never fuzzy-replace the wrong site.
    `indent_delta` is how many spaces the matched block is indented relative to
    the search block's first line (for reindenting the replacement)."""
    content_lines = content.split("\n")
    search_lines = search.split("\n")
    if search_lines and search_lines[-1] == "":
        search_lines = search_lines[:-1]  # drop trailing-newline artifact
    if not search_lines:
        return None
    norm_search = [line.strip() for line in search_lines]
    n = len(search_lines)

    offsets: list[int] = []
    pos = 0
    for line in content_lines:
        offsets.append(pos)
        pos += len(line) + 1  # +1 for the stripped "\n"

    matches: list[tuple[int, int, int]] = []
    for i in range(len(content_lines) - n + 1):
        window = content_lines[i : i + n]
        if [line.strip() for line in window] != norm_search:
            continue
        start = offsets[i]
        end = offsets[i + n - 1] + len(content_lines[i + n - 1])  # exclude trailing \n
        delta = leading_spaces(content_lines[i]) - leading_spaces(search_lines[0])
        matches.append((start, end, delta))
    if len(matches) == 1:
        return matches[0]
    return None


def locate_search_block(content: str, search: str) -> bool:
    """True if `search` can actually be applied to `content` — exact substring
    or a unique whitespace-normalized match (the same logic apply uses). A False
    means the block does not exist in the file (a hallucinated anchor)."""
    if not search:
        return False
    if search in content:
        return True
    return find_normalized_span(content, search) is not None


def closest_region(
    content: str, search: str, context: int = 3, max_chars: int = 1200
) -> str:
    """The actual file lines most similar to `search`, to feed back to a planner
    that emitted a search block which does not exist verbatim.

    Anchors on the single most distinctive (longest) search line via a cheap
    similarity scan — O(file lines) — then returns that region plus surrounding
    context so the planner can copy a real search block. Returns '' when there
    is nothing to anchor on."""
    content_lines = content.split("\n")
    search_lines = [line for line in search.split("\n") if line.strip()]
    if not content_lines or not search_lines:
        return ""
    anchor = max(search_lines, key=len)
    best_i, best_ratio = 0, -1.0
    for i, line in enumerate(content_lines):
        ratio = difflib.SequenceMatcher(None, line, anchor).quick_ratio()
        if ratio > best_ratio:
            best_ratio, best_i = ratio, i
    lo = max(0, best_i - context)
    hi = min(len(content_lines), best_i + len(search_lines) + context)
    snippet = "\n".join(content_lines[lo:hi])
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rstrip() + "\n... [truncated] ..."
    return snippet
