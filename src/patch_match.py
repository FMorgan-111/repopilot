"""Locating model-authored search/replace blocks inside real file content.

Shared by the EXECUTE apply path (fuzzy application) and the PLAN validation
gate (rejecting hallucinated search blocks before they burn a retry). Pure
functions, no I/O — safe to import from any node without cycles.
"""

from __future__ import annotations

import ast
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


_DEF_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def locate_node_span(source: str, qualname: str) -> tuple[int, int, int] | None:
    """Locate the function/method/class named by the dotted `qualname`
    (e.g. "MyClass.method") and return (start_offset, end_offset, indent) of its
    full source span — including any decorators — over complete lines, or None.

    Semantic addressing for AST-anchored edits: the model names a node instead
    of copying verbatim text, so there is no whitespace/line-drift anchoring to
    hallucinate. Returns None if the source does not parse, the name is not
    found, or it is ambiguous (defined more than once)."""
    if not qualname:
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    matches: list[ast.AST] = []

    def walk(node: ast.AST, stack: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _DEF_NODES):
                child_stack = stack + [child.name]
                if ".".join(child_stack) == qualname:
                    matches.append(child)
                walk(child, child_stack)
            else:
                walk(child, stack)

    walk(tree, [])
    if len(matches) != 1:
        return None
    target = matches[0]

    start_line = target.lineno
    decorators = getattr(target, "decorator_list", [])
    if decorators:
        start_line = min(start_line, min(d.lineno for d in decorators))
    end_line = target.end_lineno or start_line

    lines = source.splitlines(keepends=True)
    start_off = sum(len(line) for line in lines[: start_line - 1])
    end_off = sum(len(line) for line in lines[:end_line])
    return (start_off, end_off, target.col_offset)


def _sole_def_name(code: str) -> str | None:
    """If `code` (dedented) parses to exactly one top-level function/class def,
    return its name, else None. Used to detect a whole-definition replacement."""
    import textwrap

    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return None
    body = [n for n in tree.body if not isinstance(n, ast.Expr)]
    if len(body) == 1 and isinstance(body[0], _DEF_NODES):
        return body[0].name
    return None


def _node_line_count(source: str, qualname: str) -> int:
    """Line count of the located node (0 if not uniquely found)."""
    if not qualname:
        return 0
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0
    found: list[ast.AST] = []

    def walk(node: ast.AST, stack: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _DEF_NODES):
                cs = stack + [child.name]
                if ".".join(cs) == qualname:
                    found.append(child)
                walk(child, cs)
            else:
                walk(child, stack)

    walk(tree, [])
    if len(found) != 1:
        return 0
    n = found[0]
    return (n.end_lineno or n.lineno) - n.lineno + 1


def _nonblank_line_count(text: str) -> int:
    return sum(1 for ln in text.split("\n") if ln.strip())


# Replace must be at least this fraction of the located function's real size
# before we upgrade search->node_target. A much smaller replace means the model
# intended a PARTIAL edit (its search was the def line + a few lines); upgrading
# would replace the whole function with those few lines and silently delete the
# rest. The size gate is the safety core of the converter.
NODE_UPGRADE_MIN_SIZE_RATIO = 0.6


def try_upgrade_to_node_target(
    content: str, search: str, replace: str
) -> str | None:
    """If a search/replace edit is really a whole-definition rewrite, return the
    qualname to re-anchor it via AST (node_target); else None (keep search).

    Fires only when ALL hold, so it never truncates a partial edit:
      1. `replace` is exactly one function/class def named N.
      2. `search` is also a single def named N (same intent, whole def).
      3. N resolves to a unique node in `content`.
      4. `replace` is >= 60% the size of that real node (not a stub that would
         delete the body).
    """
    replace_name = _sole_def_name(replace)
    if replace_name is None:
        return None
    if _sole_def_name(search) != replace_name:
        return None
    # Resolve N to a unique qualname in the file (top-level or nested method).
    qualname = _resolve_unique_qualname(content, replace_name)
    if qualname is None:
        return None
    real_len = _node_line_count(content, qualname)
    if real_len <= 0:
        return None
    if _nonblank_line_count(replace) < NODE_UPGRADE_MIN_SIZE_RATIO * real_len:
        return None
    return qualname


def _resolve_unique_qualname(source: str, name: str) -> str | None:
    """Find the single def/class whose *last* name component is `name`, return
    its dotted qualname, or None if absent or ambiguous."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    hits: list[str] = []

    def walk(node: ast.AST, stack: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _DEF_NODES):
                cs = stack + [child.name]
                if child.name == name:
                    hits.append(".".join(cs))
                walk(child, cs)
            else:
                walk(child, stack)

    walk(tree, [])
    return hits[0] if len(hits) == 1 else None



def diagnose_node_upgrade(content: str, search: str, replace: str) -> str:
    """Human-readable reason the converter declined — diagnostic only."""
    rn = _sole_def_name(replace)
    if rn is None:
        return "replace_not_single_def"
    sn = _sole_def_name(search)
    if sn != rn:
        return f"search_name={sn!r}!=replace_name={rn!r}"
    q = _resolve_unique_qualname(content, rn)
    if q is None:
        return f"name_{rn!r}_absent_or_ambiguous"
    real = _node_line_count(content, q)
    rep = _nonblank_line_count(replace)
    ratio = (rep / real) if real else 0.0
    return (
        f"size_gate replace={rep} real={real} ratio={ratio:.2f} "
        f"(need>={NODE_UPGRADE_MIN_SIZE_RATIO})"
    )
