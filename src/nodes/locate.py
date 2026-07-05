"""LOCATE phase: Search code and read the most relevant files."""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

from ..memory import get_store
from ..retrieval import bm25_rerank, bm25_scores
from ..state import (
    AgentState,
    DecisionFrame,
    FileInfo,
    Phase,
    _as_state,
    _estimate_tokens,
    _is_budget_exceeded,
    _issue_search_terms,
    _rank_reason,
    _record_tool,
)
from ..tools import read_file, search_code

# Matches repo-relative source paths like "src/tox/tox_env/api.py" embedded in
# free-text next_checks (e.g. "Read src/tox/tox_env/runner.py to find ...").
_PATH_RE = re.compile(r"[A-Za-z0-9_./-]+\.[A-Za-z0-9_]+")

# Specific-looking identifiers (dotted modules, snake_case, CamelCase) used as
# fallback search terms when the primary issue-keyword search finds nothing.
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{3,}")

# Documentation / prose files. They match issue keywords densely (and are huge),
# so BM25 ranks them above source code — but a patch_edits + pytest agent fixes
# CODE, never prose. Excluding them keeps the planner's top-N slots on real
# source. (A docs-only bug is out of scope for this agent.)
_DOC_SUFFIXES = (".rst", ".md", ".txt", ".rdoc", ".adoc")


def _is_doc_file(path: str) -> bool:
    lowered = path.lower()
    return lowered.endswith(_DOC_SUFFIXES) or lowered.startswith("docs/") or "/docs/" in lowered


def _latest_plan_frame(state: AgentState) -> DecisionFrame | None:
    for frame in reversed(state.frame_history):
        if frame.stage == "plan":
            return frame
    return None


def _frame_candidate_paths(state: AgentState) -> list[str]:
    """Pull concrete file paths the planner asked us to inspect.

    Two sources, in priority order:
      1. The structured ``files`` array the planner stored in trace_notes.
      2. File-path-shaped tokens extracted from free-text next_checks.
    Returns a de-duplicated, order-preserving list.
    """
    frame = _latest_plan_frame(state)
    if frame is None:
        return []

    paths: list[str] = []

    try:
        notes = json.loads(frame.trace_notes) if frame.trace_notes else {}
    except (json.JSONDecodeError, TypeError):
        notes = {}
    if isinstance(notes, dict):
        for path in notes.get("files", []) or []:
            if isinstance(path, str) and path and path not in paths:
                paths.append(path)

    for check in frame.next_checks:
        for match in _PATH_RE.findall(check):
            if "/" in match and match not in paths:
                paths.append(match)

    return paths


def _issue_fallback_paths(state: AgentState) -> list[str]:
    """Repo-relative paths named verbatim in the issue/traceback text."""
    text = f"{state.issue_title}\n{state.issue_body}"
    out: list[str] = []
    for match in _PATH_RE.findall(text):
        if "/" in match and not _is_doc_file(match) and match not in out:
            out.append(match)
    return out[:6]


def _issue_fallback_terms(state: AgentState, exclude: set[str]) -> list[str]:
    """Specific identifiers (dotted/snake_case/CamelCase) from the issue text,
    used as extra code-search terms when the primary search returns nothing."""
    text = f"{state.issue_title} {state.issue_body[:1500]}"
    out: list[str] = []
    for match in _IDENT_RE.findall(text):
        if match in exclude or match in out:
            continue
        is_specific = (
            "." in match
            or "_" in match
            or (match[0].isupper() and any(c.islower() for c in match))
        )
        if is_specific:
            out.append(match)
        if len(out) >= 5:
            break
    return out


async def _locate_fallback(state: AgentState) -> list[FileInfo]:
    """Last-ditch location when the normal search found nothing: read file paths
    named directly in the issue (tracebacks list them) and search for specific
    identifiers pulled from the issue text. Bounded to a few reads."""
    candidates: dict[str, FileInfo] = {}
    for path in _issue_fallback_paths(state):
        candidates.setdefault(
            path,
            FileInfo(
                path=path,
                relevance_score=0.85,
                reason="path named directly in issue text",
                sha="",
            ),
        )

    exclude = set(_issue_search_terms(state.issue_title, state.issue_body))
    for term in _issue_fallback_terms(state, exclude):
        try:
            results = await search_code(term, state.owner, state.repo)
        except Exception:
            continue
        for result in results:
            path = result.get("path", "")
            if not path or path in candidates or _is_doc_file(path):
                continue
            score, reason = _rank_reason(path, state.issue_title, state.issue_body)
            candidates[path] = FileInfo(
                path=path,
                relevance_score=score,
                reason=f"fallback search '{term}': {reason}",
                sha=result.get("sha", ""),
            )

    ranked = sorted(
        candidates.values(), key=lambda item: item.relevance_score, reverse=True
    )[:4]
    hydrated: list[FileInfo] = []
    for info in ranked:
        try:
            file_data = await read_file(state.owner, state.repo, info.path)
        except Exception:
            continue
        info.content = file_data.get("content", "")
        info.sha = file_data.get("sha", info.sha)
        if info.content:
            hydrated.append(info)
    return hydrated


async def locate_code(state: AgentState | dict[str, Any]) -> AgentState:
    """Search code and read the most relevant files into working memory."""
    import sys
    state = _as_state(state)
    if _is_budget_exceeded(state):
        state.failure_reason = "Token budget exceeded before code location."
        state.current_phase = Phase.FAILURE
        return state

    by_path: dict[str, FileInfo] = {}

    # ── carry forward context found in earlier collect_more_context rounds ──
    # locate is otherwise stateless: it rebuilds candidates from scratch every
    # round, so good files found earlier (e.g. the real env-identity sources)
    # are discarded the moment a later round's requested paths fail to resolve,
    # starving the planner at the exact moment it must decide. Keep every file
    # already hydrated with content; new candidates are added on top.
    carried: dict[str, FileInfo] = {
        f.path: f for f in state.relevant_files if f.content and not _is_doc_file(f.path)
    }
    for path, info in carried.items():
        by_path[path] = info

    # ── memory-aided location: pull historically-modified files first ──
    store = get_store()
    try:
        memory_files = await store.get_file_index(state.owner, state.repo, limit=8)
        for mf in memory_files:
            path = mf["path"]
            if path in by_path:
                continue
            by_path[path] = FileInfo(
                path=path,
                relevance_score=0.75,  # moderately high — proven fix location
                reason=(
                    f"from memory (fixed {mf['fix_count']} time(s), "
                    f"last {mf.get('last_used', 'unknown')})"
                ),
                sha="",
            )
    except Exception:
        pass  # memory lookup is best-effort; fall through to API search

    # ── planner-directed location: read files the latest plan frame named ──
    # When the planner recommends collect_more_context it lists concrete paths
    # in next_checks / trace_notes. Seed those as high-relevance candidates so
    # each round actually pulls *new* context instead of re-searching the issue
    # text and getting the same files back.
    for path in _frame_candidate_paths(state):
        if path in by_path:
            continue
        by_path[path] = FileInfo(
            path=path,
            relevance_score=0.9,  # planner explicitly asked for this file
            reason="requested by planner next_checks",
            sha="",
        )

    terms = _issue_search_terms(state.issue_title, state.issue_body)
    print(f"  [locate] Search terms: {terms}", file=sys.stderr, flush=True)
    parallel = not os.getenv("REPOPILOT_DISABLE_PARALLEL")

    # ── search code for every term in parallel (or serial) ──
    if parallel:
        search_tasks = [
            search_code(term, state.owner, state.repo) for term in terms
        ]
        all_search_results = await asyncio.gather(
            *search_tasks, return_exceptions=True
        )
    else:
        all_search_results = []
        for term in terms:
            try:
                all_search_results.append(
                    await search_code(term, state.owner, state.repo)
                )
            except Exception as exc:
                all_search_results.append(exc)

    for term, results in zip(terms, all_search_results):
        if isinstance(results, Exception):
            _record_tool(
                state,
                "search_code",
                {"query": term, "owner": state.owner, "repo": state.repo},
                error=str(results),
            )
            continue
        _record_tool(
            state,
            "search_code",
            {"query": term, "owner": state.owner, "repo": state.repo},
            {"count": len(results)},
        )
        for result in results:
            path = result.get("path", "")
            if not path or path in by_path or _is_doc_file(path):
                continue
            score, reason = _rank_reason(path, state.issue_title, state.issue_body)
            by_path[path] = FileInfo(
                path=path,
                relevance_score=score,
                reason=reason,
                sha=result.get("sha", ""),
            )

    # Rank only candidates that still need reading; carried files already have
    # content and are preserved unconditionally below.
    ranked = sorted(
        (info for info in by_path.values() if not info.content),
        key=lambda item: item.relevance_score,
        reverse=True,
    )[:6]
    print(f"  [locate] Found {len(by_path)} candidate files, top {len(ranked)} ranked", file=sys.stderr, flush=True)
    # Start from files carried over from earlier rounds so good context is never
    # lost; newly-read files are appended.
    hydrated: list[FileInfo] = list(carried.values())

    # ── read top files in parallel (or serial) ──
    if parallel:
        read_tasks = [
            read_file(state.owner, state.repo, info.path) for info in ranked
        ]
        read_results = await asyncio.gather(
            *read_tasks, return_exceptions=True
        )
    else:
        read_results = []
        for info in ranked:
            try:
                read_results.append(
                    await read_file(state.owner, state.repo, info.path)
                )
            except Exception as exc:
                read_results.append(exc)

    newly_hydrated: list[FileInfo] = []
    for info, file_data in zip(ranked, read_results):
        if isinstance(file_data, Exception):
            _record_tool(
                state,
                "read_file",
                {"owner": state.owner, "repo": state.repo, "path": info.path},
                error=str(file_data),
            )
            continue
        info.content = file_data.get("content", "")
        info.sha = file_data.get("sha", info.sha)
        hydrated.append(info)
        newly_hydrated.append(info)
        _record_tool(
            state,
            "read_file",
            {"owner": state.owner, "repo": state.repo, "path": info.path},
            {"size": len(info.content), "sha": info.sha},
        )

    if hydrated:
        bm25_query = f"{state.issue_title}\n{state.issue_body}"
        scores = bm25_scores(bm25_query, hydrated)
        score_by_path = {score.path: score for score in scores}
        applied = any(score.bm25_score > 0 for score in scores)
        if applied:
            hydrated = bm25_rerank(bm25_query, hydrated)
        _record_tool(
            state,
            "bm25_rerank",
            {"query": bm25_query, "candidate_count": len(hydrated)},
            {
                "applied": applied,
                "ranked": [
                    {
                        "path": file.path,
                        "bm25_score": round(
                            score_by_path[file.path].bm25_score,
                            6,
                        ),
                        "normalized_score": round(
                            score_by_path[file.path].normalized_score,
                            6,
                        ),
                        "relevance_score": file.relevance_score,
                        "matched_terms": score_by_path[file.path].matched_terms,
                    }
                    for file in hydrated
                ],
            },
        )

    state.relevant_files = hydrated
    # Charge only newly-read files; carried files were already counted in the
    # round that first read them.
    state.token_usage += _estimate_tokens(
        state.issue_title,
        state.issue_body,
        "\n".join(f"{f.path}\n{f.content[:2000]}" for f in newly_hydrated),
    )

    # ── fallback location: if the normal search located nothing, mine the issue
    # text directly (paths in tracebacks + specific identifiers) before giving
    # up. Attacks the "No relevant files could be located" failure mode. ──
    if not hydrated:
        fallback = await _locate_fallback(state)
        if fallback:
            hydrated = fallback
            state.relevant_files = hydrated
            state.token_usage += _estimate_tokens(
                state.issue_title,
                state.issue_body,
                "\n".join(f"{f.path}\n{f.content[:2000]}" for f in hydrated),
            )
            print(
                f"  [locate] fallback located {len(hydrated)} file(s)",
                file=sys.stderr,
                flush=True,
            )

    # ── no-progress guard: if this round located the exact same files as the
    # previous one, collecting more context is futile — stop early instead of
    # looping PLAN↔LOCATE until the token budget runs out. ──
    signature = "|".join(sorted(f.path for f in hydrated))
    if hydrated and signature == state.last_locate_signature:
        state.current_phase = Phase.FAILURE
        state.failure_reason = (
            "Context collection made no progress (located the same files again)."
        )
        return state
    state.last_locate_signature = signature

    state.current_phase = Phase.PLAN if hydrated else Phase.FAILURE
    if not hydrated:
        state.failure_reason = "No relevant files could be located or read."
    return state
