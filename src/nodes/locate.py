"""LOCATE phase: Search code and read the most relevant files."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from ..memory import get_store
from ..state import (
    AgentState,
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


async def locate_code(state: AgentState | dict[str, Any]) -> AgentState:
    """Search code and read the most relevant files into working memory."""
    state = _as_state(state)
    if _is_budget_exceeded(state):
        state.failure_reason = "Token budget exceeded before code location."
        state.current_phase = Phase.FAILURE
        return state

    by_path: dict[str, FileInfo] = {}

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

    terms = _issue_search_terms(state.issue_title, state.issue_body)
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
            if not path or path in by_path:
                continue
            score, reason = _rank_reason(path, state.issue_title, state.issue_body)
            by_path[path] = FileInfo(
                path=path,
                relevance_score=score,
                reason=reason,
                sha=result.get("sha", ""),
            )

    ranked = sorted(
        by_path.values(), key=lambda item: item.relevance_score, reverse=True
    )[:6]
    hydrated: list[FileInfo] = []

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
        _record_tool(
            state,
            "read_file",
            {"owner": state.owner, "repo": state.repo, "path": info.path},
            {"size": len(info.content), "sha": info.sha},
        )

    state.relevant_files = hydrated
    state.token_usage += _estimate_tokens(
        state.issue_title,
        state.issue_body,
        "\n".join(f"{f.path}\n{f.content[:2000]}" for f in hydrated),
    )
    state.current_phase = Phase.PLAN if hydrated else Phase.FAILURE
    if not hydrated:
        state.failure_reason = "No relevant files could be located or read."
    return state
