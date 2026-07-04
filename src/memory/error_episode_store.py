"""Global, cross-repo error-episode store for semantic recall.

Unlike ``repo_store`` (one SQLite file per owner/repo), this is a single global
``episodes.db``: fix experience mined on one project can be recalled when
planning a fix on another. Each episode pairs an issue's semantics (the recall
key, embedded) with its outcome payload (traceback keyframe, the patch that was
tried, and whether it succeeded).

Backed by a synchronous sqlite3 connection (``check_same_thread=False`` + a lock)
so the sqlite-vec extension loads cleanly; async callers use ``arecord`` /
``arecall`` which offload to a thread.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .embedding import EMBED_DIM, Embedder
from .keyframe import extract_keyframe
from .sqlite_vec_index import SqliteVecIndex
from .vector_index import VectorIndex


def _repopilot_home() -> Path:
    configured = os.getenv("REPOPILOT_HOME")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".repopilot"


def default_episodes_db_path() -> Path:
    return _repopilot_home() / "episodes.db"


@dataclass(frozen=True)
class RecalledEpisode:
    owner: str
    repo: str
    issue_title: str
    keyframe: str
    patch: str
    success: bool
    distance: float


class ErrorEpisodeStore:
    def __init__(
        self,
        db_path: str | Path | None = None,
        embedder: Any | None = None,
        index_factory: Any | None = None,
        dim: int = EMBED_DIM,
    ) -> None:
        self.dim = dim
        self.embedder = embedder if embedder is not None else Embedder(dim=dim)
        path = Path(db_path) if db_path is not None else default_episodes_db_path()
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS episodes ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "owner TEXT, repo TEXT, issue_url TEXT, issue_title TEXT, "
            "keyframe TEXT, patch TEXT, success INTEGER, created_at TEXT)"
        )
        factory = index_factory or (lambda conn, d: SqliteVecIndex(conn, d))
        self.index: VectorIndex = factory(self.conn, dim)

    # -- write -----------------------------------------------------------------

    def record(
        self,
        *,
        owner: str,
        repo: str,
        issue_url: str,
        issue_title: str,
        issue_body: str,
        error_log: str,
        patch: str,
        success: bool,
    ) -> int:
        keyframe = extract_keyframe(error_log)
        vector = self.embedder.embed(_recall_text(issue_title, issue_body))
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO episodes"
                "(owner, repo, issue_url, issue_title, keyframe, patch, success, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    owner,
                    repo,
                    issue_url,
                    issue_title,
                    keyframe,
                    patch,
                    1 if success else 0,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            rowid = int(cur.lastrowid)
            self.index.add(rowid, vector)
            self.conn.commit()
        return rowid

    # -- read ------------------------------------------------------------------

    def recall(
        self,
        *,
        issue_title: str,
        issue_body: str,
        k: int = 3,
        exclude_issue_url: str | None = None,
    ) -> list[RecalledEpisode]:
        vector = self.embedder.embed(_recall_text(issue_title, issue_body))
        # Over-fetch so self-episodes can be dropped without shrinking the result.
        fetch = k + 5 if exclude_issue_url else k
        with self._lock:
            hits = self.index.search(vector, fetch)
            if not hits:
                return []
            by_id = {h.rowid: h.distance for h in hits}
            placeholders = ",".join("?" * len(by_id))
            rows = self.conn.execute(
                f"SELECT id, owner, repo, issue_url, issue_title, keyframe, patch, success "
                f"FROM episodes WHERE id IN ({placeholders})",
                tuple(by_id.keys()),
            ).fetchall()
        results = [
            RecalledEpisode(
                owner=r["owner"],
                repo=r["repo"],
                issue_title=r["issue_title"],
                keyframe=r["keyframe"] or "",
                patch=r["patch"] or "",
                success=bool(r["success"]),
                distance=by_id[r["id"]],
            )
            for r in rows
            if not (exclude_issue_url and r["issue_url"] == exclude_issue_url)
        ]
        results.sort(key=lambda e: e.distance)
        return results[:k]

    # -- async wrappers --------------------------------------------------------

    async def arecord(self, **kwargs: Any) -> int:
        import asyncio

        return await asyncio.to_thread(lambda: self.record(**kwargs))

    async def arecall(self, **kwargs: Any) -> list[RecalledEpisode]:
        import asyncio

        return await asyncio.to_thread(lambda: self.recall(**kwargs))

    def close(self) -> None:
        with self._lock:
            self.conn.close()


def _recall_text(issue_title: str, issue_body: str) -> str:
    return f"{issue_title}\n\n{issue_body}".strip()


_STORE: ErrorEpisodeStore | None = None


def get_episode_store() -> ErrorEpisodeStore | None:
    """Process-wide store singleton, or None when episodes are not enabled.

    Cross-repo episodes are OPT-IN via ``REPOPILOT_ENABLE_EPISODES=1`` because
    the first embed downloads the bge-small model from HuggingFace; enabling it
    by default would hang offline/CI environments. Construction is lazy and does
    not load the model (that happens on first embed), so this stays cheap."""
    global _STORE
    if os.getenv("REPOPILOT_ENABLE_EPISODES") != "1":
        return None
    if _STORE is None:
        _STORE = ErrorEpisodeStore()
    return _STORE


def reset_episode_store() -> None:
    """Test hook: drop the cached singleton."""
    global _STORE
    if _STORE is not None:
        _STORE.close()
    _STORE = None
