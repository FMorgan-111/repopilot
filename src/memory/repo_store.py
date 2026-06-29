"""Per-repo SQLite memory: file index + issue history."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import aiosqlite

logger = logging.getLogger("repopilot.memory")

_DDL_FILE_INDEX = """
CREATE TABLE IF NOT EXISTS file_index (
    owner TEXT NOT NULL,
    repo  TEXT NOT NULL,
    path  TEXT NOT NULL,
    fix_count  INTEGER DEFAULT 1,
    last_used  TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (owner, repo, path)
)
"""

_DDL_ISSUE_LOG = """
CREATE TABLE IF NOT EXISTS issue_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    owner        TEXT NOT NULL,
    repo         TEXT NOT NULL,
    issue_number INTEGER NOT NULL,
    success      INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT DEFAULT (datetime('now'))
)
"""


def _db_path(base: Path, owner: str, repo: str) -> Path:
    """Return the per-repo SQLite file path."""
    db_dir = base / owner / repo
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "memory.db"


class RepoStore:
    """Per-repo SQLite memory: file index + issue history."""

    def __init__(self, base_path: str = "~/.repopilot/memory"):
        self._base = Path(base_path).expanduser()
        self._db_cache: dict[tuple[str, str], aiosqlite.Connection] = {}

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    async def _get_conn(self, owner: str, repo: str) -> aiosqlite.Connection:
        key = (owner, repo)
        if key not in self._db_cache:
            db_path = _db_path(self._base, owner, repo)
            conn = await aiosqlite.connect(str(db_path))
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute(_DDL_FILE_INDEX)
            await conn.execute(_DDL_ISSUE_LOG)
            await conn.commit()
            self._db_cache[key] = conn
        return self._db_cache[key]

    # ------------------------------------------------------------------
    # public API — file index
    # ------------------------------------------------------------------

    async def get_file_index(
        self, owner: str, repo: str, limit: int = 20
    ) -> list[dict]:
        """Return historically-modified files, ordered by fix_count desc."""
        conn = await self._get_conn(owner, repo)
        cursor = await conn.execute(
            "SELECT path, fix_count, last_used FROM file_index "
            "WHERE owner=? AND repo=? "
            "ORDER BY fix_count DESC, last_used DESC "
            "LIMIT ?",
            (owner, repo, limit),
        )
        rows = await cursor.fetchall()
        return [
            {"path": r[0], "fix_count": r[1], "last_used": r[2]} for r in rows
        ]

    async def record_file(self, owner: str, repo: str, path: str) -> None:
        """Atomically upsert and increment fix_count."""
        conn = await self._get_conn(owner, repo)
        await conn.execute(
            "INSERT INTO file_index (owner, repo, path, fix_count, last_used) "
            "VALUES (?, ?, ?, 1, datetime('now')) "
            "ON CONFLICT(owner, repo, path) DO UPDATE SET "
            "  fix_count = fix_count + 1, "
            "  last_used = datetime('now')",
            (owner, repo, path),
        )
        await conn.commit()

    # ------------------------------------------------------------------
    # public API — issue log
    # ------------------------------------------------------------------

    async def record_issue(
        self, owner: str, repo: str, issue_number: int, success: bool
    ) -> None:
        """Log an issue processing result."""
        conn = await self._get_conn(owner, repo)
        await conn.execute(
            "INSERT INTO issue_log (owner, repo, issue_number, success) "
            "VALUES (?, ?, ?, ?)",
            (owner, repo, issue_number, 1 if success else 0),
        )
        await conn.commit()

    async def get_issue_history(
        self, owner: str, repo: str, limit: int = 10
    ) -> list[dict]:
        """Return recently-processed issues."""
        conn = await self._get_conn(owner, repo)
        cursor = await conn.execute(
            "SELECT issue_number, success, created_at FROM issue_log "
            "WHERE owner=? AND repo=? "
            "ORDER BY created_at DESC, id DESC "
            "LIMIT ?",
            (owner, repo, limit),
        )
        rows = await cursor.fetchall()
        return [
            {"issue_number": r[0], "success": bool(r[1]), "created_at": r[2]}
            for r in rows
        ]

    # ------------------------------------------------------------------
    # cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close all cached connections."""
        for conn in self._db_cache.values():
            await conn.close()
        self._db_cache.clear()


# Module-level singleton (optional convenience).
_store: RepoStore | None = None
_pending_background_tasks: set[asyncio.Task] = set()


def get_store() -> RepoStore:
    """Return a shared RepoStore singleton."""
    global _store
    if _store is None:
        _store = RepoStore()
    return _store


async def close_store() -> None:
    """Close and reset the shared RepoStore singleton."""
    global _store
    await _drain_background_tasks()
    if _store is None:
        return

    store = _store
    try:
        await store.close()
    finally:
        _store = None


async def _drain_background_tasks() -> None:
    """Wait for pending fire-and-forget writes before cleanup closes DBs."""
    current_task = asyncio.current_task()
    while True:
        pending = [
            task
            for task in _pending_background_tasks
            if not task.done() and task is not current_task
        ]
        if not pending:
            return
        await asyncio.gather(*pending, return_exceptions=True)


def _fire_and_forget(coro):
    """Schedule a coroutine in the background; log warnings on failure."""

    async def _wrapper():
        try:
            await coro
        except Exception:
            logger.warning(
                "background memory write failed", exc_info=True
            )

    try:
        task = asyncio.ensure_future(_wrapper())
        _pending_background_tasks.add(task)
        task.add_done_callback(_pending_background_tasks.discard)
    except RuntimeError:
        # No event loop running — this is fine in test contexts.
        pass
