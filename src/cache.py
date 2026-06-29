"""Simple file-cache for GitHub API responses with TTL-based expiration.

Usage::

    from .cache import cached

    @cached
    async def read_issue(owner, repo, issue_number):
        ...

Cache keys are derived from the function name and its arguments (MD5).
Cache entries live under ``~/.repopilot/cache/`` and expire after
*CACHE_TTL* seconds (default 600 s = 10 min).

Environment variables
---------------------
REPOPILOT_DISABLE_CACHE=1   Skip the cache entirely (read-through only).
REPOPILOT_CACHE_TTL=<secs>  Override the default TTL.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from functools import wraps
from pathlib import Path

CACHE_TTL = int(os.getenv("REPOPILOT_CACHE_TTL", "600"))


def _repopilot_home() -> Path:
    configured = os.getenv("REPOPILOT_HOME")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".repopilot"


def cache_dir() -> Path:
    return _repopilot_home() / "cache"


def _ensure_dir() -> None:
    cache_dir().mkdir(parents=True, exist_ok=True)


def _cache_key(func_name: str, *args, **kwargs) -> str:
    """Derive a deterministic, filesystem-safe key from call arguments."""
    payload = json.dumps(
        {"func": func_name, "args": args, "kwargs": kwargs},
        sort_keys=True,
        default=str,
    )
    return hashlib.md5(payload.encode()).hexdigest()


def _cache_path(key: str) -> Path:
    return cache_dir() / f"{key}.json"


def _load(key: str) -> dict | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if time.time() - data.get("ts", 0) > CACHE_TTL:
        path.unlink(missing_ok=True)
        return None
    return data.get("value")


def _save(key: str, value: object) -> None:
    _ensure_dir()
    _cache_path(key).write_text(
        json.dumps({"ts": time.time(), "value": value}, default=str),
        encoding="utf-8",
    )


def cached(func):
    """Decorator: cache async function results with TTL.

    Skipped when ``REPOPILOT_DISABLE_CACHE`` is truthy.
    """
    if os.getenv("REPOPILOT_DISABLE_CACHE"):
        return func

    @wraps(func)
    async def wrapper(*args, **kwargs):
        key = _cache_key(func.__name__, *args, **kwargs)
        hit = _load(key)
        if hit is not None:
            return hit
        result = await func(*args, **kwargs)
        try:
            _save(key, result)
        except OSError:
            pass
        return result

    return wrapper
