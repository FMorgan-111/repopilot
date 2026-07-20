"""File cache with fresh reads and bounded stale-on-error fallback.

Usage::

    from .cache import cached

    @cached
    async def read_issue(owner, repo, issue_number):
        ...

Cache keys are derived from the function name and its arguments (MD5).
Cache entries live under ``~/.repopilot/cache/``. They are fresh for
*CACHE_TTL* seconds (default 600 s = 10 min) and may be used as stale fallback
until *CACHE_STALE_TTL* seconds (default 86,400 s = 24 h).

Environment variables
---------------------
REPOPILOT_DISABLE_CACHE=1   Skip the cache entirely (read-through only).
REPOPILOT_CACHE_TTL=<secs>  Override the default TTL.
REPOPILOT_CACHE_STALE_TTL=<secs>  Override the maximum stale age.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Callable, Literal

CACHE_TTL = int(os.getenv("REPOPILOT_CACHE_TTL", "600"))
CACHE_STALE_TTL = int(os.getenv("REPOPILOT_CACHE_STALE_TTL", "86400"))
logger = logging.getLogger("repopilot.cache")


@dataclass(frozen=True)
class CacheEntry:
    value: object
    age_seconds: float
    state: Literal["fresh", "stale"]


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


def _load(key: str) -> CacheEntry | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    age = max(0.0, time.time() - data.get("ts", 0))
    if age > CACHE_STALE_TTL:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    state = "fresh" if age <= CACHE_TTL else "stale"
    return CacheEntry(data.get("value"), age, state)


def _save(key: str, value: object) -> None:
    _ensure_dir()
    _cache_path(key).write_text(
        json.dumps({"ts": time.time(), "value": value}, default=str),
        encoding="utf-8",
    )


def _log_cache_event(
    event: str,
    func_name: str,
    key: str,
    age_seconds: float | None = None,
) -> None:
    fields: dict[str, object] = {
        "event": event,
        "function": func_name,
        "key": key[:12],
    }
    if age_seconds is not None:
        fields["age_seconds"] = round(age_seconds, 3)
    logger.info("cache %s", json.dumps(fields, sort_keys=True))


def cached(
    func=None,
    *,
    stale_if_error: Callable[[BaseException], bool] | None = None,
):
    """Cache an async function, optionally falling back to bounded stale data.

    Skipped when ``REPOPILOT_DISABLE_CACHE`` is truthy.
    """

    def decorate(inner):
        if os.getenv("REPOPILOT_DISABLE_CACHE"):
            return inner

        @wraps(inner)
        async def wrapper(*args, **kwargs):
            key = _cache_key(inner.__name__, *args, **kwargs)
            entry = _load(key)
            if entry is not None and entry.state == "fresh":
                _log_cache_event(
                    "fresh_hit", inner.__name__, key, entry.age_seconds
                )
                return entry.value

            try:
                result = await inner(*args, **kwargs)
            except Exception as exc:
                if (
                    entry is not None
                    and entry.state == "stale"
                    and stale_if_error is not None
                    and stale_if_error(exc)
                ):
                    _log_cache_event(
                        "stale_fallback",
                        inner.__name__,
                        key,
                        entry.age_seconds,
                    )
                    return entry.value
                raise

            try:
                _save(key, result)
            except OSError:
                _log_cache_event("save_failed", inner.__name__, key)
            return result

        return wrapper

    if func is None:
        return decorate
    return decorate(func)
