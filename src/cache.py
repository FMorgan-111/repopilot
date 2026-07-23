"""File cache with fresh reads and bounded stale-on-error fallback.

Usage::

    from .cache import cached

    @cached
    async def read_issue(owner, repo, issue_number):
        ...

Cache keys are derived from the function name, its arguments, and an anonymous
or SHA-256-fingerprinted GitHub credential partition.
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
import inspect
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


def _typed_cache_value(value: object) -> object:
    value_type = type(value)
    if value is None:
        return ["none"]
    if value_type is bool:
        return ["bool", value]
    if value_type is int:
        return ["int", str(value)]
    if value_type is float:
        return ["float", value.hex()]
    if value_type is str:
        return ["str", value]
    if value_type is bytes:
        return ["bytes", value.hex()]
    if value_type is list:
        return ["list", [_typed_cache_value(item) for item in value]]
    if value_type is tuple:
        return ["tuple", [_typed_cache_value(item) for item in value]]
    if value_type in {set, frozenset}:
        items = [_typed_cache_value(item) for item in value]
        items.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
        return [value_type.__name__, items]
    if value_type is dict:
        items = [
            [_typed_cache_value(key), _typed_cache_value(item)]
            for key, item in value.items()
        ]
        items.sort(key=lambda item: json.dumps(item[0], sort_keys=True, separators=(",", ":")))
        return ["dict", items]
    if isinstance(value, Path):
        return [
            "path",
            f"{value_type.__module__}.{value_type.__qualname__}",
            str(value),
        ]
    raise TypeError(
        "unsupported cache key value: "
        f"{value_type.__module__}.{value_type.__qualname__}"
    )


def _cache_key(func: Callable, *args, **kwargs) -> str:
    """Derive a canonical key from function identity and bound typed arguments."""
    if not callable(func):
        raise TypeError("cache key function must be callable")
    target = inspect.unwrap(func)
    signature = inspect.signature(target)
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    payload = json.dumps(
        {
            "module": target.__module__,
            "qualname": target.__qualname__,
            "arguments": [
                [name, _typed_cache_value(value)]
                for name, value in bound.arguments.items()
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    call_digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"{_github_credential_partition()}-{call_digest}"


def _github_credential_partition() -> str:
    token = os.getenv("GITHUB_TOKEN", "")
    if not token:
        return "github-anonymous"
    fingerprint = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return f"github-auth-{fingerprint}"


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
            key = _cache_key(inner, *args, **kwargs)
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
