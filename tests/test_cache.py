from __future__ import annotations

import importlib
import logging
import sys

import pytest


@pytest.mark.asyncio
async def test_cached_writes_under_configurable_repopilot_home(
    monkeypatch, tmp_path
):
    repopilot_home = tmp_path / "custom-home"
    monkeypatch.setenv("REPOPILOT_HOME", str(repopilot_home))
    sys.modules.pop("src.cache", None)
    cache = importlib.import_module("src.cache")

    calls = {"count": 0}

    @cache.cached
    async def fetch_value(value: str) -> dict[str, str]:
        calls["count"] += 1
        return {"value": value}

    result = await fetch_value("alpha")

    expected_path = repopilot_home / "cache" / f"{cache._cache_key('fetch_value', 'alpha')}.json"
    assert result == {"value": "alpha"}
    assert calls["count"] == 1
    assert expected_path.exists()
    assert expected_path.is_file()
    assert cache._cache_path(cache._cache_key("fetch_value", "alpha")) == expected_path


@pytest.mark.asyncio
async def test_cached_returns_result_when_save_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("REPOPILOT_HOME", str(tmp_path / "home"))
    sys.modules.pop("src.cache", None)
    cache = importlib.import_module("src.cache")

    async def fetch_value() -> dict[str, str]:
        return {"value": "fresh"}

    def fail_save(key: str, value: object) -> None:
        raise PermissionError("cache directory is read-only")

    monkeypatch.setattr(cache, "_save", fail_save)

    cached_fetch = cache.cached(fetch_value)

    assert await cached_fetch() == {"value": "fresh"}


@pytest.mark.asyncio
async def test_cached_returns_stale_value_on_retryable_refresh_error(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.setenv("REPOPILOT_HOME", str(tmp_path))
    sys.modules.pop("src.cache", None)
    cache = importlib.import_module("src.cache")
    monkeypatch.setattr(cache, "CACHE_TTL", 10)
    monkeypatch.setattr(cache, "CACHE_STALE_TTL", 100, raising=False)
    now = {"value": 50.0}
    monkeypatch.setattr(cache.time, "time", lambda: now["value"])
    calls = 0

    @cache.cached(stale_if_error=lambda exc: isinstance(exc, ConnectionError))
    async def fetch():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ConnectionError("offline")
        return {"value": "cached"}

    caplog.set_level(logging.INFO, logger="repopilot.cache")
    assert await fetch() == {"value": "cached"}
    now["value"] = 65.0
    assert await fetch() == {"value": "cached"}
    assert calls == 2
    assert "stale_fallback" in caplog.text


@pytest.mark.asyncio
async def test_cached_does_not_hide_non_retryable_error(monkeypatch, tmp_path):
    monkeypatch.setenv("REPOPILOT_HOME", str(tmp_path))
    sys.modules.pop("src.cache", None)
    cache = importlib.import_module("src.cache")
    monkeypatch.setattr(cache, "CACHE_TTL", 10)
    monkeypatch.setattr(cache, "CACHE_STALE_TTL", 100, raising=False)
    now = {"value": 0.0}
    monkeypatch.setattr(cache.time, "time", lambda: now["value"])

    @cache.cached(stale_if_error=lambda exc: isinstance(exc, ConnectionError))
    async def fetch():
        if now["value"]:
            raise ValueError("bad schema")
        return {"value": "cached"}

    await fetch()
    now["value"] = 20.0
    with pytest.raises(ValueError, match="bad schema"):
        await fetch()


@pytest.mark.asyncio
async def test_cached_removes_value_beyond_stale_window(monkeypatch, tmp_path):
    monkeypatch.setenv("REPOPILOT_HOME", str(tmp_path))
    sys.modules.pop("src.cache", None)
    cache = importlib.import_module("src.cache")
    monkeypatch.setattr(cache, "CACHE_TTL", 10)
    monkeypatch.setattr(cache, "CACHE_STALE_TTL", 30, raising=False)
    now = {"value": 0.0}
    monkeypatch.setattr(cache.time, "time", lambda: now["value"])
    calls = 0

    @cache.cached(stale_if_error=lambda exc: True)
    async def fetch():
        nonlocal calls
        calls += 1
        return {"call": calls}

    assert await fetch() == {"call": 1}
    now["value"] = 31.0
    assert await fetch() == {"call": 2}


@pytest.mark.asyncio
async def test_expired_cache_unlink_failure_does_not_break_refresh(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("REPOPILOT_HOME", str(tmp_path))
    sys.modules.pop("src.cache", None)
    cache = importlib.import_module("src.cache")
    monkeypatch.setattr(cache, "CACHE_TTL", 10)
    monkeypatch.setattr(cache, "CACHE_STALE_TTL", 30, raising=False)
    now = {"value": 0.0}
    monkeypatch.setattr(cache.time, "time", lambda: now["value"])
    calls = 0

    @cache.cached
    async def fetch():
        nonlocal calls
        calls += 1
        return {"call": calls}

    assert await fetch() == {"call": 1}
    now["value"] = 31.0

    def fail_unlink(*args, **kwargs):
        raise PermissionError("read-only cache")

    monkeypatch.setattr(type(cache._cache_path("unused")), "unlink", fail_unlink)
    assert await fetch() == {"call": 2}


@pytest.mark.asyncio
async def test_disabled_cache_calls_function_each_time(monkeypatch, tmp_path):
    monkeypatch.setenv("REPOPILOT_HOME", str(tmp_path))
    monkeypatch.setenv("REPOPILOT_DISABLE_CACHE", "1")
    sys.modules.pop("src.cache", None)
    cache = importlib.import_module("src.cache")
    calls = 0

    @cache.cached
    async def fetch():
        nonlocal calls
        calls += 1
        return {"call": calls}

    assert await fetch() == {"call": 1}
    assert await fetch() == {"call": 2}


@pytest.mark.asyncio
async def test_cache_logs_metadata_without_cached_value(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.setenv("REPOPILOT_HOME", str(tmp_path))
    sys.modules.pop("src.cache", None)
    cache = importlib.import_module("src.cache")
    monkeypatch.setattr(cache, "CACHE_TTL", 10)
    monkeypatch.setattr(cache, "CACHE_STALE_TTL", 100, raising=False)
    now = {"value": 0.0}
    monkeypatch.setattr(cache.time, "time", lambda: now["value"])

    @cache.cached(stale_if_error=lambda exc: True)
    async def fetch():
        if now["value"]:
            raise ConnectionError("offline")
        return {"token": "sentinel-secret-token"}

    caplog.set_level(logging.INFO, logger="repopilot.cache")
    await fetch()
    now["value"] = 20.0
    await fetch()

    assert "stale_fallback" in caplog.text
    assert "fetch" in caplog.text
    assert "sentinel-secret-token" not in caplog.text


@pytest.mark.asyncio
async def test_github_cache_same_token_hits_same_partition(monkeypatch, tmp_path):
    monkeypatch.setenv("REPOPILOT_HOME", str(tmp_path))
    monkeypatch.setenv("GITHUB_TOKEN", "token-same-sentinel")
    sys.modules.pop("src.cache", None)
    cache = importlib.import_module("src.cache")
    calls = 0

    @cache.cached
    async def fetch_issue():
        nonlocal calls
        calls += 1
        return {"number": 7}

    assert await fetch_issue() == {"number": 7}
    assert await fetch_issue() == {"number": 7}
    assert calls == 1


@pytest.mark.asyncio
async def test_github_cache_token_rotation_misses_private_partition(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("REPOPILOT_HOME", str(tmp_path))
    monkeypatch.setenv("GITHUB_TOKEN", "token-first-sentinel")
    sys.modules.pop("src.cache", None)
    cache = importlib.import_module("src.cache")
    calls = 0

    @cache.cached
    async def fetch_issue():
        nonlocal calls
        calls += 1
        return {"call": calls}

    assert await fetch_issue() == {"call": 1}
    monkeypatch.setenv("GITHUB_TOKEN", "token-second-sentinel")
    assert await fetch_issue() == {"call": 2}
    assert await fetch_issue() == {"call": 2}
    assert calls == 2


@pytest.mark.asyncio
async def test_github_cache_anonymous_partition_is_separate(monkeypatch, tmp_path):
    monkeypatch.setenv("REPOPILOT_HOME", str(tmp_path))
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    sys.modules.pop("src.cache", None)
    cache = importlib.import_module("src.cache")
    calls = 0

    @cache.cached
    async def fetch_issue():
        nonlocal calls
        calls += 1
        return {"call": calls}

    assert await fetch_issue() == {"call": 1}
    monkeypatch.setenv("GITHUB_TOKEN", "token-authenticated-sentinel")
    assert await fetch_issue() == {"call": 2}
    assert calls == 2


@pytest.mark.asyncio
async def test_github_cache_never_persists_or_logs_raw_token(
    monkeypatch, tmp_path, caplog
):
    raw_token = "github-token-never-persist-sentinel"
    monkeypatch.setenv("REPOPILOT_HOME", str(tmp_path))
    monkeypatch.setenv("GITHUB_TOKEN", raw_token)
    sys.modules.pop("src.cache", None)
    cache = importlib.import_module("src.cache")

    @cache.cached
    async def fetch_issue():
        return {"number": 7}

    caplog.set_level(logging.INFO, logger="repopilot.cache")
    await fetch_issue()
    await fetch_issue()

    key = cache._cache_key("fetch_issue")
    files = list(cache.cache_dir().glob("*.json"))
    assert key.startswith("github-auth-")
    assert raw_token not in key
    assert files
    assert all(raw_token not in path.name for path in files)
    assert all(raw_token not in path.read_text(encoding="utf-8") for path in files)
    assert raw_token not in caplog.text
