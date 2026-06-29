from __future__ import annotations

import importlib
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
