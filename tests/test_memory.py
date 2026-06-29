"""Tests for RepoStore — per-repo SQLite strategy memory."""

import asyncio
import tempfile

import pytest

from src.memory import RepoStore, _fire_and_forget, close_store, get_store


@pytest.fixture
def store():
    """Return a RepoStore backed by a temp directory, cleaned up after test."""
    with tempfile.TemporaryDirectory() as tmp:
        s = RepoStore(base_path=tmp)
        yield s
        # Explicit close to avoid ResourceWarnings.
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(s.close())
        else:
            loop.run_until_complete(s.close())


@pytest.mark.asyncio
async def test_file_index_store_and_retrieve(store):
    """record_file persists; get_file_index returns the path."""
    await store.record_file("alice", "demo", "src/main.py")
    await store.record_file("alice", "demo", "src/utils.py")
    await store.record_file("alice", "demo", "tests/test_main.py")

    index = await store.get_file_index("alice", "demo")
    paths = [r["path"] for r in index]

    assert "src/main.py" in paths
    assert "src/utils.py" in paths
    assert "tests/test_main.py" in paths


@pytest.mark.asyncio
async def test_file_index_increments_fix_count(store):
    """Recording the same file twice increments fix_count."""
    await store.record_file("alice", "demo", "src/main.py")
    await store.record_file("alice", "demo", "src/main.py")

    index = await store.get_file_index("alice", "demo")
    main = next(r for r in index if r["path"] == "src/main.py")
    assert main["fix_count"] == 2


@pytest.mark.asyncio
async def test_issue_log_store_and_retrieve(store):
    """record_issue persists; get_issue_history returns entries latest-first."""
    await store.record_issue("alice", "demo", 1, success=True)
    await store.record_issue("alice", "demo", 2, success=False)
    await store.record_issue("alice", "demo", 3, success=True)

    history = await store.get_issue_history("alice", "demo")
    assert len(history) == 3
    assert history[0]["issue_number"] == 3  # latest first
    assert history[0]["success"] is True
    assert history[1]["issue_number"] == 2
    assert history[1]["success"] is False
    assert history[2]["issue_number"] == 1
    assert history[2]["success"] is True


@pytest.mark.asyncio
async def test_get_file_index_empty_repo(store):
    """An unrecorded repo returns an empty list."""
    index = await store.get_file_index("nobody", "norepo")
    assert index == []


@pytest.mark.asyncio
async def test_get_issue_history_empty_repo(store):
    """An unrecorded repo returns an empty list."""
    history = await store.get_issue_history("nobody", "norepo")
    assert history == []


@pytest.mark.asyncio
async def test_close_store_closes_and_resets_singleton(tmp_path, monkeypatch):
    """close_store closes cached connections and resets the singleton."""
    await close_store()
    monkeypatch.setenv("HOME", str(tmp_path))

    old_store = get_store()
    await old_store.record_issue("alice", "demo", 42, success=True)

    try:
        await close_store()

        assert old_store._db_cache == {}
        assert get_store() is not old_store
    finally:
        await close_store()


@pytest.mark.asyncio
async def test_close_store_waits_for_pending_background_writes(monkeypatch):
    """close_store drains fire-and-forget memory writes before closing."""
    from src.memory import repo_store

    await close_store()
    events: list[str] = []

    class FakeStore:
        async def close(self):
            events.append("close")

    async def background_write():
        await asyncio.sleep(0.01)
        events.append("background")

    monkeypatch.setattr(repo_store, "_store", FakeStore())

    _fire_and_forget(background_write())

    try:
        await close_store()
        assert events == ["background", "close"]
    finally:
        await close_store()
