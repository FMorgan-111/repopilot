"""Tests for RepoStore — per-repo SQLite strategy memory."""

import tempfile
from pathlib import Path

import pytest

from src.memory import RepoStore


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
