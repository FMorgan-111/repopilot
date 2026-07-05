"""Tests for the locate_code fallback (B): recover file location when the
primary issue-keyword search returns nothing."""

from src import new_agent
from src.nodes import locate as locate_node


class EmptyMemoryStore:
    async def get_file_index(self, owner, repo, limit=8):
        return []


def _state(title, body):
    return new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/9",
        owner="acme",
        repo="widget",
        issue_title=title,
        issue_body=body,
        current_phase=new_agent.Phase.LOCATE,
    )


# ---- pure helpers -----------------------------------------------------------

def test_issue_fallback_paths_extracts_repo_paths():
    state = _state(
        "crash in handler",
        'Traceback:\n  File "src/app/router.py", line 5\nsee also docs/guide.rst',
    )
    paths = locate_node._issue_fallback_paths(state)
    assert "src/app/router.py" in paths
    assert "docs/guide.rst" not in paths  # docs excluded


def test_issue_fallback_terms_prefers_specific_identifiers():
    state = _state(
        "RequestHandler breaks",
        "the dispatch_request method in module foo.bar.baz raises when called",
    )
    terms = locate_node._issue_fallback_terms(state, exclude=set())
    assert "dispatch_request" in terms  # snake_case
    assert "foo.bar.baz" in terms       # dotted module
    assert "RequestHandler" in terms    # CamelCase


# ---- fallback triggers when primary search is empty -------------------------

async def test_locate_fallback_reads_issue_path_when_search_empty(monkeypatch):
    read_paths = []

    async def fake_search_code(query, owner, repo):
        return []  # primary search + fallback search both find nothing

    async def fake_read_file(owner, repo, path):
        read_paths.append(path)
        return {"content": f"# {path}\ndef handle(): ...\n", "sha": f"sha-{path}"}

    monkeypatch.setenv("REPOPILOT_DISABLE_PARALLEL", "1")
    monkeypatch.setattr(locate_node, "get_store", lambda: EmptyMemoryStore())
    monkeypatch.setattr(locate_node, "search_code", fake_search_code)
    monkeypatch.setattr(locate_node, "read_file", fake_read_file)

    state = _state(
        "handler crashes",
        'Traceback (most recent call last):\n  File "src/app/router.py", line 5, in handle',
    )
    result = await locate_node.locate_code(state)

    assert "src/app/router.py" in read_paths
    assert result.current_phase == new_agent.Phase.PLAN
    assert any(f.path == "src/app/router.py" for f in result.relevant_files)


async def test_locate_fallback_searches_identifiers_when_no_paths(monkeypatch):
    async def fake_search_code(query, owner, repo):
        # primary keyword search misses; fallback identifier search hits.
        if query == "dispatch_request":
            return [{"path": "src/app/core.py", "sha": "s"}]
        return []

    async def fake_read_file(owner, repo, path):
        return {"content": "def dispatch_request(): ...\n", "sha": "r"}

    monkeypatch.setenv("REPOPILOT_DISABLE_PARALLEL", "1")
    monkeypatch.setattr(locate_node, "get_store", lambda: EmptyMemoryStore())
    monkeypatch.setattr(locate_node, "search_code", fake_search_code)
    monkeypatch.setattr(locate_node, "read_file", fake_read_file)

    state = _state("routing bug", "the dispatch_request method misbehaves")
    result = await locate_node.locate_code(state)

    assert result.current_phase == new_agent.Phase.PLAN
    assert any(f.path == "src/app/core.py" for f in result.relevant_files)


async def test_locate_still_fails_when_fallback_finds_nothing(monkeypatch):
    async def fake_search_code(query, owner, repo):
        return []

    async def fake_read_file(owner, repo, path):
        raise RuntimeError("404")

    monkeypatch.setenv("REPOPILOT_DISABLE_PARALLEL", "1")
    monkeypatch.setattr(locate_node, "get_store", lambda: EmptyMemoryStore())
    monkeypatch.setattr(locate_node, "search_code", fake_search_code)
    monkeypatch.setattr(locate_node, "read_file", fake_read_file)

    state = _state("vague bug", "it just does not work")
    result = await locate_node.locate_code(state)

    assert result.current_phase == new_agent.Phase.FAILURE
    assert "No relevant files" in result.failure_reason
