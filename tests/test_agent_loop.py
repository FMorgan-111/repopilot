import pytest

from src import agent_loop


def queue_llm(monkeypatch, responses):
    """Make agent_loop.llm_call return each queued decision in turn."""
    calls = []
    it = iter(responses)

    async def fake_llm_call(system, user, model="deepseek-v4-flash"):
        calls.append({"system": system, "user": user})
        return next(it)

    monkeypatch.setattr(agent_loop, "llm_call", fake_llm_call)
    return calls


async def test_agent_loop_calls_tools_then_finishes(monkeypatch):
    calls = queue_llm(
        monkeypatch,
        [
            {"tool": "read_issue", "args": {}},
            {"tool": "search_code", "args": {"query": "login"}},
            {"done": True, "summary": "Login bug", "files": ["src/auth.py"], "fix_plan": "Patch auth"},
        ],
    )

    async def fake_read_issue(owner, repo, n):
        return {"title": "Login crash", "body": "boom", "state": "open", "labels": [], "number": n}

    async def fake_search_code(query, owner, repo):
        return [{"path": "src/auth.py", "repository": f"{owner}/{repo}", "url": "", "sha": "1"}]

    monkeypatch.setattr(agent_loop, "read_issue", fake_read_issue)
    monkeypatch.setattr(agent_loop, "search_code", fake_search_code)

    result = await agent_loop.agent_analyze("https://github.com/acme/widget/issues/42")

    assert result["done"] is True
    assert result["summary"] == "Login bug"
    assert result["files"] == ["src/auth.py"]
    assert result["fix_plan"] == "Patch auth"
    assert len(result["trace_id"]) == 12
    assert result["turns"] == 3
    # The transcript grows: each turn sees the prior tool results.
    assert "Tool result" in calls[2]["user"]


async def test_agent_loop_invalid_url_returns_error():
    result = await agent_loop.agent_analyze("https://example.com/acme/widget/issues/42")

    assert "error" in result
    assert "Invalid GitHub issue URL" in result["error"]
    assert result["trace_id"]


async def test_agent_loop_max_turns_reached(monkeypatch):
    async def always_tool(system, user, model="deepseek-v4-flash"):
        return {"tool": "read_issue", "args": {}}

    async def fake_read_issue(owner, repo, n):
        return {"title": "x", "body": "y", "state": "open", "labels": [], "number": n}

    monkeypatch.setattr(agent_loop, "llm_call", always_tool)
    monkeypatch.setattr(agent_loop, "read_issue", fake_read_issue)

    result = await agent_loop.agent_analyze("https://github.com/acme/widget/issues/1", max_turns=3)

    assert result["done"] is True
    assert result["error"] == "Max turns reached"
    assert result["turns"] == 3


async def test_agent_loop_tool_error_is_captured_and_loop_continues(monkeypatch):
    queue_llm(
        monkeypatch,
        [
            {"tool": "search_code", "args": {"query": "x"}},
            {"done": True, "summary": "done anyway"},
        ],
    )

    async def failing_search(query, owner, repo):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(agent_loop, "search_code", failing_search)

    result = await agent_loop.agent_analyze("https://github.com/acme/widget/issues/1")

    assert result["done"] is True
    assert result["summary"] == "done anyway"


async def test_agent_loop_handles_response_without_tool_or_done(monkeypatch):
    queue_llm(
        monkeypatch,
        [
            {"thinking": "hmm"},  # malformed: neither tool nor done
            {"done": True, "summary": "recovered"},
        ],
    )

    result = await agent_loop.agent_analyze("https://github.com/acme/widget/issues/1")

    assert result["done"] is True
    assert result["summary"] == "recovered"


async def test_agent_loop_llm_failure_returns_error(monkeypatch):
    async def boom(system, user, model="deepseek-v4-flash"):
        raise RuntimeError("network down")

    monkeypatch.setattr(agent_loop, "llm_call", boom)

    result = await agent_loop.agent_analyze("https://github.com/acme/widget/issues/1")

    assert result["done"] is True
    assert "LLM call failed" in result["error"]


async def test_execute_tool_dispatches_read_file(monkeypatch):
    async def fake_read_file(owner, repo, path):
        return {"path": path, "content": "code", "sha": "s", "size": 4}

    monkeypatch.setattr(agent_loop, "read_file", fake_read_file)

    result = await agent_loop.execute_tool("read_file", {"path": "src/x.py"}, "acme", "widget", 1)

    assert result["content"] == "code"
    assert result["path"] == "src/x.py"


async def test_execute_tool_unknown_tool_raises():
    with pytest.raises(ValueError, match="Unknown tool"):
        await agent_loop.execute_tool("nope", {}, "acme", "widget", 1)
