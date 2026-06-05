import pytest

from src import agent


async def test_analyze_issue_normal_flow_returns_complete_result(monkeypatch):
    async def fake_read_issue(owner, repo, issue_number):
        return {
            "title": "Login crash",
            "body": "Crashes after submit.",
            "state": "open",
            "labels": ["bug"],
            "number": issue_number,
        }

    async def fake_classify_issue(title, body):
        return {"type": "bug", "severity": "high", "confidence": 0.91}

    async def fake_search_code(query, owner, repo):
        return [{"path": "src/auth.py", "repository": f"{owner}/{repo}", "url": "", "sha": "1"}]

    async def fake_rank_files(title, body, files):
        return [{**files[0], "relevance_score": 0.97, "reason": "Handles login."}]

    async def fake_generate_fix_plan(title, body, classification, ranked_files):
        return {
            "fix_plan": "Fix auth submit path.",
            "risk_level": "medium",
            "test_suggestions": ["Add login regression test."],
        }

    monkeypatch.setattr(agent, "read_issue", fake_read_issue)
    monkeypatch.setattr(agent, "classify_issue", fake_classify_issue)
    monkeypatch.setattr(agent, "search_code", fake_search_code)
    monkeypatch.setattr(agent, "rank_files", fake_rank_files)
    monkeypatch.setattr(agent, "generate_fix_plan", fake_generate_fix_plan)

    result = await agent.analyze_issue("https://github.com/acme/widget/issues/42")

    assert result["trace_id"]
    assert result["issue"] == {
        "title": "Login crash",
        "state": "open",
        "labels": ["bug"],
    }
    assert result["classification"] == {"type": "bug", "severity": "high", "confidence": 0.91}
    assert result["files"][0]["path"] == "src/auth.py"
    assert result["fix_plan"] == "Fix auth submit path."
    assert result["risk_level"] == "medium"
    assert result["test_suggestions"] == ["Add login regression test."]


async def test_analyze_issue_invalid_url_returns_error():
    result = await agent.analyze_issue("https://example.com/acme/widget/issues/42")

    assert "error" in result
    assert "Invalid GitHub issue URL" in result["error"]
    assert result["trace_id"]


async def test_analyze_issue_search_failure_still_returns_classification(monkeypatch):
    async def fake_read_issue(owner, repo, issue_number):
        return {
            "title": "Login crash",
            "body": "Crashes after submit.",
            "state": "open",
            "labels": ["bug"],
            "number": issue_number,
        }

    async def fake_classify_issue(title, body):
        return {"type": "bug", "severity": "high", "confidence": 0.88}

    async def failing_search_code(query, owner, repo):
        raise RuntimeError("GitHub search unavailable")

    async def fake_generate_fix_plan(title, body, classification, ranked_files):
        return {
            "fix_plan": "Inspect auth flow manually.",
            "risk_level": "low",
            "test_suggestions": [],
        }

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("rank_files should not be called without files")

    monkeypatch.setattr(agent, "read_issue", fake_read_issue)
    monkeypatch.setattr(agent, "classify_issue", fake_classify_issue)
    monkeypatch.setattr(agent, "search_code", failing_search_code)
    monkeypatch.setattr(agent, "rank_files", fail_if_called)
    monkeypatch.setattr(agent, "generate_fix_plan", fake_generate_fix_plan)

    result = await agent.analyze_issue("https://github.com/acme/widget/issues/42")

    assert "error" not in result
    assert result["classification"] == {"type": "bug", "severity": "high", "confidence": 0.88}
    assert result["files"] == []
    assert result["fix_plan"] == "Inspect auth flow manually."


async def test_analyze_issue_trace_id_non_empty(monkeypatch):
    async def fake_read_issue(owner, repo, issue_number):
        return {"title": "Docs typo", "body": "Typo", "state": "open", "labels": [], "number": issue_number}

    async def fake_classify_issue(title, body):
        return {"type": "docs", "severity": "low", "confidence": 0.8}

    async def fake_search_code(query, owner, repo):
        return []

    async def fake_generate_fix_plan(title, body, classification, ranked_files):
        return {"fix_plan": "Fix typo.", "risk_level": "low", "test_suggestions": []}

    monkeypatch.setattr(agent, "read_issue", fake_read_issue)
    monkeypatch.setattr(agent, "classify_issue", fake_classify_issue)
    monkeypatch.setattr(agent, "search_code", fake_search_code)
    monkeypatch.setattr(agent, "generate_fix_plan", fake_generate_fix_plan)

    result = await agent.analyze_issue("https://github.com/acme/widget/issues/7")

    assert isinstance(result["trace_id"], str)
    assert len(result["trace_id"]) == 12
