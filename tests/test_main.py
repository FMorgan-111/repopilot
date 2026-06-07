from fastapi.testclient import TestClient

from src import main


def test_post_analyze_normal_request(monkeypatch):
    async def fake_analyze_issue(issue_url):
        return {
            "trace_id": "abc123def456",
            "issue": {"title": "Login crash", "state": "open", "labels": ["bug"]},
            "classification": {"type": "bug", "severity": "high", "confidence": 0.9},
            "files": [],
            "fix_plan": "Fix it.",
            "risk_level": "low",
            "test_suggestions": [],
        }

    monkeypatch.setattr(main, "analyze_issue", fake_analyze_issue)
    client = TestClient(main.app)

    response = client.post(
        "/analyze",
        json={"issue_url": "https://github.com/acme/widget/issues/42"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["trace_id"] == "abc123def456"
    assert data["issue"]["title"] == "Login crash"


def test_post_analyze_invalid_url(monkeypatch):
    async def fake_analyze_issue(issue_url):
        return {"error": "Invalid GitHub issue URL: nope", "trace_id": "abc123def456"}

    monkeypatch.setattr(main, "analyze_issue", fake_analyze_issue)
    client = TestClient(main.app)

    response = client.post("/analyze", json={"issue_url": "nope"})

    assert response.status_code == 400
    assert response.json() == {
        "status": "error",
        "error": "Invalid GitHub issue URL: nope",
        "trace_id": "abc123def456",
    }


def test_get_health_returns_ok():
    client = TestClient(main.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_post_agent_routes_to_agent_loop(monkeypatch):
    async def fake_agent_analyze(issue_url, max_turns=10):
        return {
            "done": True,
            "summary": "Login bug",
            "files": ["src/auth.py"],
            "fix_plan": "Patch auth",
            "trace_id": "abc123def456",
            "turns": 2,
        }

    monkeypatch.setattr(main, "agent_analyze", fake_agent_analyze)
    client = TestClient(main.app)

    response = client.post(
        "/agent",
        json={"issue_url": "https://github.com/acme/widget/issues/42"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["done"] is True
    assert data["fix_plan"] == "Patch auth"
    assert data["turns"] == 2


def test_post_agent_invalid_url_returns_error(monkeypatch):
    async def fake_agent_analyze(issue_url, max_turns=10):
        return {"error": "Invalid GitHub issue URL: nope", "trace_id": "abc123def456"}

    monkeypatch.setattr(main, "agent_analyze", fake_agent_analyze)
    client = TestClient(main.app)

    response = client.post("/agent", json={"issue_url": "nope"})

    assert response.status_code == 400
    assert response.json() == {
        "status": "error",
        "error": "Invalid GitHub issue URL: nope",
        "trace_id": "abc123def456",
    }


def test_post_agent_v2_routes_to_new_agent(monkeypatch):
    async def fake_agent_v2(issue_url, max_retries=3, token_budget=50000):
        return {
            "done": True,
            "success": True,
            "final_phase": "DONE",
            "issue_url": issue_url,
            "fix_applied": True,
            "pr_url": "https://github.com/acme/widget/pull/42",
            "turns_taken": 6,
            "token_used": 1234,
            "error": None,
        }

    monkeypatch.setattr(main, "agent_v2", fake_agent_v2)
    client = TestClient(main.app)

    response = client.post(
        "/agent/v2",
        json={
            "issue_url": "https://github.com/acme/widget/issues/42",
            "max_retries": 2,
            "token_budget": 5000,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["final_phase"] == "DONE"
    assert data["pr_url"] == "https://github.com/acme/widget/pull/42"
