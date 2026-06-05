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

    assert response.status_code == 200
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
