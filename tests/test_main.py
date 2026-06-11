import asyncio
import json

import pytest
from httpx import ASGITransport, AsyncClient

from src import main
from src.state import AgentState


@pytest.fixture
def api_client():
    transport = ASGITransport(app=main.app)
    client = AsyncClient(transport=transport, base_url="http://testserver")
    yield client
    asyncio.run(client.aclose())


async def test_post_analyze_normal_request(monkeypatch, api_client):
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
    response = await api_client.post(
        "/analyze",
        json={"issue_url": "https://github.com/acme/widget/issues/42"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["trace_id"] == "abc123def456"
    assert data["issue"]["title"] == "Login crash"


async def test_post_analyze_invalid_url(monkeypatch, api_client):
    async def fake_analyze_issue(issue_url):
        return {"error": "Invalid GitHub issue URL: nope", "trace_id": "abc123def456"}

    monkeypatch.setattr(main, "analyze_issue", fake_analyze_issue)
    response = await api_client.post("/analyze", json={"issue_url": "nope"})

    assert response.status_code == 400
    assert response.json() == {
        "status": "error",
        "error": "Invalid GitHub issue URL: nope",
        "trace_id": "abc123def456",
    }


async def test_get_health_returns_ok(api_client):
    response = await api_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_post_agent_routes_to_agent_loop(monkeypatch, api_client):
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
    response = await api_client.post(
        "/agent",
        json={"issue_url": "https://github.com/acme/widget/issues/42"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["done"] is True
    assert data["fix_plan"] == "Patch auth"
    assert data["turns"] == 2


async def test_post_agent_invalid_url_returns_error(monkeypatch, api_client):
    async def fake_agent_analyze(issue_url, max_turns=10):
        return {"error": "Invalid GitHub issue URL: nope", "trace_id": "abc123def456"}

    monkeypatch.setattr(main, "agent_analyze", fake_agent_analyze)
    response = await api_client.post("/agent", json={"issue_url": "nope"})

    assert response.status_code == 400
    assert response.json() == {
        "status": "error",
        "error": "Invalid GitHub issue URL: nope",
        "trace_id": "abc123def456",
    }


async def test_post_agent_v2_routes_to_new_agent(monkeypatch, api_client):
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
    response = await api_client.post(
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


async def test_post_agent_v2_resume_routes_to_resume_agent(monkeypatch, api_client):
    calls = []

    async def fake_resume_agent_v2(run_id, human_answer):
        calls.append({"run_id": run_id, "human_answer": human_answer})
        return {
            "done": True,
            "success": True,
            "waiting_for_user": False,
            "final_phase": "DONE",
            "run_id": run_id,
            "trace_id": run_id,
            "human_input_request": {},
            "error": None,
        }

    monkeypatch.setattr(main, "resume_agent_v2", fake_resume_agent_v2)
    response = await api_client.post(
        "/agent/v2/resume",
        json={
            "run_id": "abc123def456",
            "human_answer": "Breaking changes are not allowed.",
        },
    )

    assert response.status_code == 200
    assert calls == [
        {
            "run_id": "abc123def456",
            "human_answer": "Breaking changes are not allowed.",
        }
    ]
    data = response.json()
    assert data["success"] is True
    assert data["final_phase"] == "DONE"
    assert data["run_id"] == "abc123def456"


async def test_post_agent_v2_resume_rejects_non_paused_run(monkeypatch, api_client):
    async def fake_resume_agent_v2(run_id, human_answer):
        return {
            "done": True,
            "success": False,
            "waiting_for_user": False,
            "final_phase": "DONE",
            "run_id": run_id,
            "trace_id": run_id,
            "human_input_request": {},
            "error": f"Run {run_id} is not waiting for user input.",
        }

    monkeypatch.setattr(main, "resume_agent_v2", fake_resume_agent_v2)

    response = await api_client.post(
        "/agent/v2/resume",
        json={
            "run_id": "abc123def456",
            "human_answer": "Breaking changes are not allowed.",
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "status": "error",
        "done": True,
        "success": False,
        "waiting_for_user": False,
        "final_phase": "DONE",
        "run_id": "abc123def456",
        "trace_id": "abc123def456",
        "human_input_request": {},
        "error": "Run abc123def456 is not waiting for user input.",
    }


async def test_post_agent_v2_resume_returns_404_for_missing_saved_run(
    monkeypatch, api_client
):
    async def fake_resume_agent_v2(run_id, human_answer):
        raise FileNotFoundError(f"No saved run file for {run_id}")

    monkeypatch.setattr(main, "resume_agent_v2", fake_resume_agent_v2)

    response = await api_client.post(
        "/agent/v2/resume",
        json={
            "run_id": "missing-run",
            "human_answer": "Continue without extra constraints.",
        },
    )

    assert response.status_code == 404
    assert response.json() == {
        "status": "error",
        "success": False,
        "run_id": "missing-run",
        "error": "Saved run missing-run was not found.",
    }


async def test_post_agent_v2_resume_returns_500_for_corrupt_saved_run_json(
    monkeypatch, api_client
):
    async def fake_resume_agent_v2(run_id, human_answer):
        raise json.JSONDecodeError("Expecting value", "not-json", 0)

    monkeypatch.setattr(main, "resume_agent_v2", fake_resume_agent_v2)

    response = await api_client.post(
        "/agent/v2/resume",
        json={
            "run_id": "corrupt-run",
            "human_answer": "Continue without extra constraints.",
        },
    )

    assert response.status_code == 500
    assert response.json() == {
        "status": "error",
        "success": False,
        "run_id": "corrupt-run",
        "error": "Saved run corrupt-run could not be loaded.",
    }


async def test_post_agent_v2_resume_returns_500_for_invalid_saved_run_state(
    monkeypatch, api_client
):
    async def fake_resume_agent_v2(run_id, human_answer):
        AgentState.model_validate(
            {"issue_url": "https://github.com/acme/widget/issues/42", "current_phase": "BAD"}
        )

    monkeypatch.setattr(main, "resume_agent_v2", fake_resume_agent_v2)

    response = await api_client.post(
        "/agent/v2/resume",
        json={
            "run_id": "invalid-run",
            "human_answer": "Continue without extra constraints.",
        },
    )

    assert response.status_code == 500
    assert response.json() == {
        "status": "error",
        "success": False,
        "run_id": "invalid-run",
        "error": "Saved run invalid-run could not be loaded.",
    }


async def test_get_agent_v2_replay_returns_json(monkeypatch, api_client):
    calls = []

    def fake_replay_run(run_id):
        calls.append(run_id)
        return {
            "run_id": run_id,
            "issue_url": "https://github.com/acme/widget/issues/7",
            "current_phase": "WAITING_FOR_USER",
            "pause": {"pending_human_input": True, "question": "Confirm API risk."},
            "timeline": [
                {
                    "index": 1,
                    "type": "decision_frame",
                    "frame_id": "df_0001",
                    "stage": "plan",
                    "summary": "Need approval.",
                }
            ],
        }

    monkeypatch.setattr(main, "replay_run", fake_replay_run)

    response = await api_client.get("/agent/v2/runs/abc123def456/replay")

    assert response.status_code == 200
    assert calls == ["abc123def456"]
    assert response.json() == {
        "run_id": "abc123def456",
        "issue_url": "https://github.com/acme/widget/issues/7",
        "current_phase": "WAITING_FOR_USER",
        "pause": {"pending_human_input": True, "question": "Confirm API risk."},
        "timeline": [
            {
                "index": 1,
                "type": "decision_frame",
                "frame_id": "df_0001",
                "stage": "plan",
                "summary": "Need approval.",
            }
        ],
    }


async def test_get_agent_v2_replay_returns_markdown(monkeypatch, api_client):
    replay = {
        "run_id": "abc123def456",
        "timeline": [],
    }

    monkeypatch.setattr(main, "replay_run", lambda run_id: replay)
    monkeypatch.setattr(
        main,
        "format_replay_markdown",
        lambda replay_data: f"# Replay {replay_data['run_id']}",
    )

    response = await api_client.get(
        "/agent/v2/runs/abc123def456/replay",
        params={"format": "markdown"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert response.text == "# Replay abc123def456"


async def test_get_agent_v2_replay_returns_404_for_missing_saved_run(
    monkeypatch, api_client
):
    def fake_replay_run(run_id):
        raise FileNotFoundError(f"No saved run file for {run_id}")

    monkeypatch.setattr(main, "replay_run", fake_replay_run)

    response = await api_client.get("/agent/v2/runs/missing-run/replay")

    assert response.status_code == 404
    assert response.json() == {
        "status": "error",
        "success": False,
        "run_id": "missing-run",
        "error": "Saved run missing-run was not found.",
    }


async def test_get_agent_v2_replay_returns_500_for_corrupt_saved_run(
    monkeypatch, api_client
):
    def fake_replay_run(run_id):
        raise json.JSONDecodeError("Expecting value", "not-json", 0)

    monkeypatch.setattr(main, "replay_run", fake_replay_run)

    response = await api_client.get("/agent/v2/runs/corrupt-run/replay")

    assert response.status_code == 500
    assert response.json() == {
        "status": "error",
        "success": False,
        "run_id": "corrupt-run",
        "error": "Saved run corrupt-run could not be loaded.",
    }
