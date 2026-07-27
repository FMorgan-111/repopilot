import asyncio
import json

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from src import main, new_agent, run_store
from src.async_safety import CancellationDrainError
from src.state import AgentState, Phase

API_TOKEN = "test-api-token"
AUTHORIZED_HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}
ISSUE_URL = "https://github.com/acme/widget/issues/42"


@pytest.fixture(autouse=True)
def api_security_env(monkeypatch):
    monkeypatch.setenv("REPOPILOT_API_TOKEN", API_TOKEN)
    monkeypatch.setenv("REPOPILOT_ALLOWED_REPOS", "acme/widget")


@pytest.fixture
def api_client():
    transport = ASGITransport(app=main.app)
    client = AsyncClient(transport=transport, base_url="http://testserver")
    yield client
    asyncio.run(client.aclose())


def _state(
    *,
    issue_url: str = ISSUE_URL,
    owner: str = "acme",
    repo: str = "widget",
) -> AgentState:
    return AgentState(
        issue_url=issue_url,
        owner=owner,
        repo=repo,
        trace_id="abc123def456",
        current_phase=Phase.WAITING_FOR_USER,
        pending_human_input=True,
        human_input_request={"question": "Proceed?"},
    )


async def test_health_is_public_when_security_configuration_is_missing(
    monkeypatch, api_client
):
    monkeypatch.delenv("REPOPILOT_API_TOKEN", raising=False)
    monkeypatch.delenv("REPOPILOT_ALLOWED_REPOS", raising=False)

    response = await api_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize("configured", [None, "", "   "])
async def test_missing_or_blank_api_token_fails_closed_before_handler(
    monkeypatch, api_client, configured
):
    called = False

    async def forbidden_call(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("authority helper must not run")

    if configured is None:
        monkeypatch.delenv("REPOPILOT_API_TOKEN", raising=False)
    else:
        monkeypatch.setenv("REPOPILOT_API_TOKEN", configured)
    monkeypatch.setattr(main, "agent_v2", forbidden_call)

    response = await api_client.post("/agent/v2", json={"issue_url": ISSUE_URL})

    assert response.status_code == 503
    assert response.json() == {"detail": "Service unavailable."}
    assert called is False


@pytest.mark.parametrize(
    "authorization",
    [None, "", "test-api-token", "Basic test-api-token", "Bearer", "Bearer wrong"],
)
async def test_missing_malformed_and_wrong_authorization_are_identical(
    api_client, authorization
):
    headers = {}
    if authorization is not None:
        headers["Authorization"] = authorization

    response = await api_client.post(
        "/agent/v2",
        headers=headers,
        json={"issue_url": ISSUE_URL},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized."}
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize(
    "allowlist",
    [
        None,
        "",
        "*",
        "https://github.com/acme/widget",
        "acme/widget/path",
        "acme/widget,",
        "acme/widget,ACME/WIDGET",
        " acme/widget",
        "acme /widget",
        "bad_owner/widget",
        "-owner/widget",
        "owner-/widget",
    ],
)
async def test_missing_or_malformed_allowlist_fails_closed_before_agent(
    monkeypatch, api_client, allowlist
):
    called = False

    async def forbidden_call(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("agent must not run")

    if allowlist is None:
        monkeypatch.delenv("REPOPILOT_ALLOWED_REPOS", raising=False)
    else:
        monkeypatch.setenv("REPOPILOT_ALLOWED_REPOS", allowlist)
    monkeypatch.setattr(main, "agent_v2", forbidden_call)

    response = await api_client.post(
        "/agent/v2",
        headers=AUTHORIZED_HEADERS,
        json={"issue_url": ISSUE_URL},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Service unavailable."}
    assert called is False


async def test_unauthorized_repository_rejected_before_agent(monkeypatch, api_client):
    called = False

    async def forbidden_call(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("agent must not run")

    monkeypatch.setattr(main, "agent_v2", forbidden_call)

    response = await api_client.post(
        "/agent/v2",
        headers=AUTHORIZED_HEADERS,
        json={"issue_url": "https://github.com/other/repo/issues/1"},
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "Repository is not authorized."}
    assert called is False


@pytest.mark.parametrize(
    "issue_url",
    [
        "http://github.com/acme/widget/issues/42",
        "https://evil.example/acme/widget/issues/42",
        "https://user@github.com/acme/widget/issues/42",
        "https://github.com:443/acme/widget/issues/42",
        "https://github.com/acme/widget/issues/0",
        "https://github.com/acme/widget/issues/42/extra",
        "https://github.com/acme/widget/issues/42?repo=other",
        "https://github.com/acme/widget/issues/42#fragment",
        "https://github.com/acme/widget/pulls/42",
        "https://github.com/bad_owner/widget/issues/42",
        "https://github.com/-owner/widget/issues/42",
        "https://github.com/owner-/widget/issues/42",
        "https://github.com/ac\nme/widget/issues/42",
        "https://github.com/ac\rme/widget/issues/42",
        "https://github.com/ac\tme/widget/issues/42",
    ],
)
async def test_strict_issue_url_rejection_precedes_agent(
    monkeypatch, api_client, issue_url
):
    called = False

    async def forbidden_call(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("agent must not run")

    monkeypatch.setattr(main, "agent_v2", forbidden_call)

    response = await api_client.post(
        "/agent/v2",
        headers=AUTHORIZED_HEADERS,
        json={"issue_url": issue_url},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid request."}
    assert called is False


async def test_repository_allowlist_match_is_case_insensitive(
    monkeypatch, api_client
):
    calls = []

    async def fake_agent(issue_url, max_retries=3, token_budget=100000):
        calls.append(issue_url)
        return {"success": True, "error": None}

    monkeypatch.setenv("REPOPILOT_ALLOWED_REPOS", "ACME/WIDGET")
    monkeypatch.setattr(main, "agent_v2", fake_agent)

    response = await api_client.post(
        "/agent/v2",
        headers=AUTHORIZED_HEADERS,
        json={"issue_url": ISSUE_URL},
    )

    assert response.status_code == 200
    assert calls == [ISSUE_URL]


async def test_intelligent_agent_never_exceeds_agent_v2_retry_cap(
    monkeypatch, api_client
):
    captured = {}

    async def fake_intelligent(issue_url, *, max_retries, token_budget):
        captured.update(
            issue_url=issue_url,
            max_retries=max_retries,
            token_budget=token_budget,
        )
        return {"success": True, "error": None}

    monkeypatch.setattr(main, "intelligent_analyze_issue", fake_intelligent)

    response = await api_client.post(
        "/intelligent-agent",
        headers=AUTHORIZED_HEADERS,
        json={"issue_url": ISSUE_URL, "max_turns": 10, "token_budget": 100_000},
    )

    assert response.status_code == 200
    assert captured == {
        "issue_url": ISSUE_URL,
        "max_retries": 3,
        "token_budget": 100_000,
    }


async def test_schema_routes_are_not_exposed_even_to_authenticated_clients(api_client):
    for path in ("/openapi.json", "/docs", "/redoc"):
        response = await api_client.get(path, headers=AUTHORIZED_HEADERS)
        assert response.status_code == 404


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("POST", "/analyze", {"issue_url": ISSUE_URL}),
        ("POST", "/agent", {"issue_url": ISSUE_URL}),
        ("POST", "/intelligent-agent", {"issue_url": ISSUE_URL}),
        ("POST", "/agent/v2", {"issue_url": ISSUE_URL}),
        (
            "POST",
            "/agent/v2/resume",
            {"run_id": "abc123def456", "human_answer": "Proceed."},
        ),
        ("GET", "/agent/v2/runs/abc123def456", None),
        ("GET", "/agent/v2/runs/abc123def456/replay", None),
    ],
)
async def test_every_authority_and_data_route_requires_bearer_authentication(
    api_client, method, path, payload
):
    response = await api_client.request(method, path, json=payload)

    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized."}
    assert response.headers["www-authenticate"] == "Bearer"


async def test_duplicate_authorization_headers_are_rejected(api_client):
    response = await api_client.post(
        "/agent/v2",
        headers=[
            ("Authorization", f"Bearer {API_TOKEN}"),
            ("Authorization", f"Bearer {API_TOKEN}"),
        ],
        json={"issue_url": ISSUE_URL},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized."}


async def test_non_ascii_authorization_header_gets_uniform_401(api_client):
    response = await api_client.post(
        "/agent/v2",
        headers=[(b"Authorization", b"Bearer \xff")],
        json={"issue_url": ISSUE_URL},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized."}
    assert response.headers["www-authenticate"] == "Bearer"


def _json_body_of_size(size: int) -> bytes:
    prefix = json.dumps({"issue_url": ISSUE_URL}, separators=(",", ":")).encode()
    assert len(prefix) <= size
    return prefix + (b" " * (size - len(prefix)))


async def test_authentication_precedes_request_body_limit(api_client):
    response = await api_client.post(
        "/agent/v2",
        content=_json_body_of_size(65_537),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized."}


async def test_declared_request_body_limit_accepts_exact_boundary(
    monkeypatch, api_client
):
    calls = []

    async def fake_agent(issue_url, max_retries=3, token_budget=100000):
        calls.append(issue_url)
        return {"success": True, "error": None}

    monkeypatch.setattr(main, "agent_v2", fake_agent)
    response = await api_client.post(
        "/agent/v2",
        content=_json_body_of_size(65_536),
        headers={**AUTHORIZED_HEADERS, "Content-Type": "application/json"},
    )

    assert response.status_code == 200
    assert calls == [ISSUE_URL]


async def test_declared_request_body_limit_rejects_one_over(api_client):
    response = await api_client.post(
        "/agent/v2",
        content=_json_body_of_size(65_537),
        headers={**AUTHORIZED_HEADERS, "Content-Type": "application/json"},
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "Request body too large."}


async def test_chunked_request_body_limit_accepts_exact_boundary(
    monkeypatch, api_client
):
    calls = []

    async def chunks():
        body = _json_body_of_size(65_536)
        yield body[:40_000]
        yield body[40_000:]

    async def fake_agent(issue_url, max_retries=3, token_budget=100000):
        calls.append(issue_url)
        return {"success": True, "error": None}

    monkeypatch.setattr(main, "agent_v2", fake_agent)
    response = await api_client.post(
        "/agent/v2",
        content=chunks(),
        headers={**AUTHORIZED_HEADERS, "Content-Type": "application/json"},
    )

    assert response.status_code == 200
    assert calls == [ISSUE_URL]


async def test_chunked_request_body_limit_rejects_one_over(api_client):
    async def chunks():
        body = _json_body_of_size(65_537)
        yield body[:40_000]
        yield body[40_000:]

    response = await api_client.post(
        "/agent/v2",
        content=chunks(),
        headers={**AUTHORIZED_HEADERS, "Content-Type": "application/json"},
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "Request body too large."}


@pytest.mark.parametrize("content_length", ["-1", "+1", "not-a-number", "1,2"])
async def test_invalid_content_length_fails_closed(api_client, content_length):
    response = await api_client.post(
        "/agent/v2",
        content=_json_body_of_size(1024),
        headers={
            **AUTHORIZED_HEADERS,
            "Content-Type": "application/json",
            "Content-Length": content_length,
        },
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid request."}


async def test_extremely_large_declared_content_length_is_rejected_without_parsing(
    api_client,
):
    response = await api_client.post(
        "/agent/v2",
        content=b"",
        headers={
            **AUTHORIZED_HEADERS,
            "Content-Type": "application/json",
            "Content-Length": "9" * 5_000,
        },
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "Request body too large."}


def test_request_model_string_limits_and_extra_fields():
    assert len(main.AnalyzeRequest(issue_url="x" * 2048).issue_url) == 2048
    with pytest.raises(ValidationError):
        main.AnalyzeRequest(issue_url="x" * 2049)

    assert len(
        main.AgentV2ResumeRequest(run_id="a" * 64, human_answer="h" * 16_384).run_id
    ) == 64
    with pytest.raises(ValidationError):
        main.AgentV2ResumeRequest(run_id="a" * 65, human_answer="ok")
    with pytest.raises(ValidationError):
        main.AgentV2ResumeRequest(run_id="unsafe/run", human_answer="ok")
    with pytest.raises(ValidationError):
        main.AgentV2ResumeRequest(run_id="safe", human_answer="")
    with pytest.raises(ValidationError):
        main.AgentV2ResumeRequest(run_id="safe", human_answer="h" * 16_385)
    with pytest.raises(ValidationError):
        main.AgentV2Request(issue_url=ISSUE_URL, unexpected=True)


@pytest.mark.parametrize(
    ("model", "field", "accepted", "rejected"),
    [
        (main.AgentRequest, "max_turns", (1, 10), (0, 11)),
        (main.IntelligentAgentRequest, "max_turns", (1, 10), (0, 11)),
        (main.AgentV2Request, "max_retries", (0, 3), (-1, 4)),
        (main.IntelligentAgentRequest, "token_budget", (1, 100_000), (0, 100_001)),
        (main.AgentV2Request, "token_budget", (1, 100_000), (0, 100_001)),
    ],
)
def test_request_model_numeric_limits(model, field, accepted, rejected):
    for value in accepted:
        assert getattr(model(issue_url=ISSUE_URL, **{field: value}), field) == value
    for value in rejected:
        with pytest.raises(ValidationError):
            model(issue_url=ISSUE_URL, **{field: value})
    with pytest.raises(ValidationError):
        model(issue_url=ISSUE_URL, **{field: str(accepted[0])})


async def test_validation_errors_are_stable_and_do_not_echo_input(api_client):
    secret_answer = "private-human-answer"
    response = await api_client.post(
        "/agent/v2/resume",
        headers=AUTHORIZED_HEADERS,
        json={"run_id": "safe", "human_answer": secret_answer, "unexpected": True},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid request."}
    assert secret_answer not in response.text


async def test_resume_checks_allowlist_configuration_before_loading_run(
    monkeypatch, api_client
):
    loaded = False

    def forbidden_load(run_id):
        nonlocal loaded
        loaded = True
        raise AssertionError("run must not be loaded")

    monkeypatch.delenv("REPOPILOT_ALLOWED_REPOS", raising=False)
    monkeypatch.setattr(main, "load_run", forbidden_load)

    response = await api_client.post(
        "/agent/v2/resume",
        headers=AUTHORIZED_HEADERS,
        json={"run_id": "abc123def456", "human_answer": "Proceed."},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Service unavailable."}
    assert loaded is False


async def test_resume_authorizes_stored_run_before_invocation(monkeypatch, api_client):
    calls = []
    state = _state()

    async def fake_resume(run_id, human_answer, *, state):
        calls.append((run_id, human_answer, state))
        return {"success": True, "run_id": run_id, "error": None}

    monkeypatch.setattr(main, "load_run", lambda run_id: state)
    monkeypatch.setattr(main, "resume_agent_v2", fake_resume)

    response = await api_client.post(
        "/agent/v2/resume",
        headers=AUTHORIZED_HEADERS,
        json={"run_id": "abc123def456", "human_answer": "Proceed."},
    )

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert calls == [("abc123def456", "Proceed.", state)]


async def test_serial_second_resume_returns_stable_conflict(
    monkeypatch, tmp_path, api_client
):
    root_dir = tmp_path / ".repopilot"
    monkeypatch.setenv("REPOPILOT_HOME", str(root_dir))
    run_store.save_run(_state(), root_dir=root_dir)

    async def fake_run_graph(graph, state):
        state.current_phase = Phase.DONE
        return state

    monkeypatch.setattr(new_agent, "run_graph", fake_run_graph)
    monkeypatch.setattr(new_agent, "_save_trace", lambda *args, **kwargs: None)

    request = {
        "run_id": "abc123def456",
        "human_answer": "Proceed.",
    }
    first = await api_client.post(
        "/agent/v2/resume",
        headers=AUTHORIZED_HEADERS,
        json=request,
    )
    second = await api_client.post(
        "/agent/v2/resume",
        headers=AUTHORIZED_HEADERS,
        json=request,
    )

    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json() == {"detail": "Run resume conflict."}


@pytest.mark.parametrize(
    "state",
    [
        _state(issue_url="https://github.com/secret/private/issues/1", owner="secret", repo="private"),
        _state(owner="other"),
        _state(repo="other"),
        _state(issue_url="not-a-github-issue", owner="", repo=""),
    ],
)
async def test_resume_rejects_unauthorized_or_inconsistent_stored_repository(
    monkeypatch, api_client, state
):
    called = False
    human_answer = "private-human-answer"

    async def forbidden_resume(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("resume must not run")

    monkeypatch.setattr(main, "load_run", lambda run_id: state)
    monkeypatch.setattr(main, "resume_agent_v2", forbidden_resume)

    response = await api_client.post(
        "/agent/v2/resume",
        headers=AUTHORIZED_HEADERS,
        json={"run_id": "abc123def456", "human_answer": human_answer},
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "Repository is not authorized."}
    assert called is False
    assert human_answer not in response.text
    assert "secret/private" not in response.text


@pytest.mark.parametrize(
    ("max_retries", "token_budget"),
    [(-1, 100_000), (4, 100_000), (3, 0), (3, 100_001)],
)
async def test_resume_rejects_saved_state_outside_hard_execution_limits(
    monkeypatch, api_client, max_retries, token_budget
):
    state = _state()
    state.max_retries = max_retries
    state.token_budget = token_budget
    called = False

    async def forbidden_resume(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("invalid saved execution limits must not resume")

    monkeypatch.setattr(main, "load_run", lambda run_id: state)
    monkeypatch.setattr(main, "resume_agent_v2", forbidden_resume)

    response = await api_client.post(
        "/agent/v2/resume",
        headers=AUTHORIZED_HEADERS,
        json={"run_id": "abc123def456", "human_answer": "Proceed."},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Saved run could not be loaded."}
    assert called is False


async def test_resume_rejects_run_id_that_does_not_match_loaded_state(
    monkeypatch, api_client
):
    state = _state()
    state.trace_id = "different-run"
    called = False

    async def forbidden_resume(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("mismatched saved state must not resume")

    monkeypatch.setattr(main, "load_run", lambda run_id: state)
    monkeypatch.setattr(main, "resume_agent_v2", forbidden_resume)

    response = await api_client.post(
        "/agent/v2/resume",
        headers=AUTHORIZED_HEADERS,
        json={"run_id": "abc123def456", "human_answer": "Proceed."},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Saved run could not be loaded."}
    assert called is False


async def test_inspect_returns_safe_summary_only_after_stored_repo_authorization(
    monkeypatch, api_client
):
    monkeypatch.setattr(main, "load_run", lambda run_id: _state())

    response = await api_client.get(
        "/agent/v2/runs/abc123def456",
        headers=AUTHORIZED_HEADERS,
    )

    assert response.status_code == 200
    assert response.json() == {
        "run_id": "abc123def456",
        "issue_url": ISSUE_URL,
        "current_phase": "WAITING_FOR_USER",
        "pending_human_input": True,
        "human_input_question": "Proceed?",
        "latest_decision_frame": None,
        "updated_at": "",
    }


async def test_replay_uses_the_authorized_stored_snapshot(monkeypatch, api_client):
    calls = []
    state = _state()

    def fake_load(run_id):
        calls.append(run_id)
        return state

    monkeypatch.setattr(main, "load_run", fake_load)

    response = await api_client.get(
        "/agent/v2/runs/abc123def456/replay",
        headers=AUTHORIZED_HEADERS,
    )

    assert response.status_code == 200
    assert calls == ["abc123def456"]
    assert response.json()["run_id"] == "abc123def456"
    assert response.json()["issue_url"] == ISSUE_URL


async def test_replay_format_is_limited_to_json_or_markdown(
    monkeypatch, api_client
):
    loaded = False

    def forbidden_load(run_id):
        nonlocal loaded
        loaded = True
        raise AssertionError("invalid query must not load run")

    monkeypatch.setattr(main, "load_run", forbidden_load)

    response = await api_client.get(
        "/agent/v2/runs/abc123def456/replay?format=html",
        headers=AUTHORIZED_HEADERS,
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid request."}
    assert loaded is False


async def test_replay_markdown_is_built_from_authorized_snapshot(
    monkeypatch, api_client
):
    monkeypatch.setattr(main, "load_run", lambda run_id: _state())

    response = await api_client.get(
        "/agent/v2/runs/abc123def456/replay?format=markdown",
        headers=AUTHORIZED_HEADERS,
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert response.text.startswith("# RepoPilot Replay: abc123def456")


@pytest.mark.parametrize(
    "path",
    [
        "/agent/v2/runs/abc123def456",
        "/agent/v2/runs/abc123def456/replay",
    ],
)
async def test_unauthorized_stored_repo_returns_no_inspect_or_replay_metadata(
    monkeypatch, api_client, path
):
    state = _state(
        issue_url="https://github.com/secret/private/issues/1",
        owner="secret",
        repo="private",
    )

    def forbidden_summary(*args, **kwargs):
        raise AssertionError("unauthorized metadata must not be summarized")

    monkeypatch.setattr(main, "load_run", lambda run_id: state)
    monkeypatch.setattr(main, "summarize_run", forbidden_summary)
    monkeypatch.setattr(main, "summarize_replay", forbidden_summary)

    response = await api_client.get(path, headers=AUTHORIZED_HEADERS)

    assert response.status_code == 403
    assert response.json() == {"detail": "Repository is not authorized."}
    assert "secret/private" not in response.text


async def test_protected_request_concurrency_rejects_third_without_queueing(
    monkeypatch, api_client
):
    entered = 0
    two_entered = asyncio.Event()
    release = asyncio.Event()

    async def blocking_agent(issue_url, max_retries=3, token_budget=100000):
        nonlocal entered
        entered += 1
        if entered == 2:
            two_entered.set()
        await release.wait()
        return {"success": True, "error": None}

    monkeypatch.setattr(main, "agent_v2", blocking_agent)
    first = asyncio.create_task(
        api_client.post(
            "/agent/v2",
            headers=AUTHORIZED_HEADERS,
            json={"issue_url": ISSUE_URL},
        )
    )
    second = asyncio.create_task(
        api_client.post(
            "/agent/v2",
            headers=AUTHORIZED_HEADERS,
            json={"issue_url": ISSUE_URL},
        )
    )
    await asyncio.wait_for(two_entered.wait(), timeout=1)
    third = asyncio.create_task(
        api_client.post(
            "/agent/v2",
            headers=AUTHORIZED_HEADERS,
            json={"issue_url": ISSUE_URL},
        )
    )

    try:
        await asyncio.sleep(0.02)
        assert third.done(), "the third request must be rejected, not queued"
        response = await third
        assert response.status_code == 429
        assert response.json() == {"detail": "Too many requests."}
        assert response.headers["retry-after"] == "1"
        assert entered == 2

        health = await api_client.get("/health")
        assert health.status_code == 200

        unauthorized = await api_client.post(
            "/agent/v2",
            headers=AUTHORIZED_HEADERS,
            json={"issue_url": "https://github.com/other/repo/issues/1"},
        )
        assert unauthorized.status_code == 403
    finally:
        release.set()
        await asyncio.gather(first, second)
        if not third.done():
            await third


async def test_concurrency_slot_is_released_after_handler_error(monkeypatch, caplog):
    calls = 0
    secret = "private ambient credential must stay hidden"

    async def flaky_agent(issue_url, max_retries=3, token_budget=100000):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError(secret)
        return {"success": True, "error": None}

    monkeypatch.setattr(main, "agent_v2", flaky_agent)
    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        failed = await client.post(
            "/agent/v2",
            headers=AUTHORIZED_HEADERS,
            json={"issue_url": ISSUE_URL},
        )
        recovered = await client.post(
            "/agent/v2",
            headers=AUTHORIZED_HEADERS,
            json={"issue_url": ISSUE_URL},
        )

    assert failed.status_code == 502
    assert failed.json() == {"detail": "Agent request failed."}
    assert secret not in failed.text
    assert secret not in caplog.text
    assert recovered.status_code == 200


@pytest.mark.parametrize(
    ("endpoint_name", "dependency_name", "request_model"),
    [
        (
            "intelligent_agent",
            "intelligent_analyze_issue",
            main.IntelligentAgentRequest(issue_url=ISSUE_URL),
        ),
        (
            "agent_v2_endpoint",
            "agent_v2",
            main.AgentV2Request(issue_url=ISSUE_URL),
        ),
        (
            "agent_v2_resume_endpoint",
            "resume_agent_v2",
            main.AgentV2ResumeRequest(
                run_id="abc123def456",
                human_answer="Proceed.",
            ),
        ),
    ],
)
async def test_agent_endpoints_reraise_cancellation_drain_by_identity(
    monkeypatch,
    endpoint_name,
    dependency_name,
    request_model,
):
    sentinel = CancellationDrainError(
        "outer endpoint",
        asyncio.CancelledError("cancel endpoint"),
        OSError("endpoint drain failed"),
    )

    async def fail_with_drain(*_args, **_kwargs):
        raise sentinel

    monkeypatch.setattr(main, dependency_name, fail_with_drain)
    monkeypatch.setattr(main, "_load_authorized_run", lambda _run_id: _state())

    with pytest.raises(CancellationDrainError) as caught:
        await getattr(main, endpoint_name)(request_model)

    assert caught.value is sentinel


@pytest.mark.parametrize(
    ("endpoint_name", "dependency_name", "request_model"),
    [
        (
            "intelligent_agent",
            "intelligent_analyze_issue",
            main.IntelligentAgentRequest(issue_url=ISSUE_URL),
        ),
        (
            "agent_v2_endpoint",
            "agent_v2",
            main.AgentV2Request(issue_url=ISSUE_URL),
        ),
        (
            "agent_v2_resume_endpoint",
            "resume_agent_v2",
            main.AgentV2ResumeRequest(
                run_id="abc123def456",
                human_answer="Proceed.",
            ),
        ),
    ],
)
async def test_agent_endpoints_keep_safe_runtime_error_response(
    monkeypatch,
    endpoint_name,
    dependency_name,
    request_model,
):
    async def fail_with_runtime_error(*_args, **_kwargs):
        raise RuntimeError("private endpoint detail")

    monkeypatch.setattr(main, dependency_name, fail_with_runtime_error)
    monkeypatch.setattr(main, "_load_authorized_run", lambda _run_id: _state())

    response = await getattr(main, endpoint_name)(request_model)

    assert response.status_code == 502
    assert json.loads(response.body) == {"detail": "Agent request failed."}


async def test_concurrency_slot_is_released_after_cancellation(
    monkeypatch, api_client
):
    entered = 0
    two_entered = asyncio.Event()
    replacement_entered = asyncio.Event()
    release = asyncio.Event()

    async def blocking_agent(issue_url, max_retries=3, token_budget=100000):
        nonlocal entered
        entered += 1
        if entered == 2:
            two_entered.set()
        if entered == 3:
            replacement_entered.set()
        await release.wait()
        return {"success": True, "error": None}

    monkeypatch.setattr(main, "agent_v2", blocking_agent)

    def request_task():
        return asyncio.create_task(
            api_client.post(
                "/agent/v2",
                headers=AUTHORIZED_HEADERS,
                json={"issue_url": ISSUE_URL},
            )
        )

    first = request_task()
    second = request_task()
    await asyncio.wait_for(two_entered.wait(), timeout=1)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    replacement = request_task()
    try:
        await asyncio.wait_for(replacement_entered.wait(), timeout=1)
    finally:
        release.set()
        await asyncio.gather(second, replacement)


@pytest.mark.parametrize("endpoint", ["/analyze", "/agent", "/intelligent-agent", "/agent/v2"])
async def test_agent_error_responses_do_not_echo_ambient_secrets(
    monkeypatch, api_client, endpoint
):
    secret = "ambient-secret-value"

    async def failed(*args, **kwargs):
        return {"error": f"provider failed with {secret}", "trace_id": secret}

    target = {
        "/analyze": "analyze_issue",
        "/agent": "agent_analyze",
        "/intelligent-agent": "intelligent_analyze_issue",
        "/agent/v2": "agent_v2",
    }[endpoint]
    monkeypatch.setattr(main, target, failed)

    response = await api_client.post(
        endpoint,
        headers=AUTHORIZED_HEADERS,
        json={"issue_url": ISSUE_URL},
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "Agent request failed."}
    assert secret not in response.text


async def test_resume_error_response_does_not_echo_human_answer_or_secret(
    monkeypatch, api_client
):
    state = _state()
    human_answer = "private-human-answer"
    secret = "ambient-secret-value"

    async def failed_resume(run_id, answer, *, state):
        return {"error": f"{answer}: {secret}", "trace_id": secret}

    monkeypatch.setattr(main, "load_run", lambda run_id: state)
    monkeypatch.setattr(main, "resume_agent_v2", failed_resume)

    response = await api_client.post(
        "/agent/v2/resume",
        headers=AUTHORIZED_HEADERS,
        json={"run_id": "abc123def456", "human_answer": human_answer},
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "Agent request failed."}
    assert human_answer not in response.text
    assert secret not in response.text
