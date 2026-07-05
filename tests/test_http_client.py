"""Tests for HTTP client retry logic."""

import asyncio

import httpx
import pytest

from src.http_client import (
    LLM_CALL_WALLCLOCK_TIMEOUT,
    LLM_CONNECT_TIMEOUT,
    LLM_MAX_ATTEMPTS,
    LLM_RETRY_BACKOFF_MAX_SECONDS,
    LLM_STREAM_IDLE_TIMEOUT,
    MAX_RETRIES,
    RETRYABLE_GITHUB_STATUS,
    RETRYABLE_LLM_STATUS,
    _is_retryable_github,
    _is_retryable_llm,
    _reset_llm_client,
    github_request,
    llm_request,
    llm_retry_budget_seconds,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _noop_sleep(*args, **kwargs):
    """Async no-op to replace asyncio.sleep in tests, avoiding real waits."""
    pass


def _sse(content: str) -> str:
    """One-chunk Server-Sent-Events body carrying `content`, terminated by DONE."""
    import json as _json

    return (
        "data: "
        + _json.dumps({"choices": [{"delta": {"content": content}}]})
        + "\n\ndata: [DONE]\n"
    )


class _FakeStreamCM:
    """Stand-in for httpx AsyncClient.stream()'s async context manager, so the
    streaming LLM path can be tested without a real SSE server."""

    def __init__(self, *, raise_exc=None, status=200, sse=""):
        self._raise = raise_exc
        self._status = status
        self._sse = sse

    async def __aenter__(self):
        if self._raise is not None:
            raise self._raise
        return self

    async def __aexit__(self, *exc):
        return False

    @property
    def status_code(self):
        return self._status

    async def aread(self):
        return b""

    def raise_for_status(self):
        if self._status >= 400:
            request = httpx.Request(
                "POST", "https://api.deepseek.com/v1/chat/completions"
            )
            raise httpx.HTTPStatusError(
                "err", request=request, response=httpx.Response(self._status, request=request)
            )

    async def aiter_lines(self):
        for line in self._sse.split("\n"):
            yield line


async def test_close_llm_client_closes_cached_client_and_clears_global():
    from src import http_client

    client = http_client._get_llm_client()

    await http_client.close_llm_client()

    assert client.is_closed is True
    assert http_client._llm_client is None


def test_llm_timeout_budget_is_explicit():
    assert LLM_STREAM_IDLE_TIMEOUT == 120.0
    assert LLM_CONNECT_TIMEOUT == 15.0
    assert LLM_CALL_WALLCLOCK_TIMEOUT == 300.0
    assert LLM_MAX_ATTEMPTS == 2
    assert LLM_RETRY_BACKOFF_MAX_SECONDS == 20.0
    # One slow attempt is killed at the wall-clock ceiling (non-retryable), so
    # the worst case is a fast transient fail + backoff + one slow attempt.
    assert llm_retry_budget_seconds() == 320.0


@pytest.fixture(autouse=True)
def _reset_llm_between_tests():
    """Reset the shared LLM client so each test gets a fresh one (important
    when httpx_mock swaps out the transport per-test)."""
    _reset_llm_client()
    yield
    _reset_llm_client()


# ---------------------------------------------------------------------------
# Retry predicate unit tests
# ---------------------------------------------------------------------------


def test_is_retryable_github_network_error():
    assert _is_retryable_github(httpx.NetworkError("boom")) is True


def test_is_retryable_github_timeout():
    assert _is_retryable_github(httpx.TimeoutException("timeout")) is True


def test_is_retryable_github_retryable_status():
    for status in RETRYABLE_GITHUB_STATUS:
        resp = httpx.Response(status, request=httpx.Request("GET", "http://x"))
        exc = httpx.HTTPStatusError("msg", request=resp.request, response=resp)
        assert _is_retryable_github(exc) is True, f"status {status} should be retryable"


def test_is_retryable_github_non_retryable_status():
    resp = httpx.Response(404, request=httpx.Request("GET", "http://x"))
    exc = httpx.HTTPStatusError("msg", request=resp.request, response=resp)
    assert _is_retryable_github(exc) is False


def test_is_retryable_github_value_error():
    assert _is_retryable_github(ValueError("not http")) is False


def test_is_retryable_llm_retryable_status():
    for status in RETRYABLE_LLM_STATUS:
        resp = httpx.Response(status, request=httpx.Request("POST", "http://x"))
        exc = httpx.HTTPStatusError("msg", request=resp.request, response=resp)
        assert _is_retryable_llm(exc) is True, f"status {status} should be retryable"


def test_is_retryable_llm_non_retryable_status():
    resp = httpx.Response(400, request=httpx.Request("POST", "http://x"))
    exc = httpx.HTTPStatusError("msg", request=resp.request, response=resp)
    assert _is_retryable_llm(exc) is False


# ---------------------------------------------------------------------------
# GitHub request retry tests (using httpx_mock for HTTP-level mocking)
# ---------------------------------------------------------------------------


async def test_llm_request_accumulates_streamed_chunks(monkeypatch):
    """Multiple SSE delta chunks are concatenated into the final content."""
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    RealAsyncClient = httpx.AsyncClient

    body = (
        'data: {"choices":[{"delta":{"content":"{\\"a\\":"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":" 1}"}}]}\n\n'
        "data: [DONE]\n"
    )

    def fake_stream(self, method, url, **kwargs):
        return _FakeStreamCM(sse=body)

    monkeypatch.setattr(RealAsyncClient, "stream", fake_stream)

    result = await llm_request([{"role": "user", "content": "hi"}])
    assert result["choices"][0]["message"]["content"] == '{"a": 1}'


async def test_github_request_retries_on_429(httpx_mock, monkeypatch):
    """Mock returns 429 twice, then 200 — should retry and eventually succeed."""
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)

    url = "https://api.github.com/repos/foo/bar/issues/1"
    httpx_mock.add_response(method="GET", url=url, status_code=429,
                            json={"message": "rate limit"})
    httpx_mock.add_response(method="GET", url=url, status_code=429,
                            json={"message": "rate limit"})
    httpx_mock.add_response(method="GET", url=url, status_code=200,
                            json={"title": "ok"})

    resp = await github_request("GET", url)

    assert resp.status_code == 200
    assert resp.json() == {"title": "ok"}
    # 3 attempts: initial + 2 retries
    assert len(httpx_mock.requests) == 3


async def test_github_request_raises_after_max_retries(httpx_mock, monkeypatch):
    """Mock always returns 429 — should exhaust all retries then raise."""
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)

    url = "https://api.github.com/repos/foo/bar/issues/1"
    for _ in range(MAX_RETRIES + 1):  # initial attempt + retries
        httpx_mock.add_response(method="GET", url=url, status_code=429,
                                json={"message": "rate limit"})

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await github_request("GET", url)

    assert exc_info.value.response.status_code == 429
    assert len(httpx_mock.requests) == MAX_RETRIES + 1


async def test_github_request_retries_on_503(httpx_mock, monkeypatch):
    """503 Service Unavailable is retryable for GitHub requests."""
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)

    url = "https://api.github.com/repos/foo/bar/issues/1"
    httpx_mock.add_response(method="GET", url=url, status_code=503,
                            json={"message": "unavailable"})
    httpx_mock.add_response(method="GET", url=url, status_code=503,
                            json={"message": "unavailable"})
    httpx_mock.add_response(method="GET", url=url, status_code=200,
                            json={"title": "ok"})

    resp = await github_request("GET", url)

    assert resp.status_code == 200
    assert len(httpx_mock.requests) == 3


async def test_github_request_does_not_retry_on_404(httpx_mock, monkeypatch):
    """404 should NOT trigger a retry — fails immediately."""
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)

    url = "https://api.github.com/repos/foo/bar/issues/999"
    httpx_mock.add_response(method="GET", url=url, status_code=404,
                            json={"message": "not found"})

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await github_request("GET", url)

    assert exc_info.value.response.status_code == 404
    # Only 1 attempt — no retry for 404
    assert len(httpx_mock.requests) == 1


async def test_github_request_retries_on_network_error(monkeypatch):
    """Mock raises NetworkError twice, then succeeds — should retry."""
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)

    url = "https://api.github.com/repos/foo/bar/issues/1"
    call_count = 0

    # Get the real AsyncClient class (before conftest monkeypatches it)
    RealAsyncClient = httpx.AsyncClient

    async def mock_request(self, method, url, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise httpx.NetworkError("connection reset")
        req = httpx.Request(method, url)
        return httpx.Response(200, json={"title": "ok"}, request=req)

    monkeypatch.setattr(RealAsyncClient, "request", mock_request)

    resp = await github_request("GET", url)

    assert resp.status_code == 200
    assert resp.json() == {"title": "ok"}
    assert call_count == 3


async def test_github_request_retries_on_timeout(monkeypatch):
    """Mock raises TimeoutException twice, then succeeds."""
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)

    url = "https://api.github.com/repos/foo/bar/issues/1"
    call_count = 0

    RealAsyncClient = httpx.AsyncClient

    async def mock_request(self, method, url, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise httpx.TimeoutException("timed out")
        req = httpx.Request(method, url)
        return httpx.Response(200, json={"title": "ok"}, request=req)

    monkeypatch.setattr(RealAsyncClient, "request", mock_request)

    resp = await github_request("GET", url)

    assert resp.status_code == 200
    assert call_count == 3


async def test_github_request_network_error_exhausts_retries(monkeypatch):
    """All attempts raise NetworkError — should eventually raise."""
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)

    url = "https://api.github.com/repos/foo/bar/issues/1"

    RealAsyncClient = httpx.AsyncClient

    async def mock_request(self, method, url, **kwargs):
        raise httpx.NetworkError("connection reset")

    monkeypatch.setattr(RealAsyncClient, "request", mock_request)

    with pytest.raises(httpx.NetworkError):
        await github_request("GET", url)


# ---------------------------------------------------------------------------
# LLM request retry tests
# ---------------------------------------------------------------------------


async def test_llm_request_retries_on_502(httpx_mock, monkeypatch):
    """Mock returns 502 once, then 200 — LLM retries should work (1 retry)."""
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    url = "https://api.deepseek.com/v1/chat/completions"
    httpx_mock.add_response(method="POST", url=url, status_code=502,
                            json={"error": "bad gateway"})
    httpx_mock.add_response(
        method="POST", url=url, status_code=200,
        json={"choices": [{"message": {"content": '{"answer":"ok"}'}}]},
    )

    result = await llm_request([{"role": "user", "content": "hello"}])

    assert result["choices"][0]["message"]["content"] == '{"answer":"ok"}'
    assert len(httpx_mock.requests) == LLM_MAX_ATTEMPTS


async def test_llm_request_raises_after_max_retries(httpx_mock, monkeypatch):
    """Mock always returns 502 — should exhaust retries then raise."""
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    url = "https://api.deepseek.com/v1/chat/completions"
    for _ in range(LLM_MAX_ATTEMPTS):
        httpx_mock.add_response(method="POST", url=url, status_code=502,
                                json={"error": "bad gateway"})

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await llm_request([{"role": "user", "content": "hello"}])

    assert exc_info.value.response.status_code == 502
    assert len(httpx_mock.requests) == LLM_MAX_ATTEMPTS


async def test_llm_request_does_not_retry_on_400(httpx_mock, monkeypatch):
    """400 Bad Request should NOT trigger retry for LLM."""
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    url = "https://api.deepseek.com/v1/chat/completions"
    httpx_mock.add_response(method="POST", url=url, status_code=400,
                            json={"error": "bad request"})

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await llm_request([{"role": "user", "content": "hello"}])

    assert exc_info.value.response.status_code == 400
    assert len(httpx_mock.requests) == 1


async def test_llm_request_retries_on_503(httpx_mock, monkeypatch):
    """LLM retries on 503 Service Unavailable."""
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    url = "https://api.deepseek.com/v1/chat/completions"
    httpx_mock.add_response(method="POST", url=url, status_code=503,
                            json={"error": "unavailable"})
    httpx_mock.add_response(method="POST", url=url, status_code=200,
                            json={"choices": [{"message": {"content": '{"ok":true}'}}]},
                            )

    result = await llm_request([{"role": "user", "content": "hello"}])

    assert result["choices"][0]["message"]["content"] == '{"ok":true}'
    assert len(httpx_mock.requests) == 2


async def test_llm_request_retries_on_network_error(monkeypatch):
    """LLM retries on NetworkError."""
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    call_count = 0
    RealAsyncClient = httpx.AsyncClient

    def fake_stream(self, method, url, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 1:
            return _FakeStreamCM(raise_exc=httpx.NetworkError("connection reset"))
        return _FakeStreamCM(sse=_sse('{"ok":true}'))

    monkeypatch.setattr(RealAsyncClient, "stream", fake_stream)

    result = await llm_request([{"role": "user", "content": "hello"}])

    assert result["choices"][0]["message"]["content"] == '{"ok":true}'
    assert call_count == LLM_MAX_ATTEMPTS


async def test_llm_request_retries_on_timeout(monkeypatch):
    """LLM retries on TimeoutException."""
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    call_count = 0
    RealAsyncClient = httpx.AsyncClient

    def fake_stream(self, method, url, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 1:
            return _FakeStreamCM(raise_exc=httpx.TimeoutException("timed out"))
        return _FakeStreamCM(sse=_sse('{"ok":true}'))

    monkeypatch.setattr(RealAsyncClient, "stream", fake_stream)

    result = await llm_request([{"role": "user", "content": "hello"}])

    assert result["choices"][0]["message"]["content"] == '{"ok":true}'
    assert call_count == 2


async def test_llm_request_wallclock_timeout_is_not_retried(monkeypatch):
    """A call exceeding the wall-clock ceiling fails fast without a retry.

    asyncio.TimeoutError is deliberately absent from the LLM retry set, so a
    genuinely-slow generation does not double the wait by retrying.
    """
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    call_count = 0
    RealAsyncClient = httpx.AsyncClient

    def timeout_stream(self, method, url, **kwargs):
        # Simulate asyncio.wait_for's ceiling firing on this attempt.
        nonlocal call_count
        call_count += 1
        return _FakeStreamCM(raise_exc=asyncio.TimeoutError())

    monkeypatch.setattr(RealAsyncClient, "stream", timeout_stream)

    with pytest.raises(asyncio.TimeoutError):
        await llm_request([{"role": "user", "content": "hello"}])

    assert call_count == 1  # NOT retried



async def test_llm_request_respects_custom_model(httpx_mock, monkeypatch):
    """Custom model parameter is passed in the payload."""
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    url = "https://api.deepseek.com/v1/chat/completions"
    httpx_mock.add_response(
        method="POST", url=url, status_code=200,
        json={"choices": [{"message": {"content": '{"answer":"custom"}'}}]},
    )

    result = await llm_request(
        [{"role": "user", "content": "hello"}], model="custom-model-v1"
    )

    assert result["choices"][0]["message"]["content"] == '{"answer":"custom"}'
    # Verify custom model was sent in the payload
    import json
    body = json.loads(httpx_mock.requests[0].content)
    assert body["model"] == "custom-model-v1"


async def test_llm_request_passes_temperature(httpx_mock, monkeypatch):
    """Temperature parameter is forwarded in the payload."""
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    url = "https://api.deepseek.com/v1/chat/completions"
    httpx_mock.add_response(
        method="POST", url=url, status_code=200,
        json={"choices": [{"message": {"content": '{"answer":"hot"}'}}]},
    )

    await llm_request(
        [{"role": "user", "content": "hello"}], temperature=0.7
    )

    import json
    body = json.loads(httpx_mock.requests[0].content)
    assert body["temperature"] == 0.7


async def test_llm_request_passes_extra_kwargs(httpx_mock, monkeypatch):
    """Extra keyword arguments are forwarded in the payload."""
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    url = "https://api.deepseek.com/v1/chat/completions"
    httpx_mock.add_response(
        method="POST", url=url, status_code=200,
        json={"choices": [{"message": {"content": '{"answer":"extra"}'}}]},
    )

    await llm_request(
        [{"role": "user", "content": "hello"}], max_tokens=100, top_p=0.9
    )

    import json
    body = json.loads(httpx_mock.requests[0].content)
    assert body["max_tokens"] == 100
    assert body["top_p"] == 0.9
