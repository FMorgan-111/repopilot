"""Tests for HTTP client retry logic."""

import asyncio
import json

import httpx
import pytest

from src.http_client import (
    LLM_CALL_WALLCLOCK_TIMEOUT,
    LLM_CONNECT_TIMEOUT,
    LLM_MAX_ATTEMPTS,
    LLM_MAX_CHOICES,
    LLM_MAX_CONTENT_BYTES,
    LLM_MAX_RESPONSE_BYTES,
    LLM_MAX_SSE_EVENTS,
    LLM_MAX_TOOL_ARGUMENT_BYTES,
    LLM_MAX_TOOL_CALLS,
    LLM_RETRY_BACKOFF_MAX_SECONDS,
    LLM_STREAM_IDLE_TIMEOUT,
    MAX_RETRIES,
    RETRYABLE_GITHUB_STATUS,
    RETRYABLE_LLM_STATUS,
    LLMResponseError,
    _get_llm_client,
    _get_llm_model,
    _is_retryable_github,
    _is_retryable_llm,
    _reset_llm_client,
    github_request,
    is_retryable_github_error,
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

    def __init__(
        self,
        *,
        raise_exc=None,
        status=200,
        sse="",
        json_body=None,
        raw_body=None,
        byte_chunks=None,
        content_type="text/event-stream",
        forbid_unbounded=False,
    ):
        self._raise = raise_exc
        self._status = status
        self._sse = sse
        self._json_body = json_body
        self._raw_body = raw_body
        self._byte_chunks = byte_chunks
        self._forbid_unbounded = forbid_unbounded
        self.headers = {"content-type": content_type}

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
        if self._forbid_unbounded:
            raise AssertionError("unbounded aread() must not be used")
        if self._raw_body is not None:
            return self._raw_body
        if self._json_body is not None:
            import json as _json

            return _json.dumps(self._json_body).encode()
        return self._sse.encode()

    def raise_for_status(self):
        if self._status >= 400:
            request = httpx.Request(
                "POST", "https://api.deepseek.com/v1/chat/completions"
            )
            raise httpx.HTTPStatusError(
                "err", request=request, response=httpx.Response(self._status, request=request)
            )

    async def aiter_lines(self):
        if self._forbid_unbounded:
            raise AssertionError("unbounded aiter_lines() must not be used")
        for line in self._sse.split("\n"):
            yield line

    async def aiter_bytes(self):
        if self._byte_chunks is not None:
            for chunk in self._byte_chunks:
                yield chunk
            return
        if self._raw_body is not None:
            yield self._raw_body
            return
        if self._json_body is not None:
            yield json.dumps(self._json_body).encode()
            return
        yield self._sse.encode()


def _padded_json(payload, size):
    candidate = dict(payload)
    candidate["padding"] = ""
    raw = json.dumps(candidate, separators=(",", ":")).encode()
    assert len(raw) <= size
    candidate["padding"] = "x" * (size - len(raw))
    raw = json.dumps(candidate, separators=(",", ":")).encode()
    assert len(raw) == size
    return raw


def _stream_from_raw(monkeypatch, raw, *, status=200, content_type="text/event-stream", chunks=None):
    monkeypatch.setattr(
        httpx.AsyncClient,
        "stream",
        lambda self, method, url, **kwargs: _FakeStreamCM(
            status=status,
            raw_body=raw,
            byte_chunks=chunks,
            content_type=content_type,
            forbid_unbounded=True,
        ),
    )


async def test_close_llm_client_closes_cached_client_and_clears_global():
    from src import http_client

    primary_client = http_client._get_llm_client(
        "primary", "https://primary.invalid/v1"
    )
    escalation_client = http_client._get_llm_client(
        "escalation", "https://escalation.invalid/v1"
    )

    await http_client.close_llm_client()

    assert primary_client.is_closed is True
    assert escalation_client.is_closed is True
    assert http_client._llm_clients == {}


def test_llm_clients_are_independent_by_provider_and_base_url():
    primary = _get_llm_client("primary", "https://primary.invalid/v1")
    same_primary = _get_llm_client("primary", "https://primary.invalid/v1")
    escalation = _get_llm_client("escalation", "https://escalation.invalid/v1")

    assert same_primary is primary
    assert escalation is not primary


def test_llm_timeout_budget_is_explicit():
    assert LLM_STREAM_IDLE_TIMEOUT == 120.0
    assert LLM_CONNECT_TIMEOUT == 15.0
    assert LLM_CALL_WALLCLOCK_TIMEOUT == 300.0
    assert LLM_MAX_ATTEMPTS == 2
    assert LLM_RETRY_BACKOFF_MAX_SECONDS == 20.0
    # One slow attempt is killed at the wall-clock ceiling (non-retryable), so
    # the worst case is a fast transient fail + backoff + one slow attempt.
    assert llm_retry_budget_seconds() == 320.0


def test_default_llm_model_is_gemini_flash(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)

    assert _get_llm_model() == "gemini-3.5-flash:stable"


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


def test_github_retry_predicate_distinguishes_503_and_404():
    request = httpx.Request("GET", "https://api.github.com/repos/acme/widget")
    transient = httpx.HTTPStatusError(
        "503", request=request, response=httpx.Response(503, request=request)
    )
    missing = httpx.HTTPStatusError(
        "404", request=request, response=httpx.Response(404, request=request)
    )

    assert is_retryable_github_error(transient) is True
    assert is_retryable_github_error(missing) is False


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


async def test_llm_request_uses_isolated_provider_configuration(monkeypatch):
    requests = []

    def fake_stream(self, method, url, **kwargs):
        requests.append((url, kwargs))
        return _FakeStreamCM(sse=_sse('{"ok": true}'))

    monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)
    monkeypatch.setenv("LLM_API_KEY", "primary-sentinel-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://primary.invalid/v1")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("LLM_MODEL", "primary-model")
    monkeypatch.setenv("LLM_ESCALATION_API_KEY", "escalation-sentinel-key")
    monkeypatch.setenv("LLM_ESCALATION_BASE_URL", "https://escalation.invalid/v1")
    monkeypatch.setenv("LLM_ESCALATION_MODEL", "escalation-model")
    monkeypatch.setenv("REPOPILOT_ESCALATION_ENABLED", "1")

    await llm_request([{"role": "user", "content": "primary"}])
    await llm_request(
        [{"role": "user", "content": "escalation"}], provider="escalation"
    )

    primary_url, primary_kwargs = requests[0]
    escalation_url, escalation_kwargs = requests[1]
    assert primary_url == "https://primary.invalid/v1/chat/completions"
    assert primary_kwargs["json"]["model"] == "primary-model"
    assert primary_kwargs["headers"]["Authorization"] == "Bearer primary-sentinel-key"
    assert escalation_url == "https://escalation.invalid/v1/chat/completions"
    assert escalation_kwargs["json"]["model"] == "escalation-model"
    assert (
        escalation_kwargs["headers"]["Authorization"]
        == "Bearer escalation-sentinel-key"
    )


async def test_disabled_escalation_request_never_constructs_an_http_client(monkeypatch):
    from src import http_client

    monkeypatch.setenv("LLM_ESCALATION_API_KEY", "escalation-sentinel-key")
    monkeypatch.delenv("REPOPILOT_ESCALATION_ENABLED", raising=False)
    monkeypatch.setattr(
        http_client,
        "_get_llm_client",
        lambda *args, **kwargs: pytest.fail("HTTP client must not be constructed"),
    )

    with pytest.raises(LLMResponseError, match="escalation provider is not configured"):
        await llm_request([{"role": "user", "content": "escalate"}], provider="escalation")


async def test_escalation_without_a_key_never_constructs_an_http_client(monkeypatch):
    from src import http_client

    monkeypatch.setenv("REPOPILOT_ESCALATION_ENABLED", "1")
    monkeypatch.delenv("LLM_ESCALATION_API_KEY", raising=False)
    monkeypatch.setattr(
        http_client,
        "_get_llm_client",
        lambda *args, **kwargs: pytest.fail("HTTP client must not be constructed"),
    )

    with pytest.raises(LLMResponseError, match="escalation provider is not configured"):
        await llm_request([{"role": "user", "content": "escalate"}], provider="escalation")


async def test_primary_without_llm_key_never_constructs_an_http_client(monkeypatch):
    from src import http_client

    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("LINOAPI_API_KEY", "cross-vendor-sentinel")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "cross-vendor-sentinel")
    monkeypatch.setattr(
        http_client,
        "_get_llm_client",
        lambda *args, **kwargs: pytest.fail("HTTP client must not be constructed"),
    )

    with pytest.raises(ValueError, match="LLM_API_KEY"):
        await llm_request([{"role": "user", "content": "primary"}])


def test_dotenv_loading_never_overrides_process_environment(monkeypatch):
    from src import http_client

    seen = {}

    def fake_load_dotenv(*_args, **kwargs):
        seen.update(kwargs)
        return True

    monkeypatch.setattr(http_client, "load_dotenv", fake_load_dotenv)
    http_client._load_dotenv_safely()

    assert seen.get("override") is False


async def test_llm_request_preserves_stream_usage_and_finish_reason(monkeypatch):
    body = (
        'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}\n\n'
        "data: [DONE]\n"
    )
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        real_client,
        "stream",
        lambda self, method, url, **kwargs: _FakeStreamCM(sse=body),
    )

    result = await llm_request([{"role": "user", "content": "hi"}])

    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 1,
        "total_tokens": 4,
    }


async def test_llm_request_accepts_non_stream_json_completion(monkeypatch):
    payload = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"total_tokens": 7},
    }
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        real_client,
        "stream",
        lambda self, method, url, **kwargs: _FakeStreamCM(
            json_body=payload, content_type="application/json"
        ),
    )

    result = await llm_request([{"role": "user", "content": "hi"}])

    assert result == payload


def test_llm_response_limits_are_explicit():
    assert LLM_MAX_RESPONSE_BYTES == 8 * 1024 * 1024
    assert LLM_MAX_SSE_EVENTS == 50_000
    assert LLM_MAX_CHOICES == 8
    assert LLM_MAX_CONTENT_BYTES == 4 * 1024 * 1024
    assert LLM_MAX_TOOL_CALLS == 64
    assert LLM_MAX_TOOL_ARGUMENT_BYTES == 2 * 1024 * 1024


@pytest.mark.parametrize("extra", [0, 1])
async def test_json_response_byte_limit_is_inclusive(monkeypatch, extra):
    payload = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    raw = _padded_json(payload, LLM_MAX_RESPONSE_BYTES + extra)
    _stream_from_raw(monkeypatch, raw, content_type="application/json")

    if extra:
        with pytest.raises(LLMResponseError, match="response byte limit"):
            await llm_request([{"role": "user", "content": "hi"}])
    else:
        result = await llm_request([{"role": "user", "content": "hi"}])
        assert result["choices"][0]["message"]["content"] == "ok"


@pytest.mark.parametrize("extra", [0, 1])
async def test_sse_response_byte_limit_is_inclusive(monkeypatch, extra):
    completion = _sse("ok").encode()
    padding_size = LLM_MAX_RESPONSE_BYTES + extra - len(completion)
    padding = b":" + (b"x" * (padding_size - 2)) + b"\n"
    raw = padding + completion
    assert len(raw) == LLM_MAX_RESPONSE_BYTES + extra
    _stream_from_raw(monkeypatch, raw)

    if extra:
        with pytest.raises(LLMResponseError, match="response byte limit"):
            await llm_request([{"role": "user", "content": "hi"}])
    else:
        result = await llm_request([{"role": "user", "content": "hi"}])
        assert result["choices"][0]["message"]["content"] == "ok"


@pytest.mark.parametrize("extra", [0, 1])
async def test_error_response_body_is_streamed_with_same_byte_limit(monkeypatch, extra):
    raw = b"x" * (LLM_MAX_RESPONSE_BYTES + extra)
    _stream_from_raw(monkeypatch, raw, status=400, content_type="application/json")

    if extra:
        with pytest.raises(LLMResponseError, match="response byte limit"):
            await llm_request([{"role": "user", "content": "hi"}])
    else:
        with pytest.raises(httpx.HTTPStatusError):
            await llm_request([{"role": "user", "content": "hi"}])


@pytest.mark.parametrize("extra", [0, 1])
async def test_sse_event_limit_is_inclusive(monkeypatch, extra):
    event_count = LLM_MAX_SSE_EVENTS + extra
    events = ['data: {"choices":[{"delta":{"content":"ok"}}]}\n\n']
    events.extend('data: {"choices":[]}\n\n' for _ in range(event_count - 2))
    events.append("data: [DONE]\n")
    raw = "".join(events).encode()
    _stream_from_raw(monkeypatch, raw)

    if extra:
        with pytest.raises(LLMResponseError, match="SSE event limit"):
            await llm_request([{"role": "user", "content": "hi"}])
    else:
        result = await llm_request([{"role": "user", "content": "hi"}])
        assert result["choices"][0]["message"]["content"] == "ok"


async def test_fragmented_sse_is_parsed_without_aiter_lines(monkeypatch):
    raw = _sse("量子").encode()
    chunks = [raw[index : index + 3] for index in range(0, len(raw), 3)]
    _stream_from_raw(monkeypatch, raw, chunks=chunks)

    result = await llm_request([{"role": "user", "content": "hi"}])

    assert result["choices"][0]["message"]["content"] == "量子"


async def test_sse_byte_limit_includes_chunks_after_done(monkeypatch):
    completion = _sse("ok").encode()
    tail = b"x" * LLM_MAX_RESPONSE_BYTES
    _stream_from_raw(
        monkeypatch,
        completion + tail,
        chunks=[completion, tail],
    )

    with pytest.raises(LLMResponseError, match="response byte limit"):
        await llm_request([{"role": "user", "content": "hi"}])


@pytest.mark.parametrize("count", [LLM_MAX_CHOICES, LLM_MAX_CHOICES + 1])
async def test_json_choice_limit_is_inclusive(monkeypatch, count):
    payload = {
        "choices": [
            {"index": index, "message": {"role": "assistant", "content": "x"}}
            for index in range(count)
        ]
    }
    raw = json.dumps(payload, separators=(",", ":")).encode()
    _stream_from_raw(monkeypatch, raw, content_type="application/json")

    if count > LLM_MAX_CHOICES:
        with pytest.raises(LLMResponseError, match="choice limit"):
            await llm_request([{"role": "user", "content": "hi"}])
    else:
        assert len((await llm_request([{"role": "user", "content": "hi"}]))["choices"]) == count


@pytest.mark.parametrize("extra", [0, 1])
async def test_json_content_limit_is_inclusive(monkeypatch, extra):
    content = "x" * (LLM_MAX_CONTENT_BYTES + extra)
    raw = json.dumps(
        {"choices": [{"message": {"role": "assistant", "content": content}}]},
        separators=(",", ":"),
    ).encode()
    _stream_from_raw(monkeypatch, raw, content_type="application/json")

    if extra:
        with pytest.raises(LLMResponseError, match="content byte limit"):
            await llm_request([{"role": "user", "content": "hi"}])
    else:
        result = await llm_request([{"role": "user", "content": "hi"}])
        assert len(result["choices"][0]["message"]["content"]) == len(content)


async def test_json_aggregate_content_limit_rejects_multiple_choices(monkeypatch):
    content = "x" * ((LLM_MAX_CONTENT_BYTES // 2) + 1)
    payload = {
        "choices": [
            {"message": {"role": "assistant", "content": content}},
            {"message": {"role": "assistant", "content": content}},
        ]
    }
    _stream_from_raw(
        monkeypatch,
        json.dumps(payload, separators=(",", ":")).encode(),
        content_type="application/json",
    )

    with pytest.raises(LLMResponseError, match="aggregate content byte limit"):
        await llm_request([{"role": "user", "content": "hi"}])


@pytest.mark.parametrize("count", [LLM_MAX_CHOICES, LLM_MAX_CHOICES + 1])
async def test_sse_choice_limit_is_inclusive(monkeypatch, count):
    choices = [
        {"index": index, "delta": {"content": "x"}}
        for index in range(count)
    ]
    raw = (
        "data: "
        + json.dumps({"choices": choices}, separators=(",", ":"))
        + "\n\ndata: [DONE]\n"
    ).encode()
    _stream_from_raw(monkeypatch, raw)

    if count > LLM_MAX_CHOICES:
        with pytest.raises(LLMResponseError, match="choice limit"):
            await llm_request([{"role": "user", "content": "hi"}])
    else:
        result = await llm_request([{"role": "user", "content": "hi"}])
        assert len(result["choices"]) == count


async def test_sse_aggregate_content_limit_rejects_multiple_choices(monkeypatch):
    content = "x" * ((LLM_MAX_CONTENT_BYTES // 2) + 1)
    choices = [
        {"index": 0, "delta": {"content": content}},
        {"index": 1, "delta": {"content": content}},
    ]
    raw = (
        "data: "
        + json.dumps({"choices": choices}, separators=(",", ":"))
        + "\n\ndata: [DONE]\n"
    ).encode()
    _stream_from_raw(monkeypatch, raw)

    with pytest.raises(LLMResponseError, match="aggregate content byte limit"):
        await llm_request([{"role": "user", "content": "hi"}])


def _tool_call(index, arguments="{}"):
    return {
        "index": index,
        "id": f"call_{index}",
        "type": "function",
        "function": {"name": "lookup", "arguments": arguments},
    }


@pytest.mark.parametrize("count", [LLM_MAX_TOOL_CALLS, LLM_MAX_TOOL_CALLS + 1])
async def test_json_tool_call_limit_is_inclusive(monkeypatch, count):
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [_tool_call(index) for index in range(count)],
                }
            }
        ]
    }
    _stream_from_raw(
        monkeypatch,
        json.dumps(payload, separators=(",", ":")).encode(),
        content_type="application/json",
    )

    if count > LLM_MAX_TOOL_CALLS:
        with pytest.raises(LLMResponseError, match="tool call limit"):
            await llm_request([{"role": "user", "content": "hi"}])
    else:
        result = await llm_request([{"role": "user", "content": "hi"}])
        assert len(result["choices"][0]["message"]["tool_calls"]) == count


@pytest.mark.parametrize("extra", [0, 1])
async def test_json_tool_argument_limit_is_inclusive(monkeypatch, extra):
    arguments = "x" * (LLM_MAX_TOOL_ARGUMENT_BYTES + extra)
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [_tool_call(0, arguments)],
                }
            }
        ]
    }
    _stream_from_raw(
        monkeypatch,
        json.dumps(payload, separators=(",", ":")).encode(),
        content_type="application/json",
    )

    if extra:
        with pytest.raises(LLMResponseError, match="tool argument byte limit"):
            await llm_request([{"role": "user", "content": "hi"}])
    else:
        result = await llm_request([{"role": "user", "content": "hi"}])
        assert len(result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]) == len(arguments)


@pytest.mark.parametrize("extra", [0, 1])
async def test_sse_content_limit_is_inclusive(monkeypatch, extra):
    raw = _sse("x" * (LLM_MAX_CONTENT_BYTES + extra)).encode()
    _stream_from_raw(monkeypatch, raw)

    if extra:
        with pytest.raises(LLMResponseError, match="content byte limit"):
            await llm_request([{"role": "user", "content": "hi"}])
    else:
        result = await llm_request([{"role": "user", "content": "hi"}])
        assert len(result["choices"][0]["message"]["content"]) == LLM_MAX_CONTENT_BYTES


@pytest.mark.parametrize("count", [LLM_MAX_TOOL_CALLS, LLM_MAX_TOOL_CALLS + 1])
async def test_sse_tool_call_limit_is_inclusive(monkeypatch, count):
    event = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [_tool_call(index) for index in range(count)]
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    raw = ("data: " + json.dumps(event, separators=(",", ":")) + "\n\ndata: [DONE]\n").encode()
    _stream_from_raw(monkeypatch, raw)

    if count > LLM_MAX_TOOL_CALLS:
        with pytest.raises(LLMResponseError, match="tool call limit"):
            await llm_request([{"role": "user", "content": "hi"}])
    else:
        result = await llm_request([{"role": "user", "content": "hi"}])
        assert len(result["choices"][0]["message"]["tool_calls"]) == count


@pytest.mark.parametrize("extra", [0, 1])
async def test_sse_tool_argument_limit_is_inclusive(monkeypatch, extra):
    event = {
        "choices": [
            {
                "delta": {"tool_calls": [_tool_call(0, "x" * (LLM_MAX_TOOL_ARGUMENT_BYTES + extra))]},
                "finish_reason": "tool_calls",
            }
        ]
    }
    raw = ("data: " + json.dumps(event, separators=(",", ":")) + "\n\ndata: [DONE]\n").encode()
    _stream_from_raw(monkeypatch, raw)

    if extra:
        with pytest.raises(LLMResponseError, match="tool argument byte limit"):
            await llm_request([{"role": "user", "content": "hi"}])
    else:
        result = await llm_request([{"role": "user", "content": "hi"}])
        assert len(result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]) == LLM_MAX_TOOL_ARGUMENT_BYTES


def test_llm_json_error_dict_fallback_redacts_every_secret_form():
    payload = {
        "error": {
            "api_key": "dict-key-sentinel",
            "Authorization": "Bearer dict-bearer-sentinel",
            "url": "https://provider.invalid/v1?api_key=dict-query-sentinel",
            "X-API-Key": "dict-header-sentinel",
        }
    }

    with pytest.raises(LLMResponseError) as exc_info:
        from src.http_client import _parse_chat_completion_json

        _parse_chat_completion_json(payload)

    message = str(exc_info.value)
    for sentinel in (
        "dict-key-sentinel",
        "dict-bearer-sentinel",
        "dict-query-sentinel",
        "dict-header-sentinel",
    ):
        assert sentinel not in message
    assert "[REDACTED]" in message


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('data: {"error":{"message":"provider denied request"}}\n\n', "provider denied"),
        ("data: {not-json}\n\ndata: [DONE]\n", "malformed SSE JSON"),
        ("data: [DONE]\n", "empty chat completion"),
    ],
)
async def test_llm_request_rejects_invalid_successful_streams(
    monkeypatch, body, message
):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        real_client,
        "stream",
        lambda self, method, url, **kwargs: _FakeStreamCM(sse=body),
    )

    with pytest.raises(LLMResponseError, match=message):
        await llm_request([{"role": "user", "content": "hi"}])


async def test_llm_request_accepts_tool_call_only_stream(monkeypatch):
    body = (
        'data: {"choices":[{"delta":{"role":"assistant","tool_calls":['
        '{"index":0,"id":"call_1","type":"function","function":'
        '{"name":"lookup","arguments":"{\\"x\\":"}}]}}]}\n\n'
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":'
        '{"arguments":"1}"}}]},"finish_reason":"tool_calls"}]}\n\n'
        "data: [DONE]\n"
    )
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        real_client,
        "stream",
        lambda self, method, url, **kwargs: _FakeStreamCM(sse=body),
    )

    result = await llm_request([{"role": "user", "content": "hi"}])

    choice = result["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"x":1}'},
        }
    ]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            'data: {"choices":[{"index":"bad","delta":{"content":"hi"}}]}\n\n',
            "invalid choice index",
        ),
        (
            'data: {"choices":[{"delta":{"tool_calls":['
            '{"index":null,"id":"call_1","type":"function","function":'
            '{"name":"lookup","arguments":"{}"}}]}}]}\n\n',
            "invalid tool call index",
        ),
    ],
)
async def test_llm_request_wraps_malformed_provider_indexes(
    monkeypatch, body, message
):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        real_client,
        "stream",
        lambda self, method, url, **kwargs: _FakeStreamCM(sse=body),
    )

    with pytest.raises(LLMResponseError, match=message):
        await llm_request([{"role": "user", "content": "hi"}])


async def test_llm_request_rejects_incomplete_streamed_tool_call(monkeypatch):
    body = (
        'data: {"choices":[{"delta":{"tool_calls":['
        '{"index":0,"type":"function","function":'
        '{"name":"lookup","arguments":"{}"}}]},'
        '"finish_reason":"tool_calls"}]}\n\n'
        "data: [DONE]\n"
    )
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        real_client,
        "stream",
        lambda self, method, url, **kwargs: _FakeStreamCM(sse=body),
    )

    with pytest.raises(LLMResponseError, match="incomplete tool call structure"):
        await llm_request([{"role": "user", "content": "hi"}])


async def test_llm_request_rejects_incomplete_json_tool_call(monkeypatch):
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"arguments": "{}"},
                        }
                    ],
                }
            }
        ]
    }
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        real_client,
        "stream",
        lambda self, method, url, **kwargs: _FakeStreamCM(
            json_body=payload, content_type="application/json"
        ),
    )

    with pytest.raises(LLMResponseError, match="incomplete tool call structure"):
        await llm_request([{"role": "user", "content": "hi"}])


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
