"""HTTP client helpers with retry and rate-limit awareness.

``github_request`` — GitHub API calls with exponential backoff + rate limiting.
``llm_request``   — LLM API calls with exponential backoff.
"""

import asyncio
import json

import httpx
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from .model_provider import (
    ProviderName,
    escalation_is_configured,
    get_model_config,
    get_model_name,
    sanitize_provider_error,
)
from .rate_limiter import get_github_limiter


def _load_dotenv_safely() -> None:
    """Load optional local defaults without replacing process configuration."""
    load_dotenv(override=False)


_load_dotenv_safely()

# ---------------------------------------------------------------------------
# Module-level connection pool (shared across all callers)
# ---------------------------------------------------------------------------

_llm_clients: dict[tuple[ProviderName, str], httpx.AsyncClient] = {}
# Connect timeout — establishing the TCP/TLS connection should be quick even
# when generation is slow.
LLM_CONNECT_TIMEOUT = 15.0
# Per-CHUNK idle timeout for the streamed response: a gap between SSE chunks
# longer than this means the gateway stalled. Because we stream, a progressing
# generation of ANY length stays under it — unlike a buffered response, whose
# whole generation had to finish within a single read timeout (the bug that
# made large prompts like tox time out at 60s / retry-double).
# 120s (not 60s): reasoning models (e.g. gpt-5.5) pause to think before emitting
# tokens, so the gap before/between chunks can exceed 60s and trip a ReadTimeout.
LLM_STREAM_IDLE_TIMEOUT = 120.0
# Hard wall-clock backstop for one streamed call. The idle timeout is the real
# bound; this only stops a pathologically long stream. Non-retryable on expiry.
# 300s covers the slow reasoning tail (gpt-5.5 single calls observed 100-123s)
# while the retry budget (320s) stays within the per-phase timeouts (see
# test_llm_phase_timeouts_cover_retry_window).
LLM_CALL_WALLCLOCK_TIMEOUT = 300.0
LLM_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
LLM_MAX_SSE_EVENTS = 50_000
LLM_MAX_CHOICES = 8
LLM_MAX_CONTENT_BYTES = 4 * 1024 * 1024
LLM_MAX_TOOL_CALLS = 64
LLM_MAX_TOOL_ARGUMENT_BYTES = 2 * 1024 * 1024


class LLMResponseError(RuntimeError):
    """The provider returned HTTP 2xx but no usable chat completion."""


def _get_llm_client(
    provider: ProviderName = "primary", base_url: str = ""
) -> httpx.AsyncClient:
    """Return the shared LLM :class:`httpx.AsyncClient` with connection pooling."""
    key = (provider, base_url)
    client = _llm_clients.get(key)
    if client is None:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(LLM_STREAM_IDLE_TIMEOUT, connect=LLM_CONNECT_TIMEOUT),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )
        _llm_clients[key] = client
    return client


def _reset_llm_client() -> None:
    """Reset the cached LLM client (useful between tests)."""
    clients = list(_llm_clients.values())
    _llm_clients.clear()
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        for client in clients:
            asyncio.run(client.aclose())
    else:
        for client in clients:
            loop.create_task(client.aclose())


async def close_llm_client() -> None:
    """Close and clear every shared LLM client."""
    clients = list(_llm_clients.values())
    _llm_clients.clear()
    for client in clients:
        await client.aclose()


# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------

RETRYABLE_GITHUB_STATUS = {429, 502, 503, 504}
RETRYABLE_LLM_STATUS = {502, 503, 504}
MAX_RETRIES = 3
LLM_MAX_ATTEMPTS = 2  # 1 initial + 1 retry (was 4 with MAX_RETRIES+1)
LLM_RETRY_BACKOFF_MAX_SECONDS = 20.0


def llm_retry_budget_seconds() -> float:
    """Worst-case wall-clock for one LLM call including a possible retry.

    A slow attempt is killed at the wall-clock ceiling with a non-retryable
    asyncio.TimeoutError, so two slow attempts can't stack. The true worst case
    is one fast transient failure + backoff + one slow attempt:
    LLM_CALL_WALLCLOCK_TIMEOUT + LLM_RETRY_BACKOFF_MAX_SECONDS.
    """
    return LLM_CALL_WALLCLOCK_TIMEOUT + LLM_RETRY_BACKOFF_MAX_SECONDS


def is_retryable_github_error(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.NetworkError, httpx.TimeoutException)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_GITHUB_STATUS
    return False


_is_retryable_github = is_retryable_github_error


def _is_retryable_llm(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.NetworkError, httpx.TimeoutException)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_LLM_STATUS
    return False


# ---------------------------------------------------------------------------
# GitHub request
# ---------------------------------------------------------------------------


async def github_request(method: str, url: str, **kwargs) -> httpx.Response:
    """GitHub API request with exponential-backoff retry and rate limiting.

    Retries on  429 / 502 / 503 / 504  plus  *NetworkError* / *TimeoutException*.
    Maximum 3 retries, exponential backoff: 1 s → 2 s → 4 s.

    The global :class:`RateLimiter` is consulted **before** every request so we
    never exceed GitHub's rate budget.  After a successful response the limiter
    is updated from the ``X-RateLimit-Remaining`` header.
    """
    limiter = get_github_limiter()
    await limiter.acquire()

    resp = await _github_request_with_retry(method, url, **kwargs)

    await limiter.update_from_headers(resp.headers)
    return resp


@retry(
    stop=stop_after_attempt(MAX_RETRIES + 1),  # 1 initial + 3 retries
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception(_is_retryable_github),
    reraise=True,
)
async def _github_request_with_retry(method: str, url: str, **kwargs) -> httpx.Response:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.request(method, url, **kwargs)
        resp.raise_for_status()
        return resp


# ---------------------------------------------------------------------------
# LLM request
# ---------------------------------------------------------------------------

def _get_llm_base_url() -> str:
    return get_model_config("primary").base_url


def _get_llm_model() -> str:
    return get_model_name("primary")


async def llm_request(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.2,
    *,
    provider: ProviderName = "primary",
    **kwargs: object,
) -> dict:
    """LLM API request with exponential-backoff retry.

    Retries on  502 / 503 / 504  plus  *NetworkError* / *TimeoutException*.
    Maximum 1 retry (2 total attempts), exponential backoff: 2 s → 4 s.
    Uses a shared connection pool (``_get_llm_client``) to avoid per-call
    TCP handshake overhead.
    """
    if provider == "escalation" and not escalation_is_configured():
        raise LLMResponseError("escalation provider is not configured")
    config = get_model_config(provider)
    url = f"{config.base_url}/chat/completions"
    payload: dict[str, object] = {
        "model": model or config.model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    payload.update(kwargs)
    headers = {
        "Authorization": f"Bearer {config.api_key.get_secret_value()}",
        "Content-Type": "application/json",
    }

    return await _llm_request_with_retry(
        url, payload, headers, provider, config.base_url
    )


@retry(
    stop=stop_after_attempt(LLM_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=2, min=2, max=LLM_RETRY_BACKOFF_MAX_SECONDS),
    retry=retry_if_exception(_is_retryable_llm),
    reraise=True,
)
async def _llm_request_with_retry(
    url: str,
    payload: dict,
    headers: dict,
    provider: ProviderName = "primary",
    base_url: str = "",
) -> dict:
    client = _get_llm_client(provider, base_url)

    async def _consume() -> dict:
        async with client.stream("POST", url, json=payload, headers=headers) as resp:
            if resp.status_code >= 400:
                await _drain_bounded_response(resp)
                resp.raise_for_status()
            content_type = resp.headers.get("content-type", "").lower()
            if "application/json" in content_type:
                raw = await _read_bounded_response(resp)
                try:
                    response = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise LLMResponseError("malformed chat completion JSON") from exc
                return _parse_chat_completion_json(response)
            return await _consume_sse_chat_completion(resp)

    # The per-chunk idle timeout (client read timeout) bounds a stall; this
    # wall-clock is a generous total backstop. On expiry asyncio.TimeoutError is
    # raised, which is intentionally NOT retryable — no doubling on a slow call.
    return await asyncio.wait_for(_consume(), timeout=LLM_CALL_WALLCLOCK_TIMEOUT)


async def _bounded_response_chunks(resp: object):
    total = 0
    async for chunk in resp.aiter_bytes():
        if not isinstance(chunk, bytes):
            raise LLMResponseError("malformed chat completion response bytes")
        total += len(chunk)
        if total > LLM_MAX_RESPONSE_BYTES:
            raise LLMResponseError("LLM response byte limit exceeded")
        if chunk:
            yield chunk


async def _read_bounded_response(resp: object) -> bytes:
    chunks: list[bytes] = []
    async for chunk in _bounded_response_chunks(resp):
        chunks.append(chunk)
    return b"".join(chunks)


async def _drain_bounded_response(resp: object) -> None:
    async for _ in _bounded_response_chunks(resp):
        pass


def _provider_error_message(error: object) -> str:
    if isinstance(error, dict):
        error = error.get("message") or error.get("detail") or error
    return sanitize_provider_error(RuntimeError(str(error)))


def _parse_chat_completion_json(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise LLMResponseError("unsupported chat completion response shape")
    if payload.get("error"):
        raise LLMResponseError(_provider_error_message(payload["error"]))
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMResponseError("empty chat completion response")
    if len(choices) > LLM_MAX_CHOICES:
        raise LLMResponseError("chat completion choice limit exceeded")
    aggregate_content_bytes = 0
    aggregate_tool_calls = 0
    usable = False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        validated_tool_calls = (
            _validate_tool_calls(tool_calls) if tool_calls is not None else []
        )
        aggregate_tool_calls += len(validated_tool_calls)
        if aggregate_tool_calls > LLM_MAX_TOOL_CALLS:
            raise LLMResponseError("chat completion tool call limit exceeded")
        if isinstance(content, str):
            content_bytes = len(content.encode("utf-8"))
            if content_bytes > LLM_MAX_CONTENT_BYTES:
                raise LLMResponseError("chat completion content byte limit exceeded")
            aggregate_content_bytes += content_bytes
            if aggregate_content_bytes > LLM_MAX_CONTENT_BYTES:
                raise LLMResponseError(
                    "chat completion aggregate content byte limit exceeded"
                )
        if (isinstance(content, str) and content) or validated_tool_calls:
            usable = True
    if usable:
        return payload
    raise LLMResponseError("empty chat completion response")


def _provider_index(value: object, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise LLMResponseError(f"invalid {label} index") from exc


def _validate_tool_calls(tool_calls: object) -> list[dict]:
    if not isinstance(tool_calls, list):
        raise LLMResponseError("incomplete tool call structure")
    if len(tool_calls) > LLM_MAX_TOOL_CALLS:
        raise LLMResponseError("chat completion tool call limit exceeded")
    for call in tool_calls:
        if not isinstance(call, dict):
            raise LLMResponseError("incomplete tool call structure")
        function = call.get("function")
        if (
            not isinstance(call.get("id"), str)
            or not call["id"]
            or call.get("type") != "function"
            or not isinstance(function, dict)
            or not isinstance(function.get("name"), str)
            or not function["name"]
            or not isinstance(function.get("arguments"), str)
        ):
            raise LLMResponseError("incomplete tool call structure")
        if (
            len(function["arguments"].encode("utf-8"))
            > LLM_MAX_TOOL_ARGUMENT_BYTES
        ):
            raise LLMResponseError("chat completion tool argument byte limit exceeded")
    return tool_calls


async def _bounded_sse_lines(resp: object):
    pending = bytearray()
    async for chunk in _bounded_response_chunks(resp):
        pending.extend(chunk)
        while True:
            newline = pending.find(b"\n")
            if newline < 0:
                break
            raw_line = bytes(pending[:newline])
            del pending[: newline + 1]
            if raw_line.endswith(b"\r"):
                raw_line = raw_line[:-1]
            try:
                yield raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise LLMResponseError("malformed SSE encoding") from exc
    if pending:
        try:
            yield bytes(pending).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise LLMResponseError("malformed SSE encoding") from exc


async def _consume_sse_chat_completion(resp: object) -> dict:
    choice_states: dict[int, dict] = {}
    usage: dict | None = None
    event_count = 0
    aggregate_content_bytes = 0
    tool_call_count = 0
    done_seen = False

    async for line in _bounded_sse_lines(resp):
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if not data:
            continue
        event_count += 1
        if event_count > LLM_MAX_SSE_EVENTS:
            raise LLMResponseError("chat completion SSE event limit exceeded")
        if data == "[DONE]":
            done_seen = True
            continue
        if done_seen:
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError as exc:
            raise LLMResponseError("malformed SSE JSON") from exc
        if not isinstance(event, dict):
            raise LLMResponseError("unsupported SSE event shape")
        if event.get("error"):
            raise LLMResponseError(_provider_error_message(event["error"]))
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]
        event_choices = event.get("choices") or []
        if not isinstance(event_choices, list):
            raise LLMResponseError("unsupported SSE choices shape")
        if len(event_choices) > LLM_MAX_CHOICES:
            raise LLMResponseError("chat completion choice limit exceeded")
        for position, choice in enumerate(event_choices):
            if not isinstance(choice, dict):
                continue
            index = _provider_index(choice.get("index", position), "choice")
            if index not in choice_states and len(choice_states) >= LLM_MAX_CHOICES:
                raise LLMResponseError("chat completion choice limit exceeded")
            state = choice_states.setdefault(
                index,
                {
                    "content": [],
                    "content_bytes": 0,
                    "role": "assistant",
                    "tool_calls": {},
                    "finish_reason": None,
                },
            )
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                raise LLMResponseError("unsupported SSE delta shape")
            if isinstance(delta.get("role"), str):
                state["role"] = delta["role"]
            if isinstance(delta.get("content"), str):
                chunk_bytes = len(delta["content"].encode("utf-8"))
                if state["content_bytes"] + chunk_bytes > LLM_MAX_CONTENT_BYTES:
                    raise LLMResponseError(
                        "chat completion content byte limit exceeded"
                    )
                aggregate_content_bytes += chunk_bytes
                if aggregate_content_bytes > LLM_MAX_CONTENT_BYTES:
                    raise LLMResponseError(
                        "chat completion aggregate content byte limit exceeded"
                    )
                state["content"].append(delta["content"])
                state["content_bytes"] += chunk_bytes
            tool_call_count += _accumulate_tool_call_deltas(
                state["tool_calls"],
                delta.get("tool_calls"),
                remaining=LLM_MAX_TOOL_CALLS - tool_call_count,
            )
            if choice.get("finish_reason") is not None:
                state["finish_reason"] = choice["finish_reason"]

    choices: list[dict] = []
    for index, state in sorted(choice_states.items()):
        content = "".join(state["content"])
        message: dict[str, object] = {
            "role": state["role"],
            "content": content or None,
        }
        tool_calls = _finalize_tool_calls(state["tool_calls"])
        if tool_calls:
            message["tool_calls"] = tool_calls
        if not content and not tool_calls:
            continue
        choices.append(
            {
                "index": index,
                "message": message,
                "finish_reason": state["finish_reason"],
            }
        )
    if not choices:
        raise LLMResponseError("empty chat completion response")
    result: dict[str, object] = {"choices": choices}
    if usage is not None:
        result["usage"] = usage
    return result


def _accumulate_tool_call_deltas(
    states: dict[int, dict],
    deltas: object,
    *,
    remaining: int = LLM_MAX_TOOL_CALLS,
) -> int:
    if deltas is None:
        return 0
    if not isinstance(deltas, list):
        raise LLMResponseError("unsupported tool call delta shape")
    created = 0
    for position, delta in enumerate(deltas):
        if not isinstance(delta, dict):
            continue
        index = _provider_index(delta.get("index", position), "tool call")
        if index not in states:
            if created >= remaining:
                raise LLMResponseError("chat completion tool call limit exceeded")
            created += 1
        state = states.setdefault(
            index,
            {
                "id": "",
                "type": "",
                "name": "",
                "arguments": "",
                "argument_bytes": 0,
            },
        )
        if isinstance(delta.get("id"), str):
            state["id"] += delta["id"]
        if isinstance(delta.get("type"), str):
            state["type"] = delta["type"]
        function = delta.get("function") or {}
        if isinstance(function, dict):
            if isinstance(function.get("name"), str):
                state["name"] += function["name"]
            if isinstance(function.get("arguments"), str):
                argument_bytes = len(function["arguments"].encode("utf-8"))
                if (
                    state["argument_bytes"] + argument_bytes
                    > LLM_MAX_TOOL_ARGUMENT_BYTES
                ):
                    raise LLMResponseError(
                        "chat completion tool argument byte limit exceeded"
                    )
                state["arguments"] += function["arguments"]
                state["argument_bytes"] += argument_bytes
    return created


def _finalize_tool_calls(states: dict[int, dict]) -> list[dict]:
    calls: list[dict] = []
    for _, state in sorted(states.items()):
        calls.append(
            {
                "id": state["id"],
                "type": state["type"],
                "function": {
                    "name": state["name"],
                    "arguments": state["arguments"],
                },
            }
        )
    return _validate_tool_calls(calls)
