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


def is_retryable_llm_error(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.NetworkError, httpx.TimeoutException)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_LLM_STATUS
    return False


_is_retryable_llm = is_retryable_llm_error


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
    aggregate_tool_argument_bytes = 0
    canonical_choices: list[dict] = []
    seen_indexes: set[int] = set()
    for position, choice in enumerate(choices):
        if not isinstance(choice, dict):
            raise LLMResponseError("incomplete chat completion choice structure")
        index = _provider_index(choice.get("index", position), "choice")
        if index in seen_indexes:
            raise LLMResponseError("duplicate choice index")
        seen_indexes.add(index)
        message = choice.get("message")
        if not isinstance(message, dict):
            raise LLMResponseError("incomplete chat completion choice structure")
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise LLMResponseError("incomplete chat completion message structure")
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise LLMResponseError("invalid chat completion content structure")
        tool_calls = message.get("tool_calls")
        validated_tool_calls = (
            _validate_tool_calls(tool_calls) if tool_calls is not None else []
        )
        aggregate_tool_calls += len(validated_tool_calls)
        if aggregate_tool_calls > LLM_MAX_TOOL_CALLS:
            raise LLMResponseError("chat completion tool call limit exceeded")
        aggregate_tool_argument_bytes += sum(
            len(call["function"]["arguments"].encode("utf-8"))
            for call in validated_tool_calls
        )
        if aggregate_tool_argument_bytes > LLM_MAX_TOOL_ARGUMENT_BYTES:
            raise LLMResponseError(
                "chat completion aggregate tool argument byte limit exceeded"
            )
        if isinstance(content, str):
            content_bytes = len(content.encode("utf-8"))
            if content_bytes > LLM_MAX_CONTENT_BYTES:
                raise LLMResponseError("chat completion content byte limit exceeded")
            aggregate_content_bytes += content_bytes
            if aggregate_content_bytes > LLM_MAX_CONTENT_BYTES:
                raise LLMResponseError(
                    "chat completion aggregate content byte limit exceeded"
                )
        if not ((isinstance(content, str) and content) or validated_tool_calls):
            raise LLMResponseError("empty chat completion choice")
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise LLMResponseError("invalid chat completion finish reason")
        canonical_message: dict[str, object] = {
            "role": role,
            "content": content,
        }
        if validated_tool_calls:
            canonical_message["tool_calls"] = validated_tool_calls
        canonical_choices.append(
            {
                "index": index,
                "message": canonical_message,
                "finish_reason": finish_reason,
            }
        )
    result: dict[str, object] = {"choices": canonical_choices}
    if "usage" in payload and payload["usage"] is not None:
        result["usage"] = _validate_usage(payload["usage"])
    return result


def _provider_index(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise LLMResponseError(f"invalid {label} index")
    return value


_USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def _validate_usage(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        raise LLMResponseError("invalid chat completion usage")
    canonical: dict[str, int] = {}
    for field in _USAGE_FIELDS:
        if field not in value:
            continue
        count = value[field]
        if type(count) is not int or not 0 <= count <= (2**63 - 1):
            raise LLMResponseError("invalid chat completion usage")
        canonical[field] = count
    return canonical


def _validate_tool_calls(tool_calls: object) -> list[dict]:
    if not isinstance(tool_calls, list):
        raise LLMResponseError("incomplete tool call structure")
    if len(tool_calls) > LLM_MAX_TOOL_CALLS:
        raise LLMResponseError("chat completion tool call limit exceeded")
    canonical: list[dict] = []
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
        canonical.append(
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": function["name"],
                    "arguments": function["arguments"],
                },
            }
        )
    return canonical


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
    aggregate_tool_argument_bytes = 0
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
            raise LLMResponseError("chat completion data after DONE")
        try:
            event = json.loads(data)
        except json.JSONDecodeError as exc:
            raise LLMResponseError("malformed SSE JSON") from exc
        if not isinstance(event, dict):
            raise LLMResponseError("unsupported SSE event shape")
        if event.get("error"):
            raise LLMResponseError(_provider_error_message(event["error"]))
        if "usage" in event and event["usage"] is not None:
            usage = _validate_usage(event["usage"])
        event_choices = event.get("choices", [])
        if not isinstance(event_choices, list):
            raise LLMResponseError("unsupported SSE choices shape")
        if len(event_choices) > LLM_MAX_CHOICES:
            raise LLMResponseError("chat completion choice limit exceeded")
        event_indexes: set[int] = set()
        for position, choice in enumerate(event_choices):
            if not isinstance(choice, dict):
                raise LLMResponseError("unsupported SSE choice shape")
            index = _provider_index(choice.get("index", position), "choice")
            if index in event_indexes:
                raise LLMResponseError("duplicate choice index in SSE event")
            event_indexes.add(index)
            if index not in choice_states and len(choice_states) >= LLM_MAX_CHOICES:
                raise LLMResponseError("chat completion choice limit exceeded")
            state = choice_states.setdefault(
                index,
                {
                    "content": [],
                    "content_bytes": 0,
                    "role": None,
                    "tool_calls": {},
                    "finish_reason": None,
                },
            )
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                raise LLMResponseError("unsupported SSE delta shape")
            if "role" in delta:
                role = delta["role"]
                if not isinstance(role, str) or not role:
                    raise LLMResponseError("invalid SSE role structure")
                if state["role"] is not None and state["role"] != role:
                    raise LLMResponseError("conflicting role in SSE stream")
                state["role"] = role
            if "content" in delta and delta["content"] is not None:
                if not isinstance(delta["content"], str):
                    raise LLMResponseError("invalid SSE content structure")
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
            created, argument_bytes = _accumulate_tool_call_deltas(
                state["tool_calls"],
                delta.get("tool_calls"),
                remaining=LLM_MAX_TOOL_CALLS - tool_call_count,
            )
            tool_call_count += created
            aggregate_tool_argument_bytes += argument_bytes
            if aggregate_tool_argument_bytes > LLM_MAX_TOOL_ARGUMENT_BYTES:
                raise LLMResponseError(
                    "chat completion aggregate tool argument byte limit exceeded"
                )
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
                if not isinstance(finish_reason, str):
                    raise LLMResponseError("invalid SSE finish reason")
                if (
                    state["finish_reason"] is not None
                    and state["finish_reason"] != finish_reason
                ):
                    raise LLMResponseError("conflicting finish reason in SSE stream")
                state["finish_reason"] = finish_reason

    if not done_seen:
        raise LLMResponseError("incomplete chat completion SSE stream")

    choices: list[dict] = []
    for index, state in sorted(choice_states.items()):
        content = "".join(state["content"])
        message: dict[str, object] = {
            "role": state["role"] or "assistant",
            "content": content or None,
        }
        tool_calls = _finalize_tool_calls(state["tool_calls"])
        if tool_calls:
            message["tool_calls"] = tool_calls
        if not content and not tool_calls:
            raise LLMResponseError("empty chat completion choice")
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
) -> tuple[int, int]:
    if deltas is None:
        return 0, 0
    if not isinstance(deltas, list):
        raise LLMResponseError("unsupported tool call delta shape")
    created = 0
    added_argument_bytes = 0
    event_indexes: set[int] = set()
    for position, delta in enumerate(deltas):
        if not isinstance(delta, dict):
            raise LLMResponseError("unsupported tool call delta shape")
        index = _provider_index(delta.get("index", position), "tool call")
        if index in event_indexes:
            raise LLMResponseError("duplicate tool call index in SSE delta")
        event_indexes.add(index)
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
        if "id" in delta:
            call_id = delta["id"]
            if not isinstance(call_id, str) or not call_id:
                raise LLMResponseError("invalid tool call id structure")
            if state["id"] and state["id"] != call_id:
                raise LLMResponseError("conflicting tool call id in SSE stream")
            state["id"] = call_id
        if "type" in delta:
            call_type = delta["type"]
            if call_type != "function":
                raise LLMResponseError("invalid tool call type structure")
            if state["type"] and state["type"] != call_type:
                raise LLMResponseError("conflicting tool call type in SSE stream")
            state["type"] = call_type
        if "function" in delta:
            function = delta["function"]
            if not isinstance(function, dict):
                raise LLMResponseError("unsupported tool call function shape")
            if "name" in function:
                name = function["name"]
                if not isinstance(name, str) or not name:
                    raise LLMResponseError("invalid tool call name structure")
                if state["name"] and state["name"] != name:
                    raise LLMResponseError("conflicting tool call name in SSE stream")
                state["name"] = name
            if "arguments" in function:
                arguments = function["arguments"]
                if not isinstance(arguments, str):
                    raise LLMResponseError("invalid tool call arguments structure")
                argument_bytes = len(arguments.encode("utf-8"))
                if (
                    state["argument_bytes"] + argument_bytes
                    > LLM_MAX_TOOL_ARGUMENT_BYTES
                ):
                    raise LLMResponseError(
                        "chat completion tool argument byte limit exceeded"
                    )
                state["arguments"] += arguments
                state["argument_bytes"] += argument_bytes
                added_argument_bytes += argument_bytes
    return created, added_argument_bytes


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
