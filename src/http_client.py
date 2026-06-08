"""HTTP client helpers with retry and rate-limit awareness.

``github_request`` — GitHub API calls with exponential backoff + rate limiting.
``llm_request``   — LLM API calls with exponential backoff.
"""

import os

import httpx
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from .rate_limiter import get_github_limiter

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Module-level connection pool (shared across all callers)
# ---------------------------------------------------------------------------

_llm_client: httpx.AsyncClient | None = None


def _get_llm_client() -> httpx.AsyncClient:
    """Return the shared LLM :class:`httpx.AsyncClient` with connection pooling."""
    global _llm_client
    if _llm_client is None:
        _llm_client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )
    return _llm_client


def _reset_llm_client() -> None:
    """Reset the cached LLM client (useful between tests)."""
    global _llm_client
    _llm_client = None


# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------

RETRYABLE_GITHUB_STATUS = {429, 502, 503, 504}
RETRYABLE_LLM_STATUS = {502, 503, 504}
MAX_RETRIES = 3
LLM_MAX_ATTEMPTS = 2  # 1 initial + 1 retry (was 4 with MAX_RETRIES+1)


def _is_retryable_github(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.NetworkError, httpx.TimeoutException)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_GITHUB_STATUS
    return False


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

def _get_llm_api_key() -> str:
    return os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY", "")


def _get_llm_base_url() -> str:
    return os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")


def _get_llm_model() -> str:
    return os.getenv("LLM_MODEL", "deepseek-v4-pro")


async def llm_request(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.2,
    **kwargs,
) -> dict:
    """LLM API request with exponential-backoff retry.

    Retries on  502 / 503 / 504  plus  *NetworkError* / *TimeoutException*.
    Maximum 1 retry (2 total attempts), exponential backoff: 2 s → 4 s.
    Uses a shared connection pool (``_get_llm_client``) to avoid per-call
    TCP handshake overhead.
    """
    url = f"{_get_llm_base_url()}/chat/completions"
    payload: dict[str, object] = {
        "model": model or _get_llm_model(),
        "messages": messages,
        "temperature": temperature,
    }
    payload.update(kwargs)
    headers = {
        "Authorization": f"Bearer {_get_llm_api_key()}",
        "Content-Type": "application/json",
    }

    return await _llm_request_with_retry(url, payload, headers)


@retry(
    stop=stop_after_attempt(LLM_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    retry=retry_if_exception(_is_retryable_llm),
    reraise=True,
)
async def _llm_request_with_retry(url: str, payload: dict, headers: dict) -> dict:
    client = _get_llm_client()
    resp = await client.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    return resp.json()
