import asyncio
import inspect
import json
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _reset_llm_global():
    """Reset the shared LLM connection-pool client between tests so each
    test gets a fresh transport (important when httpx_mock is in play)."""
    from src.http_client import _reset_llm_client
    _reset_llm_client()
    yield
    _reset_llm_client()


@pytest.fixture(autouse=True)
def _pin_llm_env(monkeypatch):
    """Pin the LLM endpoint/model to deterministic test defaults so httpx_mock's
    hard-coded api.deepseek.com URL matches. Without this, a developer's .env
    pointing OPENAI_BASE_URL at a real gateway (e.g. the Gemini proxy) makes the
    request URL mismatch every mock and the whole http_client/llm suite goes red.
    """
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("LLM_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    # Higher-priority keys must be cleared too, else a real .env key leaks into
    # the auth header and breaks the "Bearer test-key" assertion.
    monkeypatch.delenv("LINOAPI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)


class HTTPXMock:
    def __init__(self):
        self._responses = []
        self.requests = []

    def add_response(self, method="GET", url=None, status_code=200, json=None):
        self._responses.append(
            {
                "method": method.upper(),
                "url": str(url) if url is not None else None,
                "status_code": status_code,
                "json": json,
            }
        )

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        for index, response in enumerate(self._responses):
            if response["method"] != request.method:
                continue
            if response["url"] is not None and response["url"] != str(request.url):
                continue
            self._responses.pop(index)
            return self._make_response(request, response)
        raise AssertionError(f"No mocked response for {request.method} {request.url}")

    def _make_response(self, request, response) -> httpx.Response:
        status = response["status_code"]
        body = response["json"]
        # The LLM client streams (stream=True). Re-serve the mocked completion as
        # Server-Sent Events so the streaming parser reconstructs it — keeps the
        # existing json= mocks working without per-test changes.
        wants_stream = False
        try:
            wants_stream = bool(json.loads(request.content or b"{}").get("stream"))
        except Exception:
            wants_stream = False
        if wants_stream and status == 200 and isinstance(body, dict) and body.get("choices"):
            content = body["choices"][0].get("message", {}).get("content", "")
            sse = (
                "data: "
                + json.dumps({"choices": [{"delta": {"content": content}}]})
                + "\n\ndata: [DONE]\n"
            )
            return httpx.Response(
                status_code=status, content=sse.encode(), request=request
            )
        return httpx.Response(status_code=status, json=body, request=request)


@pytest.fixture
def httpx_mock(monkeypatch):
    mock = HTTPXMock()
    original_async_client = httpx.AsyncClient

    def async_client_factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(mock.handler)
        return original_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", async_client_factory)
    return mock


def pytest_pyfunc_call(pyfuncitem):
    test_func = pyfuncitem.obj
    if not inspect.iscoroutinefunction(test_func):
        return None

    fixture_names = pyfuncitem._fixtureinfo.argnames
    kwargs = {
        name: pyfuncitem.funcargs[name]
        for name in fixture_names
        if name in pyfuncitem.funcargs
    }
    asyncio.run(test_func(**kwargs))
    return True
