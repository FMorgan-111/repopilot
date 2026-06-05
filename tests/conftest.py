import asyncio
import inspect
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
            return httpx.Response(
                status_code=response["status_code"],
                json=response["json"],
                request=request,
            )
        raise AssertionError(f"No mocked response for {request.method} {request.url}")


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
