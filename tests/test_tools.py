import base64

import httpx
import pytest

from src.tools import read_file, read_issue, search_code


async def test_read_issue_success_returns_title_body_labels(httpx_mock):
    httpx_mock.add_response(
        url="https://api.github.com/repos/acme/widget/issues/42",
        json={
            "title": "Login fails",
            "body": "Stack trace here",
            "state": "open",
            "number": 42,
            "labels": [{"name": "bug"}, {"name": "high-priority"}],
        },
    )

    result = await read_issue("acme", "widget", 42)

    assert result == {
        "title": "Login fails",
        "body": "Stack trace here",
        "state": "open",
        "labels": ["bug", "high-priority"],
        "number": 42,
    }


async def test_read_issue_http_error(httpx_mock):
    httpx_mock.add_response(
        url="https://api.github.com/repos/acme/widget/issues/404",
        status_code=404,
        json={"message": "Not Found"},
    )

    with pytest.raises(httpx.HTTPStatusError):
        await read_issue("acme", "widget", 404)


async def test_search_code_success_returns_file_list(httpx_mock):
    httpx_mock.add_response(
        url=(
            "https://api.github.com/search/code"
            "?q=repo%3Aacme%2Fwidget+login+failure&per_page=10"
        ),
        json={
            "items": [
                {
                    "path": "src/auth.py",
                    "repository": {"full_name": "acme/widget"},
                    "html_url": "https://github.com/acme/widget/blob/main/src/auth.py",
                    "sha": "abc123",
                },
                {
                    "path": "tests/test_auth.py",
                    "repository": {"full_name": "acme/widget"},
                    "html_url": "https://github.com/acme/widget/blob/main/tests/test_auth.py",
                    "sha": "def456",
                },
            ]
        },
    )

    result = await search_code("login failure", "acme", "widget")

    assert result == [
        {
            "path": "src/auth.py",
            "repository": "acme/widget",
            "url": "https://github.com/acme/widget/blob/main/src/auth.py",
            "sha": "abc123",
        },
        {
            "path": "tests/test_auth.py",
            "repository": "acme/widget",
            "url": "https://github.com/acme/widget/blob/main/tests/test_auth.py",
            "sha": "def456",
        },
    ]


async def test_search_code_empty_result(httpx_mock):
    httpx_mock.add_response(
        url=(
            "https://api.github.com/search/code"
            "?q=repo%3Aacme%2Fwidget+missing+thing&per_page=10"
        ),
        json={"items": []},
    )

    assert await search_code("missing thing", "acme", "widget") == []


async def test_read_file_decodes_base64_content(httpx_mock):
    content = base64.b64encode(b"print('hi')\n").decode()
    httpx_mock.add_response(
        url="https://api.github.com/repos/acme/widget/contents/src/app.py",
        json={"path": "src/app.py", "content": content, "sha": "abc123", "size": 12, "encoding": "base64"},
    )

    result = await read_file("acme", "widget", "src/app.py")

    assert result == {
        "path": "src/app.py",
        "content": "print('hi')\n",
        "sha": "abc123",
        "size": 12,
    }


async def test_read_file_http_error(httpx_mock):
    httpx_mock.add_response(
        url="https://api.github.com/repos/acme/widget/contents/missing.py",
        status_code=404,
        json={"message": "Not Found"},
    )

    with pytest.raises(httpx.HTTPStatusError):
        await read_file("acme", "widget", "missing.py")
