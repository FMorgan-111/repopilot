import base64
import os

from .cache import cached
from .http_client import github_request

GITHUB_API = "https://api.github.com"


def _headers() -> dict:
    token = os.getenv("GITHUB_TOKEN")
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


@cached
async def read_issue(owner: str, repo: str, issue_number: int) -> dict:
    """Fetch issue title, body, labels, and state from GitHub."""
    url = f"{GITHUB_API}/repos/{owner}/{repo}/issues/{issue_number}"
    resp = await github_request("GET", url, headers=_headers())
    data = resp.json()
    return {
        "title": data.get("title", ""),
        "body": data.get("body", "") or "",
        "state": data.get("state", ""),
        "labels": [lbl["name"] for lbl in data.get("labels", [])],
        "number": data.get("number"),
    }


@cached
async def search_code(query: str, owner: str, repo: str) -> list[dict]:
    """Search GitHub code for files related to the query in the given repo."""
    q = f"repo:{owner}/{repo} {query}"
    url = f"{GITHUB_API}/search/code"
    params = {"q": q, "per_page": 10}
    resp = await github_request("GET", url, headers=_headers(), params=params)
    items = resp.json().get("items", [])
    return [
        {
            "path": item["path"],
            "repository": item["repository"]["full_name"],
            "url": item.get("html_url", ""),
            "sha": item.get("sha", ""),
        }
        for item in items
    ]


@cached
async def read_file(owner: str, repo: str, path: str) -> dict:
    """Fetch a file's contents from GitHub, decoded from base64."""
    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"
    resp = await github_request("GET", url, headers=_headers())
    data = resp.json()
    content = base64.b64decode(data.get("content", "")).decode("utf-8", errors="replace")
    return {
        "path": data.get("path", path),
        "content": content,
        "sha": data.get("sha", ""),
        "size": data.get("size", 0),
    }
