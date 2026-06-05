import os
import httpx

GITHUB_API = "https://api.github.com"


def _headers() -> dict:
    token = os.getenv("GITHUB_TOKEN")
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


async def read_issue(owner: str, repo: str, issue_number: int) -> dict:
    """Fetch issue title, body, labels, and state from GitHub."""
    url = f"{GITHUB_API}/repos/{owner}/{repo}/issues/{issue_number}"
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, headers=_headers())
    resp.raise_for_status()
    data = resp.json()
    return {
        "title": data.get("title", ""),
        "body": data.get("body", "") or "",
        "state": data.get("state", ""),
        "labels": [lbl["name"] for lbl in data.get("labels", [])],
        "number": data.get("number"),
    }


async def search_code(query: str, owner: str, repo: str) -> list[dict]:
    """Search GitHub code for files related to the query in the given repo."""
    q = f"repo:{owner}/{repo} {query}"
    url = f"{GITHUB_API}/search/code"
    params = {"q": q, "per_page": 10}
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, headers=_headers(), params=params)
    resp.raise_for_status()
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
