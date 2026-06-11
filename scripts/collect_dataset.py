#!/usr/bin/env python3
"""Collect GitHub Issue -> PR -> diff pairs for Python bug fixes."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

GITHUB_API = "https://api.github.com"
MIN_REQUEST_INTERVAL = 0.3

# Repos use current canonical owner/name (GitHub Search API rejects transferred repos)
DEFAULT_REPOS = [
    "fastapi/fastapi",
    "django/django",
    "pallets/flask",
    "pydantic/pydantic",
    "pytest-dev/pytest",
    "psf/requests",
    "encode/httpx",
    "encode/starlette",
    "celery/celery",
    "sqlalchemy/sqlalchemy",
    "python-poetry/poetry",
    "psf/black",
    "astral-sh/ruff",
    "python/mypy",
    "Textualize/rich",
    "fastapi/typer",
    "Textualize/textual",
    "numpy/numpy",
    "pandas-dev/pandas",
    "scikit-learn/scikit-learn",
    "scipy/scipy",
    "matplotlib/matplotlib",
    "scrapy/scrapy",
    "ansible/ansible",
    "home-assistant/core",
    "apache/airflow",
    "prefecthq/prefect",
    "python/cpython",
    "sphinx-doc/sphinx",
    "jupyter/notebook",
    "jupyterlab/jupyterlab",
    "ipython/ipython",
    "pallets/click",
    "pallets/jinja",
    "pallets/werkzeug",
    "tornadoweb/tornado",
    "aio-libs/aiohttp",
    "python-pillow/Pillow",
    "huggingface/transformers",
    "huggingface/datasets",
    "keras-team/keras",
    "tensorflow/tensorflow",
    "openai/openai-python",
    "langchain-ai/langchain",
    "getsentry/sentry-python",
    "wagtail/wagtail",
    "zulip/zulip",
    "open-mmlab/mmdetection",
    "boto/boto3",
    "apache/superset",
]

EXCLUDED_LABELS = {
    "enhancement",
    "feature",
    "docs",
    "documentation",
    "dependencies",
    "dependency",
    "deps",
    "invalid",
    "wontfix",
}

BOT_HINTS = (
    "[bot]",
    "bot",
    "dependabot",
    "pre-commit-ci",
    "github-actions",
    "renovate",
    "mergify",
)

GENERATED_OR_VENDOR_PARTS = {
    "vendor",
    "vendors",
    "third_party",
    "third-party",
    "node_modules",
    "generated",
    "__generated__",
    "build",
    "dist",
}

LOCKFILE_NAMES = {
    "poetry.lock",
    "pdm.lock",
    "pipfile.lock",
    "requirements.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "conda-lock.yml",
    "conda-lock.yaml",
}

CLOSING_REF_RE = re.compile(
    r"\b(?:fix(?:e[sd])?|close[sd]?|resolve[sd]?)\s+"
    r"(?:"
    r"https://github\.com/[^/\s]+/[^/\s]+/issues/(\d+)"
    r"|(?:[\w.-]+/[\w.-]+)?#(\d+)"
    r")",
    re.IGNORECASE,
)

FORMATTING_RE = re.compile(r"\b(format(?:ting)?|black|ruff format|autopep8|isort|prettier)\b", re.IGNORECASE)
DEPENDENCY_RE = re.compile(r"\b(dependenc(?:y|ies)|bump|upgrade|pin|requirements)\b", re.IGNORECASE)


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_gh_auth_token() -> str:
    try:
        proc = subprocess.run(
            ["gh", "auth", "token"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        env_token = os.environ.get("GITHUB_TOKEN", "").strip()
        if env_token:
            return env_token
        raise SystemExit("Could not get GitHub token. Run `gh auth login` first.") from exc
    token = proc.stdout.strip()
    if not token:
        raise SystemExit("`gh auth token` returned an empty token.")
    return token


def load_repo_list(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_REPOS[:]

    text = Path(path).read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        data = json.loads(stripped)
        repos = [str(item).strip() for item in data]
    else:
        repos = []
        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            repos.extend(part.strip() for part in line.split(",") if part.strip())

    seen: set[str] = set()
    unique = []
    for repo in repos:
        if "/" not in repo or repo in seen:
            continue
        seen.add(repo)
        unique.append(repo)
    return unique


def labels_from_issue(issue: dict[str, Any]) -> list[str]:
    labels = []
    for label in issue.get("labels") or []:
        if isinstance(label, dict):
            name = label.get("name")
        else:
            name = str(label)
        if name:
            labels.append(str(name))
    return labels


def has_excluded_labels(labels: list[str]) -> bool:
    lowered = {label.strip().lower() for label in labels}
    return bool(lowered & EXCLUDED_LABELS)


def has_bug_label(labels: list[str]) -> bool:
    lowered = {label.strip().lower() for label in labels}
    # Match common bug label variants across different repo conventions
    bug_patterns = {"bug", "type: bug", "kind/bug", "bug report", "t: bug", "bugfix"}
    return bool(lowered & bug_patterns) or any("bug" in label for label in lowered)


def is_bot_actor(actor: Any) -> bool:
    if isinstance(actor, dict):
        login = str(actor.get("login") or "")
        actor_type = str(actor.get("type") or "")
        if actor_type.lower() == "bot":
            return True
    else:
        login = str(actor or "")
    normalized = login.lower()
    return any(hint in normalized for hint in BOT_HINTS)


def is_meaningful_issue(issue: dict[str, Any]) -> bool:
    title = str(issue.get("title") or "").strip()
    body = str(issue.get("body") or "").strip()
    if len(title) < 4 or len(body) < 25:
        return False
    words = re.findall(r"\w+", body)
    return len(words) >= 5


def extract_closing_issue_numbers(body: str | None) -> set[int]:
    numbers: set[int] = set()
    for match in CLOSING_REF_RE.finditer(body or ""):
        number = match.group(1) or match.group(2)
        if number:
            numbers.add(int(number))
    return numbers


def pr_body_references_issue(pr: dict[str, Any], issue_number: int) -> bool:
    return issue_number in extract_closing_issue_numbers(pr.get("body") or "")


def changed_lines(files: list[dict[str, Any]]) -> int:
    return sum(int(file.get("additions") or 0) + int(file.get("deletions") or 0) for file in files)


def is_lockfile_path(path: str) -> bool:
    name = Path(path).name.lower()
    return name in LOCKFILE_NAMES or name.endswith(".lock")


def is_generated_or_vendor_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    parts = {part for part in normalized.split("/") if part}
    if parts & GENERATED_OR_VENDOR_PARTS:
        return True
    if is_lockfile_path(normalized):
        return True
    if normalized.endswith((".min.js", ".generated.py", ".pb.go", ".pb.py")):
        return True
    return False


def has_tests_changed(files: list[dict[str, Any]]) -> bool:
    for file in files:
        path = str(file.get("filename") or file.get("path") or "").replace("\\", "/").lower()
        name = Path(path).name
        if "/tests/" in f"/{path}" or name.startswith("test_") or name.endswith("_test.py"):
            return True
    return False


def fix_size_bucket(files: list[dict[str, Any]]) -> str:
    total = changed_lines(files)
    if total <= 50:
        return "small"
    if total <= 150:
        return "medium"
    return "large"


def is_probably_formatting_only(pr: dict[str, Any], files: list[dict[str, Any]]) -> bool:
    text = f"{pr.get('title') or ''}\n{pr.get('body') or ''}"
    if not FORMATTING_RE.search(text):
        return False
    return not has_tests_changed(files)


def is_probably_dependency_change(pr: dict[str, Any], files: list[dict[str, Any]]) -> bool:
    text = f"{pr.get('title') or ''}\n{pr.get('body') or ''}"
    if DEPENDENCY_RE.search(text):
        return True
    return all(is_lockfile_path(str(file.get("filename") or "")) for file in files)


def should_keep_pr_files(files: list[dict[str, Any]]) -> bool:
    if not 1 <= len(files) <= 5:
        return False
    total = changed_lines(files)
    if not 5 <= total <= 300:
        return False
    if all(str(file.get("status") or "").lower() == "renamed" for file in files):
        return False
    if all(is_generated_or_vendor_path(str(file.get("filename") or "")) for file in files):
        return False
    return True


def should_keep_issue(issue: dict[str, Any]) -> bool:
    labels = labels_from_issue(issue)
    if not has_bug_label(labels) or has_excluded_labels(labels):
        return False
    if is_bot_actor(issue.get("user")):
        return False
    return is_meaningful_issue(issue)


def should_keep_pr(issue: dict[str, Any], pr: dict[str, Any], files: list[dict[str, Any]]) -> bool:
    if not pr.get("merged_at"):
        return False
    if is_bot_actor(pr.get("user")):
        return False
    if not str(pr.get("body") or "").strip():
        return False
    if not pr_body_references_issue(pr, int(issue["number"])):
        return False
    if is_probably_dependency_change(pr, files):
        return False
    if is_probably_formatting_only(pr, files):
        return False
    return should_keep_pr_files(files)


def repo_parts(repo_full_name: str) -> tuple[str, str]:
    owner, name = repo_full_name.split("/", 1)
    return owner, name


def normalize_file(file: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": file.get("filename") or file.get("path") or "",
        "status": file.get("status") or "",
        "additions": int(file.get("additions") or 0),
        "deletions": int(file.get("deletions") or 0),
        "patch": file.get("patch") or "",
    }


def build_dataset_record(
    repo: dict[str, Any],
    issue: dict[str, Any],
    pr: dict[str, Any],
    full_diff: str,
    files: list[dict[str, Any]],
    linked_by: str,
) -> dict[str, Any]:
    full_name = repo["full_name"]
    owner, name = repo_parts(full_name)
    issue_number = int(issue["number"])
    pr_number = int(pr["number"])
    return {
        "id": f"{full_name}#{issue_number}:{pr_number}",
        "repo": {
            "owner": owner,
            "name": name,
            "stars": int(repo.get("stargazers_count") or 0),
            "language": repo.get("language") or "Python",
        },
        "issue": {
            "number": issue_number,
            "url": issue.get("html_url") or "",
            "title": issue.get("title") or "",
            "body": issue.get("body") or "",
            "labels": labels_from_issue(issue),
            "created_at": issue.get("created_at") or "",
            "closed_at": issue.get("closed_at") or "",
        },
        "pr": {
            "number": pr_number,
            "url": pr.get("html_url") or "",
            "title": pr.get("title") or "",
            "body": pr.get("body") or "",
            "merged_at": pr.get("merged_at") or "",
            "linked_by": linked_by,
        },
        "patch": {
            "full_diff": full_diff,
            "files": [normalize_file(file) for file in files],
        },
        "signals": {
            "has_tests_changed": has_tests_changed(files),
            "fix_size_bucket": fix_size_bucket(files),
        },
        "collected_at": now_utc(),
    }


class StateStore:
    def __init__(self, path: Path, resume: bool, dry_run: bool) -> None:
        self.path = path
        self.dry_run = dry_run
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init()
        if not resume and not dry_run:
            self.conn.execute("DELETE FROM processed_issues")
            self.conn.execute("DELETE FROM processed_repos")
            self.conn.commit()

    def _init(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_repos (
                repo TEXT PRIMARY KEY,
                completed_at TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_issues (
                repo TEXT NOT NULL,
                issue_number INTEGER NOT NULL,
                status TEXT NOT NULL,
                pr_number INTEGER,
                processed_at TEXT NOT NULL,
                PRIMARY KEY (repo, issue_number)
            )
            """
        )
        self.conn.commit()

    def is_repo_done(self, repo: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM processed_repos WHERE repo = ?", (repo,)).fetchone()
        return row is not None

    def is_issue_done(self, repo: str, issue_number: int) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM processed_issues WHERE repo = ? AND issue_number = ?",
            (repo, issue_number),
        ).fetchone()
        return row is not None

    def mark_issue(self, repo: str, issue_number: int, status: str, pr_number: int | None = None) -> None:
        if self.dry_run:
            return
        self.conn.execute(
            """
            INSERT OR REPLACE INTO processed_issues
                (repo, issue_number, status, pr_number, processed_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (repo, issue_number, status, pr_number, now_utc()),
        )
        self.conn.commit()

    def mark_repo(self, repo: str) -> None:
        if self.dry_run:
            return
        self.conn.execute(
            "INSERT OR REPLACE INTO processed_repos (repo, completed_at) VALUES (?, ?)",
            (repo, now_utc()),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


class RateLimiter:
    def __init__(self, interval: float = MIN_REQUEST_INTERVAL) -> None:
        self.interval = interval
        self._lock = asyncio.Lock()
        self._last_request = 0.0

    async def wait(self) -> None:
        async with self._lock:
            elapsed = time.monotonic() - self._last_request
            if elapsed < self.interval:
                await asyncio.sleep(self.interval - elapsed)
            self._last_request = time.monotonic()


class GitHubClient:
    def __init__(self, token: str) -> None:
        self.token = token
        self.rate_limiter = RateLimiter()
        self.session = None

    async def __aenter__(self) -> "GitHubClient":
        try:
            import aiohttp
        except ImportError as exc:
            raise SystemExit("Missing dependency: pip install aiohttp") from exc

        timeout = aiohttp.ClientTimeout(total=120)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.session is not None:
            await self.session.close()

    def _headers(self, accept: str | None = None) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": accept or "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "repopilot-issue-fix-collector",
        }

    async def request_text(self, path_or_url: str, accept: str | None = None) -> str:
        data = await self._request(path_or_url, accept=accept, text=True)
        return str(data)

    async def request_json(self, path_or_url: str, accept: str | None = None) -> Any:
        return await self._request(path_or_url, accept=accept, text=False)

    async def _request(self, path_or_url: str, accept: str | None, text: bool) -> Any:
        if self.session is None:
            raise RuntimeError("GitHubClient must be used as an async context manager")
        url = path_or_url if path_or_url.startswith("http") else f"{GITHUB_API}{path_or_url}"
        attempt = 0
        while True:
            attempt += 1
            await self.rate_limiter.wait()
            async with self.session.get(url, headers=self._headers(accept)) as response:
                if response.status == 429:
                    retry_after = response.headers.get("Retry-After")
                    wait_seconds = float(retry_after) if retry_after else min(60.0, 2.0 * attempt)
                    await asyncio.sleep(wait_seconds)
                    continue

                if response.status in {403, 502, 503, 504}:
                    wait_seconds = self._rate_or_backoff_wait(response, attempt)
                    if wait_seconds is not None:
                        await asyncio.sleep(wait_seconds)
                        continue

                if response.status >= 400:
                    body = await response.text()
                    raise RuntimeError(f"GitHub API {response.status} for {url}: {body[:500]}")

                if text:
                    return await response.text()
                if response.status == 204:
                    return None
                return await response.json()

    def _rate_or_backoff_wait(self, response: Any, attempt: int) -> float | None:
        remaining = response.headers.get("X-RateLimit-Remaining")
        reset = response.headers.get("X-RateLimit-Reset")
        if remaining == "0" and reset:
            return max(1.0, float(reset) - time.time() + 5.0)
        if response.status in {502, 503, 504} and attempt <= 5:
            return min(30.0, 2.0 * attempt)
        if response.status == 403 and attempt <= 3:
            return min(60.0, 5.0 * attempt)
        return None

    async def paginate(self, path: str, max_pages: int | None = None) -> list[Any]:
        items: list[Any] = []
        page = 1
        while max_pages is None or page <= max_pages:
            separator = "&" if "?" in path else "?"
            data = await self.request_json(f"{path}{separator}per_page=100&page={page}")
            batch = data.get("items") if isinstance(data, dict) and "items" in data else data
            if not batch:
                break
            items.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        return items


class Progress:
    def __init__(self, max_items: int) -> None:
        self.max_items = max_items
        self.repos = 0
        self.issues = 0
        self.written = 0
        self.skipped = 0
        self.current_repo = ""
        self._last_render = 0.0

    def render(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_render < 0.1:
            return
        self._last_render = now
        width = 28
        ratio = min(1.0, self.written / self.max_items) if self.max_items else 0.0
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        line = (
            f"\r[{bar}] {self.written}/{self.max_items} kept "
            f"| repos {self.repos} | issues {self.issues} | skipped {self.skipped} "
            f"| {self.current_repo[:36]:36}"
        )
        sys.stderr.write(line)
        sys.stderr.flush()

    def finish(self) -> None:
        self.render(force=True)
        sys.stderr.write("\n")
        sys.stderr.flush()


def search_query(query: str, sort: str = "updated", order: str = "desc") -> str:
    return f"/search/issues?q={quote(query)}&sort={sort}&order={order}"


async def get_repo_metadata(client: GitHubClient, repo: str) -> dict[str, Any]:
    return await client.request_json(f"/repos/{repo}")


async def search_closed_bug_issues(
    client: GitHubClient,
    repo: str,
    max_pages: int | None,
) -> list[dict[str, Any]]:
    query = f"repo:{repo} is:issue is:closed label:bug"
    data = await client.paginate(search_query(query), max_pages=max_pages)
    return [item for item in data if "pull_request" not in item]


async def search_prs_with_closing_keywords(
    client: GitHubClient,
    repo: str,
    issue_number: int,
) -> list[dict[str, Any]]:
    candidates: dict[int, dict[str, Any]] = {}
    refs = [
        f"#{issue_number}",
        f"{repo}#{issue_number}",
        f"https://github.com/{repo}/issues/{issue_number}",
    ]
    for ref in refs:
        query = f'repo:{repo} is:pr is:merged "{ref}"'
        for item in await client.paginate(search_query(query), max_pages=1):
            number = int(item["number"])
            candidates[number] = item
    return list(candidates.values())


async def timeline_linked_prs(
    client: GitHubClient,
    repo: str,
    issue_number: int,
) -> list[int]:
    try:
        events = await client.paginate(f"/repos/{repo}/issues/{issue_number}/timeline", max_pages=2)
    except RuntimeError:
        return []

    pr_numbers: set[int] = set()
    for event in events:
        source_issue = (event.get("source") or {}).get("issue") or {}
        pull = source_issue.get("pull_request") or {}
        if not pull:
            continue
        number = source_issue.get("number")
        if number:
            pr_numbers.add(int(number))
    return sorted(pr_numbers)


async def get_pull_request(client: GitHubClient, repo: str, number: int) -> dict[str, Any]:
    return await client.request_json(f"/repos/{repo}/pulls/{number}")


async def get_pull_files(client: GitHubClient, repo: str, number: int) -> list[dict[str, Any]]:
    return await client.paginate(f"/repos/{repo}/pulls/{number}/files", max_pages=1)


async def get_pull_diff(client: GitHubClient, repo: str, number: int) -> str:
    return await client.request_text(
        f"/repos/{repo}/pulls/{number}",
        accept="application/vnd.github.v3.diff",
    )


async def find_linked_prs(
    client: GitHubClient,
    repo: str,
    issue_number: int,
) -> list[tuple[dict[str, Any], str]]:
    found: dict[int, tuple[dict[str, Any], str]] = {}

    for item in await search_prs_with_closing_keywords(client, repo, issue_number):
        pr_number = int(item["number"])
        pr = await get_pull_request(client, repo, pr_number)
        found[pr_number] = (pr, "fixes_keyword")

    for pr_number in await timeline_linked_prs(client, repo, issue_number):
        if pr_number in found:
            continue
        pr = await get_pull_request(client, repo, pr_number)
        if pr_body_references_issue(pr, issue_number):
            found[pr_number] = (pr, "timeline")

    return list(found.values())


def open_output(path: Path, resume: bool, dry_run: bool):
    if dry_run:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume else "w"
    return path.open(mode, encoding="utf-8")


async def collect(args: argparse.Namespace) -> int:
    repos = load_repo_list(args.repo_list)
    if not repos:
        raise SystemExit("No repositories to collect.")

    output_path = Path(args.output)
    state_path = Path(args.state_db) if args.state_db else output_path.with_suffix(output_path.suffix + ".sqlite3")
    state_path.parent.mkdir(parents=True, exist_ok=True)

    token = run_gh_auth_token()
    state = StateStore(state_path, resume=args.resume, dry_run=args.dry_run)
    progress = Progress(args.max_items)
    written = 0
    repo_written = 0  # per-repo counter

    out = open_output(output_path, resume=args.resume, dry_run=args.dry_run)
    try:
        async with GitHubClient(token) as client:
            for repo_name in repos:
                if written >= args.max_items:
                    break
                repo_written = 0  # reset per repo
                progress.current_repo = repo_name
                progress.repos += 1
                progress.render(force=True)

                if args.resume and state.is_repo_done(repo_name):
                    continue

                try:
                    repo = await get_repo_metadata(client, repo_name)
                    if str(repo.get("language") or "").lower() != "python":
                        continue

                    issues = await search_closed_bug_issues(client, repo_name, max_pages=args.issue_pages)
                    repo_completed = True
                    for issue in issues:
                        if written >= args.max_items:
                            repo_completed = False
                            break

                        issue_number = int(issue["number"])
                        if args.resume and state.is_issue_done(repo_name, issue_number):
                            continue

                        progress.issues += 1
                        progress.render()

                        if not should_keep_issue(issue):
                            progress.skipped += 1
                            state.mark_issue(repo_name, issue_number, "skipped_issue")
                            continue

                        kept = False
                        try:
                            linked_prs = await find_linked_prs(client, repo_name, issue_number)
                            for pr, linked_by in linked_prs:
                                files = await get_pull_files(client, repo_name, int(pr["number"]))
                                if not should_keep_pr(issue, pr, files):
                                    continue
                                full_diff = await get_pull_diff(client, repo_name, int(pr["number"]))
                                record = build_dataset_record(repo, issue, pr, full_diff, files, linked_by)
                                if args.dry_run:
                                    sys.stdout.write(json.dumps(record, ensure_ascii=False) + "\n")
                                else:
                                    assert out is not None
                                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                                    out.flush()
                                written += 1
                                repo_written += 1
                                progress.written = written
                                state.mark_issue(repo_name, issue_number, "kept", int(pr["number"]))
                                kept = True
                                progress.render(force=True)
                                break
                        except RuntimeError as exc:
                            progress.skipped += 1
                            state.mark_issue(repo_name, issue_number, "error")
                            print(f"\nwarning: {repo_name}#{issue_number}: {exc}", file=sys.stderr)
                            continue

                        if not kept:
                            progress.skipped += 1
                            state.mark_issue(repo_name, issue_number, "skipped_pr")

                        # Per-repo cap: stop processing this repo once cap reached
                        if args.max_items_per_repo is not None and repo_written >= args.max_items_per_repo:
                            repo_completed = True
                            break

                    if repo_completed:
                        state.mark_repo(repo_name)
                except RuntimeError as exc:
                    print(f"\nwarning: {repo_name}: {exc}", file=sys.stderr)
                    continue
    finally:
        if out is not None:
            out.close()
        state.close()
        progress.finish()

    return written


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect GitHub Issue -> Fix PR diff pairs as JSONL.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--max-items", type=int, default=2000, help="Maximum records to write.")
    parser.add_argument("--repo-list", help="Optional repo list file, one owner/repo per line or JSON array.")
    parser.add_argument("--resume", action="store_true", help="Resume from the SQLite sidecar state DB.")
    parser.add_argument("--dry-run", action="store_true", help="Print matching records to stdout without writing state/output.")
    parser.add_argument("--state-db", help="Optional SQLite state DB path.")
    parser.add_argument("--issue-pages", type=int, default=3, help="Search result pages per repo, 100 issues per page.")
    parser.add_argument("--max-items-per-repo", type=int, default=None, help="Cap per repository (skip remaining issues once cap is reached for that repo).")
    args = parser.parse_args(argv)
    if args.max_items <= 0:
        parser.error("--max-items must be positive")
    if args.issue_pages <= 0:
        parser.error("--issue-pages must be positive")
    if args.max_items_per_repo is not None and args.max_items_per_repo <= 0:
        parser.error("--max-items-per-repo must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    written = asyncio.run(collect(args))
    print(f"collected {written} records", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
