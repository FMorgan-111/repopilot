#!/usr/bin/env python3
"""Collect GitHub Issue -> PR -> diff pairs for Python bug fixes (Issues API, no Search API)."""

from __future__ import annotations

import argparse
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


GITHUB_API = "https://api.github.com"
MIN_REQUEST_INTERVAL = 0.5          # 0.5s between requests
HOURLY_REQUEST_BUDGET = 5000        # GitHub authenticated Issues/PR API limit
LOW_REMAINING_THRESHOLD = 50        # pause-until-reset cushion

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

# Client-side bug-label matching across repo conventions.
BUG_LABEL_RE = re.compile(
    r"(?:^|[\s/:_-])"
    r"(?:bug(?:\s*report)?|type:\s*bug|kind/bug|defect|\U0001F41B)"
    r"(?:$|[\s/:_-])",
    re.IGNORECASE,
)

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

# PR -> issue closing references, used to confirm a PR actually fixes the issue.
CLOSING_REF_RE = re.compile(
    r"\b(?:fix(?:e[sd])?|close[sd]?|resolve[sd]?)\s+"
    r"(?:"
    r"https://github\.com/[^/\s]+/[^/\s]+/issues/(\d+)"
    r"|(?:[\w.-]+/[\w.-]+)?#(\d+)"
    r")",
    re.IGNORECASE,
)

# Issue body -> candidate PR references (URL form and bare #num form).
PR_URL_RE = re.compile(r"https://github\.com/[^/\s]+/[^/\s]+/pull/(\d+)", re.IGNORECASE)
HASH_REF_RE = re.compile(r"(?:^|[^\w/])#(\d+)\b")

FORMATTING_RE = re.compile(r"\b(format(?:ting)?|black|ruff format|autopep8|isort|prettier)\b", re.IGNORECASE)
DEPENDENCY_RE = re.compile(r"\b(dependenc(?:y|ies)|bump|upgrade|pin|requirements)\b", re.IGNORECASE)


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------- #
# Auth (reuses collect_dataset.py's gh-auth-token pattern)
# --------------------------------------------------------------------------- #
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


def resolve_token(cli_token: str | None) -> str:
    if cli_token and cli_token.strip():
        return cli_token.strip()
    env_token = os.environ.get("GITHUB_TOKEN", "").strip()
    if env_token:
        return env_token
    return run_gh_auth_token()


def get_github_session(token: str):
    try:
        import requests
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install requests") from exc

    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "repopilot-issue-fix-collector-v2",
        }
    )
    return session


# --------------------------------------------------------------------------- #
# Repo list
# --------------------------------------------------------------------------- #
def load_repo_file(path: str) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    seen: set[str] = set()
    repos: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or "/" not in line:
            continue
        if line in seen:
            continue
        seen.add(line)
        repos.append(line)
    return repos


def repo_parts(repo_full_name: str) -> tuple[str, str]:
    owner, name = repo_full_name.split("/", 1)
    return owner, name


# --------------------------------------------------------------------------- #
# Issue / PR filtering (same semantics as collect_dataset.py)
# --------------------------------------------------------------------------- #
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
    return any(BUG_LABEL_RE.search(label) for label in labels)


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


# --------------------------------------------------------------------------- #
# State store (SQLite, resume)
# --------------------------------------------------------------------------- #
class StateStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init()

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
        self.conn.execute(
            "INSERT OR REPLACE INTO processed_repos (repo, completed_at) VALUES (?, ?)",
            (repo, now_utc()),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


# --------------------------------------------------------------------------- #
# HTTP client: 0.5s spacing, 5000/hr budget, 403/429 backoff
# --------------------------------------------------------------------------- #
class GitHubClient:
    def __init__(self, session, interval: float = MIN_REQUEST_INTERVAL) -> None:
        self.session = session
        self.interval = interval
        self._last_request = 0.0
        self._window_start = time.monotonic()
        self._window_count = 0

    def _throttle(self) -> None:
        # Per-request spacing.
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)

        # Hard hourly budget: pause until the rolling window rolls over.
        window_elapsed = time.monotonic() - self._window_start
        if window_elapsed >= 3600.0:
            self._window_start = time.monotonic()
            self._window_count = 0
        elif self._window_count >= HOURLY_REQUEST_BUDGET:
            sleep_for = 3600.0 - window_elapsed + 1.0
            print(f"\nhourly budget reached; sleeping {sleep_for:.0f}s", file=sys.stderr)
            time.sleep(max(1.0, sleep_for))
            self._window_start = time.monotonic()
            self._window_count = 0

    def _account(self) -> None:
        self._last_request = time.monotonic()
        self._window_count += 1

    def _request(self, path_or_url: str, accept: str | None, text: bool) -> Any:
        url = path_or_url if path_or_url.startswith("http") else f"{GITHUB_API}{path_or_url}"
        headers = {"Accept": accept} if accept else None
        attempt = 0
        while True:
            attempt += 1
            self._throttle()
            try:
                response = self.session.get(url, headers=headers, timeout=120)
            except Exception as e:
                # Transient network errors (ConnectionError, Timeout, etc.)
                if attempt <= 5:
                    wait = min(30.0, 2.0 * attempt)
                    print(f"\n  request error (attempt {attempt}): {e}; retrying in {wait:.0f}s",
                          file=sys.stderr)
                    time.sleep(wait)
                    continue
                raise RuntimeError(
                    f"Request failed after {attempt} attempts for {url}: {e}"
                ) from e
            self._account()
            status = response.status_code

            if status == 429:
                retry_after = response.headers.get("Retry-After")
                wait_seconds = float(retry_after) if retry_after else min(60.0, 2.0 * attempt)
                time.sleep(wait_seconds)
                continue

            if status in {403, 502, 503, 504}:
                wait_seconds = self._rate_or_backoff_wait(response, attempt)
                if wait_seconds is not None:
                    time.sleep(wait_seconds)
                    continue

            if status >= 400:
                raise RuntimeError(f"GitHub API {status} for {url}: {response.text[:500]}")

            # Proactively pause if the primary rate limit is nearly exhausted.
            remaining = response.headers.get("X-RateLimit-Remaining")
            reset = response.headers.get("X-RateLimit-Reset")
            if remaining is not None and reset and remaining.isdigit():
                if int(remaining) <= LOW_REMAINING_THRESHOLD:
                    sleep_for = max(1.0, float(reset) - time.time() + 5.0)
                    print(f"\nrate limit low ({remaining}); sleeping {sleep_for:.0f}s", file=sys.stderr)
                    time.sleep(sleep_for)

            if text:
                return response.text
            if status == 204 or not response.content:
                return None
            return response.json()

    def _rate_or_backoff_wait(self, response: Any, attempt: int) -> float | None:
        remaining = response.headers.get("X-RateLimit-Remaining")
        reset = response.headers.get("X-RateLimit-Reset")
        if remaining == "0" and reset:
            return max(1.0, float(reset) - time.time() + 5.0)
        if response.status_code in {502, 503, 504} and attempt <= 5:
            return min(30.0, 2.0 * attempt)
        if response.status_code == 403 and attempt <= 3:
            return min(60.0, 5.0 * attempt)
        return None

    def request_json(self, path_or_url: str, accept: str | None = None) -> Any:
        return self._request(path_or_url, accept=accept, text=False)

    def request_text(self, path_or_url: str, accept: str | None = None) -> str:
        return str(self._request(path_or_url, accept=accept, text=True))

    def paginate(self, path: str, max_pages: int | None = None) -> list[Any]:
        items: list[Any] = []
        page = 1
        while max_pages is None or page <= max_pages:
            separator = "&" if "?" in path else "?"
            data = self.request_json(f"{path}{separator}per_page=100&page={page}")
            if not isinstance(data, list):
                break
            if not data:
                break
            items.extend(data)
            if len(data) < 100:
                break
            page += 1
        return items


# --------------------------------------------------------------------------- #
# GitHub API calls (Issues API only — no Search API)
# --------------------------------------------------------------------------- #
def get_repo_metadata(client: GitHubClient, repo: str) -> dict[str, Any]:
    return client.request_json(f"/repos/{repo}")


def list_closed_issues(client: GitHubClient, repo: str, max_pages: int | None) -> list[dict[str, Any]]:
    # /issues returns issues AND pull requests; drop PRs, keep real issues.
    path = f"/repos/{repo}/issues?state=closed&sort=updated&direction=desc"
    data = client.paginate(path, max_pages=max_pages)
    return [item for item in data if isinstance(item, dict) and "pull_request" not in item]


def get_pull_request(client: GitHubClient, repo: str, number: int) -> dict[str, Any]:
    return client.request_json(f"/repos/{repo}/pulls/{number}")


def get_pull_files(client: GitHubClient, repo: str, number: int) -> list[dict[str, Any]]:
    return client.paginate(f"/repos/{repo}/pulls/{number}/files", max_pages=1)


def get_pull_diff(client: GitHubClient, repo: str, number: int) -> str:
    return client.request_text(
        f"/repos/{repo}/pulls/{number}",
        accept="application/vnd.github.v3.diff",
    )


def is_pull_request(client: GitHubClient, repo: str, number: int) -> bool:
    try:
        client.request_json(f"/repos/{repo}/pulls/{number}")
        return True
    except RuntimeError:
        return False


def pr_candidates_from_body(body: str | None) -> set[int]:
    numbers: set[int] = set()
    for match in PR_URL_RE.finditer(body or ""):
        numbers.add(int(match.group(1)))
    for match in HASH_REF_RE.finditer(body or ""):
        numbers.add(int(match.group(1)))
    return numbers


def timeline_pr_numbers(client: GitHubClient, repo: str, issue_number: int) -> set[int]:
    try:
        events = client.paginate(f"/repos/{repo}/issues/{issue_number}/timeline", max_pages=2)
    except RuntimeError:
        return set()

    pr_numbers: set[int] = set()
    for event in events:
        if not isinstance(event, dict):
            continue
        source_issue = (event.get("source") or {}).get("issue") or {}
        if source_issue.get("pull_request"):
            number = source_issue.get("number")
            if number:
                pr_numbers.add(int(number))
    return pr_numbers


def find_linked_prs(
    client: GitHubClient,
    repo: str,
    issue: dict[str, Any],
) -> list[tuple[dict[str, Any], str]]:
    issue_number = int(issue["number"])
    found: dict[int, tuple[dict[str, Any], str]] = {}

    # 1. Timeline cross-reference / close events that point at a PR.
    for pr_number in sorted(timeline_pr_numbers(client, repo, issue_number)):
        if pr_number in found:
            continue
        try:
            pr = get_pull_request(client, repo, pr_number)
        except RuntimeError:
            continue
        found[pr_number] = (pr, "timeline")

    # 2. PR references parsed out of the issue body.
    for pr_number in sorted(pr_candidates_from_body(issue.get("body"))):
        if pr_number in found:
            continue
        if not is_pull_request(client, repo, pr_number):
            continue
        try:
            pr = get_pull_request(client, repo, pr_number)
        except RuntimeError:
            continue
        found[pr_number] = (pr, "issue_body")

    return list(found.values())


# --------------------------------------------------------------------------- #
# Collection driver
# --------------------------------------------------------------------------- #
def collect(args: argparse.Namespace) -> int:
    repos = load_repo_file(args.repo_file)
    if not repos:
        raise SystemExit("No repositories to collect.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = (
        Path(args.state_db)
        if args.state_db
        else output_path.with_suffix(output_path.suffix + ".sqlite3")
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)

    token = resolve_token(args.github_token)
    session = get_github_session(token)
    client = GitHubClient(session)
    state = StateStore(state_path)

    written = 0
    out = output_path.open("a", encoding="utf-8")
    try:
        for repo_name in repos:
            if state.is_repo_done(repo_name):
                continue

            print(f"\n== {repo_name} ==", file=sys.stderr)
            try:
                repo = get_repo_metadata(client, repo_name)
            except RuntimeError as exc:
                print(f"warning: {repo_name}: {exc}", file=sys.stderr)
                continue

            if str(repo.get("language") or "").lower() != "python":
                state.mark_repo(repo_name)
                continue

            repo_written = 0
            try:
                issues = list_closed_issues(client, repo_name, max_pages=args.max_pages)
            except RuntimeError as exc:
                print(f"warning: {repo_name}: {exc}", file=sys.stderr)
                continue

            for issue in issues:
                issue_number = int(issue["number"])
                if state.is_issue_done(repo_name, issue_number):
                    continue

                if not should_keep_issue(issue):
                    state.mark_issue(repo_name, issue_number, "skipped_issue")
                    continue

                kept = False
                try:
                    for pr, linked_by in find_linked_prs(client, repo_name, issue):
                        pr_number = int(pr["number"])
                        files = get_pull_files(client, repo_name, pr_number)
                        if not should_keep_pr(issue, pr, files):
                            continue
                        full_diff = get_pull_diff(client, repo_name, pr_number)
                        record = build_dataset_record(repo, issue, pr, full_diff, files, linked_by)
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        out.flush()
                        written += 1
                        repo_written += 1
                        state.mark_issue(repo_name, issue_number, "kept", pr_number)
                        kept = True
                        print(
                            f"  kept {record['id']} ({linked_by}) [{written} total]",
                            file=sys.stderr,
                        )
                        break
                except RuntimeError as exc:
                    state.mark_issue(repo_name, issue_number, "error")
                    print(f"  warning: {repo_name}#{issue_number}: {exc}", file=sys.stderr)
                    continue

                if not kept:
                    state.mark_issue(repo_name, issue_number, "skipped_pr")

                if args.max_items_per_repo is not None and repo_written >= args.max_items_per_repo:
                    break

            state.mark_repo(repo_name)
    finally:
        out.close()
        state.close()

    return written


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect GitHub Issue -> Fix PR diff pairs as JSONL (Issues API, no Search API)."
    )
    parser.add_argument("--repo-file", required=True, help="Repo list file, one owner/name per line (shards/repos-00.txt format).")
    parser.add_argument("--output", required=True, help="Output JSONL path (appended to; resumable).")
    parser.add_argument("--state-db", help="Optional SQLite state DB path (defaults to <output>.sqlite3).")
    parser.add_argument("--max-items-per-repo", type=int, default=None, help="Cap kept records per repository.")
    parser.add_argument("--max-pages", type=int, default=None, help="Max closed-issue pages per repo (100 issues per page).")
    parser.add_argument("--github-token", help="GitHub token override (else $GITHUB_TOKEN, else `gh auth token`).")
    args = parser.parse_args(argv)
    if args.max_items_per_repo is not None and args.max_items_per_repo <= 0:
        parser.error("--max-items-per-repo must be positive")
    if args.max_pages is not None and args.max_pages <= 0:
        parser.error("--max-pages must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    written = collect(args)
    print(f"\ncollected {written} records", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
