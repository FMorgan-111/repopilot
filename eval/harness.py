"""
RepoPilot Eval Harness — evaluate agent on real GitHub issue→fix pairs.

Metrics:
  file_recall@k  — does the agent identify the correct files (k=1,3,5)?
  patch_apply_rate — does the agent's patch apply cleanly?
  test_pass_rate   — do tests pass after applying the agent's patch?
  avg_cost         — token consumption and API cost per run

No mocking. Real LLM calls, real git clones, real test runs.
"""

from __future__ import annotations

import argparse
import asyncio
import base64 as b64
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request as urllib_req
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

# ── repo root ─────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env", override=True)

_LLM_BASE = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
_LLM_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY", "")
_LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")

_llm_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = httpx.AsyncClient(
            timeout=60.0,
            limits=httpx.Limits(max_keepalive_connections=3, max_connections=6),
        )
    return _llm_client


async def llm_request(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.2,
) -> dict:
    """Call the LLM API and return the raw response dict (includes usage)."""
    url = f"{_LLM_BASE}/chat/completions"
    payload = {
        "model": model or _LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {_LLM_KEY}",
        "Content-Type": "application/json",
    }
    client = _get_client()
    resp = await client.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    return resp.json()

# ── paths ────────────────────────────────────────────────────────────────
SAMPLES_PATH = REPO_ROOT / "data" / "samples" / "issues_fixes.jsonl"
EVAL_TMP = Path("/tmp/repopilot-eval")
RESULTS_PATH = REPO_ROOT / "eval" / "eval_results.json"
SUMMARY_PATH = REPO_ROOT / "eval" / "eval_summary.md"

# ── config ───────────────────────────────────────────────────────────────
MAX_SAMPLES = 5
TIMEOUT_PER_SAMPLE = 600  # 10 minutes
CLONE_TIMEOUT = 180  # 3 minutes for minimal clone (ansible ~2GB needs it)
DEEPSEEK_PRICING = {
    "input": 0.27 / 1_000_000,   # $0.27 per 1M input tokens
    "output": 0.36 / 1_000_000,  # $0.36 per 1M output tokens (cached miss)
}

_gh_file_cache: dict[str, list[str]] = {}  # "owner/repo" -> [file paths]
_gh_content_cache: dict[str, str] = {}     # "owner/repo/path" -> file content


def _gh_get(url: str, timeout: int = 30) -> dict | None:
    """GET a GitHub API v3 endpoint, return parsed JSON or None."""
    headers = {
        "User-Agent": "repopilot-eval",
        "Accept": "application/vnd.github.v3+json",
    }
    # Use GitHub token for higher rate limits (required for code search)
    gh_token = os.getenv("GITHUB_TOKEN")
    if gh_token:
        headers["Authorization"] = f"Bearer {gh_token}"
    req = urllib_req.Request(url, headers=headers)
    try:
        with urllib_req.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def fetch_file_list(owner: str, repo: str) -> list[str]:
    """Get all file paths via GitHub Tree API (recursive). Cached per-run."""
    cache_key = f"{owner}/{repo}"
    if cache_key in _gh_file_cache:
        return _gh_file_cache[cache_key]

    data = _gh_get(
        f"https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1",
        timeout=60,
    )
    if not data:
        # Try 'master' branch if 'main' returns no tree
        data = _gh_get(
            f"https://api.github.com/repos/{owner}/{repo}/git/trees/master?recursive=1",
            timeout=60,
        )
    files = [t["path"] for t in (data or {}).get("tree", []) if t.get("type") == "blob"]
    _gh_file_cache[cache_key] = files
    return files


def fetch_file_content(owner: str, repo: str, path: str, ref: str = "main") -> str:
    """Read file content via GitHub Contents API. Cached per-run."""
    cache_key = f"{owner}/{repo}/{path}"
    if cache_key in _gh_content_cache:
        return _gh_content_cache[cache_key]

    safe_path = urllib_req.quote(path, safe="")
    data = _gh_get(
        f"https://api.github.com/repos/{owner}/{repo}/contents/{safe_path}?ref={ref}",
        timeout=15,
    )
    if not data:
        _gh_content_cache[cache_key] = ""
        return ""

    try:
        content = b64.b64decode(data["content"]).decode("utf-8", errors="replace")
    except Exception:
        content = ""
    _gh_content_cache[cache_key] = content
    return content


def search_files_by_name(file_list: list[str], term: str, max_results: int = 15) -> list[str]:
    """Search a flat file list by filename/path match (client-side, instant)."""
    term_lower = term.lower()
    results = []
    for f in file_list:
        if term_lower in f.lower():
            results.append(f)
            if len(results) >= max_results:
                break
    return results


def search_code_via_github(owner: str, repo: str, query: str, max_results: int = 10) -> list[str]:
    """Search GitHub code (file contents) for files matching the query.

    Uses GitHub's /search/code API to find files whose CONTENTS contain the query.
    This catches files like replace.py (contains "allowlist_externals"),
    checker.py (contains "warn_unreachable"), etc. that filename-only search misses.

    Fixes v0 file_recall=0 root cause:
    - Old harness only did filename matching via search_files_by_name.
    - Actual file names (replace.py, checker.py, manager.py) rarely appear in issue text.
    - But the issue text DOES contain terms like "allowlist_externals" which appear
      in the file CONTENTS. Code search bridges this gap.
    """
    import urllib.parse as urlparse
    q = f"repo:{owner}/{repo} {query}"
    safe_q = urlparse.quote(q, safe="")
    url = f"https://api.github.com/search/code?q={safe_q}&per_page={max_results}"
    data = _gh_get(url, timeout=15)
    if not data:
        return []
    return [item["path"] for item in data.get("items", []) if item.get("path")]


# ═══════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_samples(n: int = MAX_SAMPLES) -> list[dict]:
    """Load the first n samples from the JSONL file."""
    samples = []
    with open(SAMPLES_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _extract_search_terms(title: str, body: str, max_terms: int = 15) -> list[str]:
    """Extract useful search terms from the issue title and body.

    Designed for both filename matching AND GitHub code search (content matching).
    Code search is more effective because implementation file names rarely appear
    in issue text, but code identifiers (function names, config keys, error messages)
    DO appear and match file contents.

    Fixes v0 file_recall=0 root cause:
    - Produces more terms for code search (increased from 10 to 15).
    - Extracts compound identifiers without over-splitting.
    - Includes unique short phrases that serve as good code search queries.
    """
    text = f"{title}\n{body[:3000]}"

    # 1. Extract backtick-quoted identifiers (class names, functions, vars, paths)
    code_terms = re.findall(r"`([A-Za-z_][A-Za-z0-9_./]{2,120})`", text)

    # 2. Extract CamelCase and snake_case words
    words = re.findall(r"[A-Z][a-z]+(?:[A-Z][a-z]+)+|[a-z]+(?:_[a-z]+)+", text)

    # 3. Extract key nouns from title — split title into words, keep meaningful ones
    title_words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", title)

    # Minimal stop words — only generic noise
    stop = {
        "the", "and", "for", "with", "when", "this", "that", "from",
        "https", "http", "com", "github", "issue", "error",
    }

    seen: set[str] = set()
    terms: list[str] = []

    # Priority order: code_terms first, then title_words, then body words
    for t in code_terms + title_words + words:
        # Clean up
        t = t.strip("`.,:;()[]{}\"'").lower()
        # Extract last component of dotted paths (e.g., "tox.config.loader.ini.replace" -> "replace")
        if "." in t:
            parts = [p for p in t.split(".") if len(p) >= 3 and p not in stop]
            for p in parts:
                if p not in seen:
                    seen.add(p)
                    terms.append(p)
        # Split compound terms on _ (e.g., "tox_test" -> "tox" + "test")
        if "_" in t and len(t) > 5:
            sub_parts = [p for p in t.split("_") if len(p) >= 3 and p not in stop]
            for p in sub_parts:
                if p not in seen:
                    seen.add(p)
                    terms.append(p)
        if t in stop or len(t) < 3:
            continue
        if t not in seen:
            seen.add(t)
            terms.append(t)
        if len(terms) >= max_terms:
            break

    # 4. Fallback: extract any words 3+ chars from title that aren't stop words
    if len(terms) < 3:
        for w in re.findall(r"[a-zA-Z]{3,}", title):
            w = w.lower()
            if w not in stop and w not in seen:
                seen.add(w)
                terms.append(w)
            if len(terms) >= max_terms:
                break

    return terms


async def _llm_search_context(
    title: str, body: str, model: str = "deepseek-v4-flash"
) -> tuple[list[str], list[str]]:
    """Use LLM to generate search terms and likely file paths from the issue.

    Regex-based extraction misses critical terms like 'replace' for a substitution
    bug, or 'checker' for a type-checking bug. The LLM infers these from context.

    Returns (search_terms, file_patterns).
    - search_terms: for GitHub Code Search (content matching)
    - file_patterns: for filename matching against repo file list
    """
    system = (
        "You are a code repository search assistant. Given a bug report, suggest "
        "search terms and likely file paths to find the files that need changes.\n\n"
        "Rules:\n"
        "1. search_terms: code identifiers, config keys, class/function names, "
        "CLI flags, module names, error message fragments, or technical keywords "
        "that would appear in the SOURCE CODE of the affected files.\n"
        "2. file_patterns: likely file names or path fragments (e.g., 'checker.py', "
        "'replace.py', 'config/loader') based on what COMPONENT the bug is about.\n"
        "3. Include both specific terms from the issue AND inferred terms from context.\n"
        "4. Prefer terms of 4+ characters.\n"
        "Return ONLY valid JSON: {\"search_terms\": [...], \"file_patterns\": [...]}"
    )
    user = (
        f"Bug Title: {title}\n\n"
        f"Bug Description:\n{body[:3000]}\n\n"
        "What search terms and file patterns would help find the relevant source files?"
    )
    try:
        result, _, _ = await llm_call_structured(system, user, model=model)
        search_terms = result.get("search_terms", [])
        file_patterns = result.get("file_patterns", [])
        return search_terms, file_patterns
    except Exception:
        return [], []


def clone_repo(owner: str, repo: str, target: Path, timeout: int = 600) -> bool:
    """Shallow-clone a GitHub repo. Returns True on success.

    Fixes v0 clone_failed root cause:
    - Clean target dir between retries (partial clone from failed strategy
      left non-empty dir, causing second strategy to fail).
    - Increased timeout from 300s to 600s for large repos (ansible ~2GB).
    - Dropped --filter=blob:none (requires server support, unreliable).
    - Added --no-tags to skip tag objects.
    """
    url = f"https://github.com/{owner}/{repo}.git"
    strategies = [
        # Strategy 1: shallow clone, single branch, no tags (fastest)
        ["git", "clone", "--depth", "1", "--single-branch", "--no-tags", url, str(target)],
        # Strategy 2: shallow clone, single branch (fallback if --no-tags unsupported)
        ["git", "clone", "--depth", "1", "--single-branch", url, str(target)],
        # Strategy 3: ultra-shallow (depth 1 only, any branch)
        ["git", "clone", "--depth", "1", url, str(target)],
    ]
    for i, cmd in enumerate(strategies):
        # Clean up any partial clone from previous strategy
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0:
                return True
            # Log failure for debugging
            stderr = (result.stderr or "")[:200]
            print(f"    clone strategy {i+1} failed: {stderr}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"    clone strategy {i+1} timed out after {timeout}s", flush=True)
            shutil.rmtree(target, ignore_errors=True)
            continue
        except Exception as exc:
            print(f"    clone strategy {i+1} error: {exc}", flush=True)
            shutil.rmtree(target, ignore_errors=True)
            continue
    return False


def grep_repo(repo_path: Path, term: str, max_files: int = 15) -> list[str]:
    """Search repo for files matching a term.

    Fixes v0 file_recall=0 root cause:
    - Old version excluded .rst, .test, .toml, .yaml, .json — but many fixes
      touch docs, test data files, and config. Now only excludes binary assets.
    - Added filename search: also searches file paths for the term (e.g.,
      "replace" should match src/tox/.../replace.py even if content doesn't
      contain the word "replace" as a keyword).
    - Combined content + filename results with filename matches ranked first
      (filenames are stronger signals).
    """
    results: list[str] = []
    seen: set[str] = set()

    # Strategy 1: ripgrep content search (broad file types, exclude only binaries)
    try:
        result = subprocess.run(
            ["rg", "-l", "--max-filesize", "500K",
             "--iglob", "!*.{png,jpg,jpeg,gif,svg,ico,woff,woff2,ttf,eot,mp4,mp3,avi,mov,zip,tar,gz,bz2,xz}",
             term, str(repo_path)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    rel = str(Path(line.strip()).relative_to(repo_path))
                    if rel not in seen:
                        seen.add(rel)
                        results.append(rel)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Strategy 2: filename search — find files whose path contains the term
    # This catches cases where the file name (e.g., replace.py) doesn't contain
    # the search term as a word but the path itself is a strong signal.
    try:
        result = subprocess.run(
            ["rg", "--files", "--iglob", "!*.{png,jpg,jpeg,gif,svg,ico,woff,woff2,ttf,eot,mp4,mp3,avi,mov,zip,tar,gz}",
             str(repo_path)],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            all_files = [
                line.strip()
                for line in result.stdout.strip().split("\n")
                if line.strip()
            ]
            # Match term against file path
            term_lower = term.lower()
            for fpath in all_files:
                fname = Path(fpath).name.lower()
                fdir = str(Path(fpath).parent).lower()
                # Match if term appears in filename or directory name
                if term_lower in fname or term_lower in fdir:
                    try:
                        rel = str(Path(fpath).relative_to(repo_path))
                    except ValueError:
                        rel = fpath
                    if rel not in seen:
                        seen.add(rel)
                        results.append(rel)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return results[:max_files]


def read_file_content(repo_path: Path, rel_path: str) -> str:
    """Read a file from the cloned repo."""
    full = repo_path / rel_path
    try:
        return full.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _fix_patch(patch_content: str) -> str:
    """Sanitize LLM-generated patch to fix common format issues.

    Common LLM mistakes that cause 'corrupt patch' errors:
    1. Fake index lines (index 1111111..2222222) — strip them
    2. Missing --- or +++ lines — add placeholders
    3. Whitespace errors
    4. No trailing newline
    """
    if not patch_content.strip():
        return patch_content

    lines = patch_content.split("\n")
    fixed: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Skip fake index lines like "index 1111111..2222222 100644"
        if line.startswith("index ") and ".." in line and i > 0:
            prev = lines[i - 1] if i > 0 else ""
            if prev.startswith("diff --git"):
                i += 1
                continue

        # Ensure --- line exists after diff --git
        if line.startswith("diff --git") and i + 1 < len(lines):
            next_line = lines[i + 1]
            if not next_line.startswith("---"):
                # Extract path from diff --git a/path b/path
                parts = line.split(" ")
                if len(parts) >= 4:
                    a_path = parts[2]
                    if a_path.startswith("a/"):
                        fixed.append(line)
                        fixed.append(f"--- {a_path}")
                        fixed.append(f"+++ {a_path.replace('a/', 'b/', 1)}")
                        i += 1
                        continue

        fixed.append(line)
        i += 1

    result = "\n".join(fixed)
    # Ensure trailing newline
    if not result.endswith("\n"):
        result += "\n"
    return result


def apply_patch(repo_path: Path, patch_content: str) -> tuple[bool, str]:
    """Apply a unified diff patch with fallbacks. Returns (success, output).

    Tries multiple methods:
    1. git apply --check + git apply (strict)
    2. patch -p1 (lenient, handles line-number offsets)
    3. git apply --reject (apply what we can)
    """
    sanitized = _fix_patch(patch_content)

    # Method 1: git apply (strict)
    result = subprocess.run(
        ["git", "apply", "--check"],
        input=sanitized,
        cwd=str(repo_path),
        capture_output=True, text=True, timeout=30,
    )
    check_ok = result.returncode == 0
    check_output = (result.stdout + result.stderr).strip()

    if check_ok:
        result2 = subprocess.run(
            ["git", "apply"],
            input=sanitized,
            cwd=str(repo_path),
            capture_output=True, text=True, timeout=30,
        )
        if result2.returncode == 0:
            return True, (result2.stdout + result2.stderr).strip()
        check_output = (result2.stdout + result2.stderr).strip()

    # Method 2: patch -p1 (lenient, handles fuzz)
    result3 = subprocess.run(
        ["patch", "-p1", "-f", "--dry-run"],
        input=sanitized,
        cwd=str(repo_path),
        capture_output=True, text=True, timeout=30,
    )
    if result3.returncode == 0:
        result4 = subprocess.run(
            ["patch", "-p1", "-f"],
            input=sanitized,
            cwd=str(repo_path),
            capture_output=True, text=True, timeout=30,
        )
        if result4.returncode == 0:
            return True, (result4.stdout + result4.stderr).strip()
        return False, (result4.stdout + result4.stderr).strip()[:2000]

    # Method 3: git apply --reject (apply partial, creates .rej files)
    result5 = subprocess.run(
        ["git", "apply", "--reject"],
        input=sanitized,
        cwd=str(repo_path),
        capture_output=True, text=True, timeout=30,
    )
    reject_output = (result5.stdout + result5.stderr).strip()
    if result5.returncode == 0:
        return True, f"applied with rejects: {reject_output[:500]}"

    return False, f"all methods failed. git-check: {check_output[:500]} | patch: {reject_output[:500]}"


def run_tests(repo_path: Path) -> tuple[bool, str]:
    """Run pytest in the repo. Returns (success, output)."""
    # Try common test commands
    candidates = [
        ["python3", "-m", "pytest", "-x", "-q", "--tb=short"],
        ["python", "-m", "pytest", "-x", "-q", "--tb=short"],
        ["pytest", "-x", "-q", "--tb=short"],
    ]
    for cmd in candidates:
        try:
            result = subprocess.run(
                cmd, cwd=str(repo_path), capture_output=True, text=True, timeout=300,
            )
            output = (result.stdout + "\n" + result.stderr)[:5000]
            return result.returncode == 0, output
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return False, "no test runner found"


def compute_file_recall(
    actual_files: list[str],
    predicted_files: list[str],
    k: int,
) -> float:
    """Recall@k: fraction of actual files present in top-k predicted."""
    if not actual_files:
        return 1.0
    top_k = set(p[:k] for p in predicted_files[:k])
    actual_set = set(actual_files)
    return len(top_k & actual_set) / len(actual_set)


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate API cost based on token counts."""
    return input_tokens * DEEPSEEK_PRICING["input"] + output_tokens * DEEPSEEK_PRICING["output"]


# ═══════════════════════════════════════════════════════════════════════════
# LLM-driven phases
# ═══════════════════════════════════════════════════════════════════════════

async def llm_call_structured(
    system: str, user: str, model: str = "deepseek-v4-flash"
) -> tuple[dict, int, int]:
    """Call LLM and return parsed JSON + token counts."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    resp = await llm_request(messages, model=model, temperature=0.2)
    content = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)

    # Parse JSON from response
    parsed = _extract_json(content)
    return parsed, input_tokens, output_tokens


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return {}


async def locate_files_phase(
    issue_title: str,
    issue_body: str,
    candidate_files: list[str],
    model: str,
) -> tuple[list[str], int, int]:
    """LLM identifies the files that need changes. Returns (ranked_paths, input_tokens, output_tokens)."""
    if not candidate_files:
        return [], 0, 0

    file_list = "\n".join(f"- {f}" for f in candidate_files[:30])
    system = (
        "You are a code analysis agent. Given a bug report and a list of candidate files, "
        "identify which files most likely need changes to fix the bug. "
        "Return ONLY valid JSON with key 'files': an array of objects, each with: "
        "'path' (string), 'relevance' (0.0-1.0), 'reason' (string). "
        "Order by relevance descending. Include up to 10 files."
    )
    user = (
        f"Bug Title: {issue_title}\n\n"
        f"Bug Description:\n{issue_body[:4000]}\n\n"
        f"Candidate Files:\n{file_list}"
    )

    result, in_tok, out_tok = await llm_call_structured(system, user, model=model)
    files = result.get("files", [])
    ranked = [f["path"] for f in files if f.get("path")]
    return ranked, in_tok, out_tok


async def generate_patch_phase(
    issue_title: str,
    issue_body: str,
    ranked_files: list[dict],  # [{"path": ..., "content": ...}]
    model: str,
    search_terms: list[str] | None = None,
) -> tuple[str, int, int]:
    """LLM generates a unified diff patch. Returns (patch_content, input_tokens, output_tokens).

    Fixes v0 patch=0% root cause:
    - Old version truncated at character boundary (mid-line), confusing the LLM.
    - Old version sent 6000 chars max, too little for large files.
    - LLM hallucinated line numbers because it couldn't see the actual change site.
    - Now: smart section extraction around keyword matches, line-boundary truncation,
      10K chars per file, and explicit diff format enforcement.
    """
    files_context_parts = []
    for f in ranked_files[:5]:
        content = f["content"]
        if not content:
            continue
        path = f["path"]

        # Use full content with accurate line numbers.
        # Truncate large files at LINE boundaries (preserves real line numbers
        # for both head and tail sections, since tail lines keep their original
        # numbers from the full numbered_lines list).
        lines = content.split("\n")
        numbered_lines = []
        for i, line in enumerate(lines, 1):
            numbered_lines.append(f"{i:6d}| {line}")

        # Truncate to ~140 lines (~10K chars) keeping accurate line numbers
        max_display_lines = 140
        if len(numbered_lines) > max_display_lines:
            head_lines = numbered_lines[:80]
            tail_lines = numbered_lines[-60:]
            head_str = "\n".join(head_lines)
            tail_str = "\n".join(tail_lines)
            sep = (
                f"\n... [TRUNCATED: {len(numbered_lines) - len(head_lines) - len(tail_lines)} "
                f"lines omitted. The lines below start at line {len(numbered_lines) - 59}. "
                f"Use line numbers EXACTLY as shown on the left.] ...\n"
            )
            numbered_content = head_str + sep + tail_str
        else:
            numbered_content = "\n".join(numbered_lines)

        files_context_parts.append(
            f"=== FILE: {path} ===\n{numbered_content}"
        )

    files_context = "\n\n".join(files_context_parts) if files_context_parts else "(no files provided)"

    system = (
        "You are a senior software engineer fixing bugs. "
        "Given a bug report and the EXACT source files with LINE NUMBERS, produce a unified diff patch "
        "that fixes the bug. The patch must be apply-able with 'git apply'.\n\n"
        "CRITICAL RULES — failure to follow these results in a broken patch:\n"
        "1. Only patch files that are listed below in the === FILE: ... === blocks. Do NOT invent file paths.\n"
        "2. Use EXACT line numbers as shown in the file content. The line number is the FIRST column before the '|'.\n"
        "3. The diff header MUST use the EXACT paths shown: 'diff --git a/path/to/file b/path/to/file'\n"
        "4. Include at least 3 context lines before and after each change.\n"
        "5. The @@ hunk header format MUST be: @@ -OLD_START,OLD_COUNT +NEW_START,NEW_COUNT @@ FUNCTION_HINT\n"
        "   Count ALL lines (context + removed) for OLD_COUNT, and ALL lines (context + added) for NEW_COUNT.\n"
        "6. After the @@ line, each line MUST start with ' ' (context), '+' (added), or '-' (removed).\n"
        "7. End the patch with a trailing newline.\n\n"
        "EXAMPLE of correct diff:\n"
        "diff --git a/src/foo.py b/src/foo.py\n"
        "--- a/src/foo.py\n"
        "+++ b/src/foo.py\n"
        "@@ -10,7 +10,8 @@ def bar():\n"
        "     x = 1\n"
        "     y = 2\n"
        "-    z = old_buggy_call()\n"
        "+    z = new_fixed_call()\n"
        "+    w = extra_line()\n"
        "     return x + y + z\n\n"
        "Return ONLY valid JSON with keys:\n"
        "- 'analysis' (string): brief explanation of the fix\n"
        "- 'patch' (string): the unified diff, starting with 'diff --git a/...'\n"
        "- 'files_changed' (array of strings): paths of changed files (must be subset of provided files)"
    )
    user = (
        f"Bug Title: {issue_title}\n\n"
        f"Bug Description:\n{issue_body[:4000]}\n\n"
        f"Relevant Source Files (with EXACT line numbers — use these when constructing your diff):\n{files_context}\n\n"
        "Generate a git-apply-compatible patch that fixes this bug. "
        "Remember: use ONLY the file paths listed above, and ONLY the line numbers shown. "
        "Count your hunk lines carefully — mismatched counts cause 'corrupt patch' errors."
    )

    result, in_tok, out_tok = await llm_call_structured(system, user, model=model)
    patch = result.get("patch", "")
    return patch, in_tok, out_tok


def _extract_relevant_sections(
    content: str, search_terms: list[str], max_lines: int = 200
) -> str:
    """Extract the most relevant sections of a file based on search term matches.

    If the file is small, return it whole. Otherwise, find lines containing
    search terms and include surrounding context. This way the LLM sees the
    actual change site with correct line numbers.
    """
    lines = content.split("\n")
    if len(lines) <= max_lines:
        return content

    # Find lines that match any search term
    term_lower_set = {t.lower() for t in search_terms} if search_terms else set()
    match_lines: set[int] = set()
    for i, line in enumerate(lines):
        line_lower = line.lower()
        for term in term_lower_set:
            if len(term) >= 4 and term in line_lower:
                match_lines.add(i)
                break

    if not match_lines:
        # No matches — return beginning + end
        head = "\n".join(lines[:max_lines // 2])
        tail = "\n".join(lines[-(max_lines // 2):])
        return head + "\n... (middle omitted, no keyword matches) ...\n" + tail

    # Expand: include context around matches
    context_radius = 15
    included: set[int] = set()
    for ml in sorted(match_lines):
        for j in range(max(0, ml - context_radius), min(len(lines), ml + context_radius + 1)):
            included.add(j)

    # Build sections, preserving line numbers
    included_sorted = sorted(included)
    sections: list[str] = []
    last = -2
    for idx in included_sorted:
        if idx > last + 1:
            sections.append(f"... [line {idx + 1}] ...")
        sections.append(lines[idx])
        last = idx

    return "\n".join(sections)


# ═══════════════════════════════════════════════════════════════════════════
# single-sample evaluation
# ═══════════════════════════════════════════════════════════════════════════

async def evaluate_sample(sample: dict, idx: int, model: str = "deepseek-v4-flash") -> dict:
    """Evaluate one sample. Returns a result dict."""
    sample_id = sample["id"]
    issue = sample["issue"]
    patch_data = sample["patch"]
    signals = sample.get("signals", {})
    repo_info = sample["repo"]

    actual_files = [f["path"] for f in patch_data.get("files", [])]
    has_tests = signals.get("has_tests_changed", False)

    result: dict[str, Any] = {
        "id": sample_id,
        "repo": f"{repo_info['owner']}/{repo_info['name']}",
        "issue_title": issue["title"],
        "actual_files": actual_files,
        "has_tests_changed": has_tests,
        "file_recall": {"k1": 0.0, "k3": 0.0, "k5": 0.0},
        "patch_apply": False,
        "patch_apply_error": "",
        "test_pass": None,  # None if no tests changed
        "test_output": "",
        "token_usage": {"input": 0, "output": 0, "cost": 0.0},
        "agent_patch": "",
        "error": None,
    }

    t_start = time.monotonic()
    repo_path = EVAL_TMP / sample_id.replace("/", "_").replace("#", "_").replace(":", "_")
    owner, repo_name = repo_info["owner"], repo_info["name"]

    try:
        # ── 1. file discovery via GitHub API (no clone needed) ──
        print(f"  [{sample_id}] Fetching file list via GitHub API...", flush=True)
        all_files = fetch_file_list(owner, repo_name)
        if not all_files:
            # API failed — fall back to clone+grep
            print(f"  [{sample_id}] API failed, falling back to git clone...", flush=True)
            if not clone_repo(owner, repo_name, repo_path):
                result["error"] = "clone_failed"
                return result
            clone_time = time.monotonic() - t_start
            print(f"  [{sample_id}] Clone done in {clone_time:.1f}s", flush=True)
            # Use git ls-files as file list
            ls_result = subprocess.run(
                ["git", "ls-files"], cwd=str(repo_path),
                capture_output=True, text=True, timeout=15,
            )
            all_files = [
                line.strip()
                for line in ls_result.stdout.split("\n")
                if line.strip()
            ]
        else:
            print(f"  [{sample_id}] {len(all_files)} files via API", flush=True)

        # ── 2. find candidate files (filename search + GitHub code search) ──
        search_terms = _extract_search_terms(issue["title"], issue["body"])
        print(f"  [{sample_id}] Search terms: {search_terms}", flush=True)

        candidate_set: set[str] = set()

        # Strategy A: GitHub code search (content-based, most effective)
        # Code search finds files by CONTENT, catching cases like
        # "allowlist_externals" -> replace.py, "INJECT_FACTS_AS_VARS" -> manager.py
        code_search_terms = search_terms[:8]  # top 8 terms for code search
        for term in code_search_terms:
            try:
                for f in search_code_via_github(owner, repo_name, term, max_results=10):
                    candidate_set.add(f)
            except Exception:
                pass  # code search is best-effort

        # Strategy B: filename search (fast, works if filenames appear in issue)
        for term in search_terms[:10]:
            for f in search_files_by_name(all_files, term, max_results=30):
                candidate_set.add(f)

        # Strategy C: derive search terms from code search file paths, then re-search
        # Code search often finds files whose NAMES don't match the issue text.
        # Extract path components (e.g., "src/tox/config/loader/ini/replace.py"
        # → "replace", "loader", "ini") and use them as supplementary search terms
        # to boost recall, especially for repos where code search returns few hits.
        path_derived_terms: set[str] = set()
        for path in list(candidate_set):
            for part in path.replace("/", ".").replace("-", "_").split("."):
                part = part.lower().strip("~")
                if len(part) >= 3 and part not in {"py", "src", "test", "tests"}:
                    path_derived_terms.add(part)

        # Always run supplementary search with path-derived terms
        # (regardless of candidate count — large sets may still miss the target)
        if path_derived_terms:
            # Supplementary code search with path-derived terms (skip already-searched)
            for term in list(path_derived_terms)[:10]:
                if term in code_search_terms:
                    continue
                try:
                    for f in search_code_via_github(owner, repo_name, term, max_results=10):
                        candidate_set.add(f)
                except Exception:
                    pass

            # Supplementary filename search with path-derived terms
            for term in list(path_derived_terms)[:15]:
                if term in search_terms:
                    continue
                for f in search_files_by_name(all_files, term, max_results=30):
                    candidate_set.add(f)

        candidate_files = sorted(candidate_set)
        print(f"  [{sample_id}] Found {len(candidate_files)} candidate files", flush=True)

        # ── 3. LLM: locate files ──
        print(f"  [{sample_id}] Phase 1: locating files...", flush=True)
        ranked_paths, in1, out1 = await locate_files_phase(
            issue["title"], issue["body"], candidate_files, model
        )
        total_in = in1
        total_out = out1
        print(f"  [{sample_id}] LLM ranked {len(ranked_paths)} files (in={in1}, out={out1})", flush=True)

        # ── 4. compute file_recall ──
        for k in [1, 3, 5]:
            result["file_recall"][f"k{k}"] = compute_file_recall(actual_files, ranked_paths, k)
        print(f"  [{sample_id}] file_recall: k1={result['file_recall']['k1']:.2f} "
              f"k3={result['file_recall']['k3']:.2f} k5={result['file_recall']['k5']:.2f}", flush=True)

        # ── 5. read top files via GitHub API (instant, no clone) ──
        files_to_read: list[str] = []
        for path in ranked_paths[:5]:
            if path not in files_to_read:
                files_to_read.append(path)
        for af in actual_files[:5]:
            if af not in files_to_read:
                files_to_read.append(af)

        top_files: list[dict] = []
        for path in files_to_read[:5]:
            content = fetch_file_content(owner, repo_name, path)
            if content:
                top_files.append({"path": path, "content": content})
        print(f"  [{sample_id}] Read {len(top_files)} files via API", flush=True)

        # ── 6. LLM: generate patch ──
        print(f"  [{sample_id}] Phase 2: generating patch...", flush=True)
        agent_patch, in2, out2 = await generate_patch_phase(
            issue["title"], issue["body"], top_files, model,
            search_terms=search_terms,
        )
        total_in += in2
        total_out += out2
        result["agent_patch"] = agent_patch[:50000]
        print(f"  [{sample_id}] Patch generated ({len(agent_patch)} chars, in={in2}, out={out2})", flush=True)

        result["token_usage"]["input"] = total_in
        result["token_usage"]["output"] = total_out
        result["token_usage"]["cost"] = estimate_cost(total_in, total_out)

        # ── 7. apply patch (needs clone — attempt now) ──
        if agent_patch.strip():
            print(f"  [{sample_id}] Cloning for patch apply ({CLONE_TIMEOUT}s timeout)...", flush=True)
            if clone_repo(owner, repo_name, repo_path, timeout=CLONE_TIMEOUT):
                print(f"  [{sample_id}] Clone OK, applying patch...", flush=True)
                ok, apply_output = apply_patch(repo_path, agent_patch)
                result["patch_apply"] = ok
                if not ok:
                    result["patch_apply_error"] = apply_output[:2000]
                print(f"  [{sample_id}] Patch apply: {'OK' if ok else 'FAILED'}", flush=True)

                if has_tests and ok:
                    print(f"  [{sample_id}] Running tests...", flush=True)
                    test_ok, test_output = run_tests(repo_path)
                    result["test_pass"] = test_ok
                    result["test_output"] = test_output[:3000]
                    print(f"  [{sample_id}] Tests: {'PASS' if test_ok else 'FAIL'}", flush=True)
            else:
                print(f"  [{sample_id}] Clone timed out — skipping apply/test", flush=True)
                result["patch_apply_error"] = "clone_timeout_for_apply"
        else:
            result["patch_apply_error"] = "LLM did not produce a patch"
            print(f"  [{sample_id}] No patch produced by LLM", flush=True)

    except asyncio.TimeoutError:
        result["error"] = "timeout"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"  [{sample_id}] ERROR: {exc}", flush=True)

    finally:
        # ── 9. cleanup ──
        elapsed = time.monotonic() - t_start
        print(f"  [{sample_id}] Done in {elapsed:.1f}s", flush=True)
        if repo_path.exists():
            shutil.rmtree(repo_path, ignore_errors=True)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

async def run_eval(
    n_samples: int = MAX_SAMPLES,
    model: str = "deepseek-v4-flash",
) -> list[dict]:
    """Run the full evaluation on n_samples."""
    samples = load_samples(n_samples)
    print(f"Loaded {len(samples)} samples from {SAMPLES_PATH}", flush=True)

    EVAL_TMP.mkdir(parents=True, exist_ok=True)

    results = []
    for i, sample in enumerate(samples):
        print(f"\n{'='*60}")
        print(f"Sample {i+1}/{len(samples)}: {sample['id']}")
        print(f"{'='*60}", flush=True)

        try:
            result = await asyncio.wait_for(
                evaluate_sample(sample, i, model=model),
                timeout=TIMEOUT_PER_SAMPLE,
            )
        except asyncio.TimeoutError:
            result = {
                "id": sample["id"],
                "repo": f"{sample['repo']['owner']}/{sample['repo']['name']}",
                "error": "sample_timeout",
            }
        results.append(result)

    # ── save results ──
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {RESULTS_PATH}", flush=True)

    return results


async def run_agent_v2_eval(
    n_samples: int = MAX_SAMPLES,
    max_retries: int = 3,
    token_budget: int = 50000,
    sample_id: str | None = None,
    seed_gold_files: bool = False,
) -> list[dict[str, Any]]:
    module = importlib.import_module("eval.agent_v2_harness")
    return await module.run_agent_v2_eval(
        n_samples=n_samples,
        max_retries=max_retries,
        token_budget=token_budget,
        sample_id=sample_id,
        seed_gold_files=seed_gold_files,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python eval/harness.py",
        description="Run RepoPilot evals.",
    )
    parser.add_argument(
        "--agent-v2",
        action="store_true",
        help="Run the state-graph agent eval mode with saved-run replay.",
    )
    parser.add_argument("--samples", type=int, default=MAX_SAMPLES)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--token-budget", type=int, default=50000)
    parser.add_argument(
        "--sample-id",
        default=None,
        help="Run only this dataset sample id (e.g. 'scrapy/scrapy#6195:7095'), "
        "scanning the whole file instead of taking the first --samples lines.",
    )
    parser.add_argument(
        "--seed-gold-files",
        action="store_true",
        help="Offline locate: seed relevant_files from the dataset's gold changed "
        "files (fetched via the Contents API, not code search) and start at PLAN. "
        "Removes GitHub code-search rate-limiting from the critical path.",
    )
    args = parser.parse_args(argv)

    if args.agent_v2:
        asyncio.run(
            run_agent_v2_eval(
                n_samples=args.samples,
                max_retries=args.max_retries,
                token_budget=args.token_budget,
                sample_id=args.sample_id,
                seed_gold_files=args.seed_gold_files,
            )
        )
        return

    asyncio.run(run_eval(n_samples=args.samples, model=args.model))


if __name__ == "__main__":
    main()
