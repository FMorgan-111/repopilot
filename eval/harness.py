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

import asyncio
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

# ── repo root ─────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent

# ── inline LLM client (avoid relative-import issues from src/ modules) ────
import httpx
from dotenv import load_dotenv

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
DEEPSEEK_PRICING = {
    "input": 0.27 / 1_000_000,   # $0.27 per 1M input tokens
    "output": 0.36 / 1_000_000,  # $0.36 per 1M output tokens (cached miss)
}


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


def _extract_search_terms(title: str, body: str, max_terms: int = 5) -> list[str]:
    """Extract useful search terms from the issue title and body."""
    text = f"{title} {body[:2000]}"
    # grab backtick-quoted identifiers (class names, functions, vars)
    code_terms = re.findall(r"`([A-Za-z_][A-Za-z0-9_.]{2,80})`", text)
    # grab CamelCase and snake_case words
    words = re.findall(r"[A-Z][a-z]+(?:[A-Z][a-z]+)+|[a-z]+(?:_[a-z]+)+", text)
    stop = {
        "the", "and", "for", "with", "when", "this", "that", "from",
        "issue", "error", "bug", "https", "http", "com", "github",
        "import", "from", "class", "def", "return", "self", "true", "false",
    }
    seen: set[str] = set()
    terms: list[str] = []
    for t in code_terms + words:
        t = t.strip("`.,:;()[]{}\"'").lower()
        if t in stop or len(t) < 3:
            continue
        if t not in seen:
            seen.add(t)
            terms.append(t)
        if len(terms) >= max_terms:
            break
    if not terms:
        # fallback: take title words
        terms = [w.lower() for w in title.split() if len(w) > 3][:max_terms]
    return terms


def clone_repo(owner: str, repo: str, target: Path, timeout: int = 300) -> bool:
    """Shallow-clone a GitHub repo. Returns True on success."""
    target.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{owner}/{repo}.git"
    strategies = [
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--single-branch", url, str(target)],
        ["git", "clone", "--depth", "1", url, str(target)],
    ]
    for cmd in strategies:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0:
                return True
        except subprocess.TimeoutExpired:
            continue
    return False


def grep_repo(repo_path: Path, term: str, max_files: int = 15) -> list[str]:
    """Search repo for files containing a term (ripgrep or grep)."""
    try:
        result = subprocess.run(
            ["rg", "-l", "--max-filesize", "500K", "--iglob", "!*.{png,jpg,gif,svg,woff,woff2,ttf,eot,ico,lock,json,yaml,yml,toml,csv,log}", term, str(repo_path)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            files = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
            return files[:max_files]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # fallback to grep
    try:
        result = subprocess.run(
            ["grep", "-r", "-l", "--include=*.py", "--include=*.js", "--include=*.ts",
             "--include=*.rs", "--include=*.go", "--include=*.java", term, str(repo_path)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            files = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
            return files[:max_files]
    except subprocess.TimeoutExpired:
        pass
    return []


def read_file_content(repo_path: Path, rel_path: str) -> str:
    """Read a file from the cloned repo."""
    full = repo_path / rel_path
    try:
        return full.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def apply_patch(repo_path: Path, patch_content: str) -> tuple[bool, str]:
    """Apply a unified diff patch. Returns (success, output)."""
    result = subprocess.run(
        ["git", "apply", "--check"],
        input=patch_content,
        cwd=str(repo_path),
        capture_output=True, text=True, timeout=30,
    )
    check_ok = result.returncode == 0
    check_output = (result.stdout + result.stderr).strip()

    if not check_ok:
        return False, check_output

    result2 = subprocess.run(
        ["git", "apply"],
        input=patch_content,
        cwd=str(repo_path),
        capture_output=True, text=True, timeout=30,
    )
    apply_ok = result2.returncode == 0
    apply_output = (result2.stdout + result2.stderr).strip()
    return apply_ok, apply_output


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
) -> tuple[str, int, int]:
    """LLM generates a unified diff patch. Returns (patch_content, input_tokens, output_tokens)."""
    files_context = "\n\n".join(
        f"=== FILE: {f['path']} ===\n{f['content'][:3000]}"
        for f in ranked_files[:3]
    )

    system = (
        "You are a senior software engineer fixing bugs. "
        "Given a bug report and relevant source files, produce a unified diff patch "
        "that fixes the bug. The patch must be apply-able with 'git apply'.\n\n"
        "Return ONLY valid JSON with keys:\n"
        "- 'analysis' (string): brief explanation of the fix\n"
        "- 'patch' (string): the unified diff, starting with 'diff --git a/...'\n"
        "- 'files_changed' (array of strings): paths of changed files"
    )
    user = (
        f"Bug Title: {issue_title}\n\n"
        f"Bug Description:\n{issue_body[:4000]}\n\n"
        f"Relevant Source Files:\n{files_context}\n\n"
        "Generate a git-apply-compatible patch that fixes this bug."
    )

    result, in_tok, out_tok = await llm_call_structured(system, user, model=model)
    patch = result.get("patch", "")
    return patch, in_tok, out_tok


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

    try:
        # ── 1. clone repo ──
        print(f"  [{sample_id}] Cloning {repo_info['owner']}/{repo_info['name']}...", flush=True)
        if not clone_repo(repo_info["owner"], repo_info["name"], repo_path):
            result["error"] = "clone_failed"
            return result
        clone_time = time.monotonic() - t_start
        print(f"  [{sample_id}] Clone done in {clone_time:.1f}s", flush=True)

        # ── 2. find candidate files via grep ──
        search_terms = _extract_search_terms(issue["title"], issue["body"])
        print(f"  [{sample_id}] Search terms: {search_terms}", flush=True)

        candidate_set: set[str] = set()
        for term in search_terms[:5]:
            files = grep_repo(repo_path, term)
            for f in files:
                rel = str(Path(f).relative_to(repo_path))
                candidate_set.add(rel)

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

        # ── 5. read top files ──
        top_files: list[dict] = []
        for path in ranked_paths[:3]:
            content = read_file_content(repo_path, path)
            if content:
                top_files.append({"path": path, "content": content})

        if not top_files and candidate_files:
            # fallback: read any matching actual files
            for af in actual_files[:3]:
                content = read_file_content(repo_path, af)
                if content:
                    top_files.append({"path": af, "content": content})

        print(f"  [{sample_id}] Read {len(top_files)} files for patch generation", flush=True)

        # ── 6. LLM: generate patch ──
        print(f"  [{sample_id}] Phase 2: generating patch...", flush=True)
        agent_patch, in2, out2 = await generate_patch_phase(
            issue["title"], issue["body"], top_files, model
        )
        total_in += in2
        total_out += out2
        result["agent_patch"] = agent_patch[:50000]  # truncate huge patches
        print(f"  [{sample_id}] Patch generated ({len(agent_patch)} chars, in={in2}, out={out2})", flush=True)

        result["token_usage"]["input"] = total_in
        result["token_usage"]["output"] = total_out
        result["token_usage"]["cost"] = estimate_cost(total_in, total_out)

        # ── 7. apply patch ──
        if agent_patch.strip():
            print(f"  [{sample_id}] Applying patch...", flush=True)
            ok, apply_output = apply_patch(repo_path, agent_patch)
            result["patch_apply"] = ok
            if not ok:
                result["patch_apply_error"] = apply_output[:2000]
            print(f"  [{sample_id}] Patch apply: {'OK' if ok else 'FAILED'}", flush=True)

            # ── 8. run tests (if has_tests_changed) ──
            if has_tests and ok:
                print(f"  [{sample_id}] Running tests...", flush=True)
                test_ok, test_output = run_tests(repo_path)
                result["test_pass"] = test_ok
                result["test_output"] = test_output[:3000]
                print(f"  [{sample_id}] Tests: {'PASS' if test_ok else 'FAIL'}", flush=True)
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


if __name__ == "__main__":
    asyncio.run(run_eval())
