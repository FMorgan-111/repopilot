# SWE-bench Integration and Resilient API Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run RepoPilot against reproducible SWE-bench Verified instances at their exact `base_commit`, export official prediction files, and then make live GitHub API reads resilient to temporary outages with bounded stale-cache fallback.

**Architecture:** Add a benchmark adapter that normalizes official SWE-bench rows without exposing gold data to the agent. Prepare a commit-keyed local checkout before LOCATE so search, patching, and tests all see one historical tree; retain GitHub API search for normal live runs. Extend the existing file cache with fresh/stale states and a retryable-error predicate after the benchmark path is working.

**Tech Stack:** Python 3.10+, Pydantic, asyncio subprocesses, Git, pytest/pytest-asyncio, Hugging Face `datasets` as an optional eval dependency, SWE-bench Docker harness.

## Global Constraints

- Implement SWE-bench Verified before changing the GitHub API cache.
- Never send `patch`, `test_patch`, `FAIL_TO_PASS`, or `PASS_TO_PASS` to the model or store them in RepoPilot traces.
- Benchmark checkouts must resolve exactly to the 40-character hexadecimal `base_commit`.
- Normal runs without `repo_ref` continue to target the latest default branch.
- API fresh TTL remains 600 seconds; stale fallback defaults to 86,400 seconds.
- Stale fallback is allowed only for HTTP 429/502/503/504, network errors, and timeouts.
- Credentials must remain environment-only and must not appear in logs, persisted errors, prediction files, or Git remotes after network operations.
- Existing custom JSONL samples and `--seed-gold-files` remain supported.
- Every production behavior change follows RED → GREEN → REFACTOR and receives a focused commit.

---

## File Structure

- Create `eval/swe_bench.py`: dataset loading, normalization, diverse selection, local cache, safe agent seed, prediction serialization.
- Create `tests/test_swe_bench.py`: adapter, sampling, persistence, leak-prevention, prediction tests.
- Modify `pyproject.toml`: optional `eval` dependency group.
- Modify `src/state.py`: immutable benchmark reference field.
- Modify `src/new_agent.py`: distinguish issue-only and oracle-file seeds; export applied Git diff.
- Modify `tests/test_new_agent.py`: seed routing and payload patch tests.
- Modify `src/nodes/execute.py`: commit-aware cache preparation, live refresh, prepared-repo environment setup, cache diagnostics.
- Modify `tests/test_new_agent.py`: exact-ref and live-refresh repository tests.
- Create `src/local_search.py`: bounded tracked-file search and reads for historical work trees.
- Create `tests/test_local_search.py`: local checkout search contract.
- Modify `src/nodes/locate.py`: local historical locator branch with no GitHub API mixing.
- Modify `tests/test_locate_bm25.py`: local LOCATE integration and BM25 ordering.
- Modify `eval/agent_v2_harness.py`: SWE-bench source selection, exact checkout preparation, prediction output.
- Modify `tests/test_agent_v2_eval.py`: harness source/seed/output tests.
- Modify `src/http_client.py`: public retryable-GitHub predicate.
- Modify `src/cache.py`: bounded stale entries and safe diagnostics.
- Modify `src/tools.py`: enable stale fallback for GitHub reads.
- Modify `tests/test_cache.py` and `tests/test_tools.py`: fresh/stale/error/logging coverage.
- Modify `README.md`: optional eval installation and reproducible commands.

---

### Task 1: SWE-bench Dataset Adapter

**Files:**
- Create: `eval/swe_bench.py`
- Create: `tests/test_swe_bench.py`
- Modify: `pyproject.toml:34-48`

**Interfaces:**
- Produces: `load_verified_samples(count: int, seed: int, cache_path: Path | None = None, dataset_loader: Callable | None = None) -> list[dict[str, Any]]`
- Produces: `build_agent_seed(sample: Mapping[str, Any], repo_path: str) -> dict[str, Any]`
- Produces: `write_predictions(results: Sequence[Mapping[str, Any]], path: Path) -> Path`
- Consumes later: normalized records with `id`, `instance_id`, `repo`, `issue`, `base_commit`, evaluator-only patch/test fields.

- [ ] **Step 1: Write failing adapter and leakage tests**

```python
# tests/test_swe_bench.py
import json

from eval import swe_bench


def row(instance_id="acme__widget-8", repo="acme/widget"):
    return {
        "instance_id": instance_id,
        "repo": repo,
        "issue_id": "7",
        "issue_url": "https://github.com/acme/widget/issues/7",
        "problem_statement": "Login crash\n\nThe endpoint raises ValueError.",
        "base_commit": "a" * 40,
        "patch": "diff --git a/src/auth.py b/src/auth.py\n",
        "test_patch": "diff --git a/tests/test_auth.py b/tests/test_auth.py\n",
        "FAIL_TO_PASS": '["tests/test_auth.py::test_login"]',
        "PASS_TO_PASS": '["tests/test_auth.py::test_other"]',
        "version": "1.0",
        "created_at": "2026-01-01",
        "difficulty": "medium",
    }


def test_normalize_verified_row_preserves_evaluator_fields():
    sample = swe_bench.normalize_verified_row(row())
    assert sample["id"] == "acme__widget-8"
    assert sample["repo"] == {"owner": "acme", "name": "widget"}
    assert sample["issue"]["title"] == "Login crash"
    assert sample["base_commit"] == "a" * 40
    assert sample["evaluation"]["fail_to_pass"] == ["tests/test_auth.py::test_login"]
    assert sample["evaluation"]["gold_patch"].startswith("diff --git")


def test_agent_seed_excludes_gold_and_test_material():
    seed = swe_bench.build_agent_seed(
        swe_bench.normalize_verified_row(row()), "/tmp/acme-widget"
    )
    encoded = json.dumps(seed)
    assert seed["repo_ref"] == "a" * 40
    assert seed["repo_path"] == "/tmp/acme-widget"
    assert "gold_patch" not in encoded
    assert "test_patch" not in encoded
    assert "FAIL_TO_PASS" not in encoded


def test_diverse_selection_is_deterministic_and_spreads_repositories():
    rows = [
        row("acme__widget-1", "acme/widget"),
        row("acme__widget-2", "acme/widget"),
        row("other__tool-3", "other/tool"),
    ]
    first = swe_bench.select_diverse(rows, count=2, seed=17)
    second = swe_bench.select_diverse(rows, count=2, seed=17)
    assert [item["instance_id"] for item in first] == [item["instance_id"] for item in second]
    assert {item["repo"] for item in first} == {"acme/widget", "other/tool"}
```

- [ ] **Step 2: Run adapter tests and verify RED**

Run: `python -m pytest tests/test_swe_bench.py -q`

Expected: collection fails with `ImportError: cannot import name 'swe_bench' from 'eval'`.

- [ ] **Step 3: Implement normalization, safe seed, selection, persistence, and prediction serialization**

```python
# eval/swe_bench.py
from __future__ import annotations

import json
import os
import random
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

DATASET_NAME = "SWE-bench/SWE-bench_Verified"
DATASET_REVISION = "main"


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not value:
        return []
    decoded = json.loads(value)
    return [str(item) for item in decoded]


def normalize_verified_row(row: Mapping[str, Any]) -> dict[str, Any]:
    owner, name = str(row["repo"]).split("/", 1)
    statement = str(row.get("problem_statement") or "")
    first_line, _, rest = statement.partition("\n")
    return {
        "id": str(row["instance_id"]),
        "instance_id": str(row["instance_id"]),
        "source": "swe-bench-verified",
        "repo": {"owner": owner, "name": name},
        "issue": {
            "number": int(row.get("issue_id") or 0),
            "url": str(row.get("issue_url") or ""),
            "title": first_line.strip() or str(row["instance_id"]),
            "body": rest.strip() or statement,
        },
        "base_commit": str(row["base_commit"]),
        "evaluation": {
            "gold_patch": str(row.get("patch") or ""),
            "test_patch": str(row.get("test_patch") or ""),
            "fail_to_pass": _json_list(row.get("FAIL_TO_PASS")),
            "pass_to_pass": _json_list(row.get("PASS_TO_PASS")),
        },
        "metadata": {
            "version": row.get("version"),
            "created_at": row.get("created_at"),
            "difficulty": row.get("difficulty"),
        },
    }


def select_diverse(rows: Sequence[Mapping[str, Any]], count: int, seed: int) -> list[Mapping[str, Any]]:
    rng = random.Random(seed)
    grouped: dict[str, deque[Mapping[str, Any]]] = defaultdict(deque)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    for item in shuffled:
        grouped[str(item["repo"])].append(item)
    repos = list(grouped)
    rng.shuffle(repos)
    selected: list[Mapping[str, Any]] = []
    while repos and len(selected) < count:
        next_repos: list[str] = []
        for repo in repos:
            selected.append(grouped[repo].popleft())
            if grouped[repo]:
                next_repos.append(repo)
            if len(selected) == count:
                break
        repos = next_repos
    return selected


def _default_cache_path() -> Path:
    root = Path(os.getenv("REPOPILOT_HOME", Path.home() / ".repopilot"))
    return root / "eval" / "datasets" / "swe-bench-verified.jsonl"


def load_verified_samples(count: int, seed: int, cache_path: Path | None = None, dataset_loader: Callable | None = None) -> list[dict[str, Any]]:
    path = cache_path or _default_cache_path()
    if path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    else:
        if dataset_loader is None:
            from datasets import load_dataset
            dataset_loader = load_dataset
        rows = list(
            dataset_loader(DATASET_NAME, split="test", revision=DATASET_REVISION)
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(dict(item), ensure_ascii=False) + "\n" for item in rows), encoding="utf-8")
        path.with_suffix(".metadata.json").write_text(
            json.dumps(
                {"dataset": DATASET_NAME, "revision": DATASET_REVISION},
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return [normalize_verified_row(item) for item in select_diverse(rows, count, seed)]


def build_agent_seed(sample: Mapping[str, Any], repo_path: str) -> dict[str, Any]:
    return {
        "owner": sample["repo"]["owner"],
        "repo": sample["repo"]["name"],
        "issue_number": sample["issue"]["number"],
        "issue_title": sample["issue"]["title"],
        "issue_body": sample["issue"]["body"],
        "repo_ref": sample["base_commit"],
        "repo_path": repo_path,
    }


def write_predictions(results: Sequence[Mapping[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"instance_id": item["instance_id"], "model_name_or_path": item["model"], "model_patch": item.get("model_patch", "")}) for item in results]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path
```

Add to `pyproject.toml`:

```toml
eval = [
    "datasets>=3,<5",
    "swebench>=4.1,<5",
]
```

- [ ] **Step 4: Run adapter tests and verify GREEN**

Run: `python -m pytest tests/test_swe_bench.py -q`

Expected: all adapter tests pass.

- [ ] **Step 5: Commit the adapter**

```bash
git add eval/swe_bench.py tests/test_swe_bench.py pyproject.toml
git commit -m "feat(eval): add SWE-bench Verified adapter"
```

---

### Task 2: Issue-Only Seed Routing and Benchmark Reference State

**Files:**
- Modify: `src/state.py:185-230`
- Modify: `src/new_agent.py:316-357`
- Modify: `tests/test_new_agent.py`

**Interfaces:**
- Consumes: seed keys produced by `build_agent_seed`.
- Produces: `AgentState.repo_ref: str`, issue-only seeds starting at `Phase.LOCATE`, oracle seeds starting at `Phase.PLAN`.

- [ ] **Step 1: Write failing routing tests**

```python
async def test_issue_only_seed_starts_agent_at_locate(monkeypatch):
    captured = {}

    async def fake_run_graph(graph, state):
        captured["state"] = state
        captured["start"] = graph.start
        state.current_phase = new_agent.Phase.FAILED
        return state

    monkeypatch.setattr(new_agent, "StateGraph", None)
    monkeypatch.setattr(new_agent, "run_graph", fake_run_graph)
    monkeypatch.setattr(new_agent, "_save_trace", lambda *args, **kwargs: None)
    await new_agent.agent_v2(
        "https://github.com/acme/widget/issues/7",
        seed={
            "owner": "acme", "repo": "widget", "issue_number": 7,
            "issue_title": "Login crash", "issue_body": "ValueError",
            "repo_ref": "a" * 40, "repo_path": "/tmp/widget",
        },
    )
    assert captured["start"] == "locate_code"
    assert captured["state"].repo_ref == "a" * 40
    assert captured["state"].repo_path == "/tmp/widget"


async def test_oracle_seed_with_files_starts_agent_at_plan(monkeypatch):
    captured = {}
    async def fake_run_graph(graph, state):
        captured["start"] = graph.start
        state.current_phase = new_agent.Phase.FAILED
        return state
    monkeypatch.setattr(new_agent, "StateGraph", None)
    monkeypatch.setattr(new_agent, "run_graph", fake_run_graph)
    monkeypatch.setattr(new_agent, "_save_trace", lambda *args, **kwargs: None)
    await new_agent.agent_v2(
        "https://github.com/acme/widget/issues/7",
        seed={"owner": "acme", "repo": "widget", "relevant_files": [{"path": "src/auth.py", "content": "pass"}]},
    )
    assert captured["start"] == "plan_fix"
```

- [ ] **Step 2: Run routing tests and verify RED**

Run: `python -m pytest tests/test_new_agent.py -k 'issue_only_seed or oracle_seed' -q`

Expected: issue-only seed starts at `plan_fix` and/or `AgentState` lacks `repo_ref`.

- [ ] **Step 3: Implement the seed distinction**

Add to `AgentState`:

```python
repo_ref: str = ""
```

Replace the seed routing block in `agent_v2` with:

```python
    if seed:
        state.owner = seed.get("owner", "")
        state.repo = seed.get("repo", "")
        state.issue_number = seed.get("issue_number", 0)
        state.issue_title = seed.get("issue_title", "")
        state.issue_body = seed.get("issue_body", "")
        state.repo_ref = seed.get("repo_ref", "")
        state.repo_path = seed.get("repo_path", "")
        state.relevant_files = [FileInfo(**item) for item in seed.get("relevant_files", [])]
        start_phase = Phase.PLAN if state.relevant_files else Phase.LOCATE
        state.current_phase = start_phase
```

- [ ] **Step 4: Run routing tests and verify GREEN**

Run: `python -m pytest tests/test_new_agent.py -k 'seed' -q`

Expected: issue-only and oracle routing tests pass.

- [ ] **Step 5: Commit seed routing**

```bash
git add src/state.py src/new_agent.py tests/test_new_agent.py
git commit -m "feat(agent): separate issue and oracle eval seeds"
```

---

### Task 3: Exact-Commit Repository Cache and Live Refresh

**Files:**
- Modify: `src/nodes/execute.py:66-261,724-787`
- Modify: `tests/test_new_agent.py:586-752`

**Interfaces:**
- Consumes: `AgentState.owner`, `repo`, and optional `repo_ref`.
- Produces: `git_clone(state: AgentState) -> str` whose returned work tree has `HEAD == repo_ref` in benchmark mode.
- Produces: `_repo_cache_event(event: str, state: AgentState, **details: object) -> None` with credential-free stderr diagnostics.

- [ ] **Step 1: Write failing exact-reference and live-refresh tests**

```python
async def test_git_clone_fetches_exact_benchmark_ref_and_keys_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("REPOPILOT_HOME", str(tmp_path / "home"))
    commands = []
    async def fake_run(args, timeout, cwd=None):
        commands.append(args)
        return execute_node._ProcResult(0, "", "")
    monkeypatch.setattr(execute_node, "_run_git_async", fake_run)
    monkeypatch.setattr(execute_node, "_worktree_is_healthy", lambda path: async_true())
    monkeypatch.setattr(execute_node, "_worktree_head", lambda path: async_value("a" * 40))
    state = new_agent.AgentState(issue_url="x", owner="acme", repo="widget", repo_ref="a" * 40)
    path = await execute_node.git_clone(state)
    assert "a" * 40 in path
    assert any(command[:3] == ["git", "fetch", "--depth"] and "a" * 40 in command for command in commands)


async def test_git_clone_refreshes_live_cache_before_reuse(monkeypatch, tmp_path):
    home = tmp_path / "home"
    cache = home / "repos" / "acme-widget"
    work = home / "repos" / "acme-widget-work"
    (cache / ".git").mkdir(parents=True)
    (work / ".git").mkdir(parents=True)
    monkeypatch.setenv("REPOPILOT_HOME", str(home))
    commands = []
    async def fake_run(args, timeout, cwd=None):
        commands.append(args)
        return execute_node._ProcResult(0, "", "")
    monkeypatch.setattr(execute_node, "_run_git_async", fake_run)
    monkeypatch.setattr(execute_node, "_worktree_is_healthy", lambda path: async_true())
    await execute_node.git_clone(new_agent.AgentState(issue_url="x", owner="acme", repo="widget"))
    assert ["git", "-C", str(cache), "fetch", "--prune", "origin"] in commands
    assert ["git", "-C", str(work), "reset", "--hard", "HEAD"] not in commands
```

The test file defines:

```python
async def async_true():
    return True

async def async_value(value):
    return value
```

- [ ] **Step 2: Run repository tests and verify RED**

Run: `python -m pytest tests/test_new_agent.py -k 'git_clone' -q`

Expected: `repo_ref` does not affect cache paths and live cache reuse performs no fetch.

- [ ] **Step 3: Implement commit-keyed preparation and live refresh**

Add validation and path helpers:

```python
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")

def _validated_repo_ref(state: AgentState) -> str:
    if not state.repo_ref:
        return ""
    if not _COMMIT_RE.fullmatch(state.repo_ref):
        raise ValueError("repo_ref must be a 40-character hexadecimal Git commit")
    return state.repo_ref.lower()

def _repo_key(owner: str, repo: str, repo_ref: str = "") -> str:
    base = f"{owner.replace('/', '-')}-{repo.replace('/', '-')}"
    return f"{base}-{repo_ref}" if repo_ref else base

def _repo_cache_path(owner: str, repo: str, repo_ref: str = "") -> Path:
    return _repopilot_home() / "repos" / _repo_key(owner, repo, repo_ref)

def _repo_work_path(owner: str, repo: str, repo_ref: str = "") -> Path:
    return _repopilot_home() / "repos" / f"{_repo_key(owner, repo, repo_ref)}-work"

async def _worktree_head(path: str) -> str:
    result = await _run_git_async(["git", "-C", path, "rev-parse", "HEAD"], timeout=30)
    return result.stdout.strip() if result.returncode == 0 else ""

def _repo_cache_event(
    event: str, state: AgentState, **details: object
) -> None:
    payload = {
        "event": event,
        "repo": f"{state.owner}/{state.repo}",
        "ref": state.repo_ref[:12] if state.repo_ref else "latest",
        **details,
    }
    print(f"[repo-cache] {json.dumps(payload, sort_keys=True)}", file=sys.stderr)
```

Benchmark population uses an empty repository and an exact shallow fetch:

```python
await _run_git_async(["git", "init", str(cache_path)], timeout=60)
await _run_git_async(["git", "-C", str(cache_path), "remote", "add", "origin", repo_url], timeout=60)
result = await _run_git_async(["git", "-C", str(cache_path), "fetch", "--depth", "1", "origin", repo_ref], timeout=300)
await _run_git_async(["git", "-C", str(cache_path), "checkout", "--detach", "FETCH_HEAD"], timeout=60)
await _run_git_async(["git", "-C", str(cache_path), "remote", "set-url", "origin", safe_repo_url], timeout=60)
```

Use these helpers for live refresh and exact-ref population:

```python
async def _checked_git(args: list[str], timeout: float) -> _ProcResult:
    result = await _run_git_async(args, timeout=timeout)
    if result.returncode:
        raise RuntimeError(
            _redact_sensitive_error_text((result.stderr or result.stdout).strip())
        )
    return result


async def _refresh_live_cache(state: AgentState, cache_path: Path) -> None:
    tokenized = _repo_url(state, include_token=True)
    safe = _repo_url(state, include_token=False)
    await _checked_git(
        ["git", "-C", str(cache_path), "remote", "set-url", "origin", tokenized],
        60,
    )
    try:
        await _checked_git(
            ["git", "-C", str(cache_path), "fetch", "--prune", "origin"], 300
        )
        await _checked_git(
            ["git", "-C", str(cache_path), "remote", "set-head", "origin", "-a"],
            60,
        )
        await _checked_git(
            [
                "git", "-C", str(cache_path), "reset", "--hard",
                "refs/remotes/origin/HEAD",
            ],
            60,
        )
    finally:
        await _run_git_async(
            ["git", "-C", str(cache_path), "remote", "set-url", "origin", safe],
            timeout=60,
        )


async def _populate_ref_cache(
    state: AgentState, cache_path: Path, repo_ref: str
) -> None:
    await _checked_git(["git", "init", str(cache_path)], 60)
    await _checked_git(
        [
            "git", "-C", str(cache_path), "remote", "add", "origin",
            _repo_url(state, include_token=True),
        ],
        60,
    )
    try:
        await _checked_git(
            [
                "git", "-C", str(cache_path), "fetch", "--depth", "1",
                "origin", repo_ref,
            ],
            300,
        )
        await _checked_git(
            ["git", "-C", str(cache_path), "checkout", "--detach", "FETCH_HEAD"],
            60,
        )
    finally:
        await _run_git_async(
            [
                "git", "-C", str(cache_path), "remote", "set-url", "origin",
                _repo_url(state, include_token=False),
            ],
            timeout=60,
        )
```

`git_clone` validates `repo_ref`, selects ref-keyed paths, verifies existing
benchmark cache/work-tree HEAD values, and calls `_populate_ref_cache` on a
miss. Live mode calls `_refresh_live_cache` whenever its cache exists. After a
successful refresh/population, remove the old work tree with
`shutil.rmtree(work, ignore_errors=True)` and call `_clone_local_repo_async`.
This guarantees the work tree reflects the refreshed cache while keeping the
sibling venv path reusable. Emit `worktree_hit`, `object_hit`, `remote_fetch`,
`rebuild`, or `ref_mismatch` through `_repo_cache_event` at the corresponding
branches. All partial caches are removed before re-raising.

Move venv setup out of the `if not state.repo_path` block in `execute_fix` so a repository prepared before LOCATE still receives the existing cached-venv/install behavior before patch execution.

- [ ] **Step 4: Run repository and execution tests and verify GREEN**

Run: `python -m pytest tests/test_new_agent.py tests/test_patch_preflight.py -q`

Expected: exact-ref, live refresh, corruption healing, redaction, and execution tests pass.

- [ ] **Step 5: Commit repository preparation**

```bash
git add src/nodes/execute.py tests/test_new_agent.py tests/test_patch_preflight.py
git commit -m "feat(repo): cache exact benchmark commits"
```

---

### Task 4: Historical Local Code Locator

**Files:**
- Create: `src/local_search.py`
- Create: `tests/test_local_search.py`
- Modify: `src/nodes/locate.py:1-398`
- Modify: `tests/test_locate_bm25.py`

**Interfaces:**
- Produces: `search_local_checkout(repo_path: str, terms: Sequence[str], max_files: int = 4000, max_file_bytes: int = 256_000) -> list[dict[str, str]]`.
- Consumes: `AgentState.repo_path` and existing BM25 functions.

- [ ] **Step 1: Write failing local-search tests**

```python
# tests/test_local_search.py
import subprocess

from src.local_search import search_local_checkout


def test_search_local_checkout_reads_only_tracked_source_files(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "auth.py").write_text("def login():\n    raise ValueError('login')\n")
    (tmp_path / "README.md").write_text("login ValueError")
    (tmp_path / "secret.py").write_text("login ValueError")
    subprocess.run(["git", "-C", str(tmp_path), "add", "src/auth.py", "README.md"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "base"], check=True, capture_output=True)
    results = search_local_checkout(str(tmp_path), ["login", "ValueError"])
    assert [item["path"] for item in results] == ["src/auth.py"]
    assert "raise ValueError" in results[0]["content"]
```

Add this async locator test:

```python
async def test_locate_code_uses_historical_checkout_without_github(monkeypatch, tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "target.py").write_text(
        "def login():\n    raise ValueError('login crash')\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "src/target.py"], check=True)
    async def no_api(*args, **kwargs):
        raise AssertionError("GitHub API must not be used for local benchmark locate")
    monkeypatch.setattr(locate_node, "search_code", no_api)
    monkeypatch.setattr(locate_node, "read_file", no_api)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        owner="acme", repo="widget", repo_path=str(tmp_path), repo_ref="a" * 40,
        issue_title="login crash ValueError", issue_body="login raises ValueError",
        current_phase=new_agent.Phase.LOCATE,
    )
    result = await locate_node.locate_code(state)
    assert result.current_phase == new_agent.Phase.PLAN
    assert result.relevant_files[0].path == "src/target.py"
    assert any(call.tool_name == "search_local_checkout" for call in result.tool_calls)
```

- [ ] **Step 2: Run local-search tests and verify RED**

Run: `python -m pytest tests/test_local_search.py tests/test_locate_bm25.py -q`

Expected: `ModuleNotFoundError: No module named 'src.local_search'`.

- [ ] **Step 3: Implement bounded tracked-file search and local LOCATE branch**

```python
# src/local_search.py
from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

_DOC_SUFFIXES = (".md", ".rst", ".txt", ".rdoc", ".adoc")


def search_local_checkout(repo_path: str, terms: Sequence[str], max_files: int = 4000, max_file_bytes: int = 256_000) -> list[dict[str, str]]:
    root = Path(repo_path).resolve()
    result = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True)
    candidates: list[tuple[int, dict[str, str]]] = []
    lowered_terms = [term.lower() for term in terms if term]
    for raw in result.stdout.split(b"\0")[:max_files]:
        if not raw:
            continue
        path = raw.decode("utf-8", errors="replace")
        lowered_path = path.lower()
        if lowered_path.endswith(_DOC_SUFFIXES) or lowered_path.startswith("docs/") or "/docs/" in lowered_path:
            continue
        file_path = (root / path).resolve()
        if root not in file_path.parents or not file_path.is_file() or file_path.stat().st_size > max_file_bytes:
            continue
        try:
            content = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        haystack = f"{lowered_path}\n{content.lower()}"
        score = sum(haystack.count(term) for term in lowered_terms)
        if score:
            candidates.append((score, {"path": path, "content": content, "sha": ""}))
    return [item for _, item in sorted(candidates, key=lambda pair: (-pair[0], pair[1]["path"]))[:20]]
```

Add this local branch helper to `src/nodes/locate.py`:

```python
async def _locate_local_checkout(state: AgentState) -> AgentState:
    terms = _issue_search_terms(state.issue_title, state.issue_body)
    rows = await asyncio.to_thread(search_local_checkout, state.repo_path, terms)
    _record_tool(
        state,
        "search_local_checkout",
        {"repo_path": state.repo_path, "terms": terms},
        {"count": len(rows)},
    )
    files = [
        FileInfo(
            path=row["path"], content=row["content"], sha=row.get("sha", ""),
            relevance_score=1.0, reason="matched historical local checkout",
        )
        for row in rows
    ]
    if files:
        files = bm25_rerank(f"{state.issue_title}\n{state.issue_body}", files)
    state.relevant_files = files[:6]
    state.token_usage += _estimate_tokens(
        state.issue_title,
        state.issue_body,
        "\n".join(f"{item.path}\n{item.content[:2000]}" for item in files[:6]),
    )
    state.current_phase = Phase.PLAN if files else Phase.FAILURE
    if not files:
        state.failure_reason = "No relevant files could be located in local checkout."
    return state
```

At the start of `locate_code`, after the token-budget guard, use:

```python
if state.repo_path:
    return await _locate_local_checkout(state)
```

This returns before any `search_code` or `read_file` call.

- [ ] **Step 4: Run local locator tests and verify GREEN**

Run: `python -m pytest tests/test_local_search.py tests/test_locate_bm25.py tests/test_locate_fallback.py -q`

Expected: local and API locator tests pass.

- [ ] **Step 5: Commit local historical location**

```bash
git add src/local_search.py src/nodes/locate.py tests/test_local_search.py tests/test_locate_bm25.py
git commit -m "feat(locate): search exact local benchmark checkout"
```

---

### Task 5: SWE-bench Harness Source and Prediction Export

**Files:**
- Modify: `src/new_agent.py:270-294,316-413`
- Modify: `src/nodes/execute.py`
- Modify: `eval/agent_v2_harness.py`
- Modify: `tests/test_agent_v2_eval.py`
- Modify: `tests/test_new_agent.py`

**Interfaces:**
- Consumes: `load_verified_samples`, `build_agent_seed`, `write_predictions`, and commit-aware `git_clone`.
- Produces: eval results with `instance_id`, `base_commit`, `model_patch`, and failure class.
- Produces CLI: `--dataset custom|swe-bench-verified`, `--dataset-seed`, `--predictions-file`.

- [ ] **Step 1: Write failing harness and patch-export tests**

```python
async def test_swe_bench_eval_prepares_exact_checkout_and_passes_safe_seed(monkeypatch):
    sample = swe_bench.normalize_verified_row(swe_bench_row())
    captured = {}
    async def fake_clone(state):
        captured["ref"] = state.repo_ref
        return "/tmp/exact-work"
    async def fake_agent(issue_url, **kwargs):
        captured["seed"] = kwargs["seed"]
        return {"success": False, "final_phase": "FAILED", "trace_id": "trace", "model_patch": "diff --git"}
    monkeypatch.setattr(agent_v2_harness, "git_clone", fake_clone)
    monkeypatch.setattr(agent_v2_harness, "agent_v2", fake_agent)
    monkeypatch.setattr(agent_v2_harness, "replay_run", lambda run_id: {})
    result = await agent_v2_harness.evaluate_agent_v2_sample(sample, 0)
    assert captured["ref"] == "a" * 40
    assert captured["seed"]["repo_path"] == "/tmp/exact-work"
    assert "evaluation" not in captured["seed"]
    assert result["instance_id"] == "acme__widget-8"
    assert result["model_patch"] == "diff --git"
```

Add a `agent_payload_from_state` test using a temporary Git repository with one modified tracked file and assert `payload["model_patch"]` equals `git diff --binary` output.

- [ ] **Step 2: Run harness tests and verify RED**

Run: `python -m pytest tests/test_agent_v2_eval.py tests/test_new_agent.py -k 'swe_bench or model_patch' -q`

Expected: harness does not prepare `base_commit` and payload lacks `model_patch`.

- [ ] **Step 3: Implement exact checkout preparation and prediction output**

Add a credential-safe diff helper:

```python
def git_diff(repo_path: str) -> str:
    if not repo_path:
        return ""
    result = subprocess.run(["git", "-C", repo_path, "diff", "--binary", "--no-ext-diff"], capture_output=True, text=True, timeout=30)
    return result.stdout if result.returncode == 0 else ""
```

Include `model_patch: git_diff(state.repo_path)` in `agent_payload_from_state`.

Add an exact preparation helper to `eval/agent_v2_harness.py`:

```python
async def _build_eval_seed(
    sample: dict[str, Any], seed_gold_files: bool
) -> dict[str, Any] | None:
    if sample.get("source") != "swe-bench-verified":
        return _build_gold_seed(sample) if seed_gold_files else None
    issue = sample["issue"]
    repo = sample["repo"]
    prep = AgentState(
        issue_url=issue["url"],
        owner=repo["owner"],
        repo=repo["name"],
        issue_number=issue["number"],
        repo_ref=sample["base_commit"],
        skip_commit=True,
    )
    repo_path = await git_clone(prep)
    return build_agent_seed(sample, repo_path)
```

Change `evaluate_agent_v2_sample` to await `_build_eval_seed`, pass only that
seed to `agent_v2`, and add these result fields:

```python
"instance_id": sample.get("instance_id", sample["id"]),
"base_commit": sample.get("base_commit", ""),
"model_patch": payload.get("model_patch", ""),
```

Build the result dictionary first, then classify it with the existing taxonomy:

```python
result["failure_class"] = classify_sample(result)["decisive"]
return result
```

Change `run_agent_v2_eval` to accept `dataset`, `dataset_seed`, and
`predictions_path`. It calls `load_verified_samples` for the SWE-bench source,
keeps `load_samples` for custom JSONL, and calls `write_predictions` after
writing normal results when a predictions path was supplied.

Add CLI parsing:

```python
parser.add_argument("--dataset", choices=("custom", "swe-bench-verified"), default="custom")
parser.add_argument("--dataset-seed", type=int, default=17)
parser.add_argument("--predictions-file", type=str, default=None)
```

- [ ] **Step 4: Run harness tests and verify GREEN**

Run: `python -m pytest tests/test_swe_bench.py tests/test_agent_v2_eval.py tests/test_new_agent.py -q`

Expected: custom and SWE-bench paths pass, prediction JSONL contains no gold data.

- [ ] **Step 5: Commit the SWE-bench harness path**

```bash
git add src/new_agent.py src/nodes/execute.py eval/agent_v2_harness.py tests/test_agent_v2_eval.py tests/test_new_agent.py
git commit -m "feat(eval): run agent on SWE-bench base commits"
```

---

### Task 6: Bounded Stale GitHub API Cache

**Files:**
- Modify: `src/cache.py:1-105`
- Modify: `src/http_client.py:82-105`
- Modify: `src/tools.py:1-64`
- Modify: `tests/test_cache.py`
- Modify: `tests/test_tools.py`

**Interfaces:**
- Produces: `cached(func=None, *, stale_if_error: Callable[[BaseException], bool] | None = None)` preserving `@cached` compatibility.
- Produces: `is_retryable_github_error(exc: BaseException) -> bool`.
- Adds environment variable: `REPOPILOT_CACHE_STALE_TTL`, default `86400`.

- [ ] **Step 1: Write failing stale-cache tests**

```python
@pytest.mark.asyncio
async def test_cached_returns_stale_value_on_retryable_refresh_error(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("REPOPILOT_HOME", str(tmp_path))
    monkeypatch.setattr(cache, "CACHE_TTL", 10)
    monkeypatch.setattr(cache, "CACHE_STALE_TTL", 100)
    now = {"value": 50.0}
    monkeypatch.setattr(cache.time, "time", lambda: now["value"])
    calls = 0
    @cache.cached(stale_if_error=lambda exc: isinstance(exc, ConnectionError))
    async def fetch():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ConnectionError("offline")
        return {"value": "cached"}
    assert await fetch() == {"value": "cached"}
    now["value"] = 65.0
    assert await fetch() == {"value": "cached"}
    assert "stale_fallback" in caplog.text


@pytest.mark.asyncio
async def test_cached_does_not_hide_non_retryable_error(monkeypatch, tmp_path):
    monkeypatch.setenv("REPOPILOT_HOME", str(tmp_path))
    monkeypatch.setattr(cache, "CACHE_TTL", 10)
    monkeypatch.setattr(cache, "CACHE_STALE_TTL", 100)
    now = {"value": 0.0}
    monkeypatch.setattr(cache.time, "time", lambda: now["value"])
    @cache.cached(stale_if_error=lambda exc: isinstance(exc, ConnectionError))
    async def fetch():
        if now["value"]:
            raise ValueError("bad schema")
        return {"value": "cached"}
    await fetch()
    now["value"] = 20.0
    with pytest.raises(ValueError, match="bad schema"):
        await fetch()
```

Add these expiry and predicate tests:

```python
@pytest.mark.asyncio
async def test_cached_removes_value_beyond_stale_window(monkeypatch, tmp_path):
    monkeypatch.setenv("REPOPILOT_HOME", str(tmp_path))
    monkeypatch.setattr(cache, "CACHE_TTL", 10)
    monkeypatch.setattr(cache, "CACHE_STALE_TTL", 30)
    now = {"value": 0.0}
    monkeypatch.setattr(cache.time, "time", lambda: now["value"])
    calls = 0
    @cache.cached(stale_if_error=lambda exc: True)
    async def fetch():
        nonlocal calls
        calls += 1
        return {"call": calls}
    assert await fetch() == {"call": 1}
    now["value"] = 31.0
    assert await fetch() == {"call": 2}


def test_github_retry_predicate_distinguishes_503_and_404():
    request = httpx.Request("GET", "https://api.github.com/repos/acme/widget")
    transient = httpx.HTTPStatusError(
        "503", request=request, response=httpx.Response(503, request=request)
    )
    missing = httpx.HTTPStatusError(
        "404", request=request, response=httpx.Response(404, request=request)
    )
    assert http_client.is_retryable_github_error(transient) is True
    assert http_client.is_retryable_github_error(missing) is False
```

Extend the existing disabled-cache test to set `REPOPILOT_DISABLE_CACHE=1`,
call the same decorated function twice, and assert its call count is two. Use
`caplog` to assert cache log messages contain only the event, function name,
age, and a short hexadecimal key, and do not contain a sentinel token embedded
in the cached value.

- [ ] **Step 2: Run cache tests and verify RED**

Run: `python -m pytest tests/test_cache.py tests/test_tools.py tests/test_http_client.py -q`

Expected: `cached()` rejects `stale_if_error`, and expired values are deleted before refresh.

- [ ] **Step 3: Implement fresh/stale states and GitHub predicate**

Use a small cache entry model:

```python
@dataclass(frozen=True)
class CacheEntry:
    value: object
    age_seconds: float
    state: Literal["fresh", "stale"]
```

`_load` returns `None`, fresh, or stale; it removes the file only when age exceeds `CACHE_STALE_TTL`. The wrapper logs event/function/key-prefix/age, tries refresh for stale entries, and returns stale only when `stale_if_error(exc)` is true. `_save` failures log `save_failed` but preserve the fresh function result.

Rename `_is_retryable_github` to public `is_retryable_github_error` and keep `_is_retryable_github = is_retryable_github_error` as a compatibility alias for existing tests. Decorate GitHub tool functions as:

```python
@cached(stale_if_error=is_retryable_github_error)
async def read_issue(...):
    ...
```

Apply the same predicate to `search_code` and `read_file`; their existing argument-derived keys remain unchanged.

- [ ] **Step 4: Run cache and HTTP tests and verify GREEN**

Run: `python -m pytest tests/test_cache.py tests/test_tools.py tests/test_http_client.py -q`

Expected: fresh hit, stale refresh/fallback, expiry, non-retryable propagation, retry behavior, and diagnostics pass.

- [ ] **Step 5: Commit resilient API cache**

```bash
git add src/cache.py src/http_client.py src/tools.py tests/test_cache.py tests/test_tools.py tests/test_http_client.py
git commit -m "feat(cache): serve bounded stale GitHub responses"
```

---

### Task 7: Documentation, Full Verification, and Ten-Sample Run

**Files:**
- Modify: `README.md`
- Modify: `docs/CODEX_CONTEXT.md`
- Generated locally: `${REPOPILOT_HOME}/eval/datasets/swe-bench-verified.jsonl`
- Generated locally: `eval/swe_bench_predictions.jsonl`
- Generated locally: `eval/eval_results.json`

**Interfaces:**
- Consumes: complete SWE-bench and stale-cache implementation.
- Produces: reproducible operator commands and verified eval artifacts.

- [ ] **Step 1: Add documentation contract tests before changing docs**

Extend `tests/test_documentation_contract.py` to assert README contains:

```python
assert "pip install -e '.[eval]'" in readme
assert "--dataset swe-bench-verified" in readme
assert "--predictions-file" in readme
assert "REPOPILOT_CACHE_STALE_TTL" in readme
assert "base_commit" in readme
```

- [ ] **Step 2: Run documentation test and verify RED**

Run: `python -m pytest tests/test_documentation_contract.py -q`

Expected: assertions fail because the new commands are absent.

- [ ] **Step 3: Document installation, inference, official scoring, and cache behavior**

README commands must include:

```bash
pip install -e '.[eval]'
python -u eval/agent_v2_harness.py \
  --dataset swe-bench-verified \
  --dataset-seed 17 \
  --samples 10 \
  --max-retries 3 \
  --token-budget 50000 \
  --predictions-file eval/swe_bench_predictions.jsonl

python -m swebench.harness.run_evaluation \
  --dataset_name SWE-bench/SWE-bench_Verified \
  --predictions_path eval/swe_bench_predictions.jsonl \
  --max_workers 1 \
  --run_id repopilot-gemini-10
```

Explain that inference uses exact `base_commit`, gold/test patches never enter prompts, final resolved rate comes only from the official harness, fresh API cache defaults to 600 seconds, and stale fallback defaults to 86,400 seconds.

- [ ] **Step 4: Run documentation and focused suites**

Run:

```bash
python -m pytest tests/test_documentation_contract.py tests/test_swe_bench.py tests/test_agent_v2_eval.py tests/test_local_search.py tests/test_cache.py tests/test_tools.py -q
```

Expected: all focused tests pass with no unexpected warnings.

- [ ] **Step 5: Run full static and package verification**

Run:

```bash
python -m pytest -q
python -m ruff check src tests eval/agent_v2_harness.py eval/swe_bench.py
python -m build
```

Expected: full tests, Ruff, sdist, and wheel build pass.

- [ ] **Step 6: Commit documentation**

```bash
git add README.md docs/CODEX_CONTEXT.md tests/test_documentation_contract.py
git commit -m "docs: explain reproducible SWE-bench evaluation"
```

- [ ] **Step 7: Download/select ten samples and run Gemini inference**

Run the documented inference command with `LLM_MODEL=gemini-3.5-flash:stable`, the configured LinoAPI base URL, and credentials passed only through environment variables.

Expected: ten result records identify SWE-bench instance IDs, exact base commits, Gemini model, RepoPilot commit SHA, and infrastructure/model failure classes; prediction JSONL contains no credentials or evaluator-only fields.

- [ ] **Step 8: Run official SWE-bench scoring and report evidence**

Run the documented official harness command. Report resolved, unresolved, empty-patch, and infrastructure-failure counts separately. If Docker/image preparation is unavailable, preserve predictions and report the scoring step as infrastructure-blocked without treating samples as model failures.

- [ ] **Step 9: Final branch verification**

Run `git status --short --branch`, `git log -8 --oneline`, and inspect all generated tracked diffs for credentials. The branch must be clean except for intentionally untracked local evaluation artifacts ignored by Git.
