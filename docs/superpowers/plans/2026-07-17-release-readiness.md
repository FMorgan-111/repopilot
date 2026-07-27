# RepoPilot Release Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the current feature branch installable, portable, provider-compatible, and honestly measurable before merging.

**Architecture:** Keep the existing agent graph intact. Tighten the package/CI contract, introduce a vector-index backend factory with a persistent NumPy fallback, isolate chat-completion response normalization in the HTTP client, and attach an explicit evaluation mode to every Agent V2 result and aggregate.

**Tech Stack:** Python 3.10–3.12, Pydantic, httpx, SQLite, sqlite-vec, NumPy, pytest, GitHub Actions.

## Global Constraints

- `pyproject.toml` is the installation source of truth.
- Episode memory remains opt-in through `REPOPILOT_ENABLE_EPISODES=1`.
- The default gateway is `https://linoapi.com.cn/v1` and default model is `claude-sonnet-5:stable`.
- Existing `LLM_MODEL`, `OPENAI_BASE_URL`, and API-key environment overrides remain compatible.
- Existing result files without `evaluation_mode` are interpreted as `end_to_end`.
- Do not alter the Agent graph or patch-generation strategy in this plan.

---

### Task 1: Package and CI Contract

**Files:**
- Modify: `pyproject.toml`
- Modify: `requirements.txt`
- Modify: `.github/workflows/ci.yml`
- Modify: `.gitignore`
- Test: `tests/test_packaging_contract.py`

**Interfaces:**
- Produces extras `memory` and `dev` usable as `pip install -e ".[memory,dev]"`.
- CI consumes only package metadata plus the editable source tree.

- [ ] **Step 1: Write the failing metadata tests**

```python
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dev_and_memory_extras_cover_test_and_optional_runtime_dependencies():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    extras = project["optional-dependencies"]
    assert {"fastembed", "numpy", "sqlite-vec"} <= {
        item.split(">=")[0].split("<")[0] for item in extras["memory"]
    }
    assert {"pytest", "pytest-asyncio", "ruff"} <= {
        item.split(">=")[0].split("<")[0] for item in extras["dev"]
    }
```

- [ ] **Step 2: Run the metadata test and confirm it fails because optional dependencies are absent**

Run: `.venv/bin/python -m pytest tests/test_packaging_contract.py -q`

- [ ] **Step 3: Declare extras and update CI**

Add `project.optional-dependencies.memory` for `fastembed`, `numpy`, and
`sqlite-vec`; add `project.optional-dependencies.dev` for pytest,
pytest-asyncio, ruff, and the memory test dependencies. Update CI to install
`.[memory,dev]`, run pytest, and run Ruff on Ubuntu and macOS. Ignore
`*.egg-info/`.

- [ ] **Step 4: Run the package contract test**

Run: `.venv/bin/python -m pytest tests/test_packaging_contract.py -q`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml requirements.txt .github/workflows/ci.yml .gitignore tests/test_packaging_contract.py
git commit -m "build: align package extras and CI"
```

### Task 2: Portable Episode Vector Backend

**Files:**
- Create: `src/memory/numpy_vector_index.py`
- Create: `src/memory/vector_backend.py`
- Modify: `src/memory/error_episode_store.py`
- Modify: `src/memory/sqlite_vec_index.py`
- Modify: `tests/test_error_episodes.py`

**Interfaces:**
- Produces `NumpyVectorIndex(conn: sqlite3.Connection, dim: int)` implementing `VectorIndex`.
- Produces `create_vector_index(conn, dim) -> tuple[VectorIndex, str]`.
- `ErrorEpisodeStore.backend_name: str` exposes `sqlite_vec` or `numpy`.

- [ ] **Step 1: Write failing tests for persistence, cosine ordering, and fallback selection**

```python
def test_numpy_vector_index_cosine_knn():
    conn = sqlite3.connect(":memory:")
    idx = NumpyVectorIndex(conn, dim=3)
    idx.add(1, [1, 0, 0])
    idx.add(2, [0, 1, 0])
    assert [hit.rowid for hit in idx.search([0.9, 0.1, 0], 2)] == [1, 2]


def test_backend_factory_falls_back_when_sqlite_extensions_are_unavailable(monkeypatch):
    monkeypatch.setattr(vector_backend, "SqliteVecIndex", BrokenSqliteVecIndex)
    index, name = vector_backend.create_vector_index(sqlite3.connect(":memory:"), 3)
    assert isinstance(index, NumpyVectorIndex)
    assert name == "numpy"
```

- [ ] **Step 2: Run the new tests and confirm import/behavior failures**

Run: `.venv/bin/python -m pytest tests/test_error_episodes.py -q`

- [ ] **Step 3: Implement the NumPy backend**

Persist vectors as packed float32 blobs in a normal SQLite table. Search loads
the rows, calculates cosine distance with NumPy, sorts by `(distance, rowid)`,
and returns at most `k` `VectorHit` objects. Validate dimensions on add/search.

- [ ] **Step 4: Implement the backend factory and store diagnostic**

The factory catches only backend initialization failures (`ImportError`,
`AttributeError`, `sqlite3.Error`, `RuntimeError`), emits one
`RuntimeWarning`, and returns the NumPy backend. `ErrorEpisodeStore` stores the
returned backend name. An explicitly supplied `index_factory` remains
unchanged for tests and embeddings.

- [ ] **Step 5: Run memory and full focused tests**

Run: `.venv/bin/python -m pytest tests/test_error_episodes.py tests/test_memory.py -q`
Expected: all pass on macOS without SQLite extension loading.

- [ ] **Step 6: Commit**

```bash
git add src/memory tests/test_error_episodes.py tests/test_memory.py
git commit -m "fix: add portable episode vector fallback"
```

### Task 3: LLM Defaults and Response Normalization

**Files:**
- Modify: `src/http_client.py`
- Modify: `tests/test_http_client.py`
- Modify: `tests/conftest.py`

**Interfaces:**
- Produces `LLMResponseError(RuntimeError)` for invalid successful responses.
- Produces `_parse_chat_completion_json(payload: dict) -> dict`.
- Produces streamed results shaped as `{"choices": [{"message": ...,
  "finish_reason": ...}], "usage": ...}`.

- [ ] **Step 1: Retain the passing default-model regression test**

`test_default_llm_model_is_claude_sonnet_5` must assert the new default while
`LLM_MODEL` overrides remain covered by request-body tests.

- [ ] **Step 2: Write failing tests for response variants**

Add tests for SSE usage/finish reason, ordinary JSON response, explicit SSE
error, malformed SSE JSON, empty stream, and tool-call-only response. Each
invalid response must raise `LLMResponseError`; tool-call-only is valid.

- [ ] **Step 3: Run the new response tests and confirm their expected failures**

Run: `.venv/bin/python -m pytest tests/test_http_client.py -q`

- [ ] **Step 4: Implement response normalization**

Inspect `Content-Type`. For JSON, call `await resp.aread()` then parse and
validate the completion. For SSE, accumulate content and tool-call deltas by
choice/tool index, retain the last finish reason and usage object, and reject
bad or empty terminal state. Do not retry `LLMResponseError`.

- [ ] **Step 5: Run HTTP and LLM tests**

Run: `.venv/bin/python -m pytest tests/test_http_client.py tests/test_llm.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/http_client.py tests/test_http_client.py tests/conftest.py README.md CLAUDE.md docs/superpowers/specs/2026-07-17-release-readiness-design.md
git commit -m "fix: normalize provider chat completion responses"
```

### Task 4: Honest Evaluation Modes

**Files:**
- Modify: `eval/agent_v2_harness.py`
- Modify: `eval/report.py`
- Modify: `tests/test_agent_v2_eval.py`
- Modify: `tests/test_eval_report.py`

**Interfaces:**
- Each Agent V2 result contains `evaluation_mode: Literal["end_to_end", "oracle_files"]`.
- `compute_metrics()` produces `agent_v2.by_evaluation_mode` with samples, successes, and success rate.

- [ ] **Step 1: Write failing harness tests**

Assert `seed_gold_files=False` returns `evaluation_mode="end_to_end"` and
`seed_gold_files=True` returns `evaluation_mode="oracle_files"` even when gold
seeding cannot fetch a file.

- [ ] **Step 2: Write failing report tests**

Use mixed seeded/unseeded results and assert separate sample counts and success
rates in metrics and Markdown. Assert legacy Agent V2 results without the field
are grouped as `end_to_end`.

- [ ] **Step 3: Run the focused eval tests and confirm failures**

Run: `.venv/bin/python -m pytest tests/test_agent_v2_eval.py tests/test_eval_report.py -q`

- [ ] **Step 4: Add result labels and grouped metrics**

Set the result field from the CLI intent, not whether seed construction
succeeded. Add grouping in `compute_metrics` and an Evaluation Mode column plus
summary rows in Markdown.

- [ ] **Step 5: Run focused eval tests**

Run: `.venv/bin/python -m pytest tests/test_agent_v2_eval.py tests/test_eval_report.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add eval/agent_v2_harness.py eval/report.py tests/test_agent_v2_eval.py tests/test_eval_report.py
git commit -m "feat: separate oracle and end-to-end evaluation"
```

### Task 5: Documentation Consistency

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `docs/PROGRESS_REPORT.md`
- Modify: `docs/2026-07-05-technical-adjustments.md`
- Test: `tests/test_documentation_contract.py`

**Interfaces:**
- Contributor commands match `pyproject.toml` and CI.
- Historical reports explicitly identify their date, commit context, and oracle/end-to-end mode.

- [ ] **Step 1: Write failing documentation contract tests**

Assert README contains `.[memory]`, `.[dev]`, `REPOPILOT_ENABLE_EPISODES`, the
selected default model, and no claim that `pip install -e ".[dev]"` uses an
undefined extra. Assert historical documents no longer claim current test
counts or link to tracked generated outputs.

- [ ] **Step 2: Run the documentation tests and confirm failures**

Run: `.venv/bin/python -m pytest tests/test_documentation_contract.py -q`

- [ ] **Step 3: Update documentation**

Document base install, memory extra, development install, first-model download,
macOS fallback, LinoAPI environment configuration, and the distinction between
oracle-file and end-to-end evaluation. Mark progress reports as historical
snapshots and remove stale tracked-artifact claims.

- [ ] **Step 4: Run documentation tests**

Run: `.venv/bin/python -m pytest tests/test_documentation_contract.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add README.md CLAUDE.md docs tests/test_documentation_contract.py
git commit -m "docs: align installation and evaluation guidance"
```

### Task 6: Full Verification

**Files:**
- Verify all modified files.

**Interfaces:**
- No new runtime behavior; this task proves the plan's acceptance criteria.

- [ ] **Step 1: Install from package metadata**

Run: `.venv/bin/python -m pip install -e ".[memory,dev]"`
Expected: exit 0.

- [ ] **Step 2: Run the full test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: zero failures.

- [ ] **Step 3: Run lint and diff validation**

Run: `.venv/bin/python -m ruff check src tests eval`
Run: `git diff --check HEAD~5..HEAD`
Expected: both exit 0.

- [ ] **Step 4: Verify a clean package build**

Run: `.venv/bin/python -m pip install build && .venv/bin/python -m build`
Expected: wheel and sdist build successfully without importing optional memory dependencies.

- [ ] **Step 5: Review repository state**

Run: `git status --short --branch && git log --oneline -8`
Expected: only intentionally ignored local artifacts remain; implementation commits are visible.
