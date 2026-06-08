# RepoPilot

> **AI-powered GitHub issue → fix PR, with a self-reflective agent loop.**
>
> Not a general-purpose copilot. RepoPilot does one thing: reads a GitHub Issue, searches the codebase, generates a fix, runs the tests, and opens a PR. When the fix fails, it reflects on *why* and tries again.

<p align="center">
  <img src="https://github.com/FMorgan-111/repopilot/actions/workflows/ci.yml/badge.svg" alt="CI">
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

---

## Why RepoPilot

RepoPilot is built for **professional developers** who maintain real projects with test suites, CI pipelines, and a PR workflow. It is not a black-box auto-fixer for vibe coders — it shows its work, expects review, and admits when it cannot fix something.

| | RepoPilot | Sweep | Devin | Claude Code |
|---|:---:|:---:|:---:|:---:|
| **Observable reasoning chain** | ✅ LangGraph trace | ❌ Black-box | ❌ Agent log only | ❌ Opaque loop |
| **Open source** | ✅ MIT | ✅ MIT | ❌ Proprietary | ❌ Proprietary |
| **Model freedom** | ✅ Your own API key | ✅ | ❌ | ❌ |
| **Self-reflective retry** | ✅ REFLECT node | ❌ | ❌ | ❌ |
| **Local test execution** | ✅ clone + pytest | ❌ | ✅ | ✅ |
| **Output** | Draft PR (you review) | Auto-merge | Full PR | Patch file |

**Core differentiators:**

- **Observable** — LangGraph explicit state machine with conditional edges. Every phase transition, every tool call, every reflection is logged to JSONL via the built-in Tracer. You can debug *why* the agent chose this file, *why* it planned that patch, and *what* went wrong.
- **Self-reflective** — When a fix fails tests, the agent enters `REFLECT` mode: the LLM analyzes the error log, identifies the root cause, and feeds that analysis back into the planner. It remembers what it tried and avoids repeating the same mistake.
- **Model-free** — Plug in any OpenAI-compatible API: DeepSeek, Ollama, LiteLLM proxy, or OpenAI itself. You control the model, the cost, and the data.
- **Open source by design** — The competition (Devin, Claude Code) is proprietary. RepoPilot is MIT-licensed so you can self-host, audit, and extend.

---

## Quick Start

```bash
pip install repopilot
```

Set your tokens:

```bash
export GITHUB_TOKEN=ghp_...
export LLM_API_KEY=sk-...         # or DEEPSEEK_API_KEY
export LLM_MODEL=deepseek-v4-pro  # optional, defaults to deepseek-v4-pro
```

Run it:

```bash
repopilot https://github.com/org/repo/issues/42
```

For analysis only (no PR):

```bash
repopilot https://github.com/org/repo/issues/42 --dry-run
```

Machine-readable output:

```bash
repopilot https://github.com/org/repo/issues/42 --json
```

---

## How It Works

RepoPilot implements a **six-phase state machine** with a self-reflective retry loop, built on LangGraph (with an automatic fallback runner when LangGraph is not installed).

```
                    ┌──────────────┐
                    │ UNDERSTAND   │  ← read issue, classify type/severity
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   LOCATE     │  ← GitHub code search → rank by relevance
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │    PLAN      │  ← LLM generates unified diff + test command
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   EXECUTE    │  ← git clone → apply patch → run pytest
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐     ┌──────────────┐
                    │   VERIFY     │────▶│   COMMIT     │──▶ DONE
                    └──────┬───────┘     └──────────────┘
                           │ fail
                    ┌──────▼───────┐
                    │   REFLECT    │  ← LLM analyzes failure root cause
                    └──────┬───────┘
                           │
                           └───────────▶ PLAN  (retry with reflection context)
```

**Each phase is a discrete node in a LangGraph `StateGraph`:**

| Phase | Node | What it does |
|-------|------|-------------|
| `UNDERSTAND` | `understand_issue` | Fetches the GitHub Issue via API, classifies it (bug/feature/security), extracts labels and severity |
| `LOCATE` | `locate_code` | Runs GitHub code search for keywords extracted from the issue body, ranks candidate files by relevance score, reads top-6 file contents |
| `PLAN` | `plan_fix` | Prompts the LLM with issue context + relevant file contents + any previous failure reflections → generates a unified diff and test command |
| `EXECUTE` | `execute_fix` | Clones the target repo to a temp directory, applies the patch via `git apply`, runs the project's test suite (default: `pytest`) |
| `VERIFY` | `verify_fix` | Inspects test output. Routes to `COMMIT` on success, to `REFLECT` on failure, to `FAILED` if max retries exhausted or same failure seen twice |
| `REFLECT` | `reflect_on_failure` | LLM analyzes *why* the fix failed — specific error, wrong assumption, missing edge case. Feeds analysis back into `PLAN` |
| `COMMIT` | `commit_fix` | Pushes changed files via GitHub Contents API, opens a Draft PR with the fix plan and test results |
| `FAILURE` | `handle_failure` | Posts an issue comment summarizing what was found (relevant files, attempted patches, failure reason) — partial progress, still useful |

**Guardrails built in:**

- **Token budget** — configurable per-run; agent stops gracefully when exceeded rather than burning credits
- **Duplicate failure detection** — if the same patch produces the same error twice, the agent aborts instead of looping
- **Max retry cap** — defaults to 3; reachable, not theoretical
- **Pydantic-validated structured output** — LLM responses are parsed into typed models with schema validation and an automatic retry-on-`ValidationError` fallback

---

## Architecture Deep Dive

### Agent State (`AgentState`)

The entire run is modeled as a single `Pydantic` model — typed, serializable, debuggable:

```python
class AgentState(BaseModel):
    issue_url: str
    issue_title: str
    issue_body: str
    current_phase: Phase                # enum-driven routing
    relevant_files: list[FileInfo]      # ranked with relevance scores
    fix_attempts: list[FixAttempt]      # patch + test result per attempt
    conversation_history: list[ConversationTurn]
    token_usage: int
    reflection_notes: str               # populated by REFLECT node
    # ... tool_calls, owner, repo, branch, pr_url, etc.
```

### Graph Engine (dual-backend)

RepoPilot uses LangGraph when installed, but ships with a `FallbackStateGraph` + `FallbackCompiledGraph` that satisfies the same `graph.ainvoke(state)` contract using a simple while-loop with `route_from_state`. This means the agent runs identically in environments where LangGraph cannot be installed (or where you prefer zero extra dependencies).

```python
graph = build_agent_graph()          # returns LangGraph or Fallback
final_state = await run_graph(graph, state)  # same interface
```

### Pydantic-validated LLM outputs

Instead of hoping the LLM returns well-formed JSON, every structured call goes through `validate_or_retry()`:

1. Parse raw LLM response as JSON
2. Validate against the target Pydantic model
3. On `ValidationError`: retry once with the schema error injected into the prompt
4. On second failure: fall back to the raw dict with a warning — the agent continues rather than crashing

```python
# src/llm.py
async def validate_or_retry(system: str, user: str, schema: type[BaseModel]) -> dict
```

### Tracer (observability)

Every run generates a trace ID. All phase transitions, tool calls, and errors are logged as structured JSONL:

```python
tracer.log("phase_enter", {"from": "PLAN", "to": "EXECUTE"})
tracer.log("tool_call", {"name": "search_code", "args": {...}}, result)
tracer.log("agent_v2_done", {...}, error="...")
```

### Token Budget Management

Token usage is estimated (chars ÷ 4) after every LLM call. Every node checks `_is_budget_exceeded(state)` before making additional calls. When exceeded, the agent routes to `FAILURE` and publishes a partial-progress comment on the issue.

---

## Example

Here is a trace of RepoPilot fixing a real-world issue:

```bash
$ repopilot https://github.com/cookiecutter/cookiecutter/issues/1973

🔍 RepoPilot analyzing https://github.com/cookiecutter/cookiecutter/issues/1973...

Phase: DONE
Success: True
PR: https://github.com/cookiecutter/cookiecutter/pull/2100
Turns: 7
Token used: 8420

Relevant files (3):
  cookiecutter/generate.py (score: 0.72)
  cookiecutter/config.py (score: 0.48)
  tests/test_generate.py (score: 0.35)

Fix attempts: 1
  ✅ Attempt 1: cookiecutter/generate.py
```

The agent identified that CLI boolean overrides were passed as strings (not booleans) when configuring cookiecutter templates. It generated a targeted `isinstance` conversion, applied the patch, ran `pytest`, and opened a Draft PR — all in one command.

> See `examples/candidate_issues.md` for more demo-ready issue URLs from FastAPI, Textual, and cookiecutter.

---

## Technical Choices

| Decision | Rationale |
|----------|----------|
| **LangGraph explicit state machine** | Every phase, edge, and transition is visible in code — no black-box `AgentExecutor`. Conditional routing is a pure function (`route_from_state`), not hidden in a prompt. |
| **Pydantic `AgentState`** | Single typed model for the entire run. Serialisable for debugging. Schema-driven routing (no stringly-typed phase names in control flow). |
| **Fallback graph engine** | Agent runs without LangGraph. The `FallbackCompiledGraph` is 30 lines of pure Python that honors the same `graph.ainvoke()` contract. Makes offline/CI testing trivial. |
| **Structured output + validation retry** | LLMs are stochastic. `validate_or_retry` catches malformed JSON, schema mismatches, and missing keys — retries once with the exact error injected — so the agent doesn't crash on a bad parse. |
| **Token budget with guard checks** | Budget is checked before every LLM call. Exceeding it routes to a graceful failure path (issue comment with partial findings), not an HTTP 500. |
| **Duplicate failure detection** | If `verify_fix` sees the exact same patch + exact same error log twice, it aborts rather than looping. This is defense-in-depth beyond the max-retry cap. |
| **GitHub-native workflow** | Uses GitHub Contents API for file push (no local git push), opens Draft PRs (not auto-merge), and posts analysis comments on failure. |
| **Production HTTP layer** | Exponential backoff retry (tenacity) on 429/502/503/504 and network errors. Token-bucket rate limiter respects GitHub API limits (4500 req/h authenticated). Structured logging via `logging` module. |
| **Per-repo memory (Layer 2)** | SQLite-backed file index + issue history. After fixing 5 bugs in a repo, the agent searches historically-modified files first — 10x faster than cold GitHub API search. Atomic SQL writes, fire-and-forget, WAL mode. |
| **Modular codebase** | 999-line monolith split into `src/nodes/` (one file per agent phase). Each node is ~50-180 lines. `src/state.py` holds all Pydantic models. `src/graph.py` is the LangGraph wiring. |

---

## Documentation

| Document | What it covers |
|----------|---------------|
| `docs/PRODUCT_POSITIONING.md` | User personas, market fit, competitive differentiation, why RepoPilot targets professional developers not vibe coders |
| `docs/MEMORY_DESIGN_V2.md` | Four-layer memory architecture: working memory → per-repo SQLite → cross-repo strategy learning → meta statistics. Includes concurrency analysis and three deployment scales |
| `docs/COMPETITIVE_ANALYSIS.md` | Sweep vs Devin vs Claude Code vs Cursor vs Aider — feature matrix and positioning |
| `docs/RESUME_STRATEGY.md` | How to present RepoPilot in a technical resume/interview context |

---

## Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run the test suite (60+ tests, <2s)
pytest tests/ -q
```

Tests cover:

- Full state machine transitions (UNDERSTAND → LOCATE → PLAN → EXECUTE → VERIFY → COMMIT → DONE)
- REFLECT → PLAN retry loop
- Duplicate failure detection
- Token budget exhaustion
- Pydantic validation retry logic
- GitHub API tool mocking (issue read, code search, file read)
- FastAPI endpoint routing
- Tracer JSONL logging

---

## Development

```bash
git clone https://github.com/FMorgan-111/repopilot.git
cd repopilot
pip install -e .
pytest tests/ -q
```

### Running the FastAPI server

```bash
uvicorn src.main:app --reload
```

Endpoints:

- `GET  /health` — liveness check
- `POST /analyze` — issue classification + file ranking (v1)
- `POST /agent` — legacy agent loop
- `POST /agent/v2` — current state-machine agent (recommended)

---

## License

MIT — see `pyproject.toml`.

---

<p align="center">
  <sub>Built with LangGraph, Pydantic, httpx, and DeepSeek v4-pro. Maintained by <a href="https://github.com/FMorgan-111">FMorgan-111</a>.</sub>
</p>
