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

Cross-repository semantic episode memory is optional:

```bash
pip install "repopilot[memory]"
export REPOPILOT_ENABLE_EPISODES=1
```

The first enabled recall downloads the `BAAI/bge-small-en-v1.5` embedding
model. RepoPilot prefers `sqlite-vec`; on Python builds that cannot load SQLite
extensions (including some macOS builds), it emits a warning and uses the
persistent NumPy cosine-search backend instead.

Inject tokens through the process environment or your deployment's runtime
secret store. Do not put real credentials in `.env.example`, source files,
evaluation results, or shell history:

```bash
export GITHUB_TOKEN=ghp_...
export LLM_API_KEY=sk-...
export LLM_BASE_URL=https://linoapi.com.cn/v1
export LLM_MODEL=gemini-3.5-flash:stable  # optional; this is the default
```

`LLM_BASE_URL` is paired only with `LLM_API_KEY` and must use HTTPS, except
for exact `localhost`, `127.0.0.1`, or `::1` development endpoints. The legacy
`OPENAI_BASE_URL` name remains a deprecated alias only when `LLM_BASE_URL` is
unset (or has the same value); conflicting values are rejected.

Success-first escalation is off by default. It activates only when both the
explicit flag and a separately injected escalation key are present:

```bash
export REPOPILOT_ESCALATION_ENABLED=1
export LLM_ESCALATION_MODEL=claude-opus-4-8:stable
export LLM_ESCALATION_BASE_URL=https://linoapi.com.cn/v1
# Inject LLM_ESCALATION_API_KEY with the runtime secret store.
```

`REPOPILOT_ESCALATION_AFTER_NO_PROGRESS=2` is the default bounded trigger.
Without the flag or key, evaluation remains Gemini-only and does not incur an
escalation call.

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
pip install -e ".[memory,dev]"

# Run the test suite
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
pip install -e ".[memory,dev]"
pytest tests/ -q
```

### Evaluation modes

An ordinary Agent V2 evaluation is labeled `end_to_end`: RepoPilot must locate
the relevant files and produce a fix. `--seed-gold-files` is labeled
`oracle_files`: it supplies dataset-known changed files and measures planning
and patching in isolation. Oracle-file results must not be reported as
end-to-end success rates.

### SWE-bench Verified

Install the optional evaluation dependencies and run a reproducible ten-sample
inference batch:

```bash
pip install -e '.[eval]'
python -u eval/agent_v2_harness.py \
  --dataset swe-bench-verified \
  --dataset-seed 17 \
  --samples 10 \
  --max-retries 3 \
  --token-budget 100000 \
  --predictions-file eval/swe_bench_predictions.jsonl
```

This local command is useful for inference diagnostics, but it is not the
authoritative success-rate evaluation because it does not prove the required
per-instance OCI boundary or run the official scorer.

#### Authoritative OCI matrix evaluation

The manual `SWE-bench OCI Evaluation` workflow runs exactly one official
SWE-bench image per matrix job, pins RepoPilot tool execution to that image's
local `sha256:` identity, and limits model concurrency to two jobs. The
five checkpoint IDs are a subset of the fixed 50-instance baseline. The
checkpoint must pass its infrastructure and artifact checks before starting
that baseline, whose final score denominator is exactly 50.

The dataset boundary is `SWE-bench/SWE-bench_Verified`, split `test`, at the
immutable Hugging Face revision
`c104f840cc67f8b6eec6f759ebc8b2693d585d4a`. RepoPilot serializes the exact
500-row, 13-field dataset in its fixed upstream column order and requires
content SHA-256
`f61cd55ceb35b61ad592f645abcbfc8ea4d294c6c9f3c8f15e83211a8e8db98c`.
Cached content is accepted only when its metadata, row schema, row count, and
content digest all match those code-pinned values. The public workflow cache
key is
`swe-bench-verified-c104f840cc67f8b6eec6f759ebc8b2693d585d4a-f61cd55ceb35b61ad592f645abcbfc8ea4d294c6c9f3c8f15e83211a8e8db98c-v2`.

Before the first run:

1. Revoke any API key that has appeared in a terminal transcript, chat, log,
   attachment, or commit.
2. Merge `.github/workflows/swe-bench-oci-eval.yml` to the repository's default
   branch, `master`, through an explicitly authorized PR. GitHub cannot
   manually dispatch a workflow that exists only on a feature branch.
3. Store newly rotated values as GitHub repository secrets named
   `LLM_API_KEY` and `LLM_ESCALATION_API_KEY`. Do not pass values as workflow
   inputs, command arguments, cache keys, or repository files.

Dispatch and inspect the checkpoint:

```bash
gh workflow run swe-bench-oci-eval.yml --ref fix/release-readiness-20260717 -f mode=checkpoint_5
gh run list --workflow=swe-bench-oci-eval.yml --limit 5
gh run watch RUN_ID --exit-status
gh run download RUN_ID \
  --name swe-bench-oci-checkpoint_5 \
  --dir ARTIFACT_DIR/checkpoint_5
```

Only after `checkpoint_5/summary.md` confirms all five requested artifacts,
valid hashes, matching commit identity, digest-pinned usable runtimes, and a
terminal official status for every non-infrastructure instance, run the fixed
50-instance baseline:

```bash
gh workflow run swe-bench-oci-eval.yml --ref fix/release-readiness-20260717 -f mode=baseline_50
gh run list --workflow=swe-bench-oci-eval.yml --limit 5
gh run watch RUN_ID --exit-status
gh run download RUN_ID \
  --name swe-bench-oci-baseline_50 \
  --dir ARTIFACT_DIR/baseline_50
```

Each instance upload contains only `result.json`, `prediction.jsonl`,
`official_result.json`, and `manifest.json`. The aggregate contains ordered
`results.json`, `predictions.jsonl`, `official_results.json`, and `summary.md`.
Raw official logs, evaluator patches, model credentials, RepoPilot home data,
and target repositories are not uploaded. Generated evaluation outputs are
workflow artifacts and are not committed to Git. Snapshot reads are bounded to
8 MiB per file and 16 MiB per instance bundle. Official scoring tags the
prepared image digest with a run-local alias, verifies that alias immediately
before and after the harness call, and never scores through `latest`.

The aggregate displays the official score first. Its secondary engineering
score is auditable from four explicit fractions:

```text
engineering_score =
    80 * official_resolved / requested
  + 10 * non_infrastructure_instances / requested
  +  5 * explicit_agreements / official_terminal_instances
  +  5 * completed_within_budget_instances / requested
```

An agreement requires an explicit valid internal boolean verdict. Budget
credit requires a completed official job plus complete model token and elapsed
telemetry, at most 100,000 tokens, and at most 360 minutes.
Missing or invalid verdict/telemetry receives no credit. `summary.md` reports every component's
numerator and denominator and each instance's token and elapsed usage; these
secondary metrics never replace the raw official resolved fraction.

Generate a replay-safe summary from any result path:

```bash
python eval/report.py \
  --results-file eval/success_first_10.json \
  --summary-file eval/success_first_10_summary.md
```

Each instance is cloned and checked out at its exact 40-character
`base_commit` before local code search, patching, and testing. The issue seed
contains no gold patch, test patch, or evaluator-only fields.
The primary benchmark score is the Official `resolved` status from the
SWE-bench harness. RepoPilot's internal `agent_success` is true only when
`coverage_status` is `existing_verified` or `generated_verified` and a
nonempty differential `coverage_proof` is present. It contributes only to the
explicit agreement fraction above; it is neither official benchmark
resolution nor the engineering score by itself:

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name SWE-bench/SWE-bench_Verified \
  --predictions_path eval/swe_bench_predictions.jsonl \
  --max_workers 1 \
  --run_id repopilot-gemini-10
```

Official scoring requires Docker and may download large repository images.
If Docker or an image build is unavailable, keep the prediction JSONL and
report scoring as infrastructure-blocked instead of counting it as a model
failure. Likewise, `agent_success` is RepoPilot's strict internal coverage
verdict, `resolved` is only the official harness verdict, and
`scorer_infra`/OCI failures are not unresolved model outcomes.

GitHub API responses remain fresh for 600 seconds by default. On retryable
network, timeout, 429, 502, 503, or 504 errors, RepoPilot may serve an older
cached response for up to 86,400 seconds total. Configure these bounds with
`REPOPILOT_CACHE_TTL` and `REPOPILOT_CACHE_STALE_TTL`; 404 and other
non-retryable errors never use stale data.

### Running the FastAPI server

Configure a separate API bearer token and an exact repository allowlist before
starting the service. Inject the token through the deployment secret store;
the example below uses placeholders only:

```bash
export REPOPILOT_API_TOKEN=<deployment-secret>
export REPOPILOT_ALLOWED_REPOS=owner/repo,second-owner/second-repo
uvicorn src.main:app --reload
```

`REPOPILOT_ALLOWED_REPOS` is a nonempty comma-separated list of exact
`owner/repo` names. Matching is case-insensitive. Wildcards, URLs, paths,
whitespace, blank entries, duplicates, and malformed names make the service
fail closed. Only `GET /health` is public; all other routes require
`Authorization: Bearer <deployment-secret>`. Missing API configuration returns
503, missing or invalid credentials return the same 401 response, and a
repository outside the allowlist returns 403 before GitHub, model, clone, or
write helpers run.

Endpoints:

- `GET  /health` — liveness check
- `POST /analyze` — issue classification + file ranking (v1)
- `POST /agent` — legacy agent loop
- `POST /intelligent-agent` — legacy stateful agent
- `POST /agent/v2` — current state-machine agent (recommended)
- `POST /agent/v2/resume` — resume an authorized saved run
- `GET  /agent/v2/runs/{run_id}` — inspect an authorized saved-run summary
- `GET  /agent/v2/runs/{run_id}/replay?format=json|markdown` — replay an
  authorized run

Protected request bodies are capped at exactly 65,536 bytes, including chunked
requests. At most two authorized repository operations execute concurrently;
a third is rejected immediately with 429 and `Retry-After` instead of joining
an unbounded waiter queue. Agent v2 accepts `max_retries` 0–3 and
`token_budget` 1–100,000; legacy turn counts are 1–10. Run IDs use at most 64
safe alphanumeric/underscore/hyphen characters, and resume answers are limited
to 1–16,384 characters. Public OpenAPI, Swagger UI, and ReDoc routes are
disabled.

---

## License

MIT — see `pyproject.toml`.

---

<p align="center">
  <sub>Built with LangGraph, Pydantic, httpx, and OpenAI-compatible model gateways. Maintained by <a href="https://github.com/FMorgan-111">FMorgan-111</a>.</sub>
</p>
