# RepoPilot — Architecture & Code Quality Review

**Date:** 2026-06-05
**Reviewer:** Staff-level code review pass
**Scope:** Every source file (`src/main.py`, `src/agent.py`, `src/llm.py`, `src/tools.py`, `src/tracer.py`), every test (`tests/test_*.py`, `tests/conftest.py`), and config (`requirements.txt`, `.env.example`, `README.md`, `.gitignore`).
**Test status at review time:** `22 passed in 0.36s` ✅ (run with system `python3 3.14`; see *Environment note* below).

---

## TL;DR

RepoPilot is a **clean, well-tested, well-factored LLM pipeline** — genuinely above the bar for an MVP. The module boundaries are crisp, the error handling is resilient, and the test suite is fast and isolated. The single biggest gap is conceptual, not mechanical: **it is marketed as an "Agent" but it is a fixed linear chain** — the LLM never decides anything about control flow. For an *AI Agent Engineer* resume, that framing gap matters more than any individual bug.

The most damaging concrete issues are: (1) **leftover debug code that writes to a hardcoded `/tmp` file** in the hot path, (2) a **model name default that looks wrong/typo'd** and a documented `LLM_MODEL` env var that is never read, and (3) **the fix plan is generated from file *paths* only — the code is never actually read**, so the headline output is uninformed by the repository's contents.

---

## Environment note (reproducing the tests)

The checked-in `.venv/` is a Python 3.14 virtualenv **with no `pip` and no `pytest`/deps installed** — `python -m pytest` fails out of the box. The system `python3` (3.14.4) happens to have all dependencies, so tests run via `python3 -m pytest -v`. This is a reproducibility smell: a fresh clone cannot run the tests without manual environment surgery. See *Next Steps → Make the project runnable from a clean clone*.

---

## 1. Architecture Assessment

### What's good

- **Clean separation of concerns.** The five modules map almost one-to-one onto responsibilities, and the dependency direction is correct (nothing depends "upward"):
  - `main.py` — HTTP/transport (FastAPI) only.
  - `agent.py` — orchestration / pipeline sequencing.
  - `llm.py` — LLM I/O + prompt construction + JSON extraction.
  - `tools.py` — external side-effecting integrations (GitHub).
  - `tracer.py` — observability.
- **Tools vs. reasoning are correctly split.** `tools.py` (deterministic API calls) and `llm.py` (model calls) are separate — this is the right seam and makes both independently testable.
- **The pipeline degrades gracefully.** Every stage after the issue fetch has a fallback, so a single LLM hiccup or a GitHub code-search outage still returns a useful partial result. `test_analyze_issue_search_failure_still_returns_classification` proves this is intentional, not accidental.
- **Typed signatures throughout** (`tuple[str, str, int]`, `list[dict]`, `str | None`) — readable and self-documenting.
- **The OpenAI-compatible abstraction** in `llm.py` is a smart, low-cost decision: it makes the model provider swappable (DeepSeek today, anything OpenAI-shaped tomorrow).

### What I'd change

- **`agent.py` is doing slightly too much of *one* thing — repetitively.** It is not "too many responsibilities"; it is **six near-identical `try/except + t.log(...)` blocks** (Steps 1–6). The orchestration logic is sound, but the boilerplate-to-signal ratio is high and each block hand-rolls its own fallback shape. This is the place where a *small* amount of structure would pay off (a `run_step(name, coro, fallback)` helper), without over-abstracting. ~40% of the file is repeated scaffolding.
- **It isn't really an agent — it's a DAG.** The control flow is fully hardcoded: parse → read → classify → search → rank → plan. The LLM never chooses a tool, never loops, never decides it has enough information. This is fine architecturally (deterministic pipelines are easier to reason about and test) but it should either be **renamed honestly** ("analysis pipeline") or **upgraded into a real tool-calling loop** (see *Resume Impact*).
- **The pipeline reads file *paths* but never file *contents*.** `search_code` returns paths/URLs/SHAs; `rank_files` ranks paths; `generate_fix_plan` is handed only `path + relevance_score + reason`. The model that writes the fix plan **never sees a single line of source code.** This is the architecture's most important functional limitation — the headline deliverable is generated blind.
- **Module boundaries are clean, but `agent.py` reaches into dict shapes by string key** (`issue["title"]`, `plan.get("risk_level")`) rather than typed objects. With Pydantic models for the cross-module contracts, the boundaries would be enforced, not merely conventional.

**Verdict:** Boundaries are clean and correct. `agent.py` is not overloaded with *responsibilities*, but it is **repetitive** and **leaks untyped dicts across seams**. The real architectural gap is the missing agentic loop and the un-read source code.

---

## 2. Code Quality Issues

### 🔴 High severity

**2.1 — Leftover debug code writing to a hardcoded temp file** — `src/agent.py:45-47`
```python
except Exception as e:
    import sys, datetime
    with open('/tmp/repopilot_errors.log', 'a') as f:
        f.write(f"[{datetime.datetime.now()}] classify: {type(e).__name__}: {e}\n")
```
- Hardcoded `/tmp` path (non-portable, world-readable on shared hosts, silently no-ops/crashes on read-only FS).
- Inline `import sys` is **never used**; `import datetime` shadows nothing but is inline for no reason.
- It exists **only on the `classify` branch** — the other four `except` blocks don't do this — so it's clearly a debugging artifact that was never removed. Inconsistent and surprising.
- **Fix:** delete the `open(...)` block entirely; route the error through the existing `Tracer` (already done on the next line) or a real logger.

**2.2 — Wrong/undefined default model + dead config knob** — `src/llm.py:41`, `.env.example:13-14`
```python
async def llm_call(..., model: str = "deepseek-v4-flash") -> dict:
```
- `deepseek-v4-flash` is **not a known DeepSeek model id** (DeepSeek's chat model is `deepseek-chat`). This default looks like a typo/hallucinated id and would 400 against the real API.
- `.env.example` documents `LLM_MODEL=deepseek-chat` as "the default model," but **`LLM_MODEL` is never read anywhere in the code** (confirmed by grep). The documented knob is dead.
- No caller in `agent.py` ever passes `model=`, so the broken default is the *only* model the app can use.
- **Fix:** default to `deepseek-chat`, and have `_config()`/`llm_call` actually read `os.getenv("LLM_MODEL", "deepseek-chat")` so the documented env var works.

**2.3 — The fix plan never sees the code** — `src/agent.py:76-80`, `src/llm.py:95-117`
- As noted in §1: `generate_fix_plan` receives only ranked path metadata. The single most valuable thing an issue-to-fix agent could do — read the relevant lines — never happens. This is a correctness/quality ceiling on the entire product, not a cosmetic bug.

### 🟡 Medium severity

**2.4 — Blanket `except Exception` swallows real bugs** — `src/agent.py:44, 57, 69, 81`
Every post-fetch stage catches *everything*. A `KeyError` from a renamed field, a `TypeError` from a contract change, or a programming mistake is indistinguishable from a legitimate LLM/network failure — all are silently downgraded to a fallback. The `fix_plan` fallback (`"Could not generate plan."`) in particular **masks total failure as a successful 200 response**. Catch narrowly (`httpx.HTTPError`, `ValueError`/`json.JSONDecodeError`, `KeyError`) or at least log at `error` level and surface a degraded-status flag in the response.

**2.5 — Errors are returned as HTTP 200** — `src/main.py:19-21`
```python
if "error" in result:
    return {"status": "error", **result}
```
An invalid URL or an upstream failure returns `200 OK` with an `error` field. Clients/load-balancers/monitoring can't distinguish success from failure by status code. Invalid input should be `400`, upstream failure `502/503`. (`test_post_analyze_invalid_url` actually *asserts* the 200 — so the test encodes the anti-pattern.)

**2.6 — Tracer dumps full inputs/outputs (incl. issue body) to stdout** — `src/tracer.py:14-25`, `src/agent.py:38-39, 51`
- The tracer prints `input`/`output` payloads to stdout as JSONL. Issue titles/bodies (potentially private-repo content, tokens pasted into issues, PII) are emitted to stdout with no redaction and no opt-out.
- It uses `print(..., file=sys.stdout)` rather than a `logging` logger, so it can't be levelled, filtered, or routed, and it interleaves with uvicorn's own stdout.
- **Fix:** use `logging`; log step name + sizes/ids by default, gate full payloads behind a debug flag.

**2.7 — GitHub code-search query is the raw issue title** — `src/agent.py:55`
```python
query = issue["title"][:100]
files = await search_code(query, owner, repo)
```
GitHub *code* search is keyword/qualifier-based and rejects/garbles natural-language prose; feeding it a raw English issue title (`"Login crash"`) yields poor or empty results in practice. There's also no extraction of identifiers/symbols from the body. The ranking stage is therefore often ranking noise.

**2.8 — No retries, backoff, or rate-limit handling** — `src/tools.py`, `src/llm.py`
- GitHub's code-search endpoint is rate-limited to ~10 req/min and frequently returns `403`/`422`; the LLM endpoint can return `429`. There's no retry/backoff anywhere. Transient failures become permanent stage failures.
- **Fix:** wrap external calls with bounded exponential backoff (e.g. `tenacity`) on `429`/`5xx`.

### 🟢 Low severity / polish

- **2.9 — New `httpx.AsyncClient` per call** (`tools.py:18,36`, `llm.py:57`). No connection pooling/reuse across the pipeline's ~3 LLM + 2 GitHub calls. Fine for an MVP; a shared client (FastAPI lifespan) is the idiomatic fix.
- **2.10 — `_extract_json` brace-fallback handles only one level of nesting** (`llm.py:25`). The regex `\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}` won't match `{"a":{"b":{"c":1}}}` embedded in prose. In practice the raw `json.loads` on line 14 catches well-formed responses first, so this only bites when deeply-nested JSON is wrapped in prose — but it's a latent trap. A bracket-balancing scan is more robust than regex.
- **2.11 — `requirements.txt` is fully unpinned** (`fastapi`, `httpx`, …). Non-reproducible builds. Pin with `==` or a lockfile.
- **2.12 — `.env.example` drift.** Comment says "default: deepseek-chat" (§2.2), and the example `OPENAI_BASE_URL=https://linoapi.com.cn` lacks the `/v1` suffix the code expects (the code `rstrip("/")`s and appends `/chat/completions`, so a bare host would call `https://linoapi.com.cn/chat/completions`, not `/v1/...`). Easy to misconfigure.
- **2.13 — README is a single line** (`# RepoPilot`). For a portfolio project this is the highest-leverage, lowest-effort fix (see §3).
- **2.14 — No `LICENSE`, no CI workflow, no `Dockerfile`.**
- **2.15 — Unused import** `import re` is used in both files it appears in (`agent.py`, `llm.py`) — OK. But `agent.py:45`'s inline `import sys` is dead (see §2.1).
- **2.16 — `search_code` assumes item shape** (`item["path"]`, `item["repository"]["full_name"]`). A malformed GitHub item raises `KeyError`, which the agent's blanket catch then swallows into "0 files." Use `.get` for resilience (the code already does for `url`/`sha`, just not `path`/`repository`).

### Async correctness

- **No event-loop misuse, no blocking calls inside `async def`** — the I/O is genuinely async via `httpx.AsyncClient`. Good.
- `resp.raise_for_status()` is called *after* the `async with` client closes (`llm.py:57-59`, `tools.py:18-21`). For **non-streaming** responses the body is already read inside the context, so this is safe — but it's a pattern that breaks the moment someone switches to `client.stream(...)`. Worth a comment or moving `raise_for_status` inside the block.
- The custom async test runner (`conftest.py:58-70`) calls `asyncio.run()` per test — correct and isolated, but see §Testing.

### Security

- **No secrets are committed.** `.env` is git-ignored and not tracked (verified); `.env.example` masks values with `***`. ✅
- **Token handling is fine** — `GITHUB_TOKEN` read from env, sent as Bearer; key never logged in `llm.py`.
- **Main residual risk is the tracer** (§2.6): untrusted issue content is echoed to stdout unredacted.
- **No input validation on `issue_url` beyond the regex** — the regex is reasonably tight (`github.com/.../issues/\d+`), so SSRF surface is low. Worth noting the `owner`/`repo` are interpolated into a GitHub API path; the regex's `[^/]+` prevents path traversal. Acceptable.

---

## 3. Resume Impact Assessment (for an AI Agent Engineer role)

### What already reads well to an interviewer

- **Real, fast, isolated tests** (22, 0.36s, hand-rolled `httpx` mock transport — shows you understand the boundary you're mocking).
- **Observability instinct** (a `Tracer` with trace IDs and per-step JSONL) — most candidates don't think about this at all.
- **Resilient orchestration** with deliberate fallbacks, proven by a test.
- **Provider-agnostic LLM layer** (OpenAI-compatible) — signals you've thought about vendor lock-in.

### What's missing that would make an interviewer say "wow"

1. **Make it an actual agent.** Today it's a deterministic chain. The single highest-signal upgrade is a **tool-calling / ReAct loop**: give the model the tools (`read_issue`, `search_code`, *`read_file`*) and let *it* decide which to call and when to stop. That is the literal job description. Even a small, bounded loop ("max 5 tool calls") demonstrates the core competency the title is testing for.
2. **Actually read the code.** Add a `read_file_contents` tool and feed the relevant snippets into the fix-plan prompt. The difference between "plan from filenames" and "plan from the actual buggy function" is the difference between a toy and a product.
3. **An eval harness.** AI engineers are judged on whether they can *measure* an LLM system. A `evals/` folder with ~10 labeled issues and a scorer (did it pick the right file? right severity?) — even rudimentary — is a massive differentiator. Pair it with a metric printed in CI.
4. **Structured outputs with validation.** Replace prompt-and-regex JSON with **Pydantic models** for `Classification`, `RankedFile`, `FixPlan`, and validate every LLM response against them (retry on validation failure). Shows you know how to make LLM output *trustworthy*.
5. **Cost & token observability.** Capture `usage` from each LLM response into the tracer (tokens, estimated $). "I instrumented per-trace cost" is a sentence that lands in interviews.
6. **Guardrails: retries, timeouts, backoff** (§2.8) — demonstrates production maturity.
7. **A real README with an architecture diagram and a recorded demo.** Right now the repo is a single `# RepoPilot` line; an interviewer cloning it learns nothing. This is the cheapest "wow" available.

---

## 4. Actionable Next Steps (ranked: highest impact / lowest effort first)

> Ordering optimizes for **resume signal per hour**. Items 1–4 are cheap and high-leverage; 5–8 are the substantive engineering that proves the title.

| # | Step | Why it matters | Est. |
|---|------|----------------|------|
| **1** | **Delete the `/tmp` debug block** (`agent.py:45-47`) and the dead inline `import sys`. Route the error through `Tracer`/logging like the other stages. | Removes the most obvious "left my debugging in" smell a reviewer will spot in 10 seconds. Pure cleanup, zero risk. | 10 min |
| **2** | **Fix the model config** (`llm.py:41`, `_config`): default to `deepseek-chat`, read `LLM_MODEL` from env, reconcile `.env.example`. | The app currently can't talk to a real model with its default; and a documented env var is dead. Correctness + credibility. | 20 min |
| **3** | **Write a real README** — what it does, architecture diagram (the 5-module pipeline), setup, `curl` example, sample output, limitations. | Highest "wow per minute." The repo is currently a one-liner; this is what an interviewer actually reads first. | 1–2 hr |
| **4** | **Make it runnable from a clean clone**: pin `requirements.txt`, add a working venv bootstrap (the committed `.venv` has no pip), add a `make test` / `make run`. | Right now `python -m pytest` fails on a fresh checkout. A project that won't run is an instant credibility hit. | 1 hr |
| **5** | **Add structured-output validation with Pydantic** for `Classification`, `RankedFile`, `FixPlan`; validate each LLM response and retry once on failure. Replace the brittle regex fallback path. | Turns "parse whatever the LLM says" into "enforce a contract" — core AI-eng skill, and it hardens §2.4/§2.10. | 3–4 hr |
| **6** | **Add a `read_file` tool and feed code into the fix-plan prompt** (`tools.py` + `agent.py:76`). Fetch the top-ranked files' contents (GitHub `contents` API) and include relevant snippets. | Fixes the product's central limitation (§2.3): the plan is currently written blind. Biggest jump in actual output quality. | 4–6 hr |
| **7** | **Convert the pipeline into a bounded tool-calling loop** (ReAct-style): expose `read_issue`/`search_code`/`read_file` as tools, let the model drive, cap iterations. Keep the current chain as a deterministic fallback. | This is *the* thing the "AI Agent Engineer" title is testing. Single highest-signal engineering item. | 1–2 days |
| **8** | **Add an `evals/` harness**: ~10 labeled real issues + a scorer (correct file? correct severity?) + a CI job that prints the score. Also capture token/cost `usage` into the tracer. | Demonstrates you can *measure* an LLM system, not just build one. Rare and highly valued. | 1 day |

**Smaller hardening items** (fold into the above where convenient): return proper HTTP status codes (§2.5), add `tenacity` backoff on `429`/`5xx` (§2.8), redact/gate tracer payloads behind a debug flag (§2.6), share one `httpx.AsyncClient` via FastAPI lifespan (§2.9), extract search keywords from the issue instead of using the raw title (§2.7), add `LICENSE` + GitHub Actions CI.

---

## 5. Agent Loop — New Design (`src/agent_loop.py`)

**Date added:** 2026-06-05
**Goal:** address §1 / §3.1 — turn the "Agent" from a fixed DAG into a real LLM-driven tool-calling loop, *without* touching the existing linear pipeline (which stays as a deterministic baseline/fallback).

### What changed

| Concern | Linear pipeline (`agent.py`) | Agent loop (`agent_loop.py`) — new |
|---|---|---|
| Control flow | Hardcoded: parse → read → classify → search → rank → plan | LLM decides each step; loops until `done` or `max_turns` |
| Who picks tools | The code | The model (returns `{"tool", "args"}` JSON) |
| Termination | Always 6 steps | Model returns `{"done": true, ...}`, or `max_turns` cap |
| New capability | — | Can call `read_file` to read actual source before answering |
| Status | Unchanged, all 22 original tests still pass | Additive — new module, new endpoint, new tests |

### Components

- **`TOOLS`** — declarative registry of the three tools (`read_issue`, `search_code`, `read_file`), name + description, rendered into the system prompt so the model knows what it can call.
- **`SYSTEM_PROMPT`** — instructs the model to respond with **only** a JSON object each turn: either `{"tool": "<name>", "args": {...}}` to act, or `{"done": true, "summary", "files", "fix_plan"}` to finish.
- **`execute_tool(name, args, owner, repo, issue_number)`** — pure dispatcher mapping a tool name to the corresponding `tools.py` helper. `owner`/`repo`/`issue_number` are parsed once from the URL and injected, so the model only supplies the variable args (`query`, `path`). Unknown tools raise `ValueError`.
- **`agent_analyze(issue_url, max_turns=10)`** — the loop itself.

### The loop

```
parse_issue_url(issue_url)            # fail fast on bad URL → {"error", trace_id}
transcript = "Analyze this issue: <url>"
for turn in range(max_turns):
    response = await llm_call(SYSTEM_PROMPT, transcript)   # reuses llm.py
    if response.done: return {**response, trace_id, turns}
    result = await execute_tool(response.tool, response.args, ...)
    transcript += "You called: <response>\nTool result: <result>"
return {"done": true, "error": "Max turns reached", ...}
```

**Reusing `llm.llm_call` as-is.** `llm_call(system_prompt, user_prompt)` takes two strings, not a message list. Rather than modify it (and risk the existing tests), each turn passes `SYSTEM_PROMPT` as the system message and a **running transcript** — the task plus every prior tool call and result — as the user message. This is a deliberately simple form of conversation memory that fits the existing signature; the model sees full history on every turn. (A future refinement would extend `llm_call` to accept a real message array.)

### Resilience (mirrors the pipeline's graceful-degradation instinct)

- **Bad URL** → returns `{"error", trace_id}` before any LLM call.
- **Tool raises** (e.g. GitHub rate limit) → the exception is captured into the transcript as `{"error": ...}` and the loop continues, letting the model adapt.
- **LLM call raises** → returns `{"done": true, "error": "LLM call failed: ...", trace_id, turns}`.
- **Malformed model response** (neither `tool` nor `done`) → a corrective nudge is appended and the loop continues instead of crashing.
- **Runaway model** → `max_turns` (default 10) bounds cost and guarantees termination.
- Every turn is logged through the existing `Tracer` (step name = tool name), so the new loop is observable exactly like the pipeline.

### Surface area

- **New:** `src/agent_loop.py`, `tests/test_agent_loop.py` (8 tests).
- **Modified (additive only):** `tools.py` gains `read_file` (GitHub contents API, base64-decoded); `main.py` gains `POST /agent` (`{issue_url, max_turns}`) routing to `agent_analyze`; `test_tools.py` + `test_main.py` gain coverage for the new tool and endpoint.
- **Untouched:** `agent.py`, `llm.py` (consumed, not changed), `tracer.py`.
- **Tests:** 22 → 34, all passing.

### Known limitations / next steps

- Transcript-as-string memory grows unbounded across turns; for long sessions, switch to a real message array + truncation/summarization.
- The model is trusted to emit valid JSON; `_extract_json` already tolerates fenced/embedded JSON, but Pydantic-validated tool calls with a retry (per §3.4) would harden this.
- `read_file` is available to the loop but not yet wired into the linear pipeline's fix-plan prompt (§2.3 remains open there).

---

## Appendix — File-by-file notes

- **`src/main.py`** — Minimal, correct FastAPI wiring. Only issue: error → HTTP 200 (§2.5). `/health` is a nice touch.
- **`src/agent.py`** — Sound orchestration; repetitive (§1); blanket excepts (§2.4); leftover debug code (§2.1); blind fix plan (§2.3); raw-title search query (§2.7).
- **`src/llm.py`** — Good provider-agnostic design and a thoughtful `_extract_json`. Broken model default + dead `LLM_MODEL` (§2.2); one-level-nesting regex (§2.10); no `max_tokens`/usage capture.
- **`src/tools.py`** — Clean, well-shaped GitHub client. `.get` used inconsistently (§2.16); no retry/rate-limit handling (§2.8).
- **`src/tracer.py`** — Good idea, right instinct. Should use `logging`, redact payloads, and (eventually) persist traces (§2.6).
- **`src/__init__.py`** — Empty package marker. Fine.
- **`tests/conftest.py`** — Impressive hand-rolled `MockTransport`-based `httpx` mock and a custom async runner. Works well. Consider `pytest-asyncio`/`anyio` + `respx`/`pytest-httpx` to cut bespoke harness code — but what's here is correct and fast.
- **`tests/*`** — Good coverage of happy paths and key failure paths. Gaps: `llm_call` HTTP-error path, `_config` env precedence, `_extract_json` first-brace fallback, and the search-query-construction logic. `test_post_analyze_invalid_url` encodes the 200-on-error anti-pattern (§2.5) — update alongside the status-code fix.
- **`requirements.txt`** — Unpinned (§2.11).
- **`.env.example`** — Drifted from code (§2.2, §2.12).
- **`README.md`** — One line; single highest-leverage fix (§3, Step 3).
- **`.gitignore`** — Correctly ignores `.env`, `__pycache__`, `*.pyc`. ✅
