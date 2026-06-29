# Plan Fix Timeout Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make live agent-v2 evals progress past `plan_fix` reliably by aligning LLM timeout behavior with phase timeouts, improving timeout diagnostics, and reducing planner prompt pressure.

**Architecture:** Keep LangGraph routing unchanged. Add a small timeout policy layer around LLM calls, then make `plan_fix` produce smaller, inspectable prompts with explicit diagnostics so failures are attributable to request timeout, phase cancellation, prompt size, or model output latency.

**Tech Stack:** Python async, httpx, tenacity, pytest, Pydantic state models, existing RepoPilot `node_diagnostics`.

---

## Root Cause Summary

Observed live run:

- Command: `python3 eval/harness.py --agent-v2 --samples 1 --max-retries 1 --token-budget 50000`
- Run id: `33ea071652b1`
- Final phase: `FAILED`
- Error: `Phase plan_fix timed out after 150.0s`
- Replay diagnostic: `node=plan_fix`, `event=phase`, `status=timeout`, `error_type=TimeoutError`

Current timeout shape:

- `src/http_client.py`: `LLM_REQUEST_TIMEOUT = 60.0`, `LLM_MAX_ATTEMPTS = 2`
- retry window can be approximately request timeout + backoff + request timeout
- `src/graph.py`: `PHASE_TIMEOUTS["plan_fix"] = 150.0`

Likely failure mode:

- The outer `asyncio.wait_for(plan_fix(...), 150s)` can cancel the node before `plan_fix` catches and records the inner LLM failure.
- Prompt size and requested JSON/patch output make `plan_fix` a high-latency node, especially on the tox sample.

## File Structure

- Modify `src/http_client.py`
  - Owns low-level LLM request timeout constants and request kwargs.
- Modify `src/llm.py`
  - Owns `llm_call(...)`; add optional timeout/metadata parameters only if needed.
- Modify `src/nodes/plan.py`
  - Owns planner prompt construction, planner-specific diagnostics, and prompt-size control.
- Modify `src/graph.py`
  - Owns phase timeout table; keep policy tests honest.
- Modify `src/run_store.py` and `eval/report.py` only if new diagnostic fields need replay/report rendering.
- Modify tests:
  - `tests/test_http_client.py`
  - `tests/test_new_agent.py`
  - `tests/test_decision_frame.py`
  - `tests/test_eval_report.py` or `tests/test_run_store.py` if diagnostics shape changes.

## Task 1: Add Explicit LLM Timeout Budget Diagnostics

**Files:**
- Modify: `src/http_client.py`
- Modify: `tests/test_http_client.py`

- [ ] **Step 1: Write failing test for timeout constants**

Add to `tests/test_http_client.py`:

```python
def test_llm_timeout_budget_is_explicit():
    from src import http_client

    assert http_client.LLM_REQUEST_TIMEOUT == 60.0
    assert http_client.LLM_MAX_ATTEMPTS == 2
    assert http_client.LLM_RETRY_BACKOFF_MAX_SECONDS == 20.0
    assert http_client.llm_retry_budget_seconds() == 140.0
```

- [ ] **Step 2: Run failing test**

Run:

```bash
python3 -B -m pytest -p no:cacheprovider tests/test_http_client.py::test_llm_timeout_budget_is_explicit -q
```

Expected: fail because `LLM_RETRY_BACKOFF_MAX_SECONDS` and `llm_retry_budget_seconds()` do not exist.

- [ ] **Step 3: Implement timeout budget helper**

In `src/http_client.py`, add:

```python
LLM_RETRY_BACKOFF_MAX_SECONDS = 20.0


def llm_retry_budget_seconds() -> float:
    """Worst-case LLM retry budget used by graph phase timeout checks."""
    return (LLM_REQUEST_TIMEOUT * LLM_MAX_ATTEMPTS) + LLM_RETRY_BACKOFF_MAX_SECONDS
```

Update the tenacity wait config:

```python
wait=wait_exponential(multiplier=2, min=2, max=LLM_RETRY_BACKOFF_MAX_SECONDS),
```

- [ ] **Step 4: Run test to verify pass**

Run:

```bash
python3 -B -m pytest -p no:cacheprovider tests/test_http_client.py::test_llm_timeout_budget_is_explicit -q
```

Expected: pass.

## Task 2: Align Phase Timeouts With Real LLM Budget

**Files:**
- Modify: `src/graph.py`
- Modify: `tests/test_new_agent.py`

- [ ] **Step 1: Write failing timeout alignment test**

Update `tests/test_new_agent.py::test_llm_phase_timeouts_cover_retry_window`:

```python
def test_llm_phase_timeouts_cover_retry_window():
    llm_retry_window = http_client.llm_retry_budget_seconds()
    planner_margin = 30.0

    assert graph.PHASE_TIMEOUTS["understand_issue"] >= llm_retry_window
    assert graph.PHASE_TIMEOUTS["locate_code"] >= 180.0
    assert graph.PHASE_TIMEOUTS["plan_fix"] >= llm_retry_window + planner_margin
    assert graph.PHASE_TIMEOUTS["execute_fix"] >= 600.0
    assert graph.PHASE_TIMEOUTS["reflect_on_failure"] >= llm_retry_window + planner_margin
    assert graph.PHASE_TIMEOUTS["commit_fix"] >= 600.0
    assert graph.PHASE_TIMEOUTS["handle_failure"] >= 60.0
```

- [ ] **Step 2: Run failing test**

Run:

```bash
python3 -B -m pytest -p no:cacheprovider tests/test_new_agent.py::test_llm_phase_timeouts_cover_retry_window -q
```

Expected: fail if `plan_fix`/`reflect_on_failure` are too close to the calculated retry budget.

- [ ] **Step 3: Adjust phase timeouts conservatively**

In `src/graph.py`, update:

```python
PHASE_TIMEOUTS: dict[str, float] = {
    "understand_issue": 240.0,
    "locate_code": 180.0,
    "plan_fix": 180.0,
    "execute_fix": 600.0,
    "verify_fix": 15.0,
    "reflect_on_failure": 180.0,
    "commit_fix": 600.0,
    "handle_failure": 60.0,
}
```

- [ ] **Step 4: Run test to verify pass**

Run:

```bash
python3 -B -m pytest -p no:cacheprovider tests/test_new_agent.py::test_llm_phase_timeouts_cover_retry_window -q
```

Expected: pass.

## Task 3: Record Planner Prompt Size In Phase Timeout Diagnostics

**Files:**
- Modify: `src/state.py` only if diagnostic helper needs no schema change; otherwise keep dict shape.
- Modify: `src/new_agent.py`
- Modify: `src/graph.py`
- Modify: `tests/test_new_agent.py`

- [ ] **Step 1: Write failing test for phase timeout preserving prompt estimate**

Add to `tests/test_new_agent.py`:

```python
async def test_phase_timeout_preserves_existing_node_diagnostics(monkeypatch):
    async def slow_node(state):
        state.node_diagnostics.append(
            {
                "node": "plan_fix",
                "event": "prompt_built",
                "status": "success",
                "prompt_tokens_estimate": 3456,
                "relevant_file_count": 2,
            }
        )
        await asyncio.sleep(1)
        return state

    monkeypatch.setitem(graph.PHASE_TIMEOUTS, "plan_fix", 0.01)
    compiled = graph.FallbackCompiledGraph({"plan_fix": slow_node}, "plan_fix")
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        current_phase=new_agent.Phase.PLAN,
    )

    final_state = await compiled.ainvoke(state)

    assert final_state.node_diagnostics[-2]["event"] == "prompt_built"
    assert final_state.node_diagnostics[-2]["prompt_tokens_estimate"] == 3456
    assert final_state.node_diagnostics[-1]["event"] == "phase"
    assert final_state.node_diagnostics[-1]["status"] == "timeout"
```

- [ ] **Step 2: Run test**

Run:

```bash
python3 -B -m pytest -p no:cacheprovider tests/test_new_agent.py::test_phase_timeout_preserves_existing_node_diagnostics -q
```

Expected: pass if existing state mutation survives cancellation; if it fails under LangGraph only, add an equivalent wrapper-level test in `new_agent._wrap_node`.

- [ ] **Step 3: Add planner prompt-built diagnostic**

In `src/nodes/plan.py`, after `prompt_tokens_estimate = _estimate_tokens(system, user)`, add:

```python
    _record_node_diagnostic(
        state,
        node="plan_fix",
        event="prompt_built",
        status="success",
        elapsed_seconds=0.0,
        prompt_tokens_estimate=prompt_tokens_estimate,
        relevant_file_count=len(state.relevant_files[:2]),
        issue_body_chars=len(state.issue_body[:4000]),
        previous_failure_count=len(state.fix_attempts),
        has_reflection_context=bool(reflection_context),
        has_hypothesis_continuity_context=bool(hypothesis_continuity_context),
    )
```

- [ ] **Step 4: Add direct planner test**

Add to `tests/test_decision_frame.py`:

```python
async def test_plan_fix_records_prompt_built_diagnostic(monkeypatch):
    async def fake_llm_call(system, user):
        return json.dumps(
            {
                "plan": "Patch auth submit handling.",
                "patch": "diff --git a/src/auth.py b/src/auth.py\n--- a/src/auth.py\n+++ b/src/auth.py\n",
                "files": ["src/auth.py"],
                "test_command": "pytest tests/test_auth.py -q",
                "decision_frame": {
                    "stage": "plan",
                    "summary": "Patch auth submit handling.",
                    "recommended_action": "execute",
                    "risk": "medium",
                    "confidence": 0.84,
                },
            }
        )

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        issue_title="Login crash",
        issue_body="Crashes after submit.",
        current_phase=new_agent.Phase.PLAN,
    )

    next_state = await plan_node.plan_fix(state)

    prompt_diag = next(
        item for item in next_state.node_diagnostics if item["event"] == "prompt_built"
    )
    assert prompt_diag["node"] == "plan_fix"
    assert prompt_diag["prompt_tokens_estimate"] > 0
    assert prompt_diag["previous_failure_count"] == 0
```

- [ ] **Step 5: Run tests**

Run:

```bash
python3 -B -m pytest -p no:cacheprovider tests/test_new_agent.py::test_phase_timeout_preserves_existing_node_diagnostics tests/test_decision_frame.py::test_plan_fix_records_prompt_built_diagnostic -q
```

Expected: pass.

## Task 4: Reduce Planner Prompt Pressure

**Files:**
- Modify: `src/nodes/plan.py`
- Modify: `tests/test_decision_frame.py`

- [ ] **Step 1: Write failing test for prompt truncation policy**

Add to `tests/test_decision_frame.py`:

```python
async def test_plan_fix_prompt_uses_compact_file_context(monkeypatch):
    captured = {}

    async def fake_llm_call(system, user):
        captured["user"] = user
        return json.dumps(
            {
                "plan": "Patch auth submit handling.",
                "patch": "diff --git a/src/auth.py b/src/auth.py\n--- a/src/auth.py\n+++ b/src/auth.py\n",
                "files": ["src/auth.py"],
                "test_command": "pytest tests/test_auth.py -q",
                "decision_frame": {
                    "stage": "plan",
                    "summary": "Patch auth submit handling.",
                    "recommended_action": "execute",
                    "risk": "medium",
                    "confidence": 0.84,
                },
            }
        )

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        issue_title="Login crash",
        issue_body="x" * 8000,
        current_phase=new_agent.Phase.PLAN,
        relevant_files=[
            new_agent.FileInfo(
                path="src/auth.py",
                relevance_score=0.9,
                reason="auth path",
                content="a" * 5000,
            ),
            new_agent.FileInfo(
                path="src/session.py",
                relevance_score=0.8,
                reason="session path",
                content="b" * 5000,
            ),
        ],
    )

    await plan_node.plan_fix(state)

    assert "a" * 1800 not in captured["user"]
    assert "b" * 1800 not in captured["user"]
    assert len(captured["user"]) < 7000
```

- [ ] **Step 2: Run failing test**

Run:

```bash
python3 -B -m pytest -p no:cacheprovider tests/test_decision_frame.py::test_plan_fix_prompt_uses_compact_file_context -q
```

Expected: fail with current 4000-char issue body plus two 2000-char file contents.

- [ ] **Step 3: Add prompt truncation constants and helper**

In `src/nodes/plan.py`, add near helper functions:

```python
PLAN_ISSUE_BODY_LIMIT = 2500
PLAN_FILE_CONTENT_LIMIT = 1200
PLAN_MAX_FILES = 2
PLAN_FAILURE_LOG_LIMIT = 1000
```

Update `previous_failures`:

```python
    previous_failures = "\n\n".join(
        f"Attempt {idx + 1}: {attempt.test_result}\n"
        f"{_truncate_prompt_text(attempt.error_log, PLAN_FAILURE_LOG_LIMIT)}"
        for idx, attempt in enumerate(state.fix_attempts)
    )
```

Update `files_context`:

```python
    files_context = "\n\n".join(
        f"FILE: {file.path}\nRELEVANCE: {file.relevance_score} - {file.reason}\n"
        f"CONTENT:\n{_truncate_prompt_text(file.content, PLAN_FILE_CONTENT_LIMIT)}"
        for file in state.relevant_files[:PLAN_MAX_FILES]
    )
```

Update issue body interpolation:

```python
        f"Title: {state.issue_title}\n\nBody:\n"
        f"{_truncate_prompt_text(state.issue_body, PLAN_ISSUE_BODY_LIMIT)}\n\n"
```

- [ ] **Step 4: Run truncation and existing planner tests**

Run:

```bash
python3 -B -m pytest -p no:cacheprovider tests/test_decision_frame.py::test_plan_fix_prompt_uses_compact_file_context tests/test_decision_frame.py::test_plan_fix_records_plan_decision_frame tests/test_decision_frame.py::test_plan_fix_after_patch_apply_failure_includes_hypothesis_anchor -q
```

Expected: pass.

## Task 5: Add Planner Timeout Classification To Reports

**Files:**
- Modify: `eval/report.py`
- Modify: `tests/test_eval_report.py`

- [ ] **Step 1: Write failing report test**

Add to `tests/test_eval_report.py`:

```python
def test_generate_markdown_surfaces_plan_fix_phase_timeout():
    result = {
        "id": "tox-dev/tox#3075:3748",
        "mode": "agent_v2",
        "run_id": "abc123",
        "success": False,
        "waiting_for_user": False,
        "final_phase": "FAILED",
        "turns_taken": 14,
        "token_used": 5601,
        "error": "Phase plan_fix timed out after 150.0s",
        "replay": {
            "current_phase": "FAILED",
            "timeline": [
                {
                    "type": "node_diagnostic",
                    "diagnostic": {
                        "node": "plan_fix",
                        "event": "phase",
                        "status": "timeout",
                        "error_type": "TimeoutError",
                        "error": "TimeoutError",
                        "phase_timeout_seconds": 150.0,
                    },
                }
            ],
        },
    }
    markdown = report.generate_markdown(
        [result],
        report.compute_metrics([result]),
    )

    assert "Planner timeout" in markdown
    assert "plan_fix exceeded 150.0s" in markdown
```

- [ ] **Step 2: Run failing test**

Run:

```bash
python3 -B -m pytest -p no:cacheprovider tests/test_eval_report.py::test_generate_markdown_surfaces_plan_fix_phase_timeout -q
```

Expected: fail because report only lists raw node diagnostics.

- [ ] **Step 3: Add diagnostic summary helper**

In `eval/report.py`, add:

```python
def _diagnostic_summary(replay: dict[str, Any]) -> list[str]:
    summaries: list[str] = []
    for item in replay.get("timeline", []):
        diagnostic = item.get("diagnostic") or item
        if item.get("type") != "node_diagnostic":
            continue
        if (
            diagnostic.get("node") == "plan_fix"
            and diagnostic.get("event") == "phase"
            and diagnostic.get("status") == "timeout"
        ):
            timeout = diagnostic.get("phase_timeout_seconds", "")
            summaries.append(f"Planner timeout: plan_fix exceeded {timeout}s.")
    return summaries
```

Call it in `_append_replay_diagnostics(...)` before `_append_node_diagnostics(...)`:

```python
        for summary in _diagnostic_summary(replay):
            lines.append(f"- Diagnostic summary: {summary}\n")
```

- [ ] **Step 4: Run report tests**

Run:

```bash
python3 -B -m pytest -p no:cacheprovider tests/test_eval_report.py -q
```

Expected: pass.

## Task 6: Verification And Live Eval

**Files:**
- Update generated artifacts only:
  - `eval/eval_results.json`
  - `eval/eval_summary.md`
  - `examples/traces/case_1.json`
  - `docs/CODEX_CONTEXT.md`

- [ ] **Step 1: Run affected tests**

Run:

```bash
python3 -B -m pytest -p no:cacheprovider tests/test_http_client.py tests/test_new_agent.py tests/test_decision_frame.py tests/test_eval_report.py -q
```

Expected: all pass.

- [ ] **Step 2: Run compile check**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/repopilot-pycache python3 -m py_compile src/http_client.py src/graph.py src/nodes/plan.py eval/report.py tests/test_http_client.py tests/test_new_agent.py tests/test_decision_frame.py tests/test_eval_report.py
```

Expected: exit code `0`.

- [ ] **Step 3: Run ruff check**

Run:

```bash
ruff check src/http_client.py src/graph.py src/nodes/plan.py eval/report.py tests/test_http_client.py tests/test_new_agent.py tests/test_decision_frame.py tests/test_eval_report.py --select=E,F,I --ignore=E501 --cache-dir /tmp/repopilot-ruff-cache
```

Expected: `All checks passed!`

- [ ] **Step 4: Run live tox eval with network/LLM access**

Run outside the restricted sandbox:

```bash
python3 eval/harness.py --agent-v2 --samples 1 --max-retries 1 --token-budget 50000
```

Expected:

- Process exits naturally.
- If `plan_fix` still times out, report explicitly labels it as planner timeout and includes prompt-built diagnostics.
- If `plan_fix` returns, execution should reach either patch preflight repair, patch apply, test verification, or bounded failure.

- [ ] **Step 5: Regenerate report**

Run:

```bash
python3 eval/report.py
```

Expected:

- `eval/eval_summary.md` references the latest run id.
- `Replay Diagnostics` includes either patch repair outcome or planner timeout summary.

- [ ] **Step 6: Check for credential leaks**

Run:

```bash
rg -n "x-access-token:[^<]|ghp_|github_pat_|DEEPSEEK_API_KEY|LLM_API_KEY" eval/eval_results.json eval/eval_summary.md examples/traces/case_1.json docs/CODEX_CONTEXT.md
```

Expected: no matches.

- [ ] **Step 7: Update handoff context**

Append a short section to `docs/CODEX_CONTEXT.md` with:

```markdown
## 2026-06-12 Plan Fix Timeout Reliability

- Implemented timeout budget helper and phase timeout alignment.
- Added prompt-built diagnostics for `plan_fix`.
- Reduced planner prompt size with explicit truncation limits.
- Live eval result: <run id>, <final phase>, <main failure/success reason>.
- Next recommended step: <patch repair, root-cause quality, or remaining timeout blocker>.
```

## Execution Order

1. Task 1: make LLM retry budget explicit.
2. Task 2: align graph phase timeouts.
3. Task 3: add prompt-built diagnostics.
4. Task 4: reduce planner prompt size.
5. Task 5: improve report classification.
6. Task 6: run verification and live eval.

## Risk Controls

- Do not increase `plan_fix` timeout beyond 180s in the first pass; longer timeouts hide planner/model latency instead of fixing it.
- Do not change routing semantics in this slice.
- Do not change patch repair retry rules in this slice.
- Do not run live eval inside the restricted sandbox; it needs network, LLM credentials, repo cache, and SQLite writes.
