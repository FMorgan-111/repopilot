# Agent V2 Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make agent-v2 failures explainable by recording LLM/node diagnostics, checking expected decision-frame production, and aligning phase timeouts with LLM retry behavior.

**Architecture:** Add structured diagnostics to `AgentState` as audit data separate from `DecisionFrame`. Nodes that call the LLM record success/error metadata and frame health warnings; replay and payload surfaces include diagnostics alongside frames and routes.

**Tech Stack:** Python, Pydantic, pytest, LangGraph fallback-compatible state models.

---

### Task 1: LLM And Node Diagnostics

**Files:**
- Modify: `src/state.py`
- Modify: `src/nodes/plan.py`
- Modify: `src/nodes/reflect.py`
- Modify: `src/new_agent.py`
- Test: `tests/test_decision_frame.py`

- [ ] **Step 1: Write failing tests**

Add tests proving `plan_fix` records exception type/message, and successful planning records a success diagnostic.

- [ ] **Step 2: Run tests to verify failure**

Run: `python3 -m pytest -p no:cacheprovider tests/test_decision_frame.py::test_plan_fix_records_llm_diagnostic_on_timeout tests/test_decision_frame.py::test_plan_fix_records_successful_llm_diagnostic -q`
Expected: FAIL because `AgentState` has no `node_diagnostics`.

- [ ] **Step 3: Implement diagnostics**

Add `AgentState.node_diagnostics`, `_record_node_diagnostic`, and call it from plan/reflect LLM success/error paths. Include `node`, `event`, `status`, `elapsed_seconds`, `error_type`, `error`, `prompt_tokens_estimate`, and `response_tokens_estimate` where available.

- [ ] **Step 4: Run tests to verify pass**

Run: `python3 -m pytest -p no:cacheprovider tests/test_decision_frame.py::test_plan_fix_records_llm_diagnostic_on_timeout tests/test_decision_frame.py::test_plan_fix_records_successful_llm_diagnostic -q`
Expected: PASS.

### Task 2: Replay And Payload Visibility

**Files:**
- Modify: `src/new_agent.py`
- Modify: `src/run_store.py`
- Test: `tests/test_run_store.py`
- Test: `tests/test_decision_frame.py`

- [ ] **Step 1: Write failing tests**

Add tests proving payload, saved replay JSON, and replay markdown expose diagnostics.

- [ ] **Step 2: Run tests to verify failure**

Run: `python3 -m pytest -p no:cacheprovider tests/test_run_store.py::test_replay_run_includes_node_diagnostics tests/test_run_store.py::test_format_replay_markdown_includes_node_diagnostics tests/test_decision_frame.py::test_agent_payload_exposes_node_diagnostics -q`
Expected: FAIL because diagnostics are not surfaced.

- [ ] **Step 3: Implement replay/payload support**

Include diagnostics in `agent_payload_from_state`, `_save_trace`, `summarize_replay`, and markdown formatting as timeline entries.

- [ ] **Step 4: Run tests to verify pass**

Run: same command as Step 2. Expected: PASS.

### Task 3: Decision-Frame Health Checks

**Files:**
- Modify: `src/state.py`
- Modify: `src/nodes/plan.py`
- Modify: `src/nodes/reflect.py`
- Test: `tests/test_decision_frame.py`

- [ ] **Step 1: Write failing tests**

Add tests proving plan/reflect warn when LLM output normalizes without a fresh expected-stage frame.

- [ ] **Step 2: Run tests to verify failure**

Run: `python3 -m pytest -p no:cacheprovider tests/test_decision_frame.py::test_plan_fix_records_frame_health_warning_for_legacy_output tests/test_decision_frame.py::test_reflect_records_frame_health_warning_for_legacy_output -q`
Expected: FAIL because frame health warnings are not recorded.

- [ ] **Step 3: Implement frame health helper**

Add a helper that records `decision_warnings` entries with `warning_type="frame_health"`, `node`, `expected_stage`, and `reason`. Call it when legacy flat output lacks an explicit `decision_frame`, while keeping backward compatibility.

- [ ] **Step 4: Run tests to verify pass**

Run: same command as Step 2. Expected: PASS.

### Task 4: Timeout Alignment

**Files:**
- Modify: `src/graph.py`
- Test: `tests/test_new_agent.py`

- [ ] **Step 1: Run existing failing test**

Run: `python3 -m pytest -p no:cacheprovider tests/test_new_agent.py::test_llm_phase_timeouts_cover_retry_window -q`
Expected: FAIL with `90.0 >= 120.0`.

- [ ] **Step 2: Implement timeout alignment**

Set `plan_fix` and `reflect_on_failure` phase timeouts to at least 150 seconds.

- [ ] **Step 3: Run test to verify pass**

Run: same command as Step 1. Expected: PASS.

### Task 5: Final Verification

- [ ] Run focused tests:
`python3 -m pytest -p no:cacheprovider tests/test_decision_frame.py tests/test_run_store.py tests/test_new_agent.py::test_llm_phase_timeouts_cover_retry_window -q`

- [ ] Run compile check:
`PYTHONPYCACHEPREFIX=/tmp/repopilot-pycache python3 -m py_compile src/state.py src/new_agent.py src/run_store.py src/nodes/plan.py src/nodes/reflect.py src/graph.py`

- [ ] Run lint:
`ruff check src/state.py src/new_agent.py src/run_store.py src/nodes/plan.py src/nodes/reflect.py src/graph.py tests/test_decision_frame.py tests/test_run_store.py tests/test_new_agent.py --select=E,F,I --ignore=E501 --cache-dir /tmp/repopilot-ruff-cache`
