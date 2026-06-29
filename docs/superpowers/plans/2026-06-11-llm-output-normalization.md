# LLM Output Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent agent-v2 from failing when the LLM returns common schema-adjacent shapes such as string evidence where list evidence is expected.

**Architecture:** Keep internal `DecisionFrame`/`PlanDecision`/`ReflectDecision` values strict after validation, but make input boundaries tolerant. Core frame fields normalize in `src/state.py`; node-specific outer fields normalize in plan/reflect nodes; eval reports surface node diagnostics when no frame exists.

**Tech Stack:** Python, Pydantic v2 validators, pytest, ruff.

---

### Task 1: Core DecisionFrame List Normalization

**Files:**
- Modify: `src/state.py`
- Test: `tests/test_decision_schemas.py`

- [ ] Add tests for `Hypothesis.evidence`, `DecisionFrame.evidence`, and `DecisionFrame.next_checks` accepting `str` and `None`.
- [ ] Verify the tests fail before implementation.
- [ ] Implement minimal Pydantic `field_validator(mode="before")` logic in `src/state.py`.
- [ ] Run `python3 -B -m pytest -p no:cacheprovider tests/test_decision_schemas.py -q`.
- [ ] Run `ruff check src/state.py tests/test_decision_schemas.py --select=E,F,I --ignore=E501 --cache-dir /tmp/repopilot-ruff-cache`.

### Task 2: Plan/Reflect Outer List Normalization

**Files:**
- Modify: `src/nodes/plan.py`
- Modify: `src/nodes/reflect.py`
- Test: `tests/test_decision_frame.py`

- [ ] Add tests for `plan.files` and `reflect.files_that_also_need_changes` accepting a single string.
- [ ] Verify the tests fail before implementation.
- [ ] Normalize these outer fields to `list[str]` before constructing `PlanDecision` / `ReflectDecision`.
- [ ] Run `python3 -B -m pytest -p no:cacheprovider tests/test_decision_frame.py -q`.
- [ ] Run `ruff check src/nodes/plan.py src/nodes/reflect.py tests/test_decision_frame.py --select=E,F,I --ignore=E501 --cache-dir /tmp/repopilot-ruff-cache`.

### Task 3: Eval Report Node Diagnostics

**Files:**
- Modify: `eval/report.py`
- Test: `tests/test_eval_report.py`

- [ ] Add a test where an agent-v2 replay has only `node_diagnostic` timeline entries and no decision frame.
- [ ] Verify the test fails before implementation.
- [ ] Render node diagnostics in `eval_summary.md` replay diagnostics.
- [ ] Run `python3 -B -m pytest -p no:cacheprovider tests/test_eval_report.py -q`.
- [ ] Run `ruff check eval/report.py tests/test_eval_report.py --select=E,F,I --ignore=E501 --cache-dir /tmp/repopilot-ruff-cache`.

### Task 4: Integration Verification

- [ ] Run the combined focused tests for schema, decision frame, run store, new agent timeout guards, and eval report.
- [ ] Run compile checks for touched Python files.
- [ ] Run ruff for touched Python files.
- [ ] Optionally rerun one live agent-v2 eval to confirm the original `PlanDecision` validation failure no longer stops the run at string evidence.
