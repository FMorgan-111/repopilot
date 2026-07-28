# Patch-First MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make RepoPilot return the first PatchGate-approved, `git apply --check`-clean patch immediately, while reporting test and coverage status independently.

**Architecture:** Add an explicit `patch_only` run mode that is enabled by default at the public `agent_v2` entry point but disabled on directly constructed `AgentState` objects for backwards-compatible node tests and full-PR callers. Patch-only PLAN terminates after PatchGate preflight without mutating the checkout. Reporting validates either the clean preflight snapshot or an already-applied live snapshot, exports that patch independently from terminal coverage, and preserves the existing coverage-proven meaning of `success` and `fix_applied`.

**Tech Stack:** Python 3.10+, Pydantic v2, LangGraph/fallback graph, pytest, Git/PatchGate, SWE-bench JSONL and Pydantic OCI contracts.

## Global Constraints

- Public `agent_v2` runs default to `patch_only=True`; directly constructed `AgentState` defaults to `patch_only=False` so existing full-flow node tests and explicit callers remain available.
- Keep the existing five-attempt provider order unchanged: Gemini attempts 1–2, then Opus attempts 3–5.
- A returned patch must come from PatchGate state that already passed canonical path/edit validation and `git apply --check`; never export arbitrary working-tree diffs or raw `state.patch_content` without revalidation.
- Patch-only success stops after PatchGate acceptance and must not invoke EXECUTE, VERIFY, COVERAGE, COMMIT, tests, or mutate the repository checkout.
- `patch_generated` is exactly equivalent to a nonempty safe `model_patch`.
- `tests_passed` is `true` only for a matching successful `FixAttempt`, `false` only for a matching completed failed attempt, and `null` when tests were not run for the returned patch.
- Existing `success` and `fix_applied` retain their strict terminal-coverage meaning; generating a patch alone does not set either field to true.
- SWE-bench predictions must retain a generated patch even when `agent_success=false`; official resolution remains scorer-owned.
- Preserve evaluator-metadata filtering, secret redaction, path/symlink/binary/size/edit limits, and the current PatchGate preflight.
- Do not modify, stage, or commit `run_trace.py`.

---

### Task 1: Terminate patch-only runs at the PatchGate boundary

**Files:**
- Modify: `src/state.py`
- Modify: `src/new_agent.py`
- Modify: `src/nodes/plan.py`
- Test: `tests/test_decision_frame.py`
- Test: `tests/test_new_agent.py`

**Interfaces:**
- Produces: `AgentState.patch_only: bool` with a direct-construction default of `False`.
- Produces: an `agent_v2` keyword parameter named `patch_only` with default `True`, passed into the new state.
- Produces: accepted PLAN state `Phase.DONE` when `state.patch_only` is true; the decision frame is marked consumed so graph routing returns `END`.

- [ ] **Step 1: Write the failing patch-only PLAN test**

Add a focused test beside `test_plan_fix_records_search_replace_patch_edits`:

```python
async def test_patch_only_plan_stops_after_preflight_without_mutating_checkout(
    exact_repair_state, monkeypatch
):
    async def fake_llm_call(*_args, **_kwargs):
        return _exact_plan_response()

    exact_repair_state.patch_only = True
    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)

    next_state = await plan_node.plan_fix(exact_repair_state)

    assert next_state.current_phase == new_agent.Phase.DONE
    assert next_state.tool_patch_approval is not None
    assert next_state.patch_content.startswith("diff --git a/src/widget.py")
    assert new_agent.route_from_state(next_state) == new_agent.END
    checkout_diff = subprocess.run(
        ["git", "-C", next_state.repo_path, "diff", "--exit-code"],
        check=False,
        capture_output=True,
    )
    assert checkout_diff.returncode == 0
```

Import `subprocess` in the test module. Extend the existing state-capture test for `agent_v2` with `assert captured["state"].patch_only is True`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/test_decision_frame.py::test_patch_only_plan_stops_after_preflight_without_mutating_checkout \
  tests/test_new_agent.py::test_agent_v2_initializes_primary_model_from_provider
```

Expected: the new test fails because `AgentState` has no `patch_only` behavior and accepted PLAN still routes to `EXECUTE`; the capture assertion fails because `agent_v2` does not initialize patch-only mode.

- [ ] **Step 3: Implement the minimal mode and terminal transition**

Add to `AgentState` near `skip_commit`:

```python
patch_only: bool = False
```

Add `patch_only: bool = True` immediately after the existing `seed` parameter, and add `patch_only=patch_only` to the existing `AgentState` constructor. The resulting function header is:

```python
async def agent_v2(
    issue_url: str,
    max_retries: int = DEFAULT_AGENT_V2_MAX_RETRIES,
    token_budget: int = DEFAULT_AGENT_V2_TOKEN_BUDGET,
    save_final_run: bool = False,
    skip_commit: bool = False,
    seed: dict | None = None,
    patch_only: bool = True,
) -> dict:
```

At the accepted PLAN exit, preserve normal full-flow behavior and terminate only patch-only runs:

```python
frame.recommended_action = "execute"
if state.patch_only:
    state.current_phase = Phase.DONE
    state.decision_route_checked_frame_id = frame.frame_id
else:
    state.current_phase = Phase.EXECUTE
return state
```

- [ ] **Step 4: Run focused and neighboring PLAN tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/test_decision_frame.py \
  tests/test_unified_plan_protocol.py \
  tests/test_model_escalation_integration.py \
  tests/test_new_agent.py::test_agent_v2_initializes_primary_model_from_provider
```

Expected: all selected tests pass; existing directly constructed states still route accepted plans to EXECUTE.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/state.py src/new_agent.py src/nodes/plan.py \
  tests/test_decision_frame.py tests/test_new_agent.py
git commit -m "feat: stop patch-only runs after preflight"
```

---

### Task 2: Export trusted patches independently from coverage

**Files:**
- Modify: `src/new_agent.py`
- Test: `tests/test_live_binding.py`
- Test: `tests/test_new_agent.py`

**Interfaces:**
- Produces: `_validated_patch_for_report(state: AgentState) -> str`.
- Produces: `_tests_passed_for_patch(state: AgentState, patch: str) -> bool | None`.
- Adds public payload fields `patch_generated: bool` and `tests_passed: bool | None`.

- [ ] **Step 1: Write failing reporting tests**

Change the no-proof live-approval test to require the approved patch while leaving terminal success false:

```python
def test_terminal_prediction_exports_approved_live_patch_without_proof(tmp_path):
    state = _approved_state(tmp_path)
    state.current_phase = Phase.DONE

    payload = agent_payload_from_state(state, turns_taken=1)

    assert payload["success"] is False
    assert payload["patch_generated"] is True
    assert payload["tests_passed"] is None
    assert payload["model_patch"] == state.patch_content
```

Add a clean preflight test using `exact_repair_state`, `RepairPlan`, `begin_repair_round`, `bind_repair_round_author`, and `validate_patch_batch`; assert the same four fields before `apply_approved_patch` is called. Add a matching-attempt test with this complete receipt snapshot and assert `tests_passed is True`:

```python
approval = state.tool_patch_approval
assert approval is not None
state.fix_attempts.append(
    FixAttempt(
        patch_content=state.patch_content,
        patch_edits=[edit.model_copy(deep=True) for edit in state.patch_edits],
        test_result="pytest completed",
        success=True,
        patch_gate_fingerprint=approval.patch_gate_fingerprint,
        repair_provider=state.authorized_repair_provider,
        repair_model=state.authorized_repair_model,
        repair_round_id=state.authorized_repair_round_id,
    )
)
```

Add a tamper test that modifies `state.patch_content` after approval and asserts `patch_generated is False`, `tests_passed is None`, and `model_patch == ""`.

- [ ] **Step 2: Run reporting tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/test_live_binding.py \
  tests/test_new_agent.py -k 'agent_payload or patch_only'
```

Expected: approved patches without terminal coverage still serialize as empty and the two new signal keys are absent.

- [ ] **Step 3: Implement trusted patch selection and independent signals**

Import `subprocess`, `validate_live_coverage_binding`, and `revalidate_approved_patch`. Add:

```python
def _validated_patch_for_report(state: AgentState) -> str:
    preflight_state = state.model_copy(deep=True)
    try:
        revalidate_approved_patch(preflight_state)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
        pass
    else:
        return safe_prediction_patch(preflight_state.patch_content)

    live_state = state.model_copy(deep=True)
    try:
        binding = validate_live_coverage_binding(live_state)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
        return ""
    return safe_prediction_patch(binding.patch_content)


def _tests_passed_for_patch(state: AgentState, patch: str) -> bool | None:
    if not patch:
        return None
    for attempt in reversed(state.fix_attempts):
        if attempt.patch_content == patch and attempt.test_result:
            return attempt.success
    return None
```

In `agent_payload_from_state`, calculate once and export independently:

```python
model_patch = _validated_patch_for_report(state)
tests_passed = _tests_passed_for_patch(state, model_patch)
"patch_generated": bool(model_patch),
"tests_passed": tests_passed,
"model_patch": model_patch,
```

In the graph-crash payload override, keep `success=False`, `fix_applied=False`, and `final_phase="CRASHED"`, but remove the override that resets `model_patch`; leave the two independent patch fields from `agent_payload_from_state` unchanged.

- [ ] **Step 4: Run reporting, crash, and evaluator-safety tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/test_live_binding.py \
  tests/test_new_agent.py \
  tests/test_evaluator_safety.py
```

Expected: all selected tests pass; valid preflight/live patches survive absent coverage, while tampered, unapproved, evaluator-only, and drifted patches remain empty.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/new_agent.py tests/test_live_binding.py tests/test_new_agent.py
git commit -m "feat: report approved patches before test verification"
```

---

### Task 3: Preserve patch-first predictions through SWE-bench and OCI contracts

**Files:**
- Modify: `eval/agent_v2_harness.py`
- Modify: `eval/oci_contract.py`
- Test: `tests/test_agent_v2_eval.py`
- Test: `tests/test_oci_contract.py`
- Test: `tests/test_oci_aggregate.py`

**Interfaces:**
- Adds `patch_generated: StrictBool` and `tests_passed: StrictBool | None` to `ResultRecord`.
- Keeps `success == agent_success` and verified coverage as the internal success gate.
- Allows `model_patch` when `patch_generated=True`, `agent_success=False`, and at least one successful model invocation exists.

- [ ] **Step 1: Write failing eval and OCI contract tests**

Add an eval test whose fake agent returns `success=False`, `patch_generated=True`, `tests_passed=None`, and a safe nonempty `model_patch`; assert the result and written SWE-bench prediction retain the patch while `agent_success is False`.

Change the OCI contract case that currently rejects any patch with `agent_success=False` into an accepted patch-only record:

```python
def test_patch_only_result_allows_unverified_model_patch() -> None:
    record = _result_record_model().model_validate(
        _result_payload(
            model_patch="diff --git a/a.py b/a.py\n",
            patch_generated=True,
            tests_passed=None,
            success=False,
            agent_success=False,
        )
    )

    assert record.patch_generated is True
    assert record.tests_passed is None
    assert record.agent_success is False
```

Add parameterized invalid cases for `(model_patch="", patch_generated=True)` and `(model_patch=nonempty, patch_generated=False)`. Retain the existing invalid cases for empty/all-failed model invocation history. Extend the OCI aggregate fixture helper to serialize the two new fields and add one aggregate test for a patch-only unresolved result.

- [ ] **Step 2: Run eval/OCI tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/test_agent_v2_eval.py \
  tests/test_oci_contract.py \
  tests/test_oci_aggregate.py
```

Expected: patch-only OCI validation fails because nonempty `model_patch` currently requires `agent_success`; strict records reject the two new fields until the schema is updated.

- [ ] **Step 3: Implement safe result projection and relaxed patch contract**

In the eval harness, sanitize the patch and optional boolean once:

```python
model_patch = _safe_model_patch(payload.get("model_patch", ""))
raw_tests_passed = payload.get("tests_passed")
tests_passed = raw_tests_passed if type(raw_tests_passed) is bool else None
```

Add to every normal result:

```python
"patch_generated": bool(model_patch),
"tests_passed": tests_passed,
```

Use the same `model_patch` variable for SWE-bench output. Add `patch_generated=False` and `tests_passed=None` to `safe_failed_sample_result`.

In `ResultRecord`, add:

```python
patch_generated: StrictBool
tests_passed: StrictBool | None
```

Replace the coverage-coupled patch rule with exact signal consistency:

```python
if self.patch_generated is not bool(self.model_patch):
    raise ValueError("patch_generated must match model_patch presence")
```

Retain the existing requirements that a nonempty patch has model invocation history and at least one `status="ok"` invocation. Do not change the verified-coverage requirement for `agent_success`.

- [ ] **Step 4: Run eval/OCI tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/test_agent_v2_eval.py \
  tests/test_oci_contract.py \
  tests/test_oci_aggregate.py \
  tests/test_swe_bench.py
```

Expected: all selected tests pass; prediction JSONL contains the patch independent of internal test/coverage success.

- [ ] **Step 5: Run full verification and commit Task 3**

Run:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src eval tests
git diff --check
git status --short
```

Expected: full pytest exits 0 with only the existing sqlite-vec fallback warning; Ruff and diff checks exit 0; `run_trace.py` remains the only unrelated user modification.

Commit only Task 3 files:

```bash
git add eval/agent_v2_harness.py eval/oci_contract.py \
  tests/test_agent_v2_eval.py tests/test_oci_contract.py tests/test_oci_aggregate.py
git commit -m "feat: preserve patch-first swe-bench predictions"
```
