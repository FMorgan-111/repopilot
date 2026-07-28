# Simple Patch-Edits Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make RepoPilot reliably produce PatchGate-approved patches by asking Gemini and Opus for only exact `patch_edits`, with a fixed Gemini, Gemini, Opus, Opus, Opus semantic retry order.

**Architecture:** Treat a response containing `patch_edits` as the model-facing patch variant. Parse those edits first, run the existing PatchGate once, then construct the existing internal `PlanDecision` and `DecisionFrame` so downstream routing stays unchanged; retain the current tool and stop variants. Reuse the existing repair ledger and escalation policy, changing only which failure reasons bypass the two-Gemini threshold.

**Tech Stack:** Python 3.12, Pydantic, pytest, existing RepoPilot PLAN state machine and PatchGate.

## Global Constraints

- The patch-producing model response has one required top-level field: `patch_edits`.
- Gemini and Opus use the same `patch_edits[{file_path, search, replace}]` response shape.
- Runtime, not the model, derives `plan`, `files`, `test_command`, `DecisionFrame`, risk, confidence, hashes, approval, provider, model, and repair-round metadata.
- Extra top-level narrative fields do not grant authority and do not reject otherwise valid `patch_edits`.
- Existing `node_target` and file-path aliases remain accepted for compatibility but are not advertised by the prompt.
- PatchGate remains the single execution boundary; add no validator, receipt, parser, state machine, or output format.
- Correction feedback contains only the first actionable issue.
- Normal semantic failure order is Gemini, Gemini, Opus, Opus, Opus; exhausted primary gateway transport retries may switch to Opus immediately.
- Stop the paid smoke run if zero of three fixed SWE-bench Verified instances generate a patch.
- Do not edit the user-owned `run_trace.py` in the older worktree.

---

### Task 1: Patch-only model contract with runtime-derived routing state

**Files:**
- Modify: `src/patch_authorization.py`
- Modify: `src/nodes/plan.py`
- Test: `tests/test_patch_authorization.py`
- Test: `tests/test_unified_plan_protocol.py`
- Test: `tests/test_decision_frame.py`
- Test: `tests/test_escalation_packet.py`

**Interfaces:**
- Consumes: existing `_raw_edits(value)`, `validate_patch_batch(state, plan, batch)`, `PlanDecision`, and `DecisionFrame`.
- Produces: `authorize_plan_patch(state, response)` accepts a bare patch response and returns the same `PatchAuthorizationOutcome(status="accepted", decision=PlanDecision)` used downstream.

- [ ] **Step 1: Write failing authorization tests for the bare response and ignored narrative metadata**

Add focused tests equivalent to:

```python
def test_bare_patch_edits_are_authorized_with_runtime_metadata(exact_repair_state):
    state = _bind(exact_repair_state)
    response = {
        "patch_edits": [{
            "file_path": "src/widget.py",
            "search": "return 'old-sentinel'",
            "replace": "return 'new-sentinel'",
        }]
    }

    result = authorize_plan_patch(state, response)

    assert result.status == "accepted"
    assert result.decision.files == ["src/widget.py"]
    assert result.decision.test_command == ""
    assert result.decision.decision_frame.stage == "plan"
    assert result.decision.decision_frame.recommended_action == "execute"
    assert result.decision.decision_frame.risk == "unknown"
    assert result.decision.decision_frame.confidence == 0.0


def test_patch_edits_ignore_untrusted_top_level_narrative(exact_repair_state):
    state = _bind(exact_repair_state)
    response = {
        "patch_edits": [{
            "file_path": "src/widget.py",
            "search": "return 'old-sentinel'",
            "replace": "return 'new-sentinel'",
        }],
        "commentary": "model prose must not authorize or reject the edit",
        "decision_frame": {"recommended_action": "stop", "surprise": True},
    }

    assert authorize_plan_patch(state, response).status == "accepted"
```

- [ ] **Step 2: Write failing PLAN integration and prompt-contract tests**

Make one fake `llm_call` return only the bare response above and assert:

```python
assert result.current_phase == Phase.EXECUTE
assert [edit.file_path for edit in result.patch_edits] == ["src/widget.py"]
assert result.decision_frame.stage == "plan"
assert result.decision_frame.recommended_action == "execute"
```

Replace old prompt assertions with:

```python
assert "patch_edits" in plan_node.PLAN_SYSTEM
assert "file_path" in plan_node.PLAN_SYSTEM
assert "search" in plan_node.PLAN_SYSTEM
assert "replace" in plan_node.PLAN_SYSTEM
for field in ("decision_frame", "recommended_action", "confidence", "risk", "test_command"):
    assert field not in plan_node.PLAN_SYSTEM
```

- [ ] **Step 3: Run the focused tests and confirm the old envelope-first behavior fails**

Run:

```bash
/Users/morgan/Documents/Codex/2026-07-17/https-github-com-fmorgan-111-repopilot/work/repopilot/.venv/bin/python -m pytest -q tests/test_patch_authorization.py tests/test_unified_plan_protocol.py tests/test_decision_frame.py tests/test_escalation_packet.py
```

Expected: the new bare-response and prompt-contract assertions fail before production changes.

- [ ] **Step 4: Implement the minimal patch-first authorization branch**

In `authorize_plan_patch`, when `"patch_edits" in response`, call `_raw_edits` before inspecting model-authored envelope fields. Reject invalid or empty edit lists through the existing `_reject`; otherwise create the existing `RepairPlan` and `VerifiedEditBatch`, call `validate_patch_batch`, and build the accepted internal decision from runtime values:

```python
summary = f"Apply {len(state.patch_edits)} validated structured edit(s)."
canonical = PlanDecision.model_validate({
    "plan": summary,
    "patch": "",
    "patch_edits": [edit.model_dump(mode="json") for edit in state.patch_edits],
    "files": list(target_files),
    "test_command": "",
    "decision_frame": DecisionFrame(
        stage="plan",
        summary=summary,
        recommended_action="execute",
        risk="unknown",
        confidence=0.0,
    ).model_dump(mode="json"),
})
```

Keep the existing no-`patch_edits` envelope path for legacy non-execute routing. Do not add a new schema or validator.

- [ ] **Step 5: Simplify only the model-facing patch instructions**

Set `PLAN_SYSTEM` to request the patch variant in this exact structural form while retaining the existing tool and stop variants:

```python
PLAN_SYSTEM = (
    "You are RepoPilot's patch planner. Return exactly one JSON response variant. "
    "For a code change, return only {\"patch_edits\":[{\"file_path\":\"...\","
    "\"search\":\"...\",\"replace\":\"...\"}]}. Copy search text verbatim from "
    "approved file context and make it match exactly once. Use kind='tool' with one "
    "tool_intent only when one specific repository fact is missing. Use kind='stop' "
    "with stop_reason only when no safe repair is possible."
)
```

Update final-attempt and context-pressure copy so it asks for `patch_edits` directly and does not ask the model to author `recommended_action`, `next_checks`, hypothesis IDs, risk, or confidence. Keep the context itself; remove only contradictory response-format instructions.

- [ ] **Step 6: Run the focused tests and commit**

Run the Step 3 command. Expected: PASS.

Commit:

```bash
git add src/patch_authorization.py src/nodes/plan.py tests/test_patch_authorization.py tests/test_unified_plan_protocol.py tests/test_decision_frame.py tests/test_escalation_packet.py
git commit -m "fix: accept simple patch edit responses"
```

### Task 2: First-actionable correction feedback

**Files:**
- Modify: `src/patch_authorization.py`
- Modify: `src/nodes/plan.py`
- Modify: `src/repair_rounds.py`
- Test: `tests/test_patch_authorization.py`
- Test: `tests/test_unified_plan_protocol.py`
- Test: `tests/test_decision_frame.py`
- Test: `tests/test_repair_rounds.py`

**Interfaces:**
- Consumes: `PatchAuthorizationIssue`, `render_patch_correction`, `_record_model_correctable_plan_failure`, and `record_failed_repair_round`.
- Produces: one bounded correction issue, a specific `failure_reason`, and the existing `repair_round_failed` diagnostic containing that reason.

- [ ] **Step 1: Write failing tests for one issue and specific failure reason**

Change the correction test to assert only the first item survives:

```python
payload = json.loads(render_patch_correction(issues))
assert len(payload) == 1
assert payload[0]["code"] == "invalid_0"
assert "invalid_1" not in render_patch_correction(issues)
```

In an integration test, return an invalid edit and assert:

```python
assert result.failure_reason == "invalid_replacement"
assert result.node_diagnostics[-1]["failure_reason"] == "invalid_replacement"
assert not any(
    item.get("reason") == "missing_explicit_decision_frame"
    for item in result.decision_warnings
)
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run:

```bash
/Users/morgan/Documents/Codex/2026-07-17/https-github-com-fmorgan-111-repopilot/work/repopilot/.venv/bin/python -m pytest -q tests/test_patch_authorization.py tests/test_unified_plan_protocol.py tests/test_decision_frame.py tests/test_repair_rounds.py
```

Expected: old multi-issue feedback, generic reason, or synthetic frame warning fails the new assertions.

- [ ] **Step 3: Keep only the first issue with existing types**

Change `_reject` and `render_patch_correction` to use `issues[:1]`. Serialize the same four existing fields without a new packet or truncation framework:

```python
issue = list(issues[:1])[0]
return json.dumps([{
    "code": issue.code,
    "file_path": issue.file_path,
    "message": issue.message,
    "correction_context": issue.correction_context,
}], ensure_ascii=False, separators=(",", ":"), sort_keys=True)
```

- [ ] **Step 4: Record the specific issue through the existing ledger**

In `_record_model_correctable_plan_failure`, compute:

```python
failure_reason = issues[0].code if issues else reason
```

Pass that value to `record_failed_repair_round`, label the next prompt `CORRECTION FOR THE NEXT PATCH_EDITS RESPONSE`, and treat the deterministic runtime frame as present so no missing-model-frame warning is emitted. Add `failure_reason=str(failure_reason or "")[:64]` to the existing `repair_round_failed` diagnostic call.

- [ ] **Step 5: Run focused tests and commit**

Run the Step 2 command. Expected: PASS.

Commit:

```bash
git add src/patch_authorization.py src/nodes/plan.py src/repair_rounds.py tests/test_patch_authorization.py tests/test_unified_plan_protocol.py tests/test_decision_frame.py tests/test_repair_rounds.py
git commit -m "fix: return actionable patch corrections"
```

### Task 3: Two Gemini attempts before Opus escalation

**Files:**
- Modify: `src/model_policy.py`
- Test: `tests/test_model_policy.py`
- Test: `tests/test_model_escalation_integration.py`
- Test: `tests/test_unified_plan_protocol.py`

**Interfaces:**
- Consumes: existing `should_escalate(state, immediate_reason)`, `record_failed_repair_round`, and `primary_failed_repair_rounds`.
- Produces: semantic invalid/empty responses obey the existing threshold while `primary_gateway_unavailable_after_retries` remains immediate.

- [ ] **Step 1: Write failing policy and integration tests**

Assert invalid structured and empty completions are not immediate after only one failed Gemini transaction:

```python
state.primary_failed_repair_rounds = 1
for reason in (
    "invalid_structured_response_after_retries",
    "empty_completion_after_retries",
):
    assert should_escalate(state, immediate_reason=reason).escalate is False
```

Replace the old immediate-invalid integration expectation with three PLAN calls whose fake responses are invalid, invalid, then valid. Assert provider/model order:

```python
assert [(provider, model) for _, _, model, provider in calls] == [
    ("primary", PRIMARY_MODEL),
    ("primary", PRIMARY_MODEL),
    ("escalation", ESCALATION_MODEL),
]
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run:

```bash
/Users/morgan/Documents/Codex/2026-07-17/https-github-com-fmorgan-111-repopilot/work/repopilot/.venv/bin/python -m pytest -q tests/test_model_policy.py tests/test_model_escalation_integration.py tests/test_unified_plan_protocol.py
```

Expected: the first invalid response currently escalates immediately, so the new assertions fail.

- [ ] **Step 3: Narrow the existing immediate-reason set**

Change only:

```python
_IMMEDIATE_REASONS = frozenset({"primary_gateway_unavailable_after_retries"})
```

Do not change retry counters, ledger models, no-progress logic, or the 503 path.

- [ ] **Step 4: Run focused and complete tests**

Run the Step 2 command, then:

```bash
/Users/morgan/Documents/Codex/2026-07-17/https-github-com-fmorgan-111-repopilot/work/repopilot/.venv/bin/python -m pytest -q
```

Expected: focused tests PASS; complete suite reports no failures.

- [ ] **Step 5: Commit**

```bash
git add src/model_policy.py tests/test_model_policy.py tests/test_model_escalation_integration.py tests/test_unified_plan_protocol.py
git commit -m "fix: retry Gemini before Opus escalation"
```

### Task 4: Fixed three-instance patch-generation smoke gate

**Files:**
- Create: `/Users/morgan/Documents/Codex/2026-07-17/https-github-com-fmorgan-111-repopilot/work/repopilot-eval-home/eval/runs/simple-patch-smoke-20260728/cohort_ids.txt`
- Create at runtime: `/Users/morgan/Documents/Codex/2026-07-17/https-github-com-fmorgan-111-repopilot/work/repopilot-eval-home/eval/runs/simple-patch-smoke-20260728/results.json`
- Create at runtime: `/Users/morgan/Documents/Codex/2026-07-17/https-github-com-fmorgan-111-repopilot/work/repopilot-eval-home/eval/runs/simple-patch-smoke-20260728/predictions.jsonl`

**Interfaces:**
- Consumes: the first three IDs from the existing fixed 20-instance cohort and the existing patch-first eval command.
- Produces: evidence that at least one of three instances reaches a generated patch, or a zero-of-three stop signal for diagnosis.

- [ ] **Step 1: Copy exactly the first three fixed cohort IDs into the smoke cohort file**

Use the existing cohort at `/Users/morgan/Documents/Codex/2026-07-17/https-github-com-fmorgan-111-repopilot/work/repopilot-eval-home/eval/runs/patch-first-20-20260728/cohort_ids.txt`; do not resample.

- [ ] **Step 2: Run the existing patch-first harness with the configured Gemini and Opus endpoints**

Use `--dataset swe-bench-verified`, the three-ID file, `--max-retries 4`, and `--token-budget 100000`, writing results and predictions under `simple-patch-smoke-20260728`.

Expected: all three instances finish or expose a concrete reproducible failure; never continue to a 20-instance paid run from a zero-of-three result.

- [ ] **Step 3: Inspect results and apply the gate**

Count instances with a generated patch. If the count is at least one, record the smoke evidence. If it is zero, stop and report the exact first failure reason and provider sequence without adding speculative fallback code.
