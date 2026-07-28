# Unified Patch-Edits Five-Round Switching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Gemini and Opus author patches through one `PlanDecision.patch_edits` protocol, switch one-way after two failed Gemini repair rounds, and use the default five-transaction sequence Gemini, Gemini, Opus, Opus, Opus.

**Architecture:** Add a small repair-round ledger shared by PLAN and VERIFY, plus a provider-neutral patch-authorization adapter that converts untrusted raw edits into the existing internal `RepairPlan`/`VerifiedEditBatch` types before PatchGate. PLAN uses one prompt, schema, tool loop, correction loop, and model-call path for both providers; accepted PatchGate attribution is frozen into `FixAttempt`, and VERIFY consumes each failed transaction exactly once. REFLECT stays advisory and provider-neutral, while historical two-stage Opus state remains deserializable but is no longer a production model protocol.

**Tech Stack:** Python 3.11+, Pydantic v2, LangGraph-compatible state routing, pytest/pytest-asyncio, httpx, existing PatchGate and OCI sandbox.

## Global Constraints

- The only model-facing patch payload is `PlanDecision.patch_edits`; new model output must leave legacy `patch` empty.
- Both providers receive the same PLAN system prompt, bounded user-prompt builder, response union, tool policy, and raw validation.
- PatchGate is the sole producer of canonical `patch_content`, exact-only `patch_edits`, and `tool_patch_approval`.
- Default `max_retries=4` means one initial transaction plus four retries: Gemini, Gemini, Opus, Opus, Opus; no sixth transaction is permitted.
- Tool calls, context collection, ask-user, cancellation, gateway fallback, REFLECT, and summary generation do not consume semantic repair retries.
- A model-correctable PLAN or VERIFY failure consumes one retry exactly once; environment failures consume none.
- After two counted primary failures, escalation wins over repeated-signature and assertion-diversity early stops whenever a retry remains.
- Immediate primary gateway or token-reserve fallback rebinds the current round to Opus without granting a new transaction.
- Cancellation must propagate by identity and must not be recorded as a model error or charged as token usage.
- Model output cannot author provider/model attribution, round IDs, hashes, exactness flags, evaluator metadata, raw HTTP material, or credentials.
- Historical `RepairPlan`, `VerifiedEditBatch`, `opus_no_progress_rounds`, and saved states remain loadable with safe defaults.
- Keep the configured model names, API endpoints, environment-key handling, and credential flow unchanged.
- Keep the existing 100,000-token budget and reserve policy unchanged; five transactions do not grant more tokens.
- Do not redesign API cache, coverage/test generation, SWE-bench scoring, summary content/length, or the reasoning-tool allowlist; only remove provider-specific stopping and enforce the shared existing tool cap.
- Do not run paid evals in this implementation plan.
- Never stage or modify the pre-existing user change in `run_trace.py`.
- Tasks 1–7 are one indivisible feature-branch migration: Task 1 exposes the new budget before the ledger/protocol lands, so no intermediate commit may be merged, deployed, or documented as complete. Only the Task 7 full-suite checkpoint is release-eligible; the small commits exist solely for review and rollback.

---

## File Structure

- Create `src/repair_rounds.py`: monotonic repair-round IDs, author binding, frozen attribution, central authorization retirement, exactly-once retry consumption, and threshold escalation.
- Create `src/patch_authorization.py`: raw edit validation, deterministic internal compatibility objects, PatchGate invocation, correction rendering, and authorization retirement.
- Modify `src/state.py`: canonical retry constants and backward-compatible round/attribution fields.
- Modify `src/model_policy.py`: select Opus from the explicit primary-failure counter instead of no-progress signatures.
- Modify `src/summary_safety.py`: expose the shared bounded/redacted model-context sanitizer used by PLAN and REFLECT.
- Modify `src/patch_gate.py`: attach an explicit failure class where each issue is created, atomically install runtime attribution on acceptance, bind it into the gate fingerprint, and revalidate it before execution.
- Modify `src/nodes/plan.py`: one provider-neutral PLAN call and full-decision correction loop.
- Modify `src/nodes/execute.py`: require frozen PatchGate authorization and copy attribution into `FixAttempt`.
- Modify `src/nodes/verify.py`: route all non-infrastructure failures through the shared round ledger.
- Modify `src/nodes/reflect.py` and `src/reasoning_loop.py`: provider-neutral reflection/tool behavior with no Opus-only terminal.
- Modify `src/repair_flow.py`: remove dead production two-stage Opus generation/correction code while retaining exact source helpers still used by PatchGate and EXECUTE.
- Modify production/eval entry points and README so the canonical default and upper bound are 4 everywhere.
- Add focused tests in `tests/test_repair_rounds.py` and `tests/test_patch_authorization.py`; update existing policy, PLAN, VERIFY, API, eval, OCI, reflection, and security tests.

---

### Task 1: Propagate the canonical five-transaction default

**Files:**
- Modify: `src/state.py:28,822`
- Modify: `src/new_agent.py:35-55,406-433,639-645`
- Modify: `src/cli.py:8-50`
- Modify: `src/main.py:31,275-293,358-362,404-434`
- Modify: `eval/agent_v2_harness.py:34-42,590-605,767-798,910-925`
- Modify: `eval/harness.py:38-44,1202-1262`
- Modify: `eval/oci_runner.py:31-43,407-421`
- Modify: `.env.example:28-33`
- Modify: `.github/workflows/swe-bench-oci-eval.yml:70-80`
- Modify: `README.md:70-83,145-157,250-260,493-497`
- Modify: `docs/PRODUCT_POSITIONING.md:120-130`
- Test: `tests/test_new_agent.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_main.py`
- Test: `tests/test_api_security.py`
- Test: `tests/test_agent_v2_eval.py`
- Test: `tests/test_eval_harness.py`
- Test: `tests/test_oci_runner.py`
- Test: `tests/test_documentation_contract.py`
- Test: `tests/test_swe_bench_oci_workflow.py`

**Interfaces:**
- Produces: `DEFAULT_AGENT_V2_MAX_RETRIES: int = 4` and `MAX_AGENT_V2_MAX_RETRIES: int = 4` from `src.state`.
- Consumers: every active production/eval default and every public API/saved-run upper-bound check.

- [ ] **Step 1: Write failing default-propagation tests**

Add direct assertions instead of duplicating magic numbers in mocks:

```python
import inspect

from src import new_agent
from src.state import (
    DEFAULT_AGENT_V2_MAX_RETRIES,
    MAX_AGENT_V2_MAX_RETRIES,
    AgentState,
)


def test_five_transaction_retry_defaults_are_canonical():
    assert DEFAULT_AGENT_V2_MAX_RETRIES == 4
    assert MAX_AGENT_V2_MAX_RETRIES == 4
    assert AgentState(issue_url="https://github.com/a/b/issues/1").max_retries == 4
    assert inspect.signature(new_agent.agent_v2).parameters["max_retries"].default == 4
    assert (
        inspect.signature(new_agent.intelligent_analyze_issue)
        .parameters["max_retries"]
        .default
        == 4
    )
```

Update the existing CLI default test to capture both defaults:

```python
def test_cli_issue_uses_agent_v2_defaults(monkeypatch, capsys):
    calls = []

    async def fake_agent_v2(issue_url, max_retries, token_budget):
        calls.append((max_retries, token_budget))
        return {"success": True}

    monkeypatch.setattr(cli, "agent_v2", fake_agent_v2)
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["https://github.com/acme/widget/issues/42", "--json"])

    assert exc_info.value.code == 0
    assert calls == [(4, 100_000)]
    assert json.loads(capsys.readouterr().out) == {"success": True}
```

Update API boundary cases and add resume/compatibility coverage:

```python
(main.AgentV2Request, "max_retries", (0, 4), (-1, 5)),


async def test_resume_accepts_saved_state_with_five_transaction_budget(
    monkeypatch, api_client
):
    state = _state()
    state.max_retries = 4
    observed = []

    monkeypatch.setattr(main, "load_run", lambda _run_id: state)

    async def fake_resume(run_id, human_answer, *, state=None):
        observed.append((run_id, human_answer, state))
        return {"success": True, "error": None}

    monkeypatch.setattr(main, "resume_agent_v2", fake_resume)
    response = await api_client.post(
        "/agent/v2/resume",
        headers=AUTHORIZED_HEADERS,
        json={"run_id": state.trace_id, "human_answer": "Proceed."},
    )

    assert response.status_code == 200
    assert observed == [(state.trace_id, "Proceed.", state)]
    assert observed[0][2].max_retries == 4
```

Change the intelligent compatibility expectation from 3 to 4, add a CLI `--help` assertion containing `default: 4` and `five transactions total`, and add signature/default assertions for both eval harnesses plus an OCI runner test that captures `max_retries=4` from its injected `agent_runner`.

Extend `tests/test_documentation_contract.py` to assert that current docs advertise `max_retries` 0–4 and the two-failed-Gemini-round threshold. Assert that README, `.env.example`, and the active OCI workflow no longer expose the unused `REPOPILOT_ESCALATION_AFTER_NO_PROGRESS` variable.

- [ ] **Step 2: Run the focused tests and verify the old defaults fail**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_new_agent.py tests/test_cli.py tests/test_main.py tests/test_api_security.py tests/test_agent_v2_eval.py tests/test_eval_harness.py tests/test_oci_runner.py tests/test_documentation_contract.py tests/test_swe_bench_oci_workflow.py
```

Expected: FAIL with assertions showing defaults/bounds of 3, API rejection of 4, saved-run rejection of 4, compatibility clamp 3, or OCI/eval propagation 3.

- [ ] **Step 3: Add the constants and replace every active default/boundary**

In `src/state.py`:

```python
DEFAULT_AGENT_V2_MAX_RETRIES = 4
MAX_AGENT_V2_MAX_RETRIES = 4
DEFAULT_AGENT_V2_TOKEN_BUDGET = 100_000
```

Use `DEFAULT_AGENT_V2_MAX_RETRIES` for `AgentState.max_retries`, both functions in `src/new_agent.py`, CLI parsing/help, both eval harness function/CLI defaults, and the OCI generation call. Use `MAX_AGENT_V2_MAX_RETRIES` for `AgentV2Request.le`, authorized saved-run validation, and the `/intelligent-agent` clamp:

```python
class AgentV2Request(StrictRequestModel):
    issue_url: str = Field(max_length=2_048)
    max_retries: int = Field(
        default=DEFAULT_AGENT_V2_MAX_RETRIES,
        ge=0,
        le=MAX_AGENT_V2_MAX_RETRIES,
        strict=True,
    )


max_retries=min(req.max_turns, MAX_AGENT_V2_MAX_RETRIES)
```

Update the CLI help to say `Retries after the initial transaction (default: 4; five transactions total)`, README's public range/default/guardrail descriptions, and `docs/PRODUCT_POSITIONING.md`. Remove the unused `REPOPILOT_ESCALATION_AFTER_NO_PROGRESS` setting from `.env.example` and the active OCI workflow; document the deterministic two-failed-primary-round threshold instead. Historical plan documents and explicit test/eval fixtures that intentionally pass 3 stay unchanged.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run the Step 2 command again.

Expected: PASS; API accepts 4/rejects 5, a saved run with 4 resumes, compatibility/eval/OCI paths pass 4, and explicit fixtures requesting 3 remain valid.

- [ ] **Step 5: Commit the default propagation**

```bash
git add src/state.py src/new_agent.py src/cli.py src/main.py eval/agent_v2_harness.py eval/harness.py eval/oci_runner.py .env.example .github/workflows/swe-bench-oci-eval.yml README.md docs/PRODUCT_POSITIONING.md tests/test_new_agent.py tests/test_cli.py tests/test_main.py tests/test_api_security.py tests/test_agent_v2_eval.py tests/test_eval_harness.py tests/test_oci_runner.py tests/test_documentation_contract.py tests/test_swe_bench_oci_workflow.py
git commit -m "feat: use five repair transactions by default"
```

---

### Task 2: Add monotonic repair-round accounting and deterministic switching

**Files:**
- Create: `src/repair_rounds.py`
- Modify: `src/state.py:63-71,160-240,334-365,818-895`
- Modify: `src/model_policy.py:14-180`
- Modify: `src/main.py:275-300`
- Test: `tests/test_repair_rounds.py`
- Test: `tests/test_model_policy.py`
- Test: `tests/test_run_store.py`
- Test: `tests/test_api_security.py`

**Interfaces:**
- Produces: `validate_repair_round_state(state: AgentState) -> None`, `begin_repair_round(state: AgentState) -> int`, `bind_repair_round_author(state: AgentState) -> None`, `freeze_authorized_repair_round(state: AgentState) -> None`, `clear_authorized_repair_round(state: AgentState) -> None`, `retire_patch_authorization(state, *, preserve_authorized_attribution=False) -> None`, and the fully specified `record_failed_repair_round` function in Step 4.
- Consumes: existing `apply_escalation`, `should_escalate`, `AgentState.retry_count/max_retries`, and active provider/model fields.
- Later tasks consume frozen attribution fields and the exactly-once helper. `freeze_authorized_repair_round` is an internal PatchGate acceptance primitive, not a PLAN/fixture shortcut; Task 3 binds the same snapshot into the approval fingerprint.

- [ ] **Step 1: Write failing state, idempotency, and provider-sequence tests**

Create `tests/test_repair_rounds.py` with a configured escalation fixture and these core cases:

```python
from types import SimpleNamespace


def enable_escalation(monkeypatch):
    monkeypatch.setattr(model_policy, "escalation_is_configured", lambda: True)
    monkeypatch.setattr(
        model_policy,
        "get_model_config",
        lambda provider: SimpleNamespace(
            model=(
                "claude-opus-4-8:stable"
                if provider == "escalation"
                else "gemini-3.5-flash:stable"
            )
        ),
    )


def test_failed_rounds_are_counted_once_and_switch_after_two_primary_failures(
    monkeypatch,
):
    enable_escalation(monkeypatch)
    state = AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        max_retries=4,
        active_model="gemini-3.5-flash:stable",
    )

    first = begin_repair_round(state)
    bind_repair_round_author(state)
    one = record_failed_repair_round(
        state,
        round_id=first,
        provider="primary",
        model="gemini-3.5-flash:stable",
        failure_reason="invalid_edit",
        retry_phase=Phase.PLAN,
    )
    duplicate = record_failed_repair_round(
        state,
        round_id=first,
        provider="primary",
        model="gemini-3.5-flash:stable",
        failure_reason="invalid_edit",
        retry_phase=Phase.PLAN,
    )

    assert one == RepairRetryDecision(counted=True, retry_allowed=True, round_id=1)
    assert duplicate.counted is False
    assert state.retry_count == 1
    assert state.primary_failed_repair_rounds == 1
    assert state.active_provider == "primary"

    second = begin_repair_round(state)
    bind_repair_round_author(state)
    record_failed_repair_round(
        state,
        round_id=second,
        provider="primary",
        model="gemini-3.5-flash:stable",
        failure_reason="tests_failed",
        retry_phase=Phase.REFLECT,
    )

    assert second == 2
    assert state.retry_count == 2
    assert state.primary_failed_repair_rounds == 2
    assert state.active_provider == "escalation"
    assert state.active_model == "claude-opus-4-8:stable"
    assert state.escalation_reason == "primary_repair_round_limit"
```

Also test that context/tool re-entry reuses the open round, a same-round gateway fallback rebinds author without incrementing the ID/counters, no retry after the fifth failure creates a sixth ID, `record_progress()` does not reset `primary_failed_repair_rounds`, and historical state without new fields validates with zero/empty defaults. Add a run-store round trip containing an open round, a frozen authorization, an attributed `FixAttempt`, and an already-counted ID; after reload, repeating `record_failed_repair_round` must not increment anything (Task 7 later proves the full VERIFY/resume path). Add a terminal round trip with `current_repair_round_id` deliberately cleared and prove no sixth round can be allocated. Reject a tampered saved state such as `current=sequence=6,last_counted=5,max_retries=4` before resume or model invocation.

Use these exact cases so the boundary is auditable:

- `test_max_retry_values_allocate_exact_transaction_sequences` parameterized for 0, 1, 3, and 4;
- `test_same_round_failure_is_idempotent_after_run_store_roundtrip`;
- `test_terminal_roundtrip_with_cleared_current_id_rejects_sixth_round`;
- `test_future_open_round_is_rejected_before_resume`;
- `test_gateway_fallback_rebinds_same_round_without_counter_change`;
- `test_patch_authorization_retirement_is_atomic_and_test_only_can_preserve_attribution`;
- `test_record_progress_does_not_reset_primary_failed_rounds`.

- [ ] **Step 2: Run the tests and verify the new ledger is missing**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_repair_rounds.py tests/test_model_policy.py tests/test_run_store.py tests/test_api_security.py
```

Expected: FAIL on missing state fields/module/functions and old no-progress-based escalation behavior.

- [ ] **Step 3: Add backward-compatible attribution fields**

Add to `FixAttempt`:

```python
repair_round_id: int = Field(default=0, ge=0)
repair_provider: Literal["primary", "escalation"] | None = None
repair_model: str = Field(default="", max_length=200)
```

Permit `None` only as the compatibility sentinel when `repair_round_id == 0`; every newly created positive-ID attempt must have a nonempty runtime provider/model. This lets resumed historical Opus attempts be migrated once without being silently mislabeled as primary.

Add to `AgentState`:

```python
primary_failed_repair_rounds: int = Field(default=0, ge=0)
repair_round_sequence: int = Field(default=0, ge=0)
current_repair_round_id: int = Field(default=0, ge=0)
current_repair_provider: Literal["primary", "escalation"] | None = None
current_repair_model: str = Field(default="", max_length=200)
authorized_repair_round_id: int = Field(default=0, ge=0)
authorized_repair_provider: Literal["primary", "escalation"] | None = None
authorized_repair_model: str = Field(default="", max_length=200)
last_counted_repair_round_id: int = Field(default=0, ge=0)
repair_correction_context: str = Field(default="", max_length=8_000)
```

Do not delete `opus_no_progress_rounds`; it remains a historical serialization field.

- [ ] **Step 4: Implement the single-owner repair-round ledger**

Create `src/repair_rounds.py` around this exact public surface:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .model_policy import apply_escalation, should_escalate
from .state import AgentState, Phase, _record_node_diagnostic

RepairProvider = Literal["primary", "escalation"]


@dataclass(frozen=True)
class RepairRetryDecision:
    counted: bool
    retry_allowed: bool
    round_id: int


def validate_repair_round_state(state: AgentState) -> None:
    """Reject impossible/tampered ledger relationships before routing."""


def begin_repair_round(state: AgentState) -> int:
    validate_repair_round_state(state)
    if (
        state.current_repair_round_id > 0
        and state.current_repair_round_id > state.last_counted_repair_round_id
    ):
        return state.current_repair_round_id
    if (
        max(state.repair_round_sequence, state.last_counted_repair_round_id)
        >= state.max_retries + 1
    ):
        raise ValueError("repair retry budget is exhausted")
    state.repair_round_sequence = max(
        state.repair_round_sequence,
        state.last_counted_repair_round_id,
    )
    state.repair_round_sequence += 1
    state.current_repair_round_id = state.repair_round_sequence
    bind_repair_round_author(state)
    return state.current_repair_round_id


def bind_repair_round_author(state: AgentState) -> None:
    if state.current_repair_round_id <= 0:
        raise ValueError("repair round must begin before author binding")
    state.current_repair_provider = state.active_provider
    state.current_repair_model = state.active_model


def freeze_authorized_repair_round(state: AgentState) -> None:
    if state.current_repair_round_id <= 0:
        raise ValueError("authorized patch requires an active repair round")
    if state.current_repair_provider is None or not state.current_repair_model:
        raise ValueError("authorized patch requires runtime author binding")
    if state.tool_patch_approval is None or not state.patch_content or not state.patch_edits:
        raise ValueError("authorized repair attribution requires PatchGate output")
    proposed = (
        state.current_repair_round_id,
        state.current_repair_provider,
        state.current_repair_model,
    )
    existing = (
        state.authorized_repair_round_id,
        state.authorized_repair_provider,
        state.authorized_repair_model,
    )
    if state.authorized_repair_round_id > 0:
        if existing != proposed:
            raise ValueError("authorized repair attribution is already frozen")
        return
    state.authorized_repair_round_id = state.current_repair_round_id
    state.authorized_repair_provider = state.current_repair_provider
    state.authorized_repair_model = state.current_repair_model


def clear_authorized_repair_round(state: AgentState) -> None:
    state.authorized_repair_round_id = 0
    state.authorized_repair_provider = None
    state.authorized_repair_model = ""


def retire_patch_authorization(
    state: AgentState,
    *,
    preserve_authorized_attribution: bool = False,
) -> None:
    state.patch_content = ""
    state.patch_edits = []
    state.active_repair_plan = None
    state.tool_patch_approval = None
    state.generated_test_approvals = []
    if not preserve_authorized_attribution:
        clear_authorized_repair_round(state)
```

`validate_repair_round_state` must enforce `0 <= retry_count <= max_retries`, `0 <= last_counted <= sequence <= max_retries + 1`, `last_counted - retry_count in {0, 1}`, and `sequence - last_counted in {0, 1}`. `current_repair_round_id` is zero or equals `sequence`; an open current ID never exceeds the transaction cap; `primary_failed_repair_rounds <= last_counted`; current/authorized positive IDs have complete runtime attribution; and an authorized ID is not beyond `sequence` and has a gate approval. At the start of this validator, migrate only a truly legacy empty ledger (`sequence=current=last_counted=0`) by seeding `sequence=last_counted=retry_count` when `retry_count > 0`; do not repair partially populated or future-ID states. Call the same validator from authorized saved-run validation in `src/main.py`, so corrupted state fails before graph resume.

Implement `record_failed_repair_round` with this signature and ordering:

```python
def record_failed_repair_round(
    state: AgentState,
    *,
    round_id: int,
    provider: RepairProvider,
    model: str,
    failure_reason: str,
    retry_phase: Phase,
    immediate_reason: str = "",
) -> RepairRetryDecision:
```

It must reject non-positive IDs. For every `round_id <= last_counted_repair_round_id`, return `counted=False` before checking mutable current attribution, without changing phase, counters, provider, or diagnostics; set `retry_allowed=False` only when the state is already terminal (`Phase.FAILURE` or `Phase.FAILED`). For a new ID, first call `validate_repair_round_state`; it must then equal the open `current_repair_round_id`, be no greater than `repair_round_sequence`, and its provider/model arguments must exactly match the bound `current_repair_provider/current_repair_model`, otherwise raise a state-integrity error without accounting. For a valid new failure, set the last-counted ID before routing and increment the primary counter only for `provider == "primary"`. If `retry_count >= max_retries`, set the existing bounded budget-exhaustion failure reason and `Phase.FAILURE`, return `retry_allowed=False`, and do not allocate or imply another round. Otherwise increment `retry_count` exactly once, close `current_repair_round_id`, call `should_escalate(state, immediate_reason=immediate_reason)`, apply an approved decision, set `retry_phase`, and return `retry_allowed=True`. PLAN passes the cached runtime `invoked_provider/invoked_model`; VERIFY passes the attempt attribution only after verifying it matches the frozen snapshot. Record only bounded provider/model/round/counter diagnostics. Allocation is guarded by full transactions (`max_retries + 1`), not by `retry_count`: after the fourth failed transaction under `max_retries=4`, `retry_count == 4` but the fifth initial-plus-retries transaction must still be allocated. A direct call to `begin_repair_round` after the fifth terminal failure must raise without incrementing `repair_round_sequence`, including a resumed terminal state whose `current_repair_round_id` was cleared.

- [ ] **Step 5: Make ModelPolicy use the explicit threshold**

Add `primary_repair_round_limit` and `primary_gateway_unavailable_after_retries` to `APPROVED_ESCALATION_REASONS`. Add only `primary_gateway_unavailable_after_retries` to `_IMMEDIATE_REASONS`; the threshold reason must be derived from the counted state, while the existing bounded empty/invalid-completion reasons remain immediate. Then replace the no-progress selection branch with:

```python
if state.primary_failed_repair_rounds >= 2:
    return EscalationDecision(
        escalate=True,
        reason="primary_repair_round_limit",
    )
```

Keep no-progress signatures as diagnostics, but they must no longer independently choose the repair provider. Evaluate policy in this order: approved immediate reason, two-counted-primary threshold, then primary token reserve.

Update `apply_escalation` telemetry to report only the bounded explicit reason plus current repair/primary-failure counters; do not label a threshold switch with the legacy `no_progress_rounds` value.

- [ ] **Step 6: Run the focused tests and commit**

Run the Step 2 command.

Expected: PASS, including historical-state validation and exactly-once counter behavior.

```bash
git add src/state.py src/model_policy.py src/repair_rounds.py src/main.py tests/test_repair_rounds.py tests/test_model_policy.py tests/test_run_store.py tests/test_api_security.py
git commit -m "feat: add shared repair round accounting"
```

---

### Task 3: Add provider-neutral raw edit validation and PatchGate authorization

**Files:**
- Create: `src/patch_authorization.py`
- Modify: `src/patch_gate.py:66-108,332-605,640-742`
- Modify: `tests/conftest.py`
- Test: `tests/test_patch_authorization.py`
- Test: `tests/test_patch_gate.py`
- Test: `tests/test_decision_schemas.py`
- Test: `tests/test_live_binding.py`
- Test: `tests/test_test_generator.py`

**Interfaces:**
- Produces: `PatchAuthorizationIssue`, `PatchAuthorizationOutcome`, `authorize_plan_patch(state, response)`, `render_patch_correction(issues)`, and a re-export of `retire_patch_authorization`.
- Consumes: `PlanDecision` raw dictionaries, internal `RepairPlan`/`VerifiedEditBatch`, `validate_patch_batch`, and repair-round attribution helpers.
- PLAN consumes only the outcome's sanitized decision/status/issues; it never reparses the raw response or constructs trusted edits itself.
- Later protocol tests consume the shared `exact_repair_state` pytest fixture created here.

- [ ] **Step 1: Write failing adapter tests before changing PatchGate**

Add one shared exact-checkout fixture to `tests/conftest.py`:

```python
@pytest.fixture
def exact_repair_state(tmp_path):
    from src.state import AgentState, FileInfo, Phase

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Test"],
        check=True,
    )
    (repo / "src").mkdir()
    source = "def widget():\n    return 'old-sentinel'\n"
    (repo / "src" / "widget.py").write_text(source, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "--all"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
    ref = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        issue_title="Return the new sentinel",
        issue_body="widget must return new-sentinel",
        owner="acme",
        repo="widget",
        repo_path=str(repo),
        repo_ref=ref,
        current_phase=Phase.PLAN,
        relevant_files=[FileInfo(path="src/widget.py", content=source)],
        skip_commit=True,
    )
```

Add `import subprocess` to `tests/conftest.py`. Then parameterize the active provider:

```python
@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("primary", "gemini-3.5-flash:stable"),
        ("escalation", "claude-opus-4-8:stable"),
    ],
)
def test_both_providers_receive_identical_exact_patch_authorization(
    exact_repair_state, provider, model
):
    state = exact_repair_state
    state.active_provider = provider
    state.active_model = model
    begin_repair_round(state)
    bind_repair_round_author(state)
    response = {
        "kind": "plan",
        "plan": "Return the new sentinel.",
        "patch": "",
        "patch_edits": [
            {
                "file": "src/widget.py",
                "search": "return 'old-sentinel'",
                "replace": "return 'new-sentinel'",
                "resolved_target_symbol": "model.must.not.author.this",
                "expected_content_sha256": "f" * 64,
                "exact_only": True,
            }
        ],
        "files": ["src/widget.py"],
        "test_command": "pytest -q",
        "decision_frame": {
            "stage": "plan",
            "summary": "Update the sentinel.",
            "recommended_action": "execute",
            "risk": "low",
            "confidence": 0.9,
        },
    }

    result = authorize_plan_patch(state, response)

    assert result.status == "accepted"
    assert state.patch_content.startswith("diff --git")
    assert state.patch_edits[0].exact_only is True
    assert state.patch_edits[0].expected_content_sha256 != "f" * 64
    assert state.tool_patch_approval is not None
    assert state.authorized_repair_provider == provider
    assert state.authorized_repair_model == model
```

Parameterize rejected raw proposals: conflicting `file`/`path`, both anchors, neither anchor, empty replace, `replace_all=True`, nonempty legacy `patch`, unified diff text, evaluator markers, credential-shaped text, unknown non-identity fields, execute without edits, and non-execute action carrying edits. Add acceptance for equal duplicate path aliases, plus bounds for 16 edits, 8 unique paths, and existing field-size limits. Add a trust-metadata case that supplies provider/model/round IDs, approval/fingerprint fields, hashes, and exactness claims at the response/edit levels; those values must be discarded and the runtime-bound author, round, preimage hashes, and approval must remain authoritative. Assert bounded issue code/message, no canonical patch, no approval, and no active plan.

Add `test_mixed_valid_and_invalid_edits_reject_the_entire_batch`: one valid edit plus one malformed edit must return issues with no internal plan/batch authorization, no PatchGate acceptance call, and empty patch/edits/active plan/approval/frozen attribution. Partial filtering or authorization is forbidden.

Add a gate test that marks exact-checkout/preimage drift, active-plan invariant failure, and filesystem/subprocess exceptions as `environment`, while search-missing, ambiguous target, forbidden path, empty patch, and generated `git apply --check` rejection are `model_correctable` at creation time. Add a positive-attribution tamper regression: changing any authorized round/provider/model scalar after acceptance must make `revalidate_approved_patch` fail because the snapshot is part of the gate fingerprint. Add a `test_only=True` regression proving both its accepted and rejected paths cannot clear or replace frozen production attribution.

Update accepted-production gate fixtures in `tests/test_patch_gate.py`, `tests/test_live_binding.py`, and `tests/test_test_generator.py` to begin/bind a runtime round before validation and assert the gate itself freezes the author. Rejection-only fixtures may stay unbound because no executable approval is produced. A non-`test_only` batch that would otherwise be accepted but lacks a positive complete runtime binding must fail closed as `environment`; `test_only=True` remains exempt and never changes production attribution.

- [ ] **Step 2: Run adapter and gate tests to verify failure**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_patch_authorization.py tests/test_patch_gate.py tests/test_decision_schemas.py tests/test_live_binding.py tests/test_test_generator.py
```

Expected: FAIL because the adapter and explicit PatchGate failure class do not exist, and Gemini currently bypasses PatchGate.

- [ ] **Step 3: Add explicit failure classes to PatchGate issues**

Extend `PatchGateIssue` without changing existing issue codes:

```python
RepairFailureClass = Literal["model_correctable", "environment"]


class PatchGateIssue(BaseModel):
    code: Literal[
        "target_missing",
        "target_ambiguous",
        "search_missing",
        "scope_violation",
        "apply_failed",
        "empty_patch",
        "generated_artifact",
        "binary_artifact",
        "oversized_file",
    ]
    file_path: str
    message: str
    correction_context: str = ""
    failure_class: RepairFailureClass = "model_correctable"
```

Add a keyword-only `failure_class` parameter to `_issue`. Mark exact root/base unavailability, live preimage drift, filesystem/subprocess exceptions, and post-checkout instability as `environment` where those issues are created. Keep malformed targets, scope, missing/ambiguous anchors, empty changes, artifacts, size, and generated-patch apply rejection `model_correctable`. If both appear, the adapter returns `environment`.

Split `_tree_entry`/blob lookup outcomes instead of treating every nonzero git result as a missing path: a successful `ls-tree` with empty stdout is a real absent target and remains model-correctable, while invalid/missing base refs, nonzero `ls-tree`/`show`, decode failures, timeouts, and invocation/I/O errors create an environment issue. Cover this with `test_absent_tracked_path_is_model_correctable` and `test_git_tree_or_blob_failure_is_environment`.

- [ ] **Step 4: Implement the raw-validation adapter**

Use these frozen public result types in `src/patch_authorization.py`:

```python
PatchAuthorizationStatus = Literal[
    "not_requested",
    "accepted",
    "model_correctable",
    "environment",
]


class PatchAuthorizationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1, max_length=64)
    file_path: str = Field(default="", max_length=500)
    message: str = Field(min_length=1, max_length=500)
    correction_context: str = Field(default="", max_length=1_200)
    failure_class: RepairFailureClass


class PatchAuthorizationOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: PatchAuthorizationStatus
    decision: PlanDecision | None = None
    issues: tuple[PatchAuthorizationIssue, ...] = ()
```

Add a model-level invariant: `accepted` and `not_requested` require a sanitized decision and no issues; `model_correctable` and `environment` require at least one issue and expose no executable decision. This keeps PLAN from guessing how to combine partial output.

`authorize_plan_patch(state, response)` must inspect raw `patch`, `patch_edits`, path aliases, anchors, replacement, `replace_all`, extra fields, identity metadata, evaluator/secret/unified-diff content, and raw `decision_frame.recommended_action` before `PatchEdit` or `_normalize_plan_decision` can coerce/drop anything. It is the only parse path: after raw checks, validate a copy whose legacy `patch` is empty and whose executable `patch_edits` are temporarily empty to obtain the safe `PlanDecision` envelope; never send the raw mapping through the lossy normalizer a second time. It must:

1. Retire patch content, edits, active plan, generated-test approvals, PatchGate approval, and frozen attribution atomically.
2. Return `not_requested` only for a valid non-execute action with no executable payload, carrying that sanitized `PlanDecision` in `outcome.decision`.
3. Return bounded model-correctable issues for malformed proposals.
4. Discard every model-authored trust/identity field after detecting it: at minimum `resolved_target_symbol`, `expected_content_sha256`, `exact_only`, provider/model/round attribution, base/digest/fingerprint values, and approval fields. Reject unknown non-identity fields rather than silently accepting them. No discarded value may enter state, an internal compatibility object, diagnostics, or the correction prompt.
5. Construct `VerifiedEdit` values with the fixed runtime intent `Model-proposed structured edit.`.
6. Construct a deterministic `RepairPlan` with exact deduplicated target paths/node symbols and these fixed narratives:

```python
RepairPlan(
    root_cause="Runtime-authorized structured edit proposal.",
    target_files=target_files,
    target_symbols=target_symbols,
    required_behavior="Apply only the exact validated structured edits.",
    regression_test_strategy="Run the planner-selected bounded test command.",
    rejected_approaches=[],
)
```

7. Install that plan and call `validate_patch_batch`. On acceptance, PatchGate snapshots the runtime-bound round/provider/model, includes that snapshot in `_fingerprint`, and installs canonical output, approval, and authorized attribution as one acceptance operation; `_rebuild_approved_snapshot`/`revalidate_approved_patch` must recompute the same binding and reject scalar tampering. Return the safe decision with `patch=""`, deep-copied canonical `state.patch_edits`, and `files` replaced by the exact deduplicated authorized target paths rather than retaining the model's claimed scope. Convert gate issues without reclassifying their messages. A production rejection clears frozen attribution. `test_only=True` may continue temporarily replacing gate patch/output under the existing test-generator capture/restore transaction, but it must never install, clear, or replace `authorized_repair_*` on success or failure. Implement that attribution distinction explicitly in PatchGate rather than relying on restore order.

Re-export and use the central `repair_rounds.retire_patch_authorization` operation rather than duplicating field clearing in the adapter or PatchGate. It is idempotent and deliberately does not close/count the current repair round or erase the bounded correction context. The adapter calls it before processing every raw full-decision proposal; PLAN also calls it for full-transaction outcomes such as `kind=stop` that never enter the adapter. PatchGate production rejection calls it with the default; its `test_only=True` clearing calls it with `preserve_authorized_attribution=True`, so temporary gate output may clear while `authorized_repair_*` stays intact for the outer test-generator restore transaction. This placement in `repair_rounds.py` avoids a `patch_authorization` ↔ `patch_gate` import cycle.

`render_patch_correction` serializes at most 16 issue objects containing only code, file path, concise message, and bounded real-code window; its total output is capped at 8,000 characters and is stored in `state.repair_correction_context` only for the next full PLAN decision.

- [ ] **Step 5: Run tests and commit the authorization boundary**

Run the Step 2 command.

Expected: PASS for both providers, all raw rejection cases, gate classification, trusted metadata replacement, and atomic retirement.

```bash
git add src/patch_authorization.py src/patch_gate.py tests/conftest.py tests/test_patch_authorization.py tests/test_patch_gate.py tests/test_decision_schemas.py tests/test_live_binding.py tests/test_test_generator.py
git commit -m "feat: authorize all model edits through PatchGate"
```

---

### Task 4: Collapse PLAN to one provider-neutral full-decision loop

**Files:**
- Modify: `src/nodes/plan.py:1-86,240-332,545-715,794-1188,1198-1408`
- Modify: `src/escalation.py:60-90,714-722`
- Modify: `src/http_client.py:125-150`
- Modify: `src/summary_safety.py:1-55`
- Create: `tests/test_unified_plan_protocol.py`
- Test: `tests/test_decision_frame.py`
- Test: `tests/test_patch_retry.py`
- Test: `tests/test_model_escalation_integration.py`
- Test: `tests/test_escalation_packet.py`
- Test: `tests/test_outcome_summary.py`
- Test: `tests/test_context_loop_brake.py`
- Test: `tests/test_search_hallucination_gate.py`
- Test: `tests/test_error_episodes.py`

**Interfaces:**
- Consumes: `begin_repair_round`, `bind_repair_round_author`, `record_failed_repair_round`, `authorize_plan_patch`, `render_patch_correction`, and `retire_patch_authorization`.
- Produces: one `PLAN_SYSTEM`, one `build_plan_user_prompt`, one `llm_call(PLAN_SYSTEM, user, model=invoked_model, provider=invoked_provider)` path using the policy snapshot chosen immediately before the call, and `summary_safety.sanitize_model_context(value, limit, denied_literals=()) -> str` for both model-facing nodes.
- Guarantees: every model-correctable correction exits the current transaction and lets the graph enter a new full PLAN transaction.

- [ ] **Step 1: Replace old two-stage expectations with failing unified-loop tests**

Delete tests that require `generate_opus_repair`, `request_verified_edit_correction`, two local `VerifiedEditBatch` corrections, `ESCALATED_PLAN_SYSTEM`, `_unlocatable_edits`, a separate search-correction cap, or an Opus-only two-round stop. Put the provider-neutral protocol matrix in the new focused `tests/test_unified_plan_protocol.py`, and replace the old cases with tests that assert:

```python
def plan_response(*, search: str) -> dict:
    return {
        "kind": "plan",
        "plan": "Return the new sentinel.",
        "patch": "",
        "patch_edits": [
            {
                "file": "src/widget.py",
                "search": search,
                "replace": "return 'new-sentinel'",
            }
        ],
        "files": ["src/widget.py"],
        "test_command": "pytest -q",
        "decision_frame": {
            "stage": "plan",
            "summary": "Update the sentinel.",
            "recommended_action": "execute",
            "risk": "low",
            "confidence": 0.9,
        },
    }


async def test_patch_gate_rejection_requests_one_new_full_plan_decision(
    exact_repair_state, monkeypatch
):
    responses = [
        plan_response(search="missing first anchor"),
        plan_response(search="return 'old-sentinel'"),
    ]
    calls = []

    async def fake_llm(system, user, model=None, *, provider="primary", **_kwargs):
        calls.append((system, user, model, provider))
        return responses.pop(0)

    monkeypatch.setattr(plan_node, "llm_call", fake_llm)
    first = await plan_node.plan_fix(exact_repair_state)

    assert first.current_phase == Phase.PLAN
    assert first.retry_count == 1
    assert first.repair_round_sequence == 1
    assert first.tool_patch_approval is None

    second = await plan_node.plan_fix(first)

    assert second.current_phase == Phase.EXECUTE
    assert second.repair_round_sequence == 2
    assert second.tool_patch_approval is not None
    assert len(calls) == 2
```

Add a provider-parameterized action matrix proving Gemini and already-escalated Opus use the same `PLAN_SYSTEM`, prompt builder, `PlanDecision` validation, and behavior for reasoning tool, collect-context, ask-user, stop, decision-frame, and test-command variants. Assert tool/collect/ask keep the open round and spend no retry, stop spends one shared retry, and `PatchAuthorizationOutcome.decision` preserves the bounded frame/test command without reparsing raw output. For both the initial and correction calls under each provider, inspect the final system+user prompt and assert it does not advertise `unified diff`, `replace_all`, `RepairPlan`, or `VerifiedEditBatch`, including continuity/hypothesis guidance. Add cancellation identity coverage with no model-error telemetry/token debit. Preserve tests proving exactly one bounded outcome-summary section and human/context/tool inputs.

Seed stale canonical patch/plan/approval/generated-test/frozen-attribution state before a new full transaction and assert it is atomically retired for `kind=stop`, raw rejection, gate rejection, and post-gate convergence rejection. Accepted output must be the only branch that leaves a fresh executable authorization.

Add a PLAN integration case that forces exact-root/git I/O failure inside PatchGate and asserts one model call, infrastructure routing, empty correction context, and unchanged `retry_count`, `primary_failed_repair_rounds`, and `last_counted_repair_round_id`. A mixed synthetic gate result must take environment precedence over model-correctable issues.

Name the key protocol regressions explicitly:

- `test_provider_neutral_plan_action_matrix_preserves_round_and_decision_fields`;
- `test_new_full_transaction_atomically_retires_stale_authorization`;
- `test_patchgate_environment_failure_routes_infra_without_retry_or_correction`;
- `test_mixed_patchgate_issues_prefer_environment`;
- `test_primary_503_fallback_rebinds_opus_in_same_round`;
- `test_primary_token_reserve_binds_opus_before_first_call`;
- `test_tool_reprompt_crossing_token_reserve_rebinds_opus_in_same_round`;
- `test_escalation_gateway_exhaustion_routes_environment_without_retry`;
- `test_unconfigured_gateway_or_token_fallback_never_spins_or_claims_opus`;
- `test_correction_context_is_preserved_for_tools_then_replaced_or_cleared`;
- `test_plan_cancellation_preserves_identity_without_telemetry_or_debit`.

Add an unconfigured-escalation case proving a threshold decision never sets `escalated=True` or claims an Opus model when the separate escalation key/flag is unavailable.

Update `tests/test_context_loop_brake.py` to prove an allowed tool reprompt and a collect-context round retain the current `repair_round_id` without consuming retry. Convert `tests/test_search_hallucination_gate.py` to exercise PatchGate `search_missing`, bounded real-code correction windows, shared retry consumption, and terminal global budget rather than the deleted pre-gate heuristic.

Update the PLAN fakes in `tests/test_error_episodes.py` to accept `model`/`provider` keywords. Give execute-success cases the shared exact-repo authorization fixture or stub the deterministic adapter explicitly; do not preserve a no-checkout bypass merely to keep recall tests green.

- [ ] **Step 2: Run PLAN-focused tests and verify old behavior fails**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_unified_plan_protocol.py tests/test_decision_frame.py tests/test_patch_retry.py tests/test_model_escalation_integration.py tests/test_escalation_packet.py tests/test_outcome_summary.py tests/test_context_loop_brake.py tests/test_search_hallucination_gate.py tests/test_error_episodes.py
```

Expected: FAIL because Opus still enters the two-stage branch, Gemini still stores unapproved edits, and PatchGate corrections stay inside one Opus call.

- [ ] **Step 3: Make transport exhaustion an explicit safe policy reason**

Expose `is_retryable_llm_error(exc)` from `src/http_client.py` using the current network/timeout/502/503/504 predicate. In `immediate_model_policy_reason` return `primary_gateway_unavailable_after_retries` when that retryable error escapes the gateway. Do not include status text, response bodies, URLs, or headers in state.

- [ ] **Step 4: Replace the provider-specific prompt/call setup**

Rename the common system string to `PLAN_SYSTEM`, remove the contradictory unified-diff fallback sentence and every continuity/correction instruction that advertises `unified diff`, `replace_all`, `RepairPlan`, or `VerifiedEditBatch`, and remove `ESCALATED_PLAN_SYSTEM`. Build the bounded prompt once for either provider. Define/export `sanitize_model_context(value, limit, denied_literals=())` in `src/summary_safety.py` as the public wrapper around `sanitize_summary_text`/secret redaction (do not import the private escalation `_safe_text`) and use it for issue text, failure logs, reflection text, recalled episodes, human answers, and source windows.

Immediately before every full or tool-reprompt model call:

```python
apply_escalation(state, should_escalate(state))
bind_repair_round_author(state)
invoked_provider = state.current_repair_provider
invoked_model = state.current_repair_model
assert invoked_provider is not None and invoked_model
raw_response = await llm_call(
    PLAN_SYSTEM,
    user,
    model=invoked_model,
    provider=invoked_provider,
)
```

Begin a round once at PLAN entry. Build any bounded evidence about the previous attempt, then call `retire_patch_authorization` before issuing the new full transaction so stale executable state cannot survive a stop/tool/non-execute result. Immediately before every full or tool-evidence reprompt call, evaluate/apply the normal model policy, then bind the runtime author, then call the model. Thus a tool call that crosses the primary token reserve can rebind to Opus without consuming a retry or changing the round ID. Tool evidence reprompts and context collection reuse the open ID. A primary gateway exhaustion applies escalation, rebinds author, and repeats the call within the same ID; it never calls `record_failed_repair_round`. Retry within the same ID only if `apply_escalation` actually changed `active_provider` to `escalation`: when escalation is unconfigured, gateway exhaustion takes the explicit provider-unavailable infrastructure terminal and token reserve takes the existing bounded budget terminal, with no primary spin and no false `escalated`/reason state.

Catch `CancellationDrainError` and `asyncio.CancelledError` before every broad model/tool exception handler and re-raise the same object. Cancellation must not append error telemetry, debit tokens, create correction context, or count/close the repair round. Classify retryable transport exhaustion before generic `ValueError`/structured-response failures so HTTP 502/503/504 can only take the same-round gateway fallback path.

Keep exception scopes narrow: only gateway/JSON/response-schema failures from the model boundary feed `immediate_model_policy_reason`. Patch authorization returns typed outcomes; an unexpected adapter/PatchGate programming exception is an infrastructure failure and must not be mislabeled as an invalid model completion or trigger immediate escalation.

- [ ] **Step 5: Route every PLAN outcome through the common adapter and ledger**

Remove `_drop_invalid_edits`, `_unlocatable_edits`, the `two_stage_repair` branch, synthesized Opus `PlanDecision`, local verified-edit correction loop, all production calls to `record_opus_no_progress`, `MAX_REPEATED_PATCH_BLOCKS`, `MAX_SEARCH_CORRECTIONS`, and their early-terminal branches. Repeated patch/search/assertion-diversity rejection may only atomically retire authorization and call the shared retry helper; it cannot terminate ahead of the remaining Gemini-to-Opus/global budget.

Use this branch table exactly:

| Outcome | Retry accounting | Next route |
| --- | --- | --- |
| `kind=tool` with allowed tool | none | reprompt inside same round |
| collect context | none | LOCATE/PLAN with same open round |
| ask user | none | WAITING_FOR_USER with same open round |
| hard tool stop/cancellation | none | existing terminal/direct propagation |
| primary gateway exhaustion | none | Opus call in same round after author rebind |
| escalation gateway exhaustion | authorization already retired; none | provider-unavailable infrastructure failure, no rebind loop |
| empty/invalid structured response | count once; immediate safe escalation reason if primary | new PLAN if budget remains |
| `kind=stop` or decision-frame stop | atomically retire; count once | new PLAN if budget remains |
| raw edit rejection | atomically retire; count once | new PLAN with bounded correction |
| PatchGate model-correctable rejection | atomically retire; count once | new PLAN with bounded correction |
| PatchGate environment rejection | authorization already retired; none | infrastructure failure, no correction call |
| accepted PatchGate patch | none yet | post-gate convergence checks, then EXECUTE |
| repeated failed patch/assertion target after gate | atomically retire; count once | new PLAN if budget remains |

After JSON extraction, inspect only the raw `kind` discriminator. Tool/stop variants go through the existing strict union validator; plan/legacy-outcome variants go directly to `authorize_plan_patch`, which strips trust metadata and then invokes union/envelope validation on its sanitized copy. Do not preconstruct `PatchEdit` or run the old full outcome validator on unsanitized plan data. Consume only `PatchAuthorizationOutcome.decision`; never call `_normalize_plan_decision` or parse the raw response again after the adapter. The outcome always has legacy `patch=""`; accepted decisions contain deep-copied canonical `state.patch_edits`, while non-execute decisions contain none. Never copy model-authored patch or edits over PatchGate output. `record_progress()` may reset diagnostic signatures but must not touch `primary_failed_repair_rounds`. Whenever PLAN changes `current_phase` after recording a new frame, it must also set `frame.recommended_action` to the matching route so `graph._consume_decision_recommended_phase` cannot override the correction/failure route with stale `execute` or `stop`; the rejection test must assert both `current_phase == Phase.PLAN` and `decision_frame.recommended_action == "plan"`.

Treat `repair_correction_context` as a one-next-full-decision suffix with explicit ownership. Tool reprompts, collect-context, and ask-user preserve it because the same transaction still needs the requested evidence/answer. An accepted patch, stop, or non-correctable/environment result clears the old suffix; a new unusable-response/raw/PatchGate model-correctable rejection atomically replaces it with the newly rendered bounded suffix rather than appending. After an accepted patch it must be empty, so a later VERIFY failure cannot leak an obsolete anchor correction into an unrelated transaction.

- [ ] **Step 6: Run PLAN tests and commit the unified loop**

Run the Step 2 command.

Expected: PASS; a rejected proposal causes one new full decision, both providers use the same protocol, accepted Gemini edits have exact approval, and cancellation is untouched.

```bash
git add src/nodes/plan.py src/escalation.py src/http_client.py src/summary_safety.py tests/test_unified_plan_protocol.py tests/test_decision_frame.py tests/test_patch_retry.py tests/test_model_escalation_integration.py tests/test_escalation_packet.py tests/test_outcome_summary.py tests/test_context_loop_brake.py tests/test_search_hallucination_gate.py tests/test_error_episodes.py
git commit -m "feat: unify Gemini and Opus planning"
```

---

### Task 5: Freeze attribution through EXECUTE and consume VERIFY failures once

**Files:**
- Modify: `src/nodes/execute.py:1280-1310,1345-1580`
- Modify: `src/nodes/verify.py:1-342`
- Test: `tests/test_execute_security.py`
- Test: `tests/test_oci_integration.py`
- Test: `tests/test_patch_retry.py`
- Test: `tests/test_failure_taxonomy.py`
- Test: `tests/test_convergence_diversity.py`
- Test: `tests/test_new_agent.py`
- Test: `tests/test_patch_preflight.py`

**Interfaces:**
- EXECUTE consumes frozen `authorized_repair_*` state and produces the same fields on every `FixAttempt`.
- VERIFY consumes only `FixAttempt.repair_*` attribution and calls `record_failed_repair_round` for a non-infrastructure failure.

- [ ] **Step 1: Write failing execution/verification attribution tests**

Add an exact-authorized Gemini and Opus case:

```python
@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("primary", "gemini-3.5-flash:stable"),
        ("escalation", "claude-opus-4-8:stable"),
    ],
)
async def test_execute_copies_frozen_patch_author_attribution(
    tmp_path, provider, model, monkeypatch
):
    authorized_state = _approved_state(
        tmp_path,
        command="pytest -q",
        provider=provider,
        model=model,
    )
    expected_round = authorized_state.authorized_repair_round_id
    assert expected_round > 0

    async def passing_oci(_state):
        return {
            "command": "python -m pytest -q",
            "returncode": 0,
            "stdout": "1 passed",
            "stderr": "",
            "success": True,
        }

    monkeypatch.setattr(execute_node, "_run_oci_pytest", passing_oci)

    result = await execute_node.execute_fix(authorized_state)

    attempt = result.fix_attempts[-1]
    assert attempt.repair_round_id == expected_round
    assert attempt.repair_provider == provider
    assert attempt.repair_model == model
```

Update the shared `_approved_state` helper itself to set the active provider/model and call `begin_repair_round` plus `bind_repair_round_author` before `validate_patch_batch`. Assert the accepted gate call itself installed the fingerprint-bound authorized attribution. Do not call the freeze helper or manufacture authorized fields after approval; the fixture must exercise the same atomic acceptance path as production.

Add VERIFY tests for: one failed attempt increments once even if VERIFY is resumed/called twice; one invalid Gemini proposal plus one failed Gemini verified patch reaches two primary failures and switches; Opus failures do not increment the primary counter; infrastructure/post-approval drift does not consume retry; syntax/import/assertion/general failures all share the ledger; repeated assertion sets diversity but cannot terminate before remaining Opus budget.

At minimum name and assert `test_verify_loaded_counted_attempt_is_idempotent`, `test_verify_missing_historical_attribution_is_infrastructure`, `test_invalid_then_failed_primary_patch_switches_before_next_plan`, and `test_failed_escalation_attempt_spends_global_retry_not_primary_counter`.

Update every direct `execute_fix` case in `tests/test_patch_preflight.py`: success/preflight fixtures must establish a positive fingerprint-bound authorization through the gate, while intentionally unapproved cases must now assert the common state-integrity/infrastructure failure. Remove expectations that structured/raw model edits can fall through to mutation without approval.

- [ ] **Step 2: Run focused execution/verification tests to verify failure**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_execute_security.py tests/test_oci_integration.py tests/test_patch_retry.py tests/test_failure_taxonomy.py tests/test_convergence_diversity.py tests/test_new_agent.py tests/test_patch_preflight.py
```

Expected: FAIL because `FixAttempt` attribution is not copied and VERIFY still increments retries in multiple branches or terminates on legacy convergence caps.

- [ ] **Step 3: Require frozen authorization in EXECUTE**

Replace OCI-only validation with a common execution invariant that requires: exact repo ref, active internal plan, nonempty canonical patch, nonempty all-exact edits, matching PatchGate approval/manifest, and a positive frozen authorized round with nonempty runtime provider/model. Call `revalidate_approved_patch(state)` for both execution modes.

Construct every attempt with frozen values:

```python
attempt = FixAttempt(
    patch_content=state.patch_content,
    patch_edits=[edit.model_copy(deep=True) for edit in state.patch_edits],
    file_path=state.patch_edits[0].file_path,
    repair_round_id=state.authorized_repair_round_id,
    repair_provider=state.authorized_repair_provider,
    repair_model=state.authorized_repair_model,
)
```

Remove the model-patch fallback from the agent path: EXECUTE always calls `apply_approved_patch`. Missing/stale approval and post-approval checkout drift become `infra_error`; they never ask the model to repair an environmental race.

- [ ] **Step 4: Collapse VERIFY retry branches onto the shared helper**

Keep success-to-COVERAGE and infrastructure-to-FAILURE unchanged. Preserve failure-class diagnostics and assertion-diversity signaling, but remove direct `retry_count += 1`, preflight special retry pools, `_same_failure_seen_twice` terminal routing, and the fixed repeated-assertion terminal.

For a historical attempt with `repair_round_id == 0`, migrate only from a still-valid fingerprint-bound `authorized_repair_*` snapshot for that same patch: copy that round/provider/model into the attempt once and persist it before accounting. If the trusted uncounted authorization is being resumed with `current_repair_round_id == 0`, restore the matching current ID/author from that same fingerprint-bound snapshot before validating the ledger; never infer an old patch author from mutable `active_provider`/`active_model`. If no trustworthy frozen attribution exists—or if a positive-ID attempt has missing/mismatched attribution—route an infrastructure/state-integrity failure without consuming a semantic retry. Then use:

```python
retry_phase = (
    Phase.PLAN
    if failure_class in {"syntax_error", "import_error"}
    else Phase.REFLECT
)
record_failed_repair_round(
    state,
    round_id=latest.repair_round_id,
    provider=latest.repair_provider,
    model=latest.repair_model,
    failure_reason=failure_class,
    retry_phase=retry_phase,
)
```

The helper owns both the budget check and increment. VERIFY must never read `state.active_provider` to attribute an already-executed patch.

- [ ] **Step 5: Run tests and commit the execution boundary**

Run the Step 2 command.

Expected: PASS; both providers execute only exact approved patches, attribution survives provider changes, retries are exactly once, and infrastructure failures spend none.

```bash
git add src/nodes/execute.py src/nodes/verify.py tests/test_execute_security.py tests/test_oci_integration.py tests/test_patch_retry.py tests/test_failure_taxonomy.py tests/test_convergence_diversity.py tests/test_new_agent.py tests/test_patch_preflight.py
git commit -m "feat: attribute and count verified repair rounds"
```

---

### Task 6: Make REFLECT/provider tools advisory and remove the dead Opus protocol

**Files:**
- Modify: `src/nodes/reflect.py:1-55,178-217,250-566`
- Modify: `src/reasoning_loop.py:65-90,190-280`
- Modify: `src/repair_flow.py:1-1015`
- Modify: `tests/test_repair_flow.py`
- Test: `tests/test_decision_frame.py`
- Test: `tests/test_model_escalation_integration.py`
- Test: `tests/test_patch_repair_prompt.py`
- Test: `tests/test_escalation_packet.py`
- Test: `tests/test_outcome_summary.py`
- Test: `tests/test_convergence_diversity.py`

**Interfaces:**
- REFLECT consumes active provider/model only for its advisory call and always emits one `ReflectDecision` contract.
- REFLECT consumes `summary_safety.sanitize_model_context` rather than importing PLAN/private escalation helpers.
- Reasoning tools retain `MAX_TOOL_REQUESTS_PER_ROUND` but no provider-specific no-progress cap.
- `repair_flow.py` retains only exact checkout/source/symbol helpers still referenced by production callers.

- [ ] **Step 1: Write failing provider-neutral reflection/tool tests**

Replace tests for `ESCALATED_REFLECT_SYSTEM` and `opus_no_progress_limit` with:

```python
def reflect_response() -> dict:
    return {
        "kind": "reflect",
        "root_cause": "The old target was wrong.",
        "what_went_wrong": "The assertion still failed.",
        "suggested_fix_approach": "Use the verified alternate target.",
        "files_that_also_need_changes": [],
        "decision_frame": {
            "stage": "reflect",
            "summary": "Choose a different target.",
            "recommended_action": "plan",
            "risk": "low",
            "confidence": 0.8,
        },
    }


@pytest.mark.parametrize(
    ("provider", "expected_model"),
    [
        ("primary", "gemini-3.5-flash:stable"),
        ("escalation", "claude-opus-4-8:stable"),
    ],
)
async def test_reflect_uses_one_prompt_and_schema(
    provider, expected_model, reflect_state, monkeypatch
):
    observed = []

    async def fake_llm(system, user, model=None, *, provider="primary", **_kwargs):
        observed.append((system, user, model, provider))
        return reflect_response()

    state = reflect_state
    state.active_provider = provider
    state.active_model = expected_model
    monkeypatch.setattr(reflect_node, "llm_call", fake_llm)
    result = await reflect_node.reflect_on_failure(state)

    assert observed[0][0] == reflect_node.REFLECT_SYSTEM
    assert observed[0][2:] == (expected_model, provider)
    assert result.current_phase == Phase.PLAN
```

Define `reflect_state` in the target module as an explicit failed-attempt fixture with a positive repair round/provider/model; do not rely on a repository-wide `state` fixture that does not exist.

Add cases where Opus reflection returns `kind=stop`, invalid schema, or an ordinary model error: each receives deterministic reflection fallback and continues to PLAN without changing `retry_count`. Add a no-progress Opus tool case that stays bounded only by the shared eight-call limit. Assert the lightweight summary call leaves active/current/authorized provider, model, and round fields unchanged. Keep cancellation identity tests.

Add gateway/sequence tests: (1) the second counted Gemini failure routes first through an Opus REFLECT call, then the next patch-authoring call still uses the shared `PLAN_SYSTEM`/`PlanDecision.patch_edits` contract; (2) a retryable primary gateway failure during REFLECT falls back within that same advisory call without opening, closing, or charging a repair round; (3) escalation-provider gateway exhaustion performs one deterministic reflection fallback to PLAN with no semantic charge and no Opus rebind/retry loop; (4) a reflection tool call crossing the primary token reserve reevaluates policy before the evidence reprompt, switches the active advisory model to Opus, and leaves current/authorized repair attribution unchanged.

Name them `test_second_primary_failure_opus_reflect_then_shared_opus_plan`, `test_reflect_gateway_fallback_is_same_advisory_round_without_repair_charge`, `test_escalation_reflect_gateway_exhaustion_falls_back_once_without_charge`, and `test_reflect_tool_reprompt_crossing_reserve_switches_active_model_only`; assert exact provider/model call order plus unchanged repair sequence/retry counters across REFLECT.

For both providers, inspect the final reflection system+user prompt after injecting legacy failure/continuity text and assert it does not instruct the model to emit `unified diff`, `replace_all`, `RepairPlan`, or `VerifiedEditBatch`.

- [ ] **Step 2: Run reflection tests and verify the Opus-only behavior fails**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_decision_frame.py tests/test_model_escalation_integration.py tests/test_patch_repair_prompt.py tests/test_escalation_packet.py tests/test_outcome_summary.py tests/test_convergence_diversity.py
```

Expected: FAIL on the separate escalated prompt, Opus two-round stop, and reflection stop/error terminating remaining repair budget.

- [ ] **Step 3: Use one reflection prompt and deterministic fallback**

Rename the primary system string to `REFLECT_SYSTEM`, remove `ESCALATED_REFLECT_SYSTEM`, delete the legacy diff/replace-all/internal-schema guidance from the earlier reflection context builder as well as the system prompt, build one bounded/redacted user prompt, and call:

```python
raw_response = await llm_call(
    REFLECT_SYSTEM,
    user,
    model=state.active_model,
    provider=state.active_provider,
)
```

Before every full or tool-evidence reflection model call, reevaluate/apply ModelPolicy and then snapshot the actual active provider/model for telemetry; never bind or rewrite repair-round attribution from REFLECT. Primary retryable gateway exhaustion may switch provider and retry the same reflection only when escalation was actually applied; it does not open/count a repair round. Escalation-provider gateway exhaustion and unconfigured primary fallback do not recursively retry: they produce one bounded deterministic reflection fallback and proceed to PLAN without repair accounting. `kind=stop`, invalid reflection, and ordinary model failure likewise set a safe deterministic `reflection_notes` value and then call `_finalize_reflection(state, phase=Phase.PLAN)`. Cancellation and hard tool-policy stops retain direct/terminal behavior. Summary generation remains the existing lightweight auxiliary path and may not assign any repair attribution field.

- [ ] **Step 4: Remove provider-specific reasoning limits and dead two-stage code**

Delete `record_opus_no_progress` and its branches from `route_reasoning_tool`; record provider-neutral no-progress diagnostics without calling model escalation. The shared tool counter remains eight.

Run a production caller search:

```bash
rg -n "generate_opus_repair|request_verified_edit_correction|record_opus_no_progress|REPAIR_PLAN_SYSTEM|VERIFIED_EDIT_SYSTEM|VERIFIED_EDIT_CORRECTION_SYSTEM" src
```

Expected before cleanup: only definitions/imports and the provider-specific branches being removed in this step; expected after cleanup: no matches. Delete `_call_schema`, the model prompts, `generate_opus_repair`, `request_verified_edit_correction`, verified-edit-only correction/rendering helpers, and imports used only by that path. Remove or rewrite the corresponding model-call tests in `tests/test_repair_flow.py`; preserve its pure exact-checkout/read/symbol tests. Preserve `RepairContextError`, exact checkout/read helpers, and `resolve_search_target_symbol` because PatchGate/EXECUTE still import them.

- [ ] **Step 5: Run reflection tests and commit cleanup**

Run the Step 2 command plus `tests/test_repair_flow.py`, then rerun the caller search:

```bash
.venv/bin/python -m pytest -q tests/test_decision_frame.py tests/test_model_escalation_integration.py tests/test_patch_repair_prompt.py tests/test_escalation_packet.py tests/test_outcome_summary.py tests/test_convergence_diversity.py tests/test_repair_flow.py
```

Expected: tests PASS and caller search returns no production matches.

```bash
git add src/nodes/reflect.py src/reasoning_loop.py src/repair_flow.py tests/test_repair_flow.py tests/test_decision_frame.py tests/test_model_escalation_integration.py tests/test_patch_repair_prompt.py tests/test_escalation_packet.py tests/test_outcome_summary.py tests/test_convergence_diversity.py
git commit -m "refactor: remove Opus-specific repair protocol"
```

---

### Task 7: Prove the exact G/G/O/O/O lifecycle and finish repository verification

**Files:**
- Modify: `tests/test_model_escalation_integration.py`
- Modify: `tests/test_patch_retry.py`
- Modify: `tests/test_oci_integration.py`
- Modify: `tests/test_api_security.py`
- Modify: `tests/test_run_store.py`
- Modify: `README.md`

**Interfaces:**
- Verifies all prior task interfaces together through real state transitions and an exact temporary git checkout.
- Produces no new production abstraction.

- [ ] **Step 1: Add the default all-failure integration test**

Use a valid `PlanDecision` whose exact search is absent so PatchGate rejects one full proposal per graph transaction:

```python
async def test_default_budget_is_exactly_gemini_gemini_opus_opus_opus(
    exact_repair_state, monkeypatch
):
    def plan_response(*, search: str) -> dict:
        return {
            "kind": "plan",
            "plan": "Try one exact structured edit.",
            "patch": "",
            "patch_edits": [
                {
                    "file": "src/widget.py",
                    "search": search,
                    "replace": "return 'new-sentinel'",
                }
            ],
            "files": ["src/widget.py"],
            "test_command": "pytest -q",
            "decision_frame": {
                "stage": "plan",
                "summary": "Try the next bounded anchor.",
                "recommended_action": "execute",
                "risk": "low",
                "confidence": 0.5,
            },
        }

    _enable_escalation(monkeypatch)
    observed = []
    state = exact_repair_state

    async def rejected_plan(system, user, model=None, *, provider="primary", **_kwargs):
        observed.append((provider, model, state.current_repair_round_id))
        return plan_response(search=f"missing-anchor-{len(observed)}")

    async def finish_failure(current):
        current.current_phase = Phase.FAILED
        return current

    monkeypatch.setattr(plan_node, "llm_call", rejected_plan)
    monkeypatch.setattr(new_agent, "StateGraph", None)
    monkeypatch.setattr(new_agent, "handle_failure", finish_failure)
    state = await graph.run_graph(
        new_agent.build_agent_graph(start_phase=Phase.PLAN),
        state,
    )

    assert observed == [
        ("primary", "gemini-3.5-flash:stable", 1),
        ("primary", "gemini-3.5-flash:stable", 2),
        ("escalation", "claude-opus-4-8:stable", 3),
        ("escalation", "claude-opus-4-8:stable", 4),
        ("escalation", "claude-opus-4-8:stable", 5),
    ]
    assert state.current_phase == Phase.FAILED
    assert state.retry_count == 4
    assert state.primary_failed_repair_rounds == 2
    assert state.repair_round_sequence == 5
    assert state.current_repair_round_id == 5
    assert state.last_counted_repair_round_id == 5
    assert len(state.model_history) == 5
    assert sum(
        item.get("route") == "handle_failure" for item in state.route_decisions
    ) == 1

    with pytest.raises(ValueError, match="budget is exhausted"):
        begin_repair_round(state)
    assert state.repair_round_sequence == 5
```

This must use the compiled fallback graph and real `route_from_state`, not a manual `plan_fix` loop. The deterministic failure handler only prevents external issue-comment/memory side effects; PLAN routing, DecisionFrame consumption, and the terminal edge remain real. Assert model invocation history has exactly five entries so a sixth call cannot be hidden.

- [ ] **Step 2: Add mixed and fallback attribution integration cases**

Add cases for:

- one invalid Gemini proposal followed by one PatchGate-authorized Gemini patch that fails tests, proving the next PLAN call is Opus;
- primary HTTP 503 after transport retries followed by an Opus patch in the same round, then an EXECUTE/VERIFY semantic test failure, proving one round ID, frozen Opus attribution, exactly one global retry consumed, and no increment to `primary_failed_repair_rounds`;
- primary token reserve selecting Opus before the first call and freezing Opus attribution;
- repeated failed patch and assertion-target diversity each spending one shared retry after gate authorization is cleared;
- OCI accepting an exact authorized patch from either provider and rejecting missing/stale attribution/approval;
- a saved accepted/counting state round-tripping through `run_store`, then repeated VERIFY/resume remaining idempotent; a cleared-current terminal state still rejects ID 6;
- `max_retries` 0, 1, 3, and 4 producing the documented sequences with no hidden call.

Use the exact regression names `test_invalid_then_verified_primary_failure_switches_to_opus`, `test_503_fallback_opus_verify_failure_spends_only_global_retry`, `test_token_reserve_freezes_opus_before_first_call`, `test_run_store_resume_does_not_recount_verified_round`, and `test_retry_configuration_provider_sequences` (parameterized for 0/1/3/4).

- [ ] **Step 3: Run the complete focused model/patch/execute suite**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_repair_rounds.py tests/test_patch_authorization.py tests/test_unified_plan_protocol.py tests/test_model_policy.py tests/test_decision_schemas.py tests/test_decision_frame.py tests/test_patch_gate.py tests/test_live_binding.py tests/test_test_generator.py tests/test_patch_retry.py tests/test_model_escalation_integration.py tests/test_context_loop_brake.py tests/test_search_hallucination_gate.py tests/test_error_episodes.py tests/test_execute_security.py tests/test_oci_integration.py tests/test_patch_preflight.py tests/test_failure_taxonomy.py tests/test_convergence_diversity.py tests/test_outcome_summary.py tests/test_repair_flow.py tests/test_run_store.py tests/test_new_agent.py tests/test_cli.py tests/test_main.py tests/test_api_security.py tests/test_agent_v2_eval.py tests/test_eval_harness.py tests/test_oci_runner.py tests/test_documentation_contract.py tests/test_swe_bench_oci_workflow.py
```

Expected: PASS with zero failures.

- [ ] **Step 4: Run static caller and formatting checks**

Run:

```bash
rg -n "generate_opus_repair|request_verified_edit_correction|record_opus_no_progress|opus_no_progress_limit|_unlocatable_edits|MAX_(REPEATED_PATCH_BLOCKS|SEARCH_CORRECTIONS)|ESCALATED_(PLAN|REFLECT)_SYSTEM|REPAIR_PLAN_SYSTEM|VERIFIED_EDIT(_CORRECTION)?_SYSTEM|two_stage_repair" src
rg -n "patch_correction_count|hallucinated_search_block_count" src/nodes/plan.py
rg -n "max_retries: int = 3|default=3|le=3|min\(req\.max_turns, 3\)|max_retries=3" src/state.py src/new_agent.py src/cli.py src/main.py eval/agent_v2_harness.py eval/harness.py eval/oci_runner.py
git diff --check
```

Expected: all three `rg` commands find no production legacy protocol/correction-cap or active default/boundary; `git diff --check` exits 0 with no output.

- [ ] **Step 5: Run the complete repository test suite**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: all tests pass with zero failures. Do not run live model or paid SWE-bench evals.

- [ ] **Step 6: Review scope and commit the integration proof**

Confirm `git status --short` lists only intentional task files plus the unstaged user-owned `run_trace.py`. Inspect `git diff --stat` and ensure no key/API value appears in tracked changes.

```bash
git add tests/test_model_escalation_integration.py tests/test_patch_retry.py tests/test_oci_integration.py tests/test_api_security.py tests/test_run_store.py README.md
git commit -m "test: prove five-round model switching"
```

After the commit, rerun `git status --short`; `run_trace.py` must remain unstaged and uncommitted.
