# Final-Gate Telemetry and Cancellation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` to implement this plan task-by-task,
> with a fresh implementer and an independent spec/code review after every task.

**Goal:** Close the two remaining Important release blockers by making every
test-generation model request visible to token/time accounting and by preserving
cancellation-drain failures through every async recovery boundary.

**Architecture:** Keep telemetry at the model-request producer, strict
cross-record validation at the OCI contract boundary, cancellation identity at
the leaf async operations, and bounded/redacted evidence at phase wrappers.
`CancellationDrainError` remains a normal exception; every async boundary that
can receive it must re-raise it before ordinary operational-error handling.

**Tech Stack:** Python 3.10-3.12, asyncio, Pydantic v2, pytest,
pytest-asyncio, RepoPilot Agent V2, OCI evaluation contracts.

## Global constraints

- Implement the approved design in
  `docs/superpowers/specs/2026-07-24-final-gate-telemetry-cancellation-design.md`.
- Preserve the user's unstaged `run_trace.py` modification byte-for-byte and
  never stage or commit it.
- Use red-green-refactor TDD. Run each named focused test once before the
  implementation and confirm the new assertion fails for the intended reason.
- Keep Gemini Flash primary, Claude Opus escalation, model retry behavior,
  scoring weights, and the 100,000-token estimate contract unchanged.
- Record exception classes only. Never persist exception messages, model bodies,
  credentials, environment values, or unbounded external text.
- Do not change `CancellationDrainError` inheritance or convert caller
  cancellation into a normal result.
- Stage files explicitly for every task commit; never use `git add -A` or
  `git add .`.

---

### Task 1: Account for every test-generation model request

**Files:**

- Modify: `src/test_generator.py`
- Test: `tests/test_test_generator.py`
- Test: `tests/test_coverage_node.py`

**Interfaces and behavior:**

- Add `_TEST_GENERATION_PREFLIGHT_REASON = "test_generation_preflight_failed"`.
- Add a private `_TestGenerationPreflightError(ValueError)` and wrap prompt
  construction exactly as follows:

```python
try:
    prompt = _generation_prompt(state, rejection_reason)
except (OSError, RuntimeError, ValueError) as exc:
    raise _TestGenerationPreflightError(
        _TEST_GENERATION_PREFLIGHT_REASON
    ) from exc
```

  A preflight error
  creates no `ModelInvocation`, consumes no token estimate, increments no
  generation attempt, and terminates generation without retrying forever.
- Make `request_test_batch(state, rejection_reason)` build the prompt first,
  then increment `state.test_generation_attempts` immediately before the model
  request. Snapshot `active_model` and `active_provider` before the call.
- Measure the complete logical request with `time.monotonic()`. Estimate input
  and output with `src.state._estimate_tokens`, append exactly one invocation
  through `record_model_invocation` with `node="test_generation"`, and debit
  `state.token_usage` on all three terminal paths:
  `ok`, `invalid_response`, and `error`.
- Classify a non-object response or `VerifiedEditBatch` validation failure as
  `invalid_response`; classify other `llm_call` failures as `error`; re-raise
  the original bounded exception after recording it.
- A `CancellationDrainError` is re-raised before the ordinary model-error
  telemetry path. It is cancellation/cleanup evidence, not a model response
  error. The orchestration-level recovery is closed in Task 4. The
  owner-approved whole-stage review amendment in Task 7 supersedes only the
  original absence of cancellation accounting: an in-flight cancellation gets
  a distinct `cancelled` invocation before the identical object is re-raised;
  it is never classified as `error`.
- Move attempt ownership out of `run_test_generation_attempts()` and into this
  request boundary. Update existing tests that stubbed `request_test_batch()` to
  stub `llm_call()` instead, so they exercise real telemetry and cannot loop with
  an unchanged attempt counter.
- This includes `tests/test_coverage_node.py`, whose generated-coverage fixture
  currently stubs `request_test_batch()` and would otherwise loop forever after
  attempt ownership moves.

Add this validation helper and preserve the following accounting order:

```python
def _validate_test_batch(raw: object) -> VerifiedEditBatch:
    if not isinstance(raw, dict):
        raise ValueError("test generator response must be a JSON object")
    payload = {key: value for key, value in raw.items() if key != "kind"}
    try:
        return VerifiedEditBatch.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("test generator returned an invalid edit batch") from exc


prompt = _generation_prompt(state, rejection_reason)  # no telemetry yet
state.test_generation_attempts += 1
model, provider = state.active_model, state.active_provider
input_tokens = _estimate_tokens(_SYSTEM_PROMPT, prompt)
response_text = ""
started = time.monotonic()
try:
    raw = await llm_call(
        _SYSTEM_PROMPT,
        prompt,
        model=model,
        provider=provider,
        temperature=0.0,
    )
except CancellationDrainError:
    raise
except Exception as exc:
    elapsed = time.monotonic() - started
    record_model_invocation(
        state,
        model=model,
        provider=provider,
        node="test_generation",
        elapsed_seconds=elapsed,
        input_tokens=input_tokens,
        output_tokens=0,
        status="error",
        error=exc,
    )
    state.token_usage += input_tokens
    raise
try:
    response_text = json.dumps(
        raw,
        ensure_ascii=False,
        sort_keys=True,
        default=lambda value: type(value).__name__,
    )
    batch = _validate_test_batch(raw)
except (TypeError, ValueError, RecursionError) as exc:
    elapsed = time.monotonic() - started
    output_tokens = _estimate_tokens(response_text)
    record_model_invocation(
        state,
        model=model,
        provider=provider,
        node="test_generation",
        elapsed_seconds=elapsed,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        status="invalid_response",
        error=exc,
    )
    state.token_usage += input_tokens + output_tokens
    raise
elapsed = time.monotonic() - started
output_tokens = _estimate_tokens(response_text)
record_model_invocation(
    state,
    model=model,
    provider=provider,
    node="test_generation",
    elapsed_seconds=elapsed,
    input_tokens=input_tokens,
    output_tokens=output_tokens,
    status="ok",
)
state.token_usage += input_tokens + output_tokens
return batch
```

- [ ] Add failing tests proving a successful primary request records the exact
  provider/model/node/status, nonnegative elapsed time, input/output estimates,
  one attempt, and matching total `token_usage`.
- [ ] Add failing parameterized tests for a non-object response, schema-invalid
  object, and model exceptions including a `ValueError` raised by `llm_call`.
  Each must produce one invocation and one debit; response validation uses
  `invalid_response`, while every call exception uses `error` and stores only
  its class. This proves exception type alone never confuses gateway failure
  with response validation.
- [ ] Add a failing preflight test proving an oversized/invalid prompt records
  no attempt or invocation and terminates with a distinct preflight reason.
- [ ] Extend the existing two-attempt escalation test to require two ordered
  invocations (primary then escalation), exact summed usage, and no downgrade.
- [ ] Run
  `.venv/bin/python -m pytest tests/test_test_generator.py tests/test_coverage_node.py -q`
  and confirm the new assertions fail against the current implementation.
- [ ] Implement the smallest producer-owned telemetry, attempt, and preflight
  changes. Do not add telemetry to the shared `llm_call` client.
- [ ] Re-run both focused files and require them to pass.
- [ ] Commit only the three task files with
  `fix(test-generator): account for model requests`.

---

### Task 2: Stop test generation at the existing token budget

**Files:**

- Modify: `src/test_generator.py`
- Modify: `src/nodes/coverage.py`
- Test: `tests/test_test_generator.py`
- Test: `tests/test_coverage_node.py`

**Interfaces and behavior:**

- Add `_TEST_GENERATION_BUDGET_REASON = "token_budget_exceeded"`.
- In `run_test_generation_attempts()`, call `_is_budget_exceeded(state)` before
  each request, immediately after a successful request before PatchGate/file
  mutation, and after a failed request before escalation/another iteration.
- A reached budget restores the captured production diff, PatchGate approval,
  coverage state, and generated-test approvals; it prevents validation, apply,
  escalation, and request two; it returns a failed `CoverageDecision` with the
  exact budget reason.
- Catch `_TestGenerationPreflightError` separately and break with the exact
  preflight reason. Because no attempt was counted, allowing the normal loop to
  continue would be unbounded.
- In `ensure_coverage()`, preserve budget as a budget terminal:

```python
state.coverage_failure_reason = reason[:300]
if reason == _TEST_GENERATION_BUDGET_REASON:
    state.failure_reason = "Token budget exceeded during test generation."
else:
    state.failure_reason = f"test_generation_failed:{reason[:300]}"
```

  This special case is required because failure taxonomy checks
  `test_generation_failed` before its generic budget-text rule.

- [ ] Add failing request-before-call and request-after-call budget tests. The
  after-call case crosses the budget on request one and proves PatchGate,
  application, and coverage were never called.
- [ ] Assert both cases restore exact production authorization/diff, create no
  generated file/approval, and never start request two or escalation.
- [ ] Add a coverage-node test requiring the exact coverage and terminal budget
  reasons with zero invocations when the budget is already exhausted.
- [ ] Run
  `.venv/bin/python -m pytest tests/test_test_generator.py tests/test_coverage_node.py -k 'budget or preflight' -q`
  and confirm the new budget assertions fail.
- [ ] Implement only the three budget gates, preflight hard stop, restoration,
  and terminal-reason special case.
- [ ] Re-run complete `tests/test_test_generator.py` and
  `tests/test_coverage_node.py` and require them to pass.
- [ ] Commit only the four task files with
  `fix(test-generator): enforce token budget`.

---

### Task 3: Reject contradictory generated-test telemetry in OCI artifacts

**Files:**

- Modify: `eval/oci_contract.py`
- Test: `tests/test_oci_contract.py`
- Test: `tests/test_oci_aggregate.py`

**Interfaces and behavior:**

- Extend `ResultRecord`'s after-validator with one canonical generation history:
  invocations whose node is exactly `test_generation`.
- Require `len(generation_invocations) == test_generation_attempts` for every
  result. This also keeps synthetic/non-ready results at zero/zero.
- Tighten `test_generation_attempts` to a strict integer in `[0, 2]`.
- A `coverage_status == "generated_verified"` result requires at least one
  canonical generation invocation with `status == "ok"`.
- A canonical escalation-provider generation invocation requires coherent
  result-level fields: `escalated is True`, `escalation_reason` is one of the
  existing approved nonempty values, and the configured escalation model is
  present.
- Enforce one-way provider order only inside the canonical `test_generation`
  subsequence: once a generation invocation uses `provider == "escalation"`,
  no later generation invocation may use `primary`. Other nodes retain their
  own provider policy; in particular the lightweight `outcome_summary` call may
  legally use primary after an Opus generation/escalation. Existing
  `models_used` equality does not enforce this generation-local ordering.
- The legacy allowlisted node spelling `test_generator` may remain parseable for
  non-generation historical telemetry, but it does not satisfy or increment the
  canonical producer contract.
- Validation failure must reject both package-time and aggregate-time parsing;
  it must not merely remove engineering-budget credit.

```python
generation = [item for item in self.model_invocations
              if item.node == "test_generation"]
if len(generation) != self.test_generation_attempts:
    raise ValueError("test generation attempts do not match invocation history")
if self.coverage_status == "generated_verified" and not any(
    item.status == "ok" for item in generation
):
    raise ValueError("generated coverage requires a successful invocation")
```

- [ ] Add failing contract tests for missing, extra, failed-only, and
  alias-only generation histories; an exact successful primary history passes.
- [ ] Add failing contract tests for a two-attempt primary/escalation history
  with false/missing escalation claims; add the valid coherent case.
- [ ] Add reversed and interleaved canonical-generation tests; reject every
  escalation-to-primary transition inside `test_generation`, accept primary-to-
  escalation and escalation-only generation histories, and accept a primary
  `outcome_summary` invocation after escalation.
- [ ] Add aggregate mutation tests proving the same mismatches are rejected
  when read from a packaged bundle.
- [ ] Make aggregate fixtures use `existing_verified` by default. Only fixtures
  explicitly exercising generated coverage may claim `generated_verified`, and
  those fixtures must contain canonical generation telemetry.
- [ ] Prove existing aggregate token/time accounting automatically includes the
  new invocation; do not modify `eval/oci_aggregate.py` production code for
  telemetry summation.
- [ ] Run
  `.venv/bin/python -m pytest tests/test_oci_contract.py tests/test_oci_aggregate.py -q`
  and confirm the new contract tests fail for telemetry mismatch.
- [ ] Implement only the strict cross-field validator; keep existing provider,
  model, numeric, failure-taxonomy, and non-ready validators intact.
- [ ] Re-run the focused tests and require them to pass.
- [ ] Commit only the three task files with
  `fix(eval): bind generated coverage to telemetry`.

---

### Task 4: Re-raise cancellation-drain failures at async leaf boundaries

**Files:**

- Modify: `src/nodes/commit.py`
- Modify: `src/nodes/execute.py`
- Modify: `src/nodes/plan.py`
- Modify: `src/nodes/reflect.py`
- Modify: `src/tool_router.py`
- Modify: `src/coverage_gate.py`
- Modify: `src/test_generator.py`
- Test: `tests/test_commit.py`
- Test: `tests/test_execute_security.py`
- Test: `tests/test_tool_router.py`
- Test: `tests/test_coverage_gate.py`
- Test: `tests/test_test_generator.py`
- Test: `tests/test_decision_frame.py`

**Interfaces and behavior:**

- Import `CancellationDrainError` from `src.async_safety` in each producer that
  does not already import it.
- Immediately before every broad recovery handler that can wrap an awaited
  operation, add the explicit precedence rule:

  Concretely, insert `except CancellationDrainError: raise` immediately before
  each existing `(OSError, RuntimeError, ValueError)` or `Exception` handler;
  retain the existing handler body unchanged.

- Apply that rule to these exact paths:
  `commit_fix`, both the inner clone and outer execution recoveries in
  `execute_fix`, `route_tool_intent`, the fixed-run and base-run async catches in
  `validate_differential_coverage`,
  `run_test_generation_attempts`, and the model/tool loops in `plan_fix` and
  `reflect_on_failure`.
- Do not alter synchronous validation catches that cannot receive an awaited
  drain error. Ordinary I/O/schema/tool errors must continue to return their
  current bounded failure states.
- PLAN/REFLECT must re-raise before recording a model invocation error, so a
  tool cancellation is not forged into model-gateway telemetry.

- [ ] Add a real `commit_fix` regression test whose patched `create_pr` raises a
  `PRCancellationCleanupError`; assert the identical instance escapes instead
  of becoming an ordinary commit failure.
- [ ] Add focused synthetic-drain tests for execute, targeted tool routing,
  fixed/base coverage execution, and generated-test orchestration. Each test
  asserts object identity (`raised.value is sentinel`) and that no ordinary
  failure record replaces it.
- [ ] In the generated-test case, enter the applied-test path and prove cleanup
  restores the test and production authorization before the identical drain is
  re-raised. A rollback failure must not `return` over a pending drain; preserve
  the drain's cancellation/cleanup fields and chain the rollback failure.
- [ ] Implement that path with explicit `pending_drain` and `rollback_error`
  locals: capture rather than immediately raise the drain, perform rollback in
  `finally` without returning, then execute
  `raise pending_drain from rollback_error` when both exist or re-raise the
  pending drain unchanged when rollback succeeded. Only the no-drain rollback
  failure path may return the existing `generated_test_rollback_failed`
  decision. Tests assert drain identity, `.cancellation`, `.cleanup_error`, and
  the chained rollback cause.
- [ ] Add PLAN and REFLECT tests whose model-selected tool raises the sentinel;
  assert propagation and no `status="error"` model invocation for that tool
  cancellation.
- [ ] Run
  `.venv/bin/python -m pytest tests/test_commit.py tests/test_execute_security.py tests/test_tool_router.py tests/test_coverage_gate.py tests/test_test_generator.py tests/test_decision_frame.py -q`
  and confirm the new tests fail by returning a state/decision/tool error.
- [ ] Add the explicit re-raise clauses. Keep normal handlers unchanged except
  for the generated-test cleanup restructuring above.
- [ ] Re-run the focused files and require them to pass.
- [ ] Commit only the listed implementation and test files with
  `fix(asyncio): propagate cancellation drain failures`.

---

### Task 5: Preserve drain evidence through wrappers and top-level execution

**Files:**

- Modify: `src/timeout_diagnostics.py`
- Modify: `src/graph.py`
- Modify: `src/new_agent.py`
- Test: `tests/test_async_safety.py`
- Test: `tests/test_new_agent.py`

**Interfaces and behavior:**

- Define `TimeoutFailureKind = Literal["pr_cleanup", "pr_transaction",
  "generic_drain"]` and extend the evidence interface exactly to:

```python
@dataclass(frozen=True)
class TimeoutCleanupEvidence:
    failure_kind: TimeoutFailureKind
    cause_type: str
    cleanup_error_type: str
    cleanup_error: str
    operation: str = ""
    pr_number: int | None = None
```

  It represents `PRCancellationCleanupError`,
  `PRCancellationTransactionError`, and a generic `CancellationDrainError`.
- A generic record includes bounded/redacted `operation`, composite
  `cause_type`, `cleanup_error_type`, and `cleanup_error`; PR records retain
  `pr_number`, adding it to diagnostics only when it is a positive strict int.
  Transaction evidence identifies a transaction failure rather than calling it
  cleanup. In every case `cause_type` is exactly `type(drain).__name__` after
  the existing exception-class normalization; it is never derived from a
  message or `repr`.
- Visit at most `_MAX_CAUSE_DEPTH` candidate objects with one FIFO queue and an
  identity-based `seen` set. For each visited object enqueue distinct
  `__cause__`, then `__context__`, then a drain's exception-valued
  `cleanup_error`. Save the first generic drain as fallback, continue the
  bounded traversal to prefer the first PR-specific drain, then return the
  fallback only if no PR-specific candidate was found. This exact queue order
  handles branches and cycles without an unbounded recursive walk.
- Redact with `redact_secrets`, normalize whitespace, bound the operation and
  summary, and never include `cancellation.args` or raw exception reprs. Set
  `_MAX_OPERATION = 120`, retain `_MAX_ERROR_SUMMARY = 300`, and cap the final
  rendered summary at `_MAX_EVIDENCE_SUMMARY = 480` characters.
- Preserve every existing diagnostic key and add an exact kind:

  - PR cleanup emits `timeout_cleanup_kind="pr_cleanup"`,
    `timeout_cause_type`, `cleanup_error_type`, `cleanup_error`, and positive
    `cleanup_pr_number` when available; its summary starts
    `PR cancellation cleanup failed for pull request N` for a positive number,
    otherwise `PR cancellation cleanup failed for an unknown pull request`.
  - PR transaction emits the same common keys with
    `timeout_cleanup_kind="pr_transaction"`; its error fields derive from
    `.transaction_error`, and its summary uses the same numbered/unknown target
    rule after `PR cancellation transaction failed for`.
  - Generic drain emits `timeout_cleanup_kind="generic_drain"`, the common
    cause/error keys, and `cleanup_operation`; its summary starts
    `Cancellation cleanup failed during OPERATION`.
- Add `except CancellationDrainError: raise` before the generic handler in both
  `FallbackCompiledGraph.ainvoke()` and `new_agent._wrap_node()`.
- Add the same precedence in `agent_v2()` before its generic crash-to-payload
  mapping. `resume_agent_v2()` already has no normalizing catch and remains a
  propagation path.
- An internally owned phase timeout still returns a failed state, but its
  `failure_reason` and node diagnostic include the bounded evidence extracted
  from the timeout cause. Direct/external cancellation-drain failures escape.

- [ ] Add unit tests for generic OCI drain evidence, PR transaction evidence,
  nested generic-to-PR preference, cycle/depth bounds, redaction, and maximum
  summary length.
- [ ] Add direct wrapper and fallback-graph tests proving a sentinel
  `CancellationDrainError` is re-raised unchanged.
- [ ] Add a real wrapped `commit_fix` timeout test proving the returned timeout
  state preserves PR number, composite cause class, and cleanup class.
- [ ] Add an `agent_v2` test proving a graph-raised drain error escapes instead
  of returning `final_phase="CRASHED"`; retain the existing ordinary exception
  crash-payload test.
- [ ] Run
  `.venv/bin/python -m pytest tests/test_async_safety.py tests/test_new_agent.py -q`
  and confirm the new wrapper/evidence tests fail against the current code.
- [ ] Implement the strict exception precedence and bounded evidence extension.
- [ ] Re-run the two focused files and require them to pass.
- [ ] Commit only the five task files with
  `fix(asyncio): retain timeout cleanup evidence`.

---

### Task 6: Verify the final-gate remediation across supported runtimes

**Files:**

- No production changes expected.
- Update tests only if verification exposes a genuine cross-version contract
  bug; any such fix receives its own TDD commit and independent review.

- [ ] Run the affected suite on the project Python:

  ```bash
  .venv/bin/python -m pytest \
    tests/test_test_generator.py tests/test_coverage_node.py \
    tests/test_coverage_gate.py tests/test_commit.py \
    tests/test_execute_security.py tests/test_tool_router.py \
    tests/test_decision_frame.py tests/test_async_safety.py \
    tests/test_new_agent.py tests/test_oci_contract.py \
    tests/test_oci_aggregate.py -q
  ```

- [ ] Run the same affected suite with the repository's Python 3.10, 3.11, and
  3.12 isolated environments. Require zero failures; only the already-approved
  sqlite-vec skip/warning may differ by runtime.
- [ ] Run `.venv/bin/python -m pytest -q` and record the exact pass/skip count.
- [ ] Run `.venv/bin/python -m ruff check src eval tests`.
- [ ] Run `git diff --check` and verify `git status --short` shows only the
  user's pre-existing `run_trace.py` change after task commits.
- [ ] Ask a fresh reviewer to inspect the implementation against the approved
  spec. Do not begin pilot workflow implementation with any unresolved
  Critical or Important finding.

---

### Task 7: Close whole-stage review findings and obtain Python 3.10 CI evidence

**Owner-approved choices:** `A,A` on 2026-07-24. Use a canonical
`status="cancelled"` invocation for in-flight generation cancellation. After
all code findings are fixed and freshly reviewed, push the feature branch only
to obtain the existing CI matrix evidence; this does not authorize Pilot
implementation, secrets, paid evaluation, any environment action, merge,
prompt changes, or cohort changes.

**Files:**

- Modify: `src/state.py`
- Modify: `src/escalation.py`
- Modify: `src/test_generator.py`
- Modify: `src/new_agent.py`
- Modify: `src/repair_flow.py`
- Modify: `eval/oci_contract.py`
- Test: `tests/test_test_generator.py`
- Test: `tests/test_new_agent.py`
- Test: `tests/test_agent_v2_eval.py`
- Test: `tests/test_repair_flow.py`
- Test: `tests/test_oci_contract.py`
- Test: `tests/test_oci_aggregate.py`
- Test: `tests/test_escalation_packet.py`
- Test: `tests/test_decision_frame.py`

**Cancelled generation accounting:**

- Extend the runtime and OCI invocation status literals with exactly
  `"cancelled"`.
- In `request_test_batch()`, after prompt success, attempt increment, model and
  provider snapshot, input estimate, and timer start, catch both
  `asyncio.CancelledError` and `CancellationDrainError` before ordinary model
  errors. Append exactly one `test_generation` invocation with the snapshotted
  model/provider, nonnegative elapsed time, estimated input tokens, zero output
  tokens, `status="cancelled"`, and only the normalized cancellation class.
  Debit exactly the input estimate, then use bare `raise` so object identity,
  cancellation fields, cleanup fields, and cause chains remain intact.
- `cancelled` is not `error`, cannot prove `generated_verified`, and requires a
  nonempty normalized exception class at the OCI boundary. Keep the canonical
  equality unchanged: all `test_generation` records, including cancelled
  records, must equal `test_generation_attempts`.
- Add RED tests for direct `asyncio.CancelledError`, generic
  `CancellationDrainError`, and a real internally timed-out generation path.
  Assert one attempt, one cancelled invocation, input-only debit, no output
  estimate, original exception identity for direct cancellation, bounded
  exception-class-only persistence, and package/aggregate parsing of the
  internal timeout artifact. Prove a cancelled record alone cannot support
  generated coverage.
- Whole-stage review addendum: applied generated-test cleanup must also capture
  direct `asyncio.CancelledError`. Successful rollback re-raises the identical
  pending cancellation or drain. If rollback fails under direct cancellation,
  raise a new `CancellationDrainError("generated test rollback", cancellation,
  rollback_error)` from the rollback error; if the pending object is already a
  drain, re-raise that identical object from the rollback error. Preserve the
  ordinary no-cancellation `generated_test_rollback_failed` result. Cover direct
  success, direct rollback failure, and a real `wait_for_phase` timeout whose
  extracted evidence reports `generic_drain`, `generated test rollback`, and
  the rollback error class.

**Public payload fallback:**

- Add exact `"test_generation_attempts": state.test_generation_attempts` to
  `agent_payload_from_state()` beside coverage/model telemetry. Do not derive
  attempts from history or coerce it at the producer.
- Add RED tests for the payload field and for `eval/agent_v2_harness.py` when
  durable `load_run()` fails: canonical one-attempt and two-attempt generation
  histories from the public payload must retain their exact count and pass
  package/aggregate validation.

**Graph-crash telemetry persistence:**

- Whole-stage review continuation: replace the ordinary `run_graph()` crash
  branch's handcrafted response with a safe projection of the mutated
  `AgentState`. Set the durable state phase to `FAILED`, clear pending human
  input, redact and bound the crash reason, call `agent_payload_from_state()`,
  and then expose public `final_phase="CRASHED"`, `done=true`, `success=false`,
  `waiting_for_user=false`, no successful patch claim, and empty model patch.
- When `save_final_run=True`, run the existing best-effort save before writing
  the trace. Preserve exact token usage, cancelled generation invocation,
  generation-attempt count, and coverage fields in both payload and saved
  state; no exception secret may reach payload, trace, or stderr.
- Add a unit RED test for a graph that records one canonical cancelled
  generation invocation with seven input tokens and then raises. Add an
  end-to-end RED test through `evaluate_agent_v2_sample()`,
  `package_instance()`, and `aggregate_artifacts()` with durable loading forced
  unavailable; the exact call/count/token telemetry must survive.

**Outermost cancellation-drain consumers:**

- Whole-stage review continuation: add an explicit bare re-raise for
  `CancellationDrainError` before the ordinary generic fallback at exactly six
  newly reachable consumers: `intelligent_agent()`, `agent_v2_endpoint()`, and
  `agent_v2_resume_endpoint()` in `src/main.py`;
  `run_exact_verified_instance()` and the per-sample catch in
  `run_agent_v2_eval()`; and `generate_instance()` in `eval/oci_runner.py`.
  Keep the eval harness import compatible with direct script execution.
- Do not expand this change to synchronous inspect/replay routes or unrelated
  legacy catches. At every named boundary assert sentinel identity; paired
  ordinary `RuntimeError` controls must retain the existing safe 502, failed
  eval result, or OCI infrastructure artifact behavior.

**Escalated PLAN tool cancellation:**

- Import `CancellationDrainError` in `src/repair_flow.py` and re-raise it
  immediately before `_call_schema()`'s broad `except Exception` handler.
  Preserve ordinary model/schema/tool handling unchanged.
- Add a RED test that uses the real escalated/two-stage PLAN route, has a
  model-selected tool raise a sentinel drain, and asserts identical propagation
  with no forged `status="error"` invocation and no token debit for that schema
  call.

**TDD and commits:**

- Run the new focused tests before production changes and retain the exact RED
  failures in the task report.
- Implement the three findings as the smallest compatible changes. Suggested
  commit boundaries:
  1. `fix(telemetry): account for cancelled generation requests`
  2. `fix(eval): preserve generation attempts in public payload`
  3. `fix(plan): propagate escalated tool cancellation`
- Run the complete affected suite from Task 6, then the full suite, Ruff, and
  `git diff --check`. Verify status shows only the original `run_trace.py` user
  change and its recorded SHA-256 is unchanged.
- Generate a fresh whole-stage review package from `7636b13` through the new
  head. Resolve every finding attributable to that range and fresh re-review
  the resulting latest head; repeat until it has no unresolved Critical,
  Important, or Minor range finding. The design's explicitly deferred pre-base
  Minor items remain out of scope and separately tracked.
- Push only `fix/release-readiness-20260717`, bind the GitHub Actions `CI` run
  to the exact pushed head SHA, and require overall `success`: all six Ubuntu
  and macOS Python 3.10/3.11/3.12 matrix jobs, `lint`, and `oci-integration`
  must pass. A CI failure returns to TDD and fresh review; it never authorizes
  Pilot work, paid inference, secrets, any environment action, merge,
  prompt/cohort changes, or a rerun of the evaluation workflow.

## Completion evidence

Record in the handoff:

- focused and full-suite command outputs;
- Python 3.10/3.11/3.12 counts;
- Ruff and `git diff --check` results;
- all task and whole-stage remediation commit SHAs;
- reviewer verdict and any follow-up commit;
- the exact pushed head SHA and its green Python 3.10/3.11/3.12 CI run URL;
- confirmation that `run_trace.py` was neither staged nor modified.
