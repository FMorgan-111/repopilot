# Final Gate Telemetry and Cancellation Remediation

Date: 2026-07-24

Status: approved approach; implementation pending written-spec review

## Context

The final whole-branch review found two Important release blockers after the
Python 3.10/3.12 and OCI compatibility fixes had passed locally:

1. Regression-test generation calls the model without adding a
   `ModelInvocation` or debiting `AgentState.token_usage`. The OCI aggregator
   derives its 100,000-token and 360-minute budget result exclusively from the
   serialized invocation history, so these calls can be omitted from both the
   agent budget and the engineering score.
2. `CancellationDrainError` is an ordinary exception so that it can retain both
   cancellation and terminal cleanup failures. Several async node boundaries
   catch it as a generic operational error and return normally. A phase timeout
   can therefore lose evidence that PR or OCI cleanup failed.

The selected approach is a targeted repair at the producers and async
boundaries. It does not change the exception hierarchy or refactor the global
LLM client.

## Goals

- Account for every logical test-generation model request in the same bounded
  telemetry format used by PLAN and REFLECT.
- Make test generation obey the existing hard token-budget semantics before a
  second request or terminal success.
- Preserve cancellation-drain failures until the phase timeout wrapper can
  chain and serialize their safe cleanup evidence.
- Cover the real node and wrapper paths that previously hid the failures.
- Keep Gemini Flash primary, Opus escalation, OCI isolation, scoring weights,
  and evaluation datasets unchanged.

## Non-goals

- Refactoring `llm_call` or changing how all existing nodes estimate tokens.
- Changing `CancellationDrainError` to inherit from `BaseException` or
  `asyncio.CancelledError`.
- Addressing the deferred Minor cache, scorer-report, or dataset-row binding
  hardening items.
- Merging the PR or starting paid SWE-bench jobs.

## Design

### 1. Test-generation model telemetry

`request_test_batch()` will own the telemetry for one logical generation
request:

1. Build and bound the prompt before starting telemetry. Prompt construction
   failures are neither model invocations nor generation attempts; they stop
   generation with a distinct bounded preflight reason.
2. Increment `test_generation_attempts` immediately before the model request,
   after prompt construction succeeds. This field therefore counts logical
   model-backed generation attempts rather than local preflight failures.
3. Snapshot `state.active_model` and `state.active_provider`, then measure the
   entire `llm_call()` duration with `time.monotonic()`.
4. Estimate input and output tokens with the same `_estimate_tokens()` helper
   used by PLAN and REFLECT. The measured duration includes any internal JSON
   retry performed by `llm_call`; token accounting follows the existing
   logical-call estimation contract.
5. Append exactly one invocation with node `test_generation` for each logical
   request. Record `ok` on a valid `VerifiedEditBatch`, `invalid_response` for
   malformed or schema-invalid output, and `error` for other model failures.
   Store only the normalized exception class.
6. Debit `state.token_usage` on every success, invalid-response, and error path,
   then return or re-raise the original error.

`run_test_generation_attempts()` will check the budget before each request and
again immediately after a successful request, before validating or applying
the generated edit. Once `token_usage >= token_budget`, it will stop without a
second request, restore the production patch authorization, and return a
bounded token-budget failure. `ensure_coverage()` will preserve that as an
explicit budget terminal reason rather than misclassifying it as an invalid
generated test.

This makes the payload's `model_history`, `token_used`, and the OCI aggregate's
model token/time totals derive from the same calls.

The strict OCI result contract will cross-check the generation claim against
that telemetry. A `generated_verified` result must contain at least one
successful `test_generation` invocation; the number of generation invocations
must agree with `test_generation_attempts`; and an escalation-provider
generation invocation must agree with the run's escalation fields. Missing,
extra, or contradictory generation telemetry rejects the artifact bundle
instead of merely removing budget credit.

### 2. Cancellation-drain propagation

`CancellationDrainError` remains a normal exception. Known async operational
boundaries will add an explicit `except CancellationDrainError: raise` before
their broad recovery handlers. The required paths are:

- PR creation through `commit_fix`;
- clone and OCI test execution through `execute_fix`;
- model-selected targeted tests through `route_tool_intent`;
- fixed/base OCI coverage runs through `validate_differential_coverage` and
  generated-test orchestration;
- PLAN and REFLECT reasoning loops after a model-selected tool call;
- phase wrappers, fallback graph execution, and `agent_v2`'s top-level crash
  mapping, which could otherwise normalize an externally propagated drain
  failure.

Synchronous validation catches that cannot receive an async drain failure will
remain unchanged. Ordinary operational errors will continue to use the current
failure-state behavior.

With this propagation rule, `wait_for_phase()` sees the terminal child error
while draining a timed-out phase and raises `TimeoutError` with the
`CancellationDrainError` in its cause chain. Cancellation identity and cleanup
failure identity remain available in memory.

### 3. Bounded timeout evidence

`extract_timeout_cleanup_evidence()` will recognize
`PRCancellationCleanupError`, `PRCancellationTransactionError`, and generic
`CancellationDrainError` instances within its existing bounded cause walk.

- PR cleanup and transaction failures retain the PR number, the specific
  composite cause type, and the appropriate cleanup or transaction error type
  and redacted summary.
- Other drain failures add a redacted, length-bounded operation, cause type,
  cleanup error type, and cleanup error summary.
- Nested drain errors are traversed without loops and prefer the most specific
  PR cleanup evidence when present.
- Exception messages are never copied without the existing secret redaction and
  size bound.

The phase wrapper will continue returning a failure state for an internal phase
timeout, but that state must now prove when cleanup also failed. External caller
cancellation will continue to propagate rather than being converted to a normal
agent result.

## Error handling

- Model or schema failures are recorded once and then retain their existing
  control flow.
- Cancellation-drain failures are never converted into tool errors, coverage
  failures, ordinary commit failures, or generic API errors.
- A generated test is never applied after the token budget reaches its terminal
  boundary.
- Production patch state and generated-test authorization are restored on every
  failed or budget-stopped generation attempt.
- Diagnostic output contains exception classes and bounded redacted summaries,
  not credentials or full external responses.

## Test strategy

Implementation will follow red-green-refactor with these regression cases:

1. A successful primary test-generation request records model/provider/node,
   elapsed time, input/output estimates, and total token usage.
2. Invalid structured output and a model error each record one appropriately
   classified invocation and debit tokens.
3. A primary failure followed by Opus escalation records two invocations and
   the complete summed usage.
4. Crossing the token budget after the first request prevents a second request,
   does not apply generated edits, restores production authorization, and ends
   with a budget reason.
5. OCI contract tests reject generated coverage with missing, extra, or
   provider-inconsistent generation invocation history.
6. A real `commit_fix` under the real phase wrapper preserves
   `PRCancellationCleanupError` evidence, including the PR number and cleanup
   error class.
7. Execute, tool-router, PLAN, REFLECT, and coverage paths re-raise synthetic generic
   `CancellationDrainError` instances instead of returning ordinary failures.
8. A generic OCI cleanup failure appears as bounded timeout-cleanup evidence;
   existing PR-specific evidence tests remain unchanged.

After focused tests pass, run the affected suites on Python 3.10, 3.11, and
3.12, then the complete suite, Ruff, `git diff --check`, clean-archive tests and
build, credential scan, and a fresh independent whole-branch review. Preserve
the unstaged `run_trace.py` modification throughout.

## Acceptance criteria

- Every logical test-generation request appears in serialized
  `model_invocations` and contributes to token/time budget calculations.
- Strict artifact validation rejects any generated-coverage claim whose
  generation attempts, invocation history, or escalation fields disagree.
- No second generation request or generated-test success occurs after the hard
  token budget is reached.
- PR and OCI cancellation cleanup failures survive real node and phase-wrapper
  boundaries and are represented in bounded diagnostics.
- Existing ordinary-error behavior and model-selection policy remain stable.
- No unresolved Critical or Important review findings remain before push.
- The branch is pushed and PR evidence is updated only after all final gates
  pass; merge and paid evaluation remain separately authorized actions.
