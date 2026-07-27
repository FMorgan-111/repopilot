# Unified Primary PatchGate Design

**Date:** 2026-07-27

**Status:** Approved for implementation planning

## Goal

Make every model-produced production patch pass through the same exact-checkout
verification boundary before EXECUTE. Gemini remains the primary model and
Claude Opus remains the one-way escalation model, but provider selection no
longer changes the patch protocol.

This closes the current SWE-bench OCI blocker where the primary PLAN path
clears `tool_patch_approval`, installs legacy `PatchEdit` values, and routes to
EXECUTE even though OCI execution requires an exact PatchGate approval.

## Success Priority

RepoPilot continues to optimize for repair success before latency or model
cost. The primary path may use two structured model calls instead of one when
that is required to produce an evidence-bound edit. No implementation may
restore a late or implicit EXECUTE-side authorization shortcut.

## Current Failure

The planning node has two incompatible production paths:

1. The primary Gemini path requests one `PlanDecision`, clears any existing
   PatchGate authorization, stores unapproved `patch_edits`, and routes to
   EXECUTE.
2. The escalated Opus path requests a patch-free `RepairPlan`, requests a
   `VerifiedEditBatch` against bounded exact source evidence, passes the batch
   through PatchGate, and only then routes to EXECUTE.

OCI execution correctly rejects the first path because no exact approval is
present. VERIFY classifies that rejection as an infrastructure failure and
terminates the instance, so a normal first-round Gemini patch cannot reach the
test runner or later escalation.

The existing regression
`test_primary_plan_replacement_atomically_retires_old_gate_approval` currently
asserts the incompatible legacy behavior and must be replaced with assertions
for a newly authorized primary patch.

## Chosen Approach

Use one provider-neutral verified-repair transaction for Gemini and Opus:

```text
ModelPolicy
    |
    v
active provider/model
    |
    v
RepairPlan -- exact bounded source evidence --> VerifiedEditBatch
    |                                             |
    +---------------- PatchGate ------------------+
                         |
              exact patch + frozen approval
                         |
                         v
                      EXECUTE
```

The existing two-stage repair implementation is the reference path. It will be
generalized instead of copied. `ModelPolicy` remains responsible for choosing
the active provider and preserving one-way Gemini-to-Opus escalation. The
repair transaction, evidence limits, tool policy, PatchGate, and executor no
longer branch on provider identity.

## Provider-Neutral Repair Interface

`src/repair_flow.py` will expose a provider-neutral
`generate_verified_repair(...)` function with the existing two-stage return
contract:

```python
async def generate_verified_repair(
    state: AgentState,
    packet: EscalationPacket,
    *,
    first_stage_prompt: str | None = None,
    first_stage_suffix: str = "",
    validate_edits: bool = True,
    router: Callable[..., Awaitable[ToolRouteResult]] = route_tool_intent,
    policy_hook: Callable[[AgentState], None] | None = None,
    tool_counter: list[int] | None = None,
    first_stage_reprompt: Callable[[tuple[str, ...]], str] | None = None,
) -> tuple[RepairPlan, VerifiedEditBatch]:
    ...
```

The function accepts either the primary or escalation provider. It continues
to use the existing bounded `EscalationPacket` representation because that
packet already strips evaluator-only fields, raw HTTP data, and secrets. The
type is not renamed in this change to avoid an unrelated serialization and
artifact migration.

`generate_opus_repair(...)` remains as a compatibility wrapper. It preserves
its escalation-only precondition and then delegates to
`generate_verified_repair(...)`. New production code uses the neutral name;
existing callers outside the planning node do not break abruptly or silently
gain primary-provider behavior through an Opus-named API.

`request_verified_edit_correction(...)` will likewise accept either active
provider. `_call_schema(...)` remains the only model-call boundary and records
the model and provider actually used for every stage.

## PLAN Data Flow

`src/nodes/plan.py` will stop using `PlanDecision` as the primary patch-authoring
protocol. For both providers it will:

1. Apply the deterministic model policy.
2. Build the existing bounded repair packet and rolling-summary suffix.
3. Request a patch-free `RepairPlan` from the active provider.
4. Resolve exact target evidence for the plan.
5. Request a `VerifiedEditBatch` from the active provider.
6. Clear any authorization belonging to an older patch.
7. Install the new `RepairPlan` as `state.active_repair_plan`.
8. Run `validate_patch_batch(...)`.
9. Route to EXECUTE only when PatchGate has installed a matching
   `state.patch_content`, exact-only `state.patch_edits`, and
   `state.tool_patch_approval`.

After acceptance, PLAN may synthesize the existing `PlanDecision` and
`DecisionFrame` presentation fields from the verified `RepairPlan`. Those
objects are reporting and routing views; they are not a second source of patch
truth and must never overwrite the PatchGate output.

The legacy primary structured-response helpers may remain only if another
non-patch planning response still consumes them. Dead patch-authoring branches
should be removed rather than retained behind an unreachable provider check.

## Test Command Preservation

The primary `PlanDecision` currently carries a model-selected `test_command`,
while `RepairPlan` carries only a narrative regression strategy. To avoid a
success regression, `RepairPlan` gains an optional bounded
`test_command: str = ""` field. Both providers may propose it during the first
stage. PLAN copies the value into `state.test_command` after PatchGate accepts
the batch.

The command does not gain execution authority from the model. Local and OCI
execution retain their existing deterministic command parsing, allowlist, and
fallback behavior. An empty or rejected command continues to fall back to the
existing fixed pytest command.

## PatchGate Rejection and Escalation

PatchGate remains fail-closed. A rejected batch clears all patch output and
authorization before any retry.

For each active provider:

1. The initial batch is checked once.
2. At most two bounded correction calls are allowed using the PatchGate issue
   code and real bounded source window.
3. The shared reasoning-tool counter remains monotonic across planning, edit
   generation, and correction calls; the existing eight-call bound is
   unchanged.

If Gemini still has no acceptable batch after its correction allowance, PLAN
records deterministic no-progress evidence and applies the existing one-way
model policy. When Opus is configured and the policy escalates, PLAN clears the
failed transaction and restarts the complete two-stage repair with Opus in the
same PLAN invocation. It must not send Gemini-authored edits directly to Opus
or grant them an approval after escalation.

If escalation is unavailable, the state routes to REFLECT or FAILURE using the
existing retry budget without exposing a patch. If Opus exhausts its bounded
corrections, the existing Opus no-progress limit and failure behavior remain in
force.

Any provider change caused by the token reserve between the two model stages
is allowed because escalation is one-way and `_call_schema(...)` records the
provider used for each call. The second stage still receives only the validated
plan and bounded evidence, never hidden conversation state.

## State Invariants

Before PLAN can route to EXECUTE, all of the following must hold:

- `state.active_repair_plan` is present.
- `state.patch_content` is nonempty.
- `state.patch_edits` is nonempty and every edit is exact-only.
- `state.tool_patch_approval` is present.
- The approval base ref equals the exact checkout commit.
- The approval patch digest equals `state.patch_content`.
- The approval manifest fingerprint remains valid.

`_has_valid_exact_patch_gate_approval(...)` remains the reusable local check.
The OCI preflight in `src/nodes/execute.py` remains unchanged as defense in
depth. EXECUTE must not synthesize, repair, or infer an approval.

Whenever a plan, batch, provider transaction, or patch is discarded,
`_clear_patch_authorization(...)` clears the plan, patch, exact edits, generated
test approvals, and PatchGate authorization atomically.

## Telemetry and Budget Accounting

Every structured model request continues through `_call_schema(...)`, which
must preserve:

- the actual primary or escalation model identity;
- one invocation record per completed or failed model request;
- matching input/output token accounting in the invocation and public totals;
- secret-redacted bounded errors;
- direct propagation of `CancellationDrainError` without false model-error
  telemetry or token debit;
- the existing primary token ceiling and 40,000-token Opus reserve.

The change must not merge two physical model calls into one telemetry record.
The extra verified-edit call is intentional and visible.

## Compatibility

- Gemini remains `gemini-3.5-flash:stable` by default.
- Claude Opus remains `claude-opus-4-8:stable` for one-way escalation.
- Existing environment overrides and independent API keys remain unchanged.
- Local PR repair and SWE-bench OCI repair use the same patch authorization
  protocol.
- Saved historical runs are not rewritten.
- Existing PatchGate, coverage, commit, prediction, and scorer contracts remain
  unchanged.
- No new dependency is introduced.

## Testing Strategy

### Repair-flow unit tests

- A primary state can produce a `RepairPlan` and `VerifiedEditBatch` through
  `generate_verified_repair(...)`.
- The two stages record the primary model/provider and exact token totals.
- A reserve-boundary escalation between stages is one-way and recorded using
  the actual provider for each call.
- Primary and escalation correction requests use the same bounded issue
  payload and tool counter.
- Invalid targets, evaluator metadata, secret-shaped text, unified diffs,
  duplicate anchors, and checkout drift remain rejected.

### PLAN regression tests

- Replace the legacy test that expects primary authorization to be absent.
- A valid Gemini repair reaches `Phase.EXECUTE` with an active plan, exact-only
  edits, a nonempty canonical patch, and matching PatchGate approval.
- PLAN never overwrites the accepted PatchGate patch with a model-authored
  legacy patch.
- Two failed primary correction rounds either restart the full transaction
  under Opus or fail closed when escalation is unavailable.
- Opus correction exhaustion retains the current bounded failure behavior.
- The optional `test_command` survives the verified transaction without
  bypassing executor policy.

### OCI integration regression

- A valid primary repair followed by OCI `execute_fix(...)` passes the exact
  approval preflight and invokes the mocked OCI test runner.
- Missing, stale, or mismatched approval still produces the existing
  `infra_error` failure.
- Invalid primary output produces no prediction patch.

### Verification commands

Run focused repair-flow, planning, PatchGate, execute-security, OCI integration,
model-escalation, and telemetry tests first. Then run Ruff and the complete
pytest suite using the repository `.venv` commands documented in `CLAUDE.md`.

## Acceptance Criteria

1. There is no provider branch that installs model-authored patch data and
   routes to EXECUTE without PatchGate approval.
2. A valid primary Gemini repair passes OCI approval preflight in an automated
   integration test.
3. Invalid or stale primary repairs fail closed with no exported patch.
4. Gemini-to-Opus escalation restarts a complete verified-repair transaction
   and never reuses an unapproved Gemini edit.
5. Tool-call limits, token reserve, cancellation semantics, and telemetry
   identity remain covered by tests.
6. Ruff and the complete local pytest suite pass.
7. No model API call or paid SWE-bench run is required for the code-completion
   claim. A checkpoint-5 live run is a separate, explicitly authorized
   post-merge validation step.

## Non-Goals

- Changing the primary or escalation models.
- Relaxing PatchGate, exact-checkout, OCI, differential-coverage, or scoring
  requirements.
- Moving authorization into EXECUTE.
- Improving repository search, cache policy, generated-test policy, or official
  SWE-bench aggregation.
- Claiming a higher benchmark score before an official scorer run.

## Rollout

Implement the change on a fresh branch/worktree based on the latest target
branch. Preserve the existing uncommitted `run_trace.py` change in the current
worktree. Land the regression tests and implementation in reviewable commits,
run the full local verification suite, and merge only after the approval
invariants are demonstrated. Dispatch checkpoint-5 only after separate user
confirmation because it invokes paid model APIs.
