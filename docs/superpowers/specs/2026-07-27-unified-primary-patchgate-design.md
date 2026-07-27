# Stable Primary Patch Generation Design

**Date:** 2026-07-27

**Status:** Design approved; awaiting written-spec review

## Goal

Make the existing Gemini primary path reliably produce a canonical patch that
can pass the exact PatchGate authorization boundary and reach EXECUTE.

This is a deliberately narrow repair. Gemini keeps its current
`PlanDecision` protocol and all existing control actions. Claude Opus keeps
its existing two-stage escalation flow. The change only closes the gap between
a primary model edit proposal and the PatchGate approval required by OCI.

## Current Failure

The primary and escalation paths currently have different execution
preconditions:

1. Gemini returns a `PlanDecision`. PLAN clears old authorization, copies the
   model's `patch` and `patch_edits` directly into state, and routes toward
   EXECUTE.
2. Opus returns a `RepairPlan` and `VerifiedEditBatch`. PatchGate resolves
   the edits against the exact checkout, creates a canonical diff, and installs
   a matching `ToolPatchApproval`.
3. OCI EXECUTE correctly requires the second form. It rejects the normal
   Gemini path because the primary edits have no exact approval.
4. That rejection is then surfaced as an infrastructure failure, so the patch
   never reaches tests or the normal retry/escalation loop.

The result is deterministic empty-patch behavior even when Gemini proposed a
plausible edit.

## Alternatives Considered

### A. Primary authorization adapter — chosen

Keep `PlanDecision` as Gemini's planning and control protocol. Validate raw
edit data before normalizing that decision; when the data is valid, convert
only those edits into an untrusted `VerifiedEditBatch`, build a narrow
deterministic authorization plan, and run the existing PatchGate.

This has the smallest regression surface and adds no second model call when the
initial edit is valid.

### B. Move Gemini to the Opus two-stage protocol

Require Gemini to emit `RepairPlan` and then `VerifiedEditBatch` on every
planning round. This is architecturally uniform, but it removes or duplicates
the current control-action protocol, adds latency, and changes many tested
call sites.

### C. Relax the OCI or PatchGate preflight

Allow legacy primary edits to execute without exact approval. This would make
patches run sooner but weakens the security boundary and permits checkout
drift. It is rejected.

## Scope

The implementation includes:

- deterministic validation of raw primary edit proposals before lossy
  `PlanDecision` normalization;
- alignment of the primary prompt with PatchGate-supported edit forms;
- a narrow authorization adapter for valid primary structured edits;
- PatchGate authorization before primary patches enter EXECUTE;
- explicit proposal-versus-environment PatchGate issue classification;
- at most two bounded correction attempts for correctable primary patch
  failures;
- normal no-progress and Gemini-to-Opus escalation after correction
  exhaustion;
- regression tests for PLAN and the OCI preflight boundary.

The implementation does not include:

- changing the default Gemini or Opus models;
- replacing the primary `PlanDecision` protocol;
- changing tool selection, context summaries, API caching, coverage
  generation, or SWE-bench scoring;
- weakening PatchGate or OCI validation;
- adding dependencies;
- running paid evaluation jobs.

## Architecture

```text
raw Gemini JSON
      |
      +-- tool/context/ask/stop ---------------------> existing behavior
      |
      v
PrimaryProposalValidator  <---- full-plan correction --+
      |                                               |
      +-- invalid / raw-diff-only --------------------+
      |
      v
PlanDecision + deterministic RepairPlan + VerifiedEditBatch
      |
      v
PatchGate  <-------------- edit correction ------------+
      |                                               |
      +-- proposal / full_plan -----------------------+
      |
      +-- proposal / edit ----------------------------+
      |
      +-- environment issue -------------------------> infrastructure path
      |
      v
canonical patch + exact edits + approval
      |
      v
EXECUTE
```

Model-specific planning remains at the entrance. The exact-checkout execution
boundary is common to both providers.

## Components

### Primary prompt contract

Gemini keeps the same `PlanDecision` and control variants, but its system
prompt stops advertising behaviors that the authorization path cannot accept.
It requires `patch=""`, `replace_all=false`, exactly one `search` or
`node_target` anchor per edit, and `recommended_action="execute"` only when
at least one structured edit is present. The contradictory legacy instruction
to fall back to a unified diff is removed.

### Raw primary proposal validator

Validation occurs after reasoning-response kind and tool routing, but before
`_normalize_plan_decision(...)` can discard anchorless edits or silently
prefer one of two anchors. It receives the raw primary JSON and produces one
of:

1. the existing non-patch control result;
2. a normalized `PlanDecision` plus a valid primary authorization
   transaction; or
3. bounded proposal issues with no `RepairPlan` or `VerifiedEditBatch`.

A response has executable-patch intent when it contains a nonempty raw
`patch`, a nonempty raw `patch_edits` or legacy `edits` array, or a
decision frame recommending `execute`. The two edit-array aliases may not
conflict. An ordinary context, ask-user, or stop decision does not enter patch
validation. An explicit `execute` action without usable structured edits, or
a non-execute action accompanied by edit payloads, is a correctable
`decision_action_conflict` instead of being silently rerouted.

For every raw edit, the validator:

- resolves the existing `file_path`, `file`, and `path` aliases, rejects
  conflicting nonempty aliases, and accepts only one canonical
  repository-relative result;
- requires exactly one supported anchor: `search` or `node_target`;
- requires nonempty replacement text;
- rejects `replace_all` because PatchGate authorizes one exact target;
- enforces the existing `VerifiedEditBatch` and `RepairPlan` edit, file,
  symbol, and field-size limits before constructing either model;
- applies the existing evaluator-metadata, secret-shaped payload, and
  unified-diff-content safety checks;
- copies only `file_path`, `search`, `node_target`, and `replace`;
- discards model-supplied `resolved_target_symbol`,
  `expected_content_sha256`, `exact_only`, and other identity metadata;
- assigns a bounded non-sensitive intent string locally.

Malformed raw edits are not silently removed or normalized into a different
operation. They produce bounded proposal issues such as
`structured_edits_required`, `invalid_edit_format`, or
`unsupported_edit_mode`. Issue payloads never echo secrets or an entire raw
model response.

Raw failures use a small `PrimaryProposalIssue` schema with bounded
`code`, `file_path`, and `message` fields, with origin fixed to
`proposal` and recovery fixed to `full_plan`. They are not represented as
a fake PatchGate result.

### Primary patch authorization adapter

Only after every raw edit passes validation does the adapter construct the
existing verified transaction types. It builds a deterministic `RepairPlan`
used only as the authorization scope:

- `target_files` is the unique set of actual proposed edit paths;
- `target_symbols` is the unique set of actual nonempty node targets;
- narrative fields are fixed local descriptions rather than copied from
  model-authored patch text;
- the user-facing plan remains `PlanDecision.plan` in `state.fix_plan`.

This plan does not claim independent semantic validation. Its purpose is to
bind the exact proposed paths and nodes to PatchGate without letting the model
author trusted checkout identity.

A raw unified diff is never execution authority. If a response has a raw patch
but no usable structured edits, no authorization transaction is created.
Creating new files from a raw diff is outside this minimal change.

### Full-Plan correction

When raw validation fails before a transaction exists, or PatchGate reports an
issue requiring a new target scope, PLAN makes a normal primary
`PlanDecision` request with a bounded corrective suffix. If a transaction
already exists, its active plan and partial gate output are discarded first,
but its correction count is retained.

The suffix contains only sanitized issue codes, concise messages, and trusted
bounded source evidence already present in `state.relevant_files` or the
evidence store. It does not require a fabricated `RepairPlan`, an empty
`VerifiedEditBatch`, or the invalid raw path as trusted scope.

The replacement response re-enters raw validation from the beginning, so it
may select a different canonical file or symbol. Tool, context, ask-user, and
stop variants continue through the existing primary protocol. This call
consumes the same two-attempt proposal budget as later edit corrections.

### PatchGate

`validate_patch_batch(...)` remains the sole producer of:

- canonical `state.patch_content`;
- exact-only `state.patch_edits` bound to whole-file preimages;
- `state.tool_patch_approval` bound to the base ref, patch digest, and result
  manifest.

PLAN must not overwrite any of those values with the original model response.
EXECUTE must not synthesize or repair approval.

PatchGate issues gain two explicit fields:

- `origin: Literal["proposal", "environment"]`;
- `recovery: Literal["edit", "full_plan", "none"]`.

Origin semantics are:

- `proposal` covers path, policy, anchor, content, size, and generated-diff
  failures that a new model edit can correct;
- `environment` covers missing or mismatched exact checkout state,
  repository I/O failures, unavailable git preimages, and checkout drift.

Recovery is assigned by the branch that creates the issue, not inferred from
its code:

- `edit` retains the active plan for errors such as a missing or ambiguous
  search block, an empty replacement, or a generated diff that does not apply;
- `full_plan` discards the active plan when the selected path or scope itself
  must change, including a missing planned path, sensitive/generated/binary
  target, or source file that exceeds the allowed size;
- `none` is mandatory for every environment-origin issue.

Existing issue codes remain for compatibility; routing does not string-match
`code` or `message`. Both new fields have backward-compatible proposal/edit
defaults, while every full-plan and environment branch sets them explicitly.
Cancellation continues to propagate directly and is never converted into an
issue.

Routing precedence for a mixed result is `environment/none`, then
`proposal/full_plan`, then `proposal/edit`. Therefore a canonical but wrong
path can be replaced by a different legal path instead of being retried forever
inside its original `RepairPlan`.

### Post-transaction edit correction

The existing bounded verified-edit correction request is generalized to the
primary provider. It is invoked only after a valid authorization plan and
batch exist and PatchGate selects `proposal/edit` recovery.

Each correction receives:

- the deterministic authorization plan;
- bounded issue codes and messages;
- the previous untrusted edit batch;
- PatchGate's bounded real-code correction windows.

It returns only a `VerifiedEditBatch`; it does not replace Gemini's normal
`PlanDecision` protocol. The shared reasoning-tool counter remains monotonic
and the existing global tool-call bound remains unchanged.

The primary correction transaction is provider-frozen. Before each correction,
PLAN reapplies the existing model policy. If policy has switched to Opus, the
remaining primary corrections are abandoned and the normal Opus two-stage
transaction starts from a clean state. Opus never performs a correction-only
call against a Gemini-authored transaction. After that boundary check, the
correction call uses the frozen primary provider and must not independently
reapply escalation policy.

### Correction budget lifecycle

The initial proposal gets at most two total corrections across full-plan and
edit correction types. One of each exhausts the budget.

The count is transaction-local: it is initialized once when an executable
primary proposal first appears, is not reset by an intermediate PatchGate
rejection, and is reset on acceptance, final discard, or provider restart.
A later independent PLAN invocation starts at zero. The existing serialized
`state.patch_correction_count` may mirror the local count for diagnostics, but
must not carry an exhausted count into a new proposal or a new Opus
transaction.

Only a completed replacement patch proposal consumes one correction. Tool and
context actions inside a correction use the existing reasoning-tool budget but
do not consume an additional patch correction. An ask-user or stop outcome
ends and resets the patch transaction before following its existing branch.

## PLAN Data Flow

1. PLAN applies the existing model policy and calls Gemini exactly as it does
   today.
2. Reasoning kind and tool routing are validated first. Tool calls, context
   requests, user questions, and stop responses follow their existing branches.
3. Raw executable edit data is inspected before `PlanDecision` normalization.
4. The first executable proposal begins a transaction: PLAN clears approval
   belonging to an older proposal and initializes its correction count to zero.
5. Raw proposal issues trigger a bounded full-`PlanDecision` correction. No
   authorization plan or empty edit batch is fabricated.
6. Once raw edits are valid, PLAN stores the user-facing plan and
   model-proposed test-command suggestion, while discarding the raw patch as
   execution authority.
7. The adapter creates the deterministic `RepairPlan` and
   `VerifiedEditBatch`, then installs that plan as
   `state.active_repair_plan`.
8. PatchGate validates the batch against the exact checkout.
9. On acceptance, PLAN retains only PatchGate's canonical patch, exact edits,
   and approval, resets the transaction counter, records the normal decision
   frame, and routes to EXECUTE.
10. A `proposal/edit` rejection clears only partial gate output and requests
    a bounded verified-edit correction against the same active plan.
11. A `proposal/full_plan` rejection discards the active authorization scope
    and requests a complete `PlanDecision`, retaining the transaction count.
12. An `environment/none` rejection ends the transaction and follows the
    existing infrastructure path without spending a model correction.
13. After two total unsuccessful corrections, PLAN records deterministic
    no-progress evidence, clears and resets the transaction, and returns to the
    existing retry/escalation policy.
14. If policy selects Opus at any correction boundary, the primary transaction
    is cleared and reset before the existing Opus two-stage flow starts.

An empty decision remains an ordinary non-patch planning result. It never
receives a fabricated approval.

## Error Classification

Proposal-origin, correctable patch-generation failures include:

- missing structured edits;
- malformed or noncanonical paths;
- unsupported `replace_all`;
- missing, duplicated, or ambiguous anchors;
- edits outside allowed repository scope;
- empty or non-applying edits;
- generated diffs that fail `git apply --check`.

Raw-format failures and proposal issues whose target scope must change use
full-plan correction. Proposal issues that can be repaired within the existing
files and symbols use edit correction. Both clear partial patch output, consume
the same bounded correction budget, and become no-progress evidence if
unresolved.

Environment-origin failures include exact-checkout unavailability or mismatch,
repository I/O failure, unavailable git preimages, and checkout drift. They do
not consume model corrections and retain infrastructure semantics.
Cancellation propagates directly. OCI preflight remains a defense-in-depth
check, not a normal place to discover an unapproved primary patch.

## State Invariants

PLAN may route a primary patch to EXECUTE only when all of these hold:

- `state.active_repair_plan` is present;
- `state.patch_content` is nonempty;
- `state.patch_edits` is nonempty and every edit is exact-only;
- `state.tool_patch_approval` is present;
- approval base ref equals the exact checkout ref;
- approval patch digest equals `state.patch_content`;
- approval manifest fingerprint is valid.

A `proposal/edit` rejection clears partial gate output while retaining the
active plan and transaction-local count. A `proposal/full_plan` rejection
also clears the active plan but retains the count for the replacement proposal.
Ending the transaction atomically clears the active plan, patch content, exact
edits, generated-test approvals, and PatchGate approval, and resets the count.
Acceptance also resets the count to zero.

## Test Command and Telemetry

The accepted primary decision may continue to supply `test_command`.
Existing command parsing, allowlists, and fallbacks remain authoritative; patch
approval does not authorize arbitrary commands.

Every correction model call uses the existing primary or structured model-call
boundary and records the actual provider, model, latency, status, and token
counts. Direct propagation of cancellation and secret-redacted bounded errors
remains unchanged. A valid initial primary patch adds no extra model call.

## Testing Strategy

### Adapter unit tests

- Raw validation observes anchorless and dual-anchor input before the current
  lossy normalizer and returns bounded issues.
- Conflicting decision actions and edit payloads require correction.
- Existing path aliases remain compatible, while conflicting aliases are
  rejected.
- Converts a search edit and a node-target edit into a bounded verified batch.
- Derives exact file and symbol scope without trusting model identity fields.
- Rejects raw-diff-only, anchorless, noncanonical, and `replace_all` inputs.
- Cannot authorize sensitive paths or evaluator metadata.

### PatchGate classification tests

- Exact-checkout absence, mismatch, repository I/O, and preimage failure return
  `environment/none`.
- Wrong planned path, forbidden artifact, and oversized source return
  `proposal/full_plan`.
- Local anchor, replacement, and generated-diff failures that remain repairable
  inside the active scope return `proposal/edit`.
- A mixed result containing an environment issue takes the environment route
  without requesting a correction; otherwise full-plan recovery takes
  precedence over edit recovery.
- Cancellation still propagates without a result.

### PLAN regression tests

- The primary prompt no longer advertises unified-diff fallback or
  `replace_all`.
- Replace the legacy assertion that a primary plan reaches EXECUTE without
  approval.
- A valid Gemini edit reaches `Phase.EXECUTE` with a matching canonical patch,
  exact-only edits, active authorization plan, and PatchGate approval.
- The raw model patch cannot overwrite PatchGate output.
- A correctable failure can succeed on either of two bounded corrections.
- A raw-diff-only response can be repaired by a full-`PlanDecision` format
  correction without fabricating an empty repair transaction.
- One full-plan correction plus one edit correction exhausts the shared budget;
  a later independent proposal starts again at zero.
- A canonical but nonexistent or forbidden planned path can be replaced by a
  different legal path through full-plan correction.
- Correction exhaustion clears all executable patch state and records
  no-progress/escalation evidence rather than an infrastructure failure.
- Environment-origin PatchGate failure performs no model correction.
- Provider change during correction discards the primary transaction before
  the full Opus flow begins.
- Existing tool, context, ask-user, stop, decision-frame, and test-command
  behavior remains unchanged.

### Execution boundary tests

- OCI EXECUTE accepts a PatchGate-authorized primary patch and invokes the
  mocked runner.
- OCI EXECUTE still rejects missing, stale, or mismatched approval.
- Opus authorization and correction tests remain green without behavioral
  changes.

### Verification

Run focused adapter, PLAN, PatchGate, and execution-security tests first, then
the complete repository test suite. Paid SWE-bench evaluation is a separate,
explicitly approved checkpoint after deterministic tests pass.

## Acceptance Criteria

The change is complete when:

1. A valid initial Gemini structured edit reaches EXECUTE with no extra model
   call.
2. Every primary patch reaching EXECUTE has a current exact PatchGate approval.
3. Invalid primary edits never reach EXECUTE and receive at most two bounded
   correction attempts.
4. Correctable patch failures no longer terminate as OCI infrastructure
   failures.
5. Scope-changing proposal failures can select a new legal target instead of
   being trapped inside the rejected plan.
6. Environment failures never consume the proposal correction budget.
7. A new proposal cannot inherit an exhausted correction count.
8. Gemini control actions and the Opus escalation flow retain existing
   behavior.
9. Focused tests and the full suite pass.
