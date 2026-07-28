# Unified Patch-Edits Model Switching Design

**Date:** 2026-07-27

**Status:** Design approved; awaiting written-spec review

## Goal

Use one model-facing repair protocol for both Gemini and Claude Opus:
`PlanDecision.patch_edits`.

RepoPilot starts with Gemini. After two unsuccessful Gemini repair rounds it
switches one-way to Opus. The default repair budget is five full transactions:
Gemini, Gemini, Opus, Opus, Opus. Provider selection changes only the active
repair model. Patch-authoring prompts, tools, response schema, patch validation,
execution, and verification remain common.

The immediate objective is reliable production of a canonical, executable
patch. Repair success takes priority over latency and model cost.

## Current Failure

PLAN currently has two model protocols:

1. Gemini returns one `PlanDecision` containing legacy `PatchEdit` values.
   PLAN copies those values into state without the exact PatchGate approval
   required by OCI.
2. Opus uses a separate two-stage `RepairPlan` then `VerifiedEditBatch`
   protocol. Only that path runs PatchGate and receives exact approval.

This split duplicates planning behavior, prevents a literal model-only switch,
and causes valid-looking Gemini edits to be rejected before tests.

The current escalation policy also waits for repeated no-progress signatures.
The new policy uses an explicit two-failed-round Gemini threshold, while
retaining immediate escalation for exhausted Gemini gateway retries and the
primary token reserve.

## Alternatives Considered

### A. Unified `PlanDecision.patch_edits` — chosen

Both models use the current primary planning and control protocol. A common
adapter treats every model edit as untrusted input and sends it through
PatchGate. A provider switch changes only `model` and `provider`.

This is the smallest model-facing protocol and keeps the existing tool,
context, ask-user, stop, decision-frame, and test-command protocol surface.
Success-first stop routing is defined explicitly below.

### B. Unified `VerifiedEditBatch`

Both models could emit `RepairPlan` followed by `VerifiedEditBatch`. This
would preserve the current Opus design but require extra model calls and either
remove or duplicate the existing control-action protocol.

### C. Keep both protocols

The existing dual path could be repaired independently. It would retain the
branching and make model switching continue to mean changing both model and
protocol.

## Scope

The implementation includes:

- one shared PLAN prompt and structured-response protocol for Gemini and Opus;
- raw edit validation before lossy `PatchEdit` normalization;
- one common `patch_edits` authorization adapter and PatchGate boundary;
- one full-`PlanDecision` correction loop for all model-correctable patch
  failures;
- an explicit failed-repair-round counter for Gemini;
- one-way Gemini-to-Opus switching after two failed Gemini rounds;
- immediate Opus fallback after exhausted Gemini gateway retries or primary
  token reserve;
- a five-transaction default repair budget by changing the canonical
  `max_retries` default and public upper bound from 3 to 4 while preserving its
  retries-after-initial semantics;
- Opus use of the remaining three default repair transactions after two failed
  Gemini transactions;
- consistent propagation of that default through state construction, public
  Python entry points, CLI, API, saved-run validation, compatibility endpoints,
  and active evaluation harnesses;
- removal of the Opus-specific production planning and correction branch;
- provider-neutral reflection and removal of Opus-only no-progress terminals;
- focused regression and execution-boundary tests.

The implementation does not include:

- changing default model names, API endpoints, or key handling;
- changing context summaries, tool selection policy, API cache, coverage
  generation, or SWE-bench scoring;
- weakening PatchGate or OCI exact-approval checks;
- granting more than the approved five default repair transactions or changing
  the token budget;
- running paid evaluation jobs;
- rewriting historical serialized run artifacts.

## Architecture

```text
                     ModelPolicy
                         |
          +--------------+--------------+
          |                             |
   Gemini (start)                 Opus (one-way)
          |                             |
          +--------------+--------------+
                         |
             shared PlanDecision prompt
                         |
             kind=tool / plan / stop
                         |
                 plan.patch_edits
                         |
                common validator
                         |
             common PatchGate adapter
                         |
                    PatchGate
             +-----------+-----------+
             |                       |
         accepted              model-correctable
             |                       |
   canonical patch + approval   next full PlanDecision
             |
           EXECUTE
             |
           VERIFY
```

Model policy is orthogonal to patch protocol. There is no Opus-specific
model-output branch.

## Unified Model Contract

Gemini and Opus receive the same system prompt, bounded user prompt, reasoning
tool protocol, and response validation. Both may return:

- `kind="tool"` with one constrained tool intent;
- `kind="plan"` with `PlanDecision`;
- `kind="stop"`.

`PlanDecision` retains:

- `plan`;
- `patch_edits`;
- `files`;
- `test_command`;
- `decision_frame`.

The legacy `patch` field remains present only for serialized compatibility and
must be empty in new model output. It never grants execution authority.

Every edit must:

- identify one canonical repository-relative path, accepting the existing
  `file_path`, `file`, or `path` aliases only when they do not conflict;
- provide exactly one anchor: `search` or `node_target`;
- provide nonempty `replace` text;
- use `replace_all=false`;
- stay within existing edit-count and field-size limits;
- exclude unified-diff payloads, evaluator-only metadata, secret-shaped text,
  model-authored hashes, and model-authored exactness claims.

An explicit `recommended_action="execute"` requires at least one usable
structured edit. Non-execute actions may not carry executable edit payloads.

The shared prompt removes the contradictory unified-diff fallback and never
asks either model for `RepairPlan` or `VerifiedEditBatch`.

## Common Patch Authorization

Raw `patch_edits` are inspected before the current normalizer can discard an
anchorless item or silently choose one of two anchors. Malformed edits produce
bounded model-correctable issues; they are not silently changed into a
different operation.

The common authorization adapter:

1. normalizes only the supported path aliases and fields;
2. discards `resolved_target_symbol`, `expected_content_sha256`,
   `exact_only`, and all other model-supplied identity metadata;
3. creates the existing internal `VerifiedEditBatch`;
4. creates a deterministic internal `RepairPlan` whose file and symbol scope
   is exactly the normalized proposed edits and whose narrative fields are
   fixed safe runtime text rather than model-authored plan or patch content;
5. installs that plan as the active authorization scope;
6. calls `validate_patch_batch(...)`.

`RepairPlan` and `VerifiedEditBatch` remain internal PatchGate compatibility
types. They are constructed deterministically without another model call and
are not a second model protocol.

PatchGate remains the sole producer of:

- canonical `state.patch_content`;
- exact-only `state.patch_edits` bound to checkout preimages;
- `state.tool_patch_approval` bound to the base ref, patch digest, and result
  manifest.

PLAN never overwrites accepted gate output with raw model content. EXECUTE
never infers or repairs approval.

## Unified Correction Loop

Every model-correctable raw-validation or PatchGate failure returns to the same
full-`PlanDecision` call. The bounded correction suffix contains:

- sanitized issue codes and concise messages;
- bounded real-code windows when PatchGate has them;
- trusted repository evidence already available in state.

The next decision may change the file, symbol, hypothesis, or complete edit
batch. There is no provider-specific `VerifiedEditBatch` correction request
and no edit-only scope that can trap the model on a bad path.

Tool calls and context collection inside a correction use the existing
reasoning-tool budget. They do not count as completed repair rounds.

## Repair-Round Accounting

Add a bounded `state.primary_failed_repair_rounds` counter. It controls only
provider selection and never grants additional global retries.

Each full repair transaction receives a monotonic `repair_round_id`. Reasoning
tool calls and their evidence reprompts remain inside that transaction. A new
full `PlanDecision` correction receives a new ID. At transaction start, state
persists that ID. Immediately before each full PLAN model call, it binds the
round's current author provider/model. An immediate gateway or token-reserve
fallback keeps the same round ID but rebinds the author to Opus before the Opus
call.

When PatchGate accepts a patch, it freezes the current round ID and author as
the authorized patch attribution. That frozen attribution cannot change before
EXECUTE copies it into the resulting `FixAttempt`; VERIFY reads only the
`FixAttempt` attribution. Pre-execution failure accounting reads the bound
author of the failed model call. All new fields use backward-compatible
defaults for historical states.

One unsuccessful Gemini repair round is recorded exactly once when:

- a completed response requests execute or stop but contains no usable
  structured edit;
- raw edit validation rejects the executable proposal;
- PatchGate rejects the proposal for a model-correctable reason;
- PatchGate accepts the proposal but a deterministic convergence guard rejects
  it before EXECUTE, including repeated failed-patch replay or required
  assertion-target diversification;
- a PatchGate-approved Gemini patch executes but VERIFY reports a non-
  infrastructure failure.

A valid patch is not counted when PLAN accepts it. If its tests pass, the run
continues to coverage and completion. If its tests fail, VERIFY records the
single failed round before returning to repair.

Accepting a patch and the existing `record_progress(...)` diagnostics do not
reset `primary_failed_repair_rounds`; only a verified successful run makes the
counter irrelevant. This preserves the sequence “one invalid Gemini proposal,
then one Gemini patch that fails tests” as two failed primary rounds.

The following do not increment the counter:

- reasoning tool calls;
- collect-more-context;
- ask-user;
- cancellation;
- exact-checkout, repository I/O, sandbox, or other infrastructure failures;
- Gemini gateway failure after the gateway's own retries are exhausted.

The last item triggers immediate provider fallback instead of pretending that
Gemini reasoned unsuccessfully.

Counter updates must be idempotent across graph routing and resume. The same
plan outcome or `FixAttempt` cannot be counted once in PLAN and again in
VERIFY. A single `last_counted_repair_round_id` is sufficient because graph
round IDs are monotonic and repair execution is sequential.

### Global retry consumption

One shared `record_failed_repair_round(...)` helper owns retry consumption.
For a model-correctable failure under either provider it:

1. ignores an already-counted round ID;
2. increments `primary_failed_repair_rounds` only when the recorded provider
   is primary;
3. checks whether another repair is allowed by the existing
   `retry_count/max_retries` contract;
4. when allowed, consumes exactly one retry before routing to the next full
   repair transaction;
5. otherwise routes to the existing terminal failure path.

PLAN uses this helper for unusable model output, PatchGate rejection, or any
post-gate convergence rejection that clears approval before EXECUTE. VERIFY
uses it for a non-infrastructure failed `FixAttempt`. Existing scattered retry
increments and fixed blockers on those paths are removed or delegated so one
failure cannot spend the global retry twice.

Tool/context/ask-user actions, environment failures, and a provider-availability
fallback do not spend a semantic repair retry. A Gemini gateway failure after
its own retries switches provider within the current global repair attempt.

The existing retry contract treats `max_retries` as retries after the initial
repair transaction. When every transaction fails and no earlier token or
infrastructure limit intervenes:

| `max_retries` | Full repair transactions | Normal provider sequence |
| ---: | ---: | --- |
| 0 | 1 | Gemini |
| 1 | 2 | Gemini, Gemini |
| 3 (legacy default or explicit configuration) | 4 | Gemini, Gemini, Opus, Opus |
| 4 (new default) | 5 | Gemini, Gemini, Opus, Opus, Opus |

This table counts full patch-producing transactions, not bounded reasoning-tool
calls inside one transaction. Immediate gateway fallback or primary token
reserve may select Opus earlier without granting another transaction.
`max_retries=4` is therefore the required default for five total transactions;
setting it to 5 would incorrectly grant a sixth transaction.

### Default and limit propagation

Define one canonical default, `DEFAULT_AGENT_V2_MAX_RETRIES = 4`, and one
canonical public upper bound, `MAX_AGENT_V2_MAX_RETRIES = 4`, beside the
existing token-budget constant. Active code imports these values rather than
repeating numeric defaults.

The implementation updates every active default or boundary that can otherwise
override or reject the state value:

- `AgentState.max_retries`;
- `agent_v2(...)` and the backward-compatible
  `intelligent_analyze_issue(...)` Python entry points;
- the CLI default and help text;
- `AgentV2Request` default and upper bound;
- authorized saved-run validation, so a paused run with `max_retries=4` can be
  resumed;
- the `/intelligent-agent` compatibility clamp;
- `eval/agent_v2_harness.py` and `eval/harness.py` function and CLI defaults;
- the OCI generation runner's explicit repair budget; and
- current README/API documentation that advertises the public range.

New eval runs use 4 so they exercise the same success-first production policy.
Historical eval plans, stored results, and explicit test fixtures that request
3 remain unchanged and reproducible; an explicit `max_retries=3` is still a
valid four-transaction configuration.

## Model Switching

Before every repair-model call, policy applies these one-way rules:

1. Start with the configured primary Gemini model.
2. Continue Gemini while `primary_failed_repair_rounds < 2`.
3. After the second failed Gemini repair round, switch immediately to the
   configured Opus provider and model.
4. Also switch immediately when Gemini gateway retries are exhausted or the
   existing primary token-reserve boundary is reached.
5. Never switch the active repair provider back to Gemini during the run.

Existing safe reasons for empty and invalid structured completions remain.
Retryable transport exhaustion, including HTTP 503 after gateway retries, adds
one bounded allowlisted reason such as
`primary_gateway_unavailable_after_retries`; raw response bodies and
credentials never enter policy state or diagnostics.

The fixed two-round threshold likewise uses one explicit allowlisted reason,
`primary_repair_round_limit`, rather than reusing a changing no-progress
signature.

When the second primary failure is recorded and a global retry remains,
escalation takes precedence over same-signature or repeated-failure early
termination. There is no intervening Gemini repair call. Existing
`no_progress_rounds` and signature fields remain useful diagnostics, but they
no longer select the repair model independently of the explicit primary-round
counter.

The active provider changes immediately after the second counted primary
failure. If the graph's next phase is REFLECT, the first Opus call may be a
reflection call; REFLECT does not author patches. The first Opus
patch-authoring call always uses the same PLAN prompt plus the existing bounded
attempt summary, failure evidence, and code context. It does not receive hidden
evaluator data or a Gemini-authored approval.

Opus receives no additional fixed two-round cap. Each unsuccessful Opus plan,
stop, PatchGate proposal, or verified patch consumes one existing global retry
through the same helper. It continues while a retry remains, subject also to
the existing token budget, phase timeout, and graph guard. Under the new
default, a normal all-failure run gives Opus exactly three full transactions
after the two Gemini transactions. Explicit non-default retry configurations
continue to follow the same shared retry contract. The token budget is
unchanged.

If Opus is not configured, the existing explicit unavailability behavior
remains authoritative; state must not claim that escalation occurred.

## Reflection, Stop, and Summary Semantics

The single patch protocol applies to patch-authoring PLAN calls. REFLECT remains
a separate, non-patch `ReflectDecision` schema, but its prompt, response
validation, and tool behavior become provider-neutral. After escalation it uses
the active Opus model without an Opus-specific reflection protocol.

The shared reflection input uses the stricter existing bounded/redacted issue,
attempt, patch, failure, and evidence fields. Neither provider receives raw
HTTP bodies, credentials, evaluator-only fields, or an unbounded conversation.

REFLECT is advisory inside an already-counted failed repair round. It does not
consume another global repair retry. If reflection schema or tool reasoning
fails after its bounded gateway/tool retries, RepoPilot creates the existing
deterministic reflection fallback and proceeds to PLAN when the global repair
budget permits. A reflection `kind="stop"` has the same fallback behavior; it
does not silently cancel the remaining patch budget.

The lightweight outcome-summary call is an auxiliary summarizer, not the active
repair model. It continues through the existing lightweight summary path after
escalation, but it must not mutate `active_provider`, `active_model`, or
repair-round attribution. Summary failure retains the deterministic fallback.

A PLAN `kind="stop"` means that the active repair model produced no patch. For
both providers it is recorded as one model-correctable failed repair round.
When a global retry remains, RepoPilot requests another full PLAN decision and
applies the normal Gemini-to-Opus policy; otherwise it terminates. Human stop,
runtime cancellation, and hard tool-policy stops retain their terminal
semantics.

## Failure Classification

Only two routing classes are needed:

- `model_correctable`: malformed edit format, raw diff, wrong or forbidden
  path, missing or ambiguous anchor, empty change, oversized proposal, or
  generated patch that fails exact apply validation, plus deterministic
  post-gate rejection of a repeated failed patch or insufficient target
  diversification;
- `environment`: exact checkout unavailable or changed, repository or git
  I/O failure, sandbox/OCI failure, or post-approval drift.

When a global retry remains, model-correctable failures request a new full
`PlanDecision`; otherwise they terminate through the existing budget failure
path. Environment failures do not consume a model repair round and retain
existing infrastructure handling. If a result contains both classes,
environment routing takes precedence.

Cancellation is not converted into either result class; it propagates directly.

Classification is assigned where the failure is created, not inferred later
from human-readable error strings.

## State and Compatibility

Before either model can route to EXECUTE:

- `state.active_repair_plan` is present;
- `state.patch_content` is nonempty;
- `state.patch_edits` is nonempty and exact-only;
- `state.tool_patch_approval` matches the checkout ref and patch digest;
- the approval manifest fingerprint is valid.

Starting a new proposal atomically retires older patch content, edits, active
scope, generated-test approvals, and PatchGate approval. A rejected proposal
cannot leak executable state into the next round.

`active_provider`, `active_model`, `escalated`, and
`escalation_reason` remain the public provider state. Model invocation
telemetry records the actual model and provider for every call.

The state also persists the monotonic round sequence, current authorized round
attribution, and last-counted round ID so async routing and resume cannot
double-count an outcome.

Historical `RepairPlan`, `VerifiedEditBatch`, and
`opus_no_progress_rounds` state fields may remain for deserialization
compatibility. They are no longer model-facing production protocols.

New round-identification and `FixAttempt` attribution fields use safe defaults
so historical state remains loadable.

## Production Cleanup

`src/nodes/plan.py` removes the `two_stage_repair` branch and calls the same
planner for both providers.

After a repository-wide caller check, remove production-only use of:

- `generate_opus_repair(...)`;
- `request_verified_edit_correction(...)`;
- Opus-specific planning/correction prompts;
- Opus-only PLAN decision synthesis.

Remove or provider-neutralize every production `record_opus_no_progress(...)`
call in PLAN, REFLECT, and the reasoning-tool loop. Opus patch failures consume
the shared global repair budget; reflection falls back deterministically; tool
reasoning remains bounded by the existing shared tool-call limit. No hidden
Opus-only two-round terminal remains.

REFLECT also removes its Opus-specific prompt/data-flow branch and uses one
bounded, sanitized `ReflectDecision` contract for either active repair model.

Shared repair-context, evidence, sanitization, and internal PatchGate types
remain when still used. Cleanup must not remove historical state parsing or
unrelated escalation telemetry.

## Testing Strategy

### Unified protocol tests

- The same `PlanDecision.patch_edits` response fixture works under Gemini and
  Opus.
- Both providers support the same tool, context, ask-user, stop, decision-frame,
  and test-command variants.
- Neither prompt advertises raw diff, `replace_all`, `RepairPlan`, or
  `VerifiedEditBatch`.
- Raw validation observes missing and dual anchors before lossy normalization.

### Patch authorization tests

- A valid Gemini edit and an identical Opus edit both produce canonical patch
  content, exact-only edits, and matching PatchGate approval.
- Model-supplied hashes and exactness flags are discarded.
- Raw patch content cannot overwrite PatchGate output.
- Model-correctable rejection clears executable state and returns bounded
  correction evidence.
- Environment failure performs no model correction and does not increment the
  primary counter.

### Switching tests

- With retry budget available, one failed Gemini round is followed by Gemini.
- With Opus configured and retry budget available, two failed Gemini rounds
  are followed immediately by Opus.
- The two-round transition records `primary_repair_round_limit` independent
  of no-progress signature contents.
- One invalid proposal plus one failed verified attempt also switches to Opus.
- Tool, context, and ask-user actions do not advance the counter.
- Exhausted Gemini gateway retries switch immediately without incrementing the
  reasoning-failure counter.
- HTTP 503 exhaustion records only the bounded allowlisted escalation reason.
- Gemini gateway failure followed by an Opus patch in the same round freezes
  Opus attribution; a later test failure spends one global retry and does not
  increment the primary failure counter.
- Token-reserve fallback performs the same pre-call rebind and attribution
  freeze.
- Repeated-patch and assertion-diversity post-gate rejection each consume one
  shared retry, clear approval, and contribute to the Gemini two-round limit.
- Switching is one-way; the first Opus patch-authoring call uses the shared
  PLAN schema and prompt even when an Opus REFLECT call occurs first.
- With the new default, the exact all-failure sequence is Gemini, Gemini, Opus,
  Opus, Opus, followed by terminal failure; no sixth model transaction occurs.
- Opus is limited only by the remaining shared global and token budgets.
- A counted outcome cannot increment twice across PLAN, EXECUTE, VERIFY, or
  resume.
- Failed Opus plan and stop outcomes consume one remaining global retry rather
  than spinning inside PLAN.
- `max_retries` values 0, 1, 3, and 4 produce the documented normal provider
  sequences without granting hidden calls.
- State, both public Python entry points, CLI, API, both active eval harnesses,
  and the OCI generation runner all resolve their default to the canonical
  value 4.
- API validation accepts 4 and rejects 5; saved-run authorization accepts and
  resumes a valid run configured with 4.
- The `/intelligent-agent` compatibility route clamps `max_turns` to 4 rather
  than silently reducing the approved default to 3.
- A default all-failure graph run allocates five distinct monotonic
  `repair_round_id` values and reaches terminal failure before allocating a
  sixth.

### Reflection and auxiliary-model tests

- REFLECT uses one `ReflectDecision` contract under either repair provider.
- Failed or stopped reflection falls back deterministically without consuming a
  second repair retry.
- No production REFLECT or reasoning-tool path enforces an Opus-only two-round
  terminal.
- The lightweight summarizer cannot mutate active repair provider, model, or
  round attribution.

### Execution boundary tests

- OCI accepts a PatchGate-authorized patch from either provider.
- OCI still rejects missing, stale, or mismatched approval.
- A successful patch reaches the existing coverage and commit flow unchanged.

### Verification

Run focused model-policy, PLAN, PatchGate, VERIFY, and execute-security tests,
then the complete repository test suite. A live model smoke test and paid
evaluation require separate explicit approval.

## Acceptance Criteria

The change is complete when:

1. Gemini and Opus have one model-facing output protocol:
   `PlanDecision.patch_edits`.
2. A valid patch from either model reaches EXECUTE with exact PatchGate
   approval.
3. No Opus-specific model planning or correction branch remains in production
   PLAN.
4. When Opus is configured and a global retry remains, two unsuccessful Gemini
   repair rounds deterministically select Opus for the next model call.
5. The default `max_retries` is 4, yielding exactly five full repair
   transactions in the sequence Gemini, Gemini, Opus, Opus, Opus, with no
   sixth transaction.
6. Every active production and evaluation entry point uses the canonical
   default 4; the public API and saved-run validator accept 4 and reject values
   above the canonical upper bound.
7. Opus uses the remaining shared global retry and token budgets without a new
   provider-specific round limit.
8. Correctable patch failures request another full `PlanDecision`; environment
   failures do not spend a model round.
9. PLAN and VERIFY failures consume retry budget exactly once, including across
   resume.
10. REFLECT and auxiliary summarization cannot create a second patch protocol or
   an Opus-only two-round terminal.
11. Tool and context behavior remains unchanged.
12. Focused tests and the complete test suite pass.
