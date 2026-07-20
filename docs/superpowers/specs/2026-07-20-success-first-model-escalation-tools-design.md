# Success-First Model Escalation and Constrained Tools Design

**Date:** 2026-07-20  
**Status:** Proposed for implementation  
**Branch:** `fix/release-readiness-20260717`

## Goal

Increase RepoPilot's real issue-resolution rate before optimizing latency. Keep
Gemini Flash as the default model, escalate difficult samples to Claude Opus,
and let the active model choose which evidence-gathering tool it needs within a
deterministic safety and progress policy.

The initial acceptance target is to improve the existing deterministic
SWE-bench Verified ten-sample run from 1/10 internal successes to at least 4/10.
Internal success remains separate from official SWE-bench resolution, which is
reported only by the official Docker harness.

## Context

The first ten-sample Gemini run produced:

- 1 internal success;
- 4 `search_not_found` failures;
- 4 `test_failed` failures;
- 1 model/API infrastructure failure caused by an empty completion;
- several expensive PLAN/REFLECT self-loops; and
- background memory warnings because `RepoStore` ignored `REPOPILOT_HOME`.

Exact-commit repository caching and local search worked: no sample failed from
GitHub clone or API 503. The next bottleneck is model convergence and edit
precision rather than repository access.

## Confirmed Decisions

1. Success rate is the primary objective; latency is secondary.
2. Gemini remains the default model.
3. Difficult samples escalate to `claude-opus-4-8:stable` through the same
   OpenAI-compatible LinoAPI gateway.
4. Once escalated, a sample does not downgrade to Gemini.
5. Opus uses a compact escalation packet, not the unbounded conversation.
6. Models choose tool intent; deterministic code controls permission, scope,
   execution, repetition, and stopping.
7. Models cannot bypass exact-commit checkout, PatchGate, secret redaction, or
   official evaluator boundaries.
8. Gold patches, test patches, `FAIL_TO_PASS`, and `PASS_TO_PASS` never enter
   model prompts.
9. Every completed PLAN-to-REFLECT failure loop produces a rolling, bounded
   outcome summary for the next PLAN prompt.
10. A code fix is not successful until an existing or generated test proves
    the behavioral difference between the base commit and the fixed checkout.
    Failure to generate such a test after two attempts terminates the task.

## Alternatives Considered

### Gemini only

This keeps cost and integration complexity low, but the live run showed repeated
invalid anchors and self-loops after feedback. Prompt changes alone are unlikely
to provide the required success-rate improvement.

### Gemini to GPT escalation

GPT's flagship coding model is a viable alternative, but the chosen first
experiment is Opus because the dominant failures are agentic code navigation,
historical-code interpretation, and precise edit generation. The model layer
will remain provider-neutral so GPT can be tested later without changing graph
semantics.

### Fully autonomous tool agent

This is flexible but permits repeated searches, arbitrary commands, uncontrolled
cost, and difficult-to-debug loops. RepoPilot will instead use constrained
autonomy: model-selected evidence intent behind a deterministic ToolPolicy.

### Full attempt history in every PLAN prompt

This preserves detail but grows on every retry and repeatedly spends context on
failed edits. The selected design retains full structured history for replay
while injecting one rolling 200-character outcome summary into PLAN.

### Static or mixed coverage detection

Static symbol references are fast but do not prove that a test observes the
reported behavior. A mixed static/dynamic gate reduces execution cost, but the
static prefilter can miss indirect tests and adds a second definition of
coverage. The selected design uses dynamic differential behavior only: the same
targeted test must pass on the fix and fail by assertion on the exact base code.

## Architecture

The graph keeps the existing high-level phases. Four bounded components are
added:

```text
PLAN / REFLECT
      |
      v
ModelPolicy ------> Gemini or Opus
      |
      v
ToolRouter <------ ToolIntent from active model
      |
      +--> ToolPolicy --> local tool --> EvidenceStore
      |
      v
RepairPlan --> ContextBuilder --> VerifiedEdit --> PatchGate --> EXECUTE
     ^                                                        |
     |                                                        v
OutcomeSummarizer <-- REFLECT <-- failed test <-- VERIFY <-- applied patch
                                                               |
                                                               v passed
                                                        CoverageGate
                                                               |
                                    existing proof or generated verified test
                                                               |
                                                               v
                                                        COMMIT / DONE
```

Each component has one responsibility:

- `ModelPolicy`: deterministic model selection and escalation.
- `ToolRouter`: translates an approved ToolIntent into one local tool call.
- `ToolPolicy`: repository, command, repetition, evidence, and budget guards.
- `EvidenceStore`: deduplicated, bounded evidence referenced by stable IDs.
- `ContextBuilder`: exact symbol and code windows from the prepared checkout.
- `PatchGate`: proves an edit is scoped and applicable before execution.
- `OutcomeSummarizer`: rolls completed attempt outcomes into a 200-character
  PLAN context without replacing structured trace history.
- `CoverageGate`: proves that targeted tests distinguish base behavior from
  fixed behavior.
- `TestGenerator`: creates test-only edits when no existing test proves the
  fix, then submits them to PatchGate and differential validation.

## State and Trace Model

`AgentState` gains serializable routing fields:

```python
active_model: str
escalated: bool
escalation_reason: str
no_progress_rounds: int
last_plan_signature: str
last_context_fingerprint: str
last_test_failure_signature: str
model_history: list[ModelInvocation]
tool_history: list[ToolInvocation]
evidence: list[Evidence]
attempt_outcome_summary: str
coverage_status: Literal[
    "pending",
    "existing_verified",
    "generated_verified",
    "failed",
]
coverage_test_files: list[str]
coverage_test_command: str
coverage_failure_reason: str
test_generation_attempts: int
```

`ModelInvocation` records model name, node, elapsed time, estimated input/output
tokens, status, and a credential-free error class. It never stores headers,
keys, or full raw requests.

Every escalation emits a trace event:

```json
{
  "event": "model_escalated",
  "from": "gemini-3.5-flash:stable",
  "to": "claude-opus-4-8:stable",
  "reason": "repeated_unlocatable_edit",
  "round": 2
}
```

Saved-run replay must restore the active model and escalation state exactly.
It must also restore the rolling outcome summary, coverage decision, selected
test command/files, coverage failure reason, and test-generation attempt count.

## Rolling Attempt Outcome Summary

Each failed repair cycle follows PLAN, EXECUTE, VERIFY, and REFLECT. At the end
of REFLECT, `OutcomeSummarizer` receives only:

- the previous rolling summary;
- the current plan and applied edit signatures;
- a bounded test-result summary; and
- the new reflection conclusion.

It calls Gemini with temperature zero and asks for a factual summary of at most
200 Unicode characters. The result replaces `attempt_outcome_summary`; summaries
do not accumulate as an unbounded list. The next PLAN prompt injects it under a
single `Completed attempts` section. Full structured attempts, evidence, model
history, and route history remain in state for trace and replay, but are not
re-expanded into the PLAN prompt through this summary.

Summary input and output pass through the same secret and evaluator-field
redaction used by escalation packets. An empty, malformed, or overlong model
result does not fail the repair. A deterministic local fallback composes the
latest plan signature, patch outcome, test failure class, and reflection action,
then truncates the result to 200 characters. Summary calls use the primary
Gemini provider and do not consume the token budget reserved for Opus.

## Model Policy

The model never chooses whether to escalate. `ModelPolicy` evaluates state
before each PLAN or REFLECT call.

Immediate escalation occurs when Gemini exhausts request retries and returns an
empty completion or repeatedly invalid structured output.

Escalation after two consecutive no-progress events occurs for:

- nonexistent search blocks;
- repeated patch or anchor signatures;
- unchanged file/context evidence;
- unchanged root-cause hypotheses;
- identical test-failure signatures without a materially different edit; or
- PLAN/REFLECT rounds that add no evidence and do not produce an applicable
  patch.

After escalation, Opus remains active for the sample. If Opus makes no progress
for two consecutive rounds, it must emit a RepairPlan or stop with an explicit
failure; it may not continue an unbounded PLAN/REFLECT loop.

The default total budget becomes 100,000 estimated tokens. Gemini may consume
at most approximately 55,000 while at least 40,000 are reserved for Opus. At
the reserve boundary, an unresolved Gemini sample escalates rather than failing
for total budget exhaustion. The total budget and existing execution retry cap
still bound Opus.

## Escalation Packet

Opus receives an allowlisted `EscalationPacket` containing:

- issue title and body;
- repository and exact `base_commit`;
- relevant evidence IDs and real code windows;
- failed edit signatures and application errors;
- bounded test-error summaries;
- rejected root causes and already-failed approaches;
- required behavioral outcome; and
- remaining token and execution budgets.

It excludes credentials, full raw HTTP payloads, generated files, evaluator
fields, gold data, and unrelated conversation history.

## Constrained Tool Selection

The active model may request one structured `ToolIntent` at a time:

```python
class ToolIntent:
    action: Literal[
        "search_symbol",
        "search_text",
        "read_symbol",
        "read_range",
        "find_references",
        "list_related_tests",
        "run_targeted_test",
        "inspect_git_diff",
        "validate_patch",
        "request_repair",
        "finish_investigation",
    ]
    args: dict[str, object]
    reason: str
    expected_evidence: str
```

The model chooses the evidence it needs, not how the operation is implemented.
`ToolPolicy` enforces:

- all paths resolve inside the prepared repository;
- SWE-bench tools use only the exact historical checkout;
- no arbitrary shell command or network access;
- test commands are parsed and matched to an allowlist;
- identical tool/argument fingerprints cannot repeat;
- at most eight local tool calls per reasoning round and thirty per sample;
- output is size-bounded, text-only, and secret-redacted; and
- a call must add a new evidence fingerprint to count as progress.

The model cannot select clone, checkout, model escalation, API configuration,
cache policy, Git push, PR creation, repository-external writes, destructive
operations, PatchGate bypass, or official benchmark scoring.

## Evidence Store

Tool results are normalized and deduplicated:

```python
class Evidence:
    evidence_id: str
    tool: str
    file_path: str | None
    symbol: str | None
    summary: str
    content: str
    fingerprint: str
```

Prompts refer to evidence IDs instead of repeatedly appending entire files.
Evidence content is bounded and persisted with the saved run. A repeated
fingerprint is a no-progress event and is not appended again.

## Two-Stage Opus Repair

Opus first produces intent without a patch:

```python
class RepairPlan:
    root_cause: str
    target_files: list[str]
    target_symbols: list[str]
    required_behavior: str
    regression_test_strategy: str
    rejected_approaches: list[str]
```

`ContextBuilder` then resolves each target symbol in the exact checkout and
returns the real signature, complete definition, and a bounded surrounding
window. Opus uses that evidence to produce `VerifiedEdit` records:

```python
class VerifiedEdit:
    file_path: str
    node_target: str | None
    search: str
    replace: str
    intent: str
```

The preferred order is a unique node target, an exact verbatim search block,
or an intentional new text source file. Free-form fuzzy unified diffs are not
the Opus repair interface.

## PatchGate

Before EXECUTE, PatchGate verifies:

- the target is tracked at the exact base commit or an intentional new source
  file;
- node targets resolve uniquely;
- search blocks exist verbatim;
- the edit is limited to RepairPlan target files;
- an in-memory or `git apply --check` dry run succeeds;
- the resulting diff has at least one substantive source change; and
- archives, wheels, virtual environments, egg metadata, caches, build output,
  oversized files, and untracked binary artifacts are excluded.

An invalid edit returns the exact validation error and real code window to the
same active model for at most two local correction attempts. Patch application
corrections do not consume normal test-debug retries and do not invoke a full
REFLECT round. Only an applied patch that fails tests returns to root-cause
reflection.

## Test Execution and Failure Classification

Verification proceeds from narrow to broad:

1. syntax, import, or compile check;
2. model-selected targeted tests after command policy validation;
3. affected module tests; and
4. broader repository tests when supported by the project and remaining
   budget.

Environment installation, network, missing interpreter, and model gateway
errors are `infra` and do not consume model debug retries. Syntax/import
failures receive a direct patch repair. Assertion failures enter root-cause
reflection. A repeated failure signature requires a different target symbol;
another unchanged failure terminates the sample.

RepoPilot's internal result remains `agent_success`. Official SWE-bench
`resolved` and `unresolved` values come only from the official harness.

## Dynamic Coverage Gate and Test Generation

A successful VERIFY routes to a new COVERAGE phase before COMMIT or eval DONE.
Internal `agent_success` now requires this phase to produce a differential test
proof; merely passing the repository's existing tests is insufficient.

`CoverageGate` derives changed production files and symbols from the final
validated diff and locates related tests. It runs the same targeted test command
against two isolated trees:

1. the fixed checkout, which must pass twice; and
2. a temporary checkout at the exact `state.repo_ref`, with the same candidate
   test present, which must fail twice with the same test ID and a stable
   assertion-failure fingerprint.

The temporary base tree never receives the production fix. Test execution uses
fixed argv approved by ToolPolicy and cannot contain shell metacharacters,
environment assignments, network operations, absolute paths, or checkout
mutations. Temporary trees are removed after validation. Import failures,
dependency errors, interpreter failures, timeouts, and process crashes are
`coverage_infra`; they never count as proof that the base behavior is wrong.

If an existing test satisfies the differential condition, coverage becomes
`existing_verified`. If the base also passes, no related test exists, or the
failure is not attributable to a stable assertion, RepoPilot invokes
`TestGenerator`.

The generator receives an allowlisted packet containing the issue, changed
symbols, exact source windows, the fixed behavior, candidate test conventions,
and the coverage rejection reason. It returns structured test-only edits. The
edits may touch only existing test roots or paths that match the repository's
established test naming layout. They cannot alter production files, fixtures
outside the approved test scope, generated artifacts, dependencies, CI, or test
configuration. PatchGate validates the edit before it reaches the checkout.

After application, the generated test must pass the same fixed/base two-run
differential protocol and the affected fixed-checkout suite must remain green.
There are at most two generation or correction attempts. The first uses the
active model. If it is Gemini and the first test fails validation, ModelPolicy
escalates the second attempt to Opus. If the sample is already on Opus, both
attempts use Opus. If escalation is disabled or lacks credentials, the second
attempt remains on the current model. A second failure sets
`coverage_status="failed"`, classifies the task as `test_generation_failed`, and
terminates without a PR or internal success.

Once verified, a generated test remains in the worktree and final PR/prediction
patch. Prediction serialization still includes only the official output fields
and must not include the SWE-bench-provided test patch or any evaluator-only
data.

## Secrets and Provider Configuration

Configuration uses separate environment variables:

```text
LLM_MODEL=gemini-3.5-flash:stable
LLM_ESCALATION_MODEL=claude-opus-4-8:stable
LLM_ESCALATION_BASE_URL=https://linoapi.com.cn/v1
# LLM_ESCALATION_API_KEY is injected by the runtime secret store.
REPOPILOT_ESCALATION_ENABLED=1
REPOPILOT_ESCALATION_AFTER_NO_PROGRESS=2
```

No concrete key is stored in source, documentation, state, traces, evaluation
artifacts, exceptions, or Git remotes. Provider errors are sanitized before
persistence. Evaluation enables escalation when a key is present; without one,
Gemini-only behavior remains compatible. Normal PR workflows require explicit
configuration so escalation cannot introduce surprise cost.

## Repository Memory Path

`RepoStore` must derive its default from the same central RepoPilot home helper:

```text
${REPOPILOT_HOME:-~/.repopilot}/memory
```

This removes the sandbox warning observed during evaluation and makes memory,
cache, runs, repositories, and datasets share one configurable root.

## Testing Strategy

Implementation follows test-driven development. Required coverage includes:

- every ModelPolicy escalation trigger and the no-downgrade rule;
- token reservation at the Gemini/Opus boundary;
- EscalationPacket allowlisting and gold/secret exclusion;
- allowed, rejected, repeated, and out-of-repository ToolIntent cases;
- tool and evidence budgets plus fingerprint deduplication;
- EvidenceStore persistence and saved-run replay;
- unique node targets, exact searches, empty patches, generated archives,
  binary artifacts, and scope violations in PatchGate;
- independent Gemini and Opus clients, retry behavior, and sanitized errors;
- Gemini failure followed by Opus repair and passing verification using mocked
  clients;
- rolling summary replacement, the 200-character limit, deterministic fallback,
  PLAN injection, and save/replay persistence;
- fixed/base differential validation with stable assertion fingerprints;
- existing-test acceptance, base-also-passes rejection, and coverage
  infrastructure classification;
- generated test path/scope enforcement and PatchGate validation;
- Gemini test-generation failure followed by an Opus-generated valid test;
- hard failure after two invalid test-generation attempts;
- final PR/prediction patches retaining only verified generated tests;
- infrastructure failures that do not consume model retries;
- `RepoStore` honoring `REPOPILOT_HOME`; and
- all existing tests remaining enabled.

## Rollout and Evaluation

Implementation order:

1. centralize the memory path;
2. add independent provider configuration and redaction;
3. add state records and ModelPolicy;
4. add EvidenceStore, ToolPolicy, and ToolRouter;
5. add EscalationPacket;
6. add RepairPlan, ContextBuilder, and VerifiedEdit;
7. add PatchGate correction rounds;
8. add rolling outcome summarization and PLAN injection;
9. add the COVERAGE phase, differential runner, and test generation gate;
10. wire failure taxonomy, replay, and eval reporting;
11. run a five-sample checkpoint; and
12. rerun the deterministic ten-sample evaluation and official scoring when
    Docker is available.

The checkpoint uses the previous pytest success as a control plus xarray,
Django, SymPy, and seaborn as failed cases. It verifies that escalation and
tool selection operate before spending a full ten-sample budget.

## Acceptance Criteria

- The deterministic ten-sample internal success count reaches at least 4/10,
  and every counted success has a stable fixed/base differential test proof.
- At least two of the four previous `search_not_found` samples are recovered.
- The empty-Gemini-completion sample escalates rather than immediately failing.
- No sample has more than two consecutive no-progress PLAN/REFLECT rounds.
- Every completed failure loop leaves a secret-free summary of at most 200
  characters, and the next PLAN receives exactly one rolling summary section.
- A missing coverage proof triggers test generation; two failed generation
  attempts terminate as `test_generation_failed` without a PR.
- Predictions contain exactly the official output fields, contain no gold data
  or credentials, exclude generated artifacts, and retain only locally
  generated tests that passed differential validation.
- Every model/tool decision is replayable from safe structured trace records.
- The complete unit and integration suite passes without removing or skipping
  existing tests.
- Official resolved rate is reported separately when Docker scoring is
  available; Docker absence is reported as scoring infrastructure blockage.

## Out of Scope

- A third fallback model in the first implementation.
- Arbitrary shell access for the model.
- Autonomous push, merge, PR creation, deletion, or benchmark scoring.
- Replacing the state graph with a general-purpose agent framework.
- Optimizing latency before the success-rate checkpoint is complete.
