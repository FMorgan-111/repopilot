# Simple Patch-Edits Protocol Design

**Date:** 2026-07-28

**Status:** Approved for implementation

## Goal

Make patch generation reliable before optimizing speed or official SWE-bench
score. Gemini and Opus must share one model-facing response shape containing
only `patch_edits`. RepoPilot, not the model, derives routing and authorization
metadata.

The first acceptance target is at least one PatchGate-approved patch in a
three-instance SWE-bench Verified smoke run. A zero-of-three result stops the
paid run for another diagnosis.

## Evidence

The interrupted patch-first batch completed six instances and generated no
patches after 30 model calls and 197,996 reported tokens. Twenty-four calls
were recorded as successful model responses, while all six final run states
ended with `invalid_plan_envelope`. No completed instance failed because of
HTTP 503, authentication, repository checkout, or PatchGate infrastructure.

The current model contract requires a narrative plan, duplicated file list,
test command, and a nested `DecisionFrame` with hypotheses, evidence, routing,
risk, and confidence. Patch authorization validates that envelope before it
parses `patch_edits`, so non-executable metadata can reject an otherwise useful
edit proposal. The exact nested Pydantic failure is currently collapsed into
the generic `invalid_plan_envelope` correction.

## Approaches Considered

### A. Minimal `patch_edits` response — chosen

The model returns one JSON object:

```json
{
  "patch_edits": [
    {
      "file_path": "src/example.py",
      "search": "old code",
      "replace": "new code"
    }
  ]
}
```

The runtime parses the edits, derives the target files, creates an internal
`DecisionFrame(recommended_action="execute")`, and passes the existing
internal `VerifiedEditBatch` through the existing PatchGate.

This retains the current exact-edit safety boundary while removing model-authored
self-authorization fields from the hot path.

### B. Bash-only agent with Git-generated patch

This follows mini-SWE-agent closely and is a viable later fallback, but it
would replace RepoPilot's current PLAN/EXECUTE boundary and is larger than the
observed failure requires.

### C. Multiple model-specific patch formats

Aider uses model-specific whole/diff formats. Supporting several parsers now
would add complexity before measuring whether the shared `patch_edits` shape
works for the configured Gemini and Opus endpoints.

## Model Contract

The patch-producing PLAN response has one required top-level field:
`patch_edits`.

Each edit uses the existing exact search/replace representation:

- `file_path`: non-empty repository-relative path;
- `search`: non-empty source text copied from the supplied checkout context;
- `replace`: non-empty replacement text.

The existing `node_target` form may remain accepted for compatibility, but the
prompt does not advertise it in the initial MVP. Existing path aliases may be
normalized by current code; no new alias or format is added.

The model does not author `decision_frame`, `recommended_action`, `files`,
`risk`, `confidence`, provider identity, repair round identity, hashes,
approval, or exactness metadata. Extra narrative fields at the top level do
not grant authority and must not prevent extraction of a valid `patch_edits`
list.

Control responses for the existing bounded reasoning tools and explicit stop
remain separate variants. They do not share the patch response schema.

## Runtime Data Flow

For a patch response, PLAN performs these operations in order:

1. Extract and validate `patch_edits` with the existing raw edit parser.
2. Derive target files and symbols from the parsed edits.
3. Build the existing deterministic internal `RepairPlan` and
   `VerifiedEditBatch`.
4. Call the existing `validate_patch_batch(...)` PatchGate once.
5. On acceptance, construct the internal `PlanDecision` and
   `DecisionFrame(stage="plan", recommended_action="execute")` required by
   downstream routing.
6. Preserve PatchGate's canonical patch, canonical edits, and approval receipt.

The model-facing response is not the internal routing state. Runtime-derived
metadata may remain in `PlanDecision` and saved traces for compatibility.

## Validation Boundary

PatchGate remains the single execution boundary. This change does not add a
validator, receipt, or second authorization layer.

The retained checks are the current repository-relative path restriction,
exact target matching, edit/file limits, exact checkout binding, canonical
patch construction, and `git apply --check` preflight. Model-authored trust
fields remain ignored.

Unknown narrative metadata must not reject valid edits. Invalid edit objects,
ambiguous paths, missing anchors, empty replacements, multiple exact matches,
and missing exact matches remain model-correctable failures.

## Correction Feedback

One failed transaction returns the first actionable issue only:

- stable issue code;
- target path when available;
- concise message;
- bounded real-code correction context when PatchGate provides it.

The correction does not include a synthetic `DecisionFrame`, provider/round
attestations, a multi-issue authorization packet, or a generic
`invalid_plan_envelope` when a more specific edit error is known.

The runtime may keep complete diagnostics internally, but the model prompt
receives only the actionable first issue.

## Retry and Model Switching

The maximum remains five full patch-producing transactions. Normal semantic
failure order is fixed:

```text
Gemini, Gemini, Opus, Opus, Opus
```

An invalid structured response counts as a failed primary repair transaction;
it does not trigger immediate Opus escalation after the first Gemini failure.
Exhausted transport retries, unavailable primary gateway, and primary token
reserve may still switch to Opus immediately within the existing policy.

Existing no-progress logic may stop repeated identical failures early, but no
new fingerprinting or retry state is introduced in this change.

## Testing

Implementation follows red-green TDD with focused behavioral tests:

1. A legal response containing only `patch_edits` is accepted.
2. Runtime derives `files` and an execute `DecisionFrame`.
3. A top-level narrative field cannot reject otherwise valid edits.
4. A concrete invalid edit returns its specific issue instead of
   `invalid_plan_envelope`.
5. The first invalid Gemini structured response keeps the next semantic repair
   transaction on Gemini; the second switches to Opus.
6. Existing PatchGate rejection, exact checkout, patch-only reporting, and
   cancellation behavior remain green.

After unit and repository tests pass, run three fixed SWE-bench Verified
patch-first smoke instances. Stop if no patch is generated.

## Explicit Non-Goals

- no new model-output format;
- no Bash-agent rewrite;
- no new authorization or receipt layer;
- no model-generated risk, confidence, hypothesis, or approval fields;
- no cache, resume, coverage-generation, scorer, or repository-clone work;
- no 20-instance paid run before the three-instance smoke gate;
- no edits to the user-owned `run_trace.py` in the older worktree.
