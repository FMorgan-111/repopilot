# Pre-merge Pilot-20 SWE-bench Evaluation

Date: 2026-07-24

Status: approved; independent design review passed; implementation pending;
owner-approved CI-only pre-Pilot push exception recorded 2026-07-24

Branch: `fix/release-readiness-20260717`

Pull request: `FMorgan-111/repopilot#2`

## Context

RepoPilot needs one reproducible success measurement before the release branch
is merged to `master`. The existing 150 custom samples cannot supply it: they do
not record immutable base commits and their Agent V2 path has no official
resolved verdict. SWE-bench Verified already supplies pinned dataset rows,
official OCI images, and the official scorer.

The user selected a real 20-instance model run and the same-repository
PR-label path so the run can happen before merge. The user also selected this
release gate:

- the fixed denominator is 20;
- all 20 cases must have a valid terminal outcome;
- infrastructure failures must be zero;
- official resolved rate is reported with no minimum passing score;
- systemic failures are repaired and the complete fixed cohort is rerun;
- after all release gates pass, merge PR 2 to `master` and end the project.

This design depends on the two Important fixes in
`2026-07-24-final-gate-telemetry-cancellation-design.md`. Complete
test-generation telemetry is required for honest budget diagnostics, and
cancellation-drain propagation is required so OCI and PR cleanup failures
cannot disappear behind an ordinary timeout.

## Goals

- Freeze 20 real SWE-bench Verified instances before observing results.
- Use Gemini Flash primary and Claude Opus escalation through the existing
  prepare, generate, score, package, and aggregate pipeline.
- Count success only from the official SWE-bench `resolved` projection.
- Bind every instance and aggregate artifact to one candidate, workflow,
  GitHub run, attempt, event, PR base, and authorization.
- Permit paid inference only from PR 2 or trusted `master`, through an
  externally protected GitHub environment.
- Prevent fork execution, arbitrary-ref dispatch, re-run amplification,
  cross-run artifact composition, and result-based cherry-picking.
- Independently verify one atomic aggregate completion marker before merge.

## Non-goals

- Treating the non-reproducible custom 150 as a success benchmark.
- Inventing a minimum resolved score without a prior baseline.
- Tuning prompts, samples, or model policy after seeing pilot outcomes.
- Caching model responses, patches, target checkouts, credentials, Docker
  state, or scorer logs.
- Using `pull_request_target`, executing fork code with secrets, or accepting
  arbitrary IDs, refs, models, budgets, or concurrency values.
- Combining instances from different workflow runs or attempts.
- Claiming that the Agent's estimated token budget is an exact provider billing
  cap.

## Approaches considered

### A. Same-repository `pull_request: labeled` with a protected environment

Selected. A no-checkout job validates the immutable event identity. Execution
jobs explicitly check out the authorized PR head. The model credentials live
only in a protected GitHub environment whose deployment refs admit PR 2's merge
ref and `master`, not general same-repository branches or PRs.

This environment is essential. A same-repository PR workflow comes from the PR
merge commit and can modify its own guards; repository-level secrets would
therefore be available to any collaborator able to create another privileged
workflow. Static conditions are defense in depth, not a substitute for an
external secret boundary.

### B. Merge a trusted dispatcher first

This gives the default branch ownership of the privileged dispatcher but needs
a preliminary merge and a second release merge. It remains the fallback if the
protected environment cannot be configured as specified.

### C. External Linux/Docker runner

This avoids GitHub PR secrets but requires new compute. The current macOS/arm64
machine has no Docker-compatible runtime, while official images target Linux
x86_64. It is not selected.

## Fixed cohort

Add `eval/pilot_20_ids.txt`. Its normalized tuple must equal the first 20
normalized IDs in `eval/baseline_50_ids.txt`, in order. Normalization is the
existing rule: strip each line and discard empty lines. The pilot loader then
requires:

1. exactly 20 IDs;
2. all IDs unique;
3. ordered tuple equality with `baseline_50[:20]`;
4. the baseline's twenty-first ID is not admitted by `pilot_20`.

The cohort is committed before any model call. There is no runtime sampling or
seed and no post-result selection.

`EvalMode`, mode-file mapping, runner CLI, aggregate CLI, workflow choice,
packaging contract, and documentation gain `pilot_20`. The existing
prepare/generate/score/package behavior otherwise remains unchanged.

## Run provenance contract

Add one strict, extra-forbidden `WorkflowProvenance` model. For a pilot it is
required in `RuntimeRecord`, copied unchanged into `InstanceManifest`, and
copied into the aggregate `PilotRunManifest`. Aggregation also receives an
independently constructed expected provenance and rejects any difference.

The model contains these exact fields:

```text
repository           Literal["FMorgan-111/repopilot"]
event_name           Literal["pull_request", "workflow_dispatch"]
event_ref            bounded non-empty string
workflow_ref         bounded non-empty string
workflow_sha         40-lower-hex SHA
run_id               strict integer >= 1
run_attempt          Literal[1]
candidate_sha        40-lower-hex SHA
pr_number            Literal[2] or null
pr_base_ref          Literal["master"] or empty
pr_base_sha          40-lower-hex SHA or empty
pr_head_repository   Literal["FMorgan-111/repopilot"] or empty
pr_head_ref          Literal["fix/release-readiness-20260717"] or empty
pr_head_sha          40-lower-hex SHA or empty
authorization_label  Literal["repopilot-pilot-20-approved"] or empty
```

For a PR event, every PR field is populated, `event_ref` is
`refs/pull/2/merge`, and `candidate_sha == pr_head_sha`. For manual dispatch,
all PR fields are empty/null, `event_ref == refs/heads/master`, and the
candidate is `github.sha`. Cross-field validation enforces the appropriate
shape.

The workflow writes a bounded provenance JSON file from GitHub contexts before
calling repository code. `prepare` validates it and persists it in the runtime.
`package` copies the same object into the instance manifest. The aggregate CLI
receives the expected object separately and verifies every manifest. This makes
mixing artifacts from two label events, run IDs, attempts, base commits, or
workflow revisions detectable even when the RepoPilot head SHA is unchanged.

Legacy `checkpoint_5` and `baseline_50` artifacts may omit provenance for
backward-compatible diagnostics. `pilot_20` requires it; old pilot-shaped
artifacts cannot exist because the mode is new.

## Official verdict and release gate

The denominator is always 20. The valid non-infrastructure statuses are:

- `resolved`: one official success;
- `unresolved`: zero official successes;
- `empty_patch`: zero official successes, still in the denominator.

`empty_patch` is a determined zero-credit model outcome even though the current
`OfficialResult.completed` flag is false. The release gate must not reuse that
flag. It computes:

```text
resolved_count    = count(status == "resolved")
unresolved_count  = count(status == "unresolved")
empty_patch_count = count(status == "empty_patch")
determined_count  = resolved_count + unresolved_count + empty_patch_count

instance_infra = (
    runtime_status != "ready"
    or official.status == "scorer_infra"
    or result.failure_class in {
        "infra", "model_gateway_infra", "coverage_infra"
    }
)
infrastructure_count = count(instance_infra)

gate_passed = (
    requested == 20
    and determined_count == 20
    and resolved_count + unresolved_count + empty_patch_count == 20
    and infrastructure_count == 0
)
official_resolved_rate = resolved_count / 20
```

The per-instance infrastructure expression is one boolean OR, so one instance
is counted at most once. An empty patch with an infrastructure failure class
fails the gate. A valid 20-empty-patch `0/20` run passes.

Internal success, agreement, engineering score, token estimates, and latency
remain diagnostic. Only `OfficialResult.resolved` contributes to success.

## Aggregate completion contract

The pilot aggregation path emits these exact files:

```text
results.json
predictions.jsonl
official_results.json
manifests.json
summary.md
run_manifest.json
```

`manifests.json` contains the 20 complete sanitized `InstanceManifest` models
in tracked order, including dataset row hash, OCI image digest, safe payload
hashes, commit, and workflow provenance.

`run_manifest.json` is a strict, extra-forbidden `PilotRunManifest` with:

```text
schema_version                   Literal[1]
mode                             Literal["pilot_20"]
evaluated_commit                 40-lower-hex SHA
provenance                       WorkflowProvenance
instance_ids                     exact ordered list of 20 IDs
selection_sha256                 lowercase SHA-256
requested                        Literal[20]
determined                       Literal[20]
official_resolved                strict integer in [0, 20]
official_unresolved              strict integer in [0, 20]
empty_patch                      strict integer in [0, 20]
infrastructure_failures          Literal[0]
official_resolved_rate           finite float in [0, 1]
files                            exact map of the other five names to SHA-256
```

Cross-field validation requires the three verdict counts to sum to 20, rate to
equal `official_resolved / 20`, candidate and evaluated commit to match, ordered
IDs to equal the tracked cohort, and `selection_sha256` to equal SHA-256 of:

```text
UTF-8("\n".join(instance_ids) + "\n")
```

The pilot output directory must not exist. Aggregation writes to a unique
same-filesystem staging directory, writes `run_manifest.json` last, and
atomically renames the complete directory to the requested output only after
all validation succeeds. Gate failure, stale target, marker-write failure, or
unexpected output leaves no published pilot directory or stale marker.

This release gate, strict directory publication, and completion marker apply
only to `pilot_20`. Existing modes retain diagnostic aggregation, existing
infrastructure behavior, no completion marker, and the existing
`aggregate_artifacts() -> summary_path` interface.

Add `python -m eval.oci_verify` as an independent, read-only verifier. It accepts
both the downloaded aggregate directory and all 20 downloaded per-instance
artifact directories, plus the expected commit and run ID. Using bounded
no-follow reads, it revalidates every instance manifest and payload, recomputes
the tracked-order aggregate, and rejects symlinks, extra/missing files, bad
hashes, malformed schemas, wrong provenance, wrong cohort/order, inconsistent
counts/rate, aggregate differences, or a recomputed gate failure. It does not
trust the workflow summary or aggregate producer.

## GitHub trigger and protected environment

The workflow retains manual dispatch and adds:

```yaml
pull_request:
  types: [labeled]
  branches: [master]
```

Before pushing the candidate, configure the GitHub environment
`repopilot-expensive-eval` outside repository code:

- move `LLM_API_KEY` and `LLM_ESCALATION_API_KEY` into environment secrets;
- remove any repository-level copies after the environment values are saved;
- require approval from the trusted repository owner;
- disallow administrator bypass where the repository plan supports it;
- restrict deployment branches/refs to `master` and `refs/pull/2/merge` only.

The credentials are revocable inference-only credentials. They grant no
GitHub, repository, registry, cloud, or deployment permission. Provider-side
rate/spend limits are used when the gateway supports them.

If the environment, its ref restrictions, or its secrets cannot be configured
and verified through GitHub's API without reading secret values, approach A is
blocked and the implementation returns to approach B. Repository-level model
secrets are not an acceptable fallback.

Before merge, configure an active `master` branch protection/ruleset outside
repository code. It must:

- require changes through a pull request;
- require branches to be up to date with `master` before merging;
- require the stable pilot completion check named `pilot_release_gate` from the
  GitHub Actions app;
- apply to administrators and have no bypass actor for this release;
- reject force pushes and direct branch deletion.

The workflow exposes `pilot_release_gate` as a stable final job/check that can
succeed only after an authorized `pilot_20` aggregation publishes a valid
completion marker. GitHub's strict up-to-date rule is the atomic expected-base
guard: if `master` changes, the PR becomes out of date and the server rejects
the merge before writing. Updating the PR changes the candidate SHA and
requires the complete pilot again.

If this strict rule or required check cannot be configured and verified through
the GitHub API, merge is blocked; a read-then-merge client-side check is not an
acceptable substitute.

## Authorization job

The first job has `permissions: {}`, performs no checkout, invokes no
repository code, and references no secrets.

A PR-label event is admitted only when all are true:

- repository is `FMorgan-111/repopilot`;
- event/action is `pull_request/labeled` and the label is exactly
  `repopilot-pilot-20-approved`;
- event ref is `refs/pull/2/merge`;
- PR number is 2 and state is open;
- PR author, `github.actor`, and `github.triggering_actor` are the trusted owner;
- base ref is `master` and base SHA is valid;
- head repository and branch are the exact same-repository release branch;
- head SHA is valid and equals repository variable
  `REPOPILOT_EVAL_CANDIDATE_SHA`;
- `github.run_attempt == 1`.

A manual-dispatch event is admitted only when all are true:

- repository and both actor identities are the trusted owner;
- mode is one of the three tracked enums;
- `github.ref_type == "branch"`;
- `github.ref == "refs/heads/master"`;
- `github.workflow_sha == github.sha`;
- workflow ref names the expected workflow on `refs/heads/master`;
- `github.run_attempt == 1`.

An arbitrary branch or tag dispatch therefore fails both the workflow guard and
the protected environment's deployment rule. A branch-controlled workflow
cannot remove the external environment rule to obtain the keys.

The authorization job emits only the allowlisted mode, exact candidate SHA,
and strict provenance fields. An empty or malformed output is never a checkout
fallback.

## Execution jobs and live admission

Job permissions are narrowed independently:

- authorization: `{}`;
- prepare: `contents: read`;
- instance: `contents: read`, `pull-requests: read`, and the exact protected
  environment;
- aggregate: `contents: read`, `pull-requests: read`;
- `pilot_release_gate`: `{}`.

Every execution job depends on successful authorization, validates a non-empty
40-hex candidate output before checkout, explicitly checks out that SHA with
`persist-credentials: false`, and verifies `git rev-parse HEAD` equals it.

The secret-bearing matrix job and its `Generate patch` step independently
repeat the owner, triggering actor, and `run_attempt == 1` guards. A GitHub
re-run, including a re-run of one matrix job, cannot expose the environment
secrets or spend model tokens. A new label-add event is a new explicit cost
authorization and receives a new run ID at attempt 1.

Immediately before `Generate patch`, a read-only GitHub API check revalidates
the live PR state, number, author, base ref/SHA, head repository/ref/SHA, and the
continued presence of the approval label. Any change fails before secret
exposure. There is an unavoidable small API-check-to-process-start TOCTOU
window; the final aggregate check below ensures such a change cannot produce an
authoritative completion marker.

Immediately before publishing `run_manifest.json`, the aggregate job performs
the same full live admission check. For manual dispatch it instead verifies
that current `master` still equals the candidate. If the head, base, label,
state, author, or branch changed during the run, no completion marker is
published.

The aggregate job condition is equivalent to:

```text
always()
and authorization succeeded
and prepare succeeded
and candidate output is valid
and run_attempt == 1
and triggering actor is the trusted owner
```

It may diagnose incomplete instance jobs, but cannot run after failed
authorization. Only upload steps inside an already authorized job use
unconditional `always()`. No empty-ref checkout is possible.

`pilot_release_gate` is non-skippable for every PR-label workflow run. It uses
`always()` and performs no checkout or secret access. It explicitly fails for
an unapproved label, failed authorization, wrong attempt/actor, or any failed,
cancelled, or skipped prepare/instance/aggregate dependency. It succeeds only
when an admitted PR `pilot_20` attempt-1 aggregate reports a published marker
whose commit and provenance match authorization. This matters because GitHub
treats a skipped required check as satisfying branch protection. The exact
job/check name is stable and its expected source is the GitHub Actions app.

## Secret, artifact, and execution boundaries

- External Actions remain pinned to audited full commit SHAs.
- Model environment secrets are referenced only by `Generate patch`.
- Dependency installation, dataset preparation, scoring, packaging,
  aggregation, caching, and upload do not reference model secrets.
- Score retains its credential-free environment preflight.
- No shell tracing, environment dumps, raw model bodies, full scorer logs, core
  dumps, or RepoPilot home directory enters logs or artifacts.
- Only the fixed public dataset cache is restored. Its key stays bound to the
  dataset revision and locked dependency identity.
- Repository commands remain inside the digest-pinned, networkless,
  capability-dropped OCI boundary. Host execution remains disabled.
- Root workflow permissions do not grant write access.
- Complete workflow runs are serialized, matrix `max-parallel` remains 2, and
  in-progress runs are not automatically canceled.

## Cost semantics

The open-source dataset and scorer are free; model inference may be billed.
Each authorized run has:

- 20 tracked instances;
- at most two instance jobs concurrently;
- an Agent soft-stop threshold of 100,000 estimated tokens per instance;
- three Agent repair retries under the existing policy;
- a 360-minute job timeout per instance;
- Gemini Flash primary and one-way Opus escalation.

The 100,000 value is not a hard provider billing cap. One request can cross the
threshold, and internal gateway/structured-output retries may consume tokens
that the current logical-call estimate cannot measure exactly. The telemetry
remediation makes RepoPilot's diagnostics complete under its documented
estimation contract, not provider invoices. Before adding the label, the owner
must accept the gateway account's balance/spend settings; this workflow cannot
promise a dollar ceiling.

## Failure and rerun policy

- Resolved, unresolved, empty patch, ordinary test failure, invalid diff,
  wrong-file selection, search failure, and budget stop are terminal model
  outcomes. They never authorize a score-improving rerun.
- Dataset, OCI image/boundary, model gateway, coverage infrastructure, scorer,
  artifact, or cancellation-cleanup failure invalidates the run.
- GitHub's re-run action is forbidden by `run_attempt == 1`. After an invalid
  run is diagnosed and any code fix passes all local gates, update the
  candidate variable and remove/re-add the label to start a complete new run.
- Never combine prior successful instances with a new run. Provenance checks
  reject such a bundle.
- For one unchanged candidate and base, the first complete infrastructure-free
  run recorded in the release evidence is authoritative, regardless of score.
- Any candidate or base change after an authoritative run requires a new exact
  provenance and complete pilot.

## Merge and final provenance

After independent aggregate verification, no repository commit or base-branch
change is allowed before merge.

Immediately before merging, read PR metadata again and require:

- open PR 2;
- current head SHA equals `run_manifest.evaluated_commit`;
- current base SHA equals `run_manifest.provenance.pr_base_sha`;
- label and all live admission properties still match;
- the active no-bypass `master` ruleset still requires an up-to-date branch and
  the successful `pilot_release_gate` check;
- all standard required checks pass.

Use GitHub's merge operation with merge method `merge` and the expected head SHA
precondition. The server-side strict up-to-date and required-check rules supply
the atomic base precondition. Do not squash, rebase, force-push, or bypass
protection. If merge commits or the strict rules are unavailable, stop rather
than silently weaken provenance.

After merge, verify the returned merge commit has the evaluated head and the
expected base as its direct parents, not merely that the evaluated commit is an
ancestor. The final release record binds:

- GitHub run ID and workflow SHA/ref;
- downloaded `run_manifest.json` SHA-256;
- cohort selection SHA-256;
- evaluated head SHA and expected base SHA;
- merge commit SHA;
- official `R/20` score and failure taxonomy.

The post-merge parent check is defense in depth. The strict server-side rule
must prevent a concurrent base update from being written in the first place; a
parent mismatch is therefore a release-integrity failure, not an accepted race.

## Implementation sequence

1. Implement the approved final-gate telemetry and cancellation remediation
   with focused TDD.
2. Run the final-gate affected and complete suites, Ruff, and
   `git diff --check`, then obtain a fresh independent whole-stage review. Do
   not push until every finding attributable to `7636b13..latest` is fixed and
   the resulting latest head has a fresh re-review with no unresolved Critical,
   Important, or Minor range finding. Explicitly deferred pre-base Minor items
   remain separately tracked.
3. Because no local Python 3.10 runtime is available, use the owner-approved
   exception to push the feature branch solely to bind and run the existing CI
   workflow. If CI fails, return to focused TDD and fresh review before a
   follow-up evidence push. This step does not authorize Pilot implementation,
   secrets, paid inference, any environment action, merge, prompt changes, or
   cohort changes. Before step 4, the exact-head run must finish with overall
   `success`: all six Ubuntu and macOS Python 3.10/3.11/3.12 matrix jobs,
   `lint`, and `oci-integration` must pass.
4. Add the fixed cohort, strict provenance, pilot gate, atomic aggregate,
   independent verifier, protected-environment workflow, and tests with focused
   TDD.
5. Run focused suites on Python 3.10, 3.11, and 3.12, then the complete suite,
   Ruff, `git diff --check`, clean-archive tests/build, packaging, workflow, and
   credential checks.
6. Obtain a fresh independent whole-branch review. Do not proceed with any
   Critical or Important finding.
7. Push or confirm the already pushed exact branch head. Configure and verify
   the protected environment and move the two model secrets there without
   reading or printing their values. Remove repository-level copies.
8. Set `REPOPILOT_EVAL_CANDIDATE_SHA` to the exact pushed head. Add the exact
   approval label once and approve the protected environment deployment.
9. Monitor the run. Download its aggregate and all 20 instance artifacts, then
   use `eval.oci_verify` to verify the candidate, run ID, hashes, fixed cohort,
   20 determined outcomes, zero infrastructure failures, and score.
10. For an invalid run, diagnose and start a complete new authorized run. For a
   valid run, freeze candidate and base.
11. Configure/verify the no-bypass strict `master` ruleset and required
   `pilot_release_gate`, recheck live PR identity, then merge PR 2 with the
   expected-head precondition. Verify direct parents and remote `master`.
12. Record `R/20` and final provenance. The project is complete.

## Test strategy

Implementation follows red-green-refactor.

### Cohort and mode

- Exact 20 unique IDs equal the normalized baseline prefix.
- Reorder, duplicate, substitution, extra ID, or baseline ID 21 is rejected.
- Runtime/manifest/runner/aggregate/package contracts accept `pilot_20` and
  require provenance only where designed.

### Provenance and aggregation

- Runtime, instance manifest, aggregate expected context, and run manifest must
  match field for field.
- Mixing two run IDs, attempts, workflow SHAs, base SHAs, events, or PRs fails.
- Exactly 20 valid bundles emit tracked-order outputs and the marker.
- Twenty empty patches pass as `0/20`; empty patch plus infrastructure fails.
- Missing/extra/duplicate/cross-commit/cross-mode/unsafe bundles fail.
- Release gating and marker publication apply only to `pilot_20`; legacy mode
  behavior and return type remain unchanged.
- A stale target directory, stale marker, marker-write error, gate failure, or
  unexpected output file publishes no pilot directory.
- `eval.oci_verify` rejects every instance or aggregate file, hash, schema,
  count, rate, selection, commit, provenance, or cross-output mutation.

### Workflow security

- Triggers are exactly manual dispatch and `pull_request: labeled`; there is no
  `pull_request_target`, push, schedule, or comment trigger.
- Manual choices are exactly the three tracked modes.
- PR guards cover exact repository, event ref, open PR, number, author, actors,
  base, head, label, external candidate SHA, and attempt 1.
- Manual guards reject arbitrary branch/tag refs and require trusted `master`.
- Re-run attempts and a different triggering actor cannot reach the environment
  or generation step.
- Live admission rejects label removal, close, base change, head change, author
  change, or branch/repository change before generation and marker publication.
- Authorization failure cannot reach checkout or aggregate; empty SHA is never
  a checkout default.
- The exact `pilot_release_gate` check can pass only after every required job
  and marker condition succeeds.
- An unauthorized label or dependency failure makes `pilot_release_gate` fail;
  it never becomes a branch-protection-satisfying skipped check.
- Job permissions are exact, actions immutable, secrets generation-only, cache
  public-only, and concurrency two.
- Mutation tests remove or weaken each predicate and prove the static contract
  fails.

### GitHub environment acceptance

Before the paid run, read-only GitHub API evidence must prove:

- `repopilot-expensive-eval` exists;
- its allowed refs are only `master` and `refs/pull/2/merge`;
- required review and bypass settings match this design;
- the two environment secret names exist;
- repository-level copies do not exist.

Secret values are never read, echoed, passed as CLI arguments, or written to
files.

### GitHub branch-rule acceptance

Before merge, read-only GitHub API evidence must prove that `master` requires a
PR, requires an up-to-date branch, requires `pilot_release_gate` from the
GitHub Actions app, applies to administrators without a release bypass, and
disallows force pushes/deletion.
The workflow contract fixes that exact check name and proves it cannot succeed
without a valid pilot completion marker.

## Acceptance criteria

- The two prior Important release findings are fixed and independently reviewed.
- All focused, cross-version, full-suite, lint, archive, build, packaging,
  credential, workflow, and merge-tree gates pass with the user's unstaged
  `run_trace.py` modification untouched.
- Protected environment evidence satisfies every external rule above.
- Active `master` rules atomically require the current base and successful
  `pilot_release_gate` with no release bypass.
- Candidate, base, workflow, run, attempt, and event provenance agree across
  GitHub, all 20 instance artifacts, and the aggregate.
- One atomic verified aggregate has 20 determined outcomes, zero infrastructure
  failures, and a strict completion marker.
- The reported score is official `R/20`; no internal metric or later run replaces
  it based on a better result.
- PR 2 is merged with the evaluated head and expected base as direct parents of
  the verified merge commit.

## Platform references

- Pull request `labeled`, merge-ref, SHA, and fork behavior:
  <https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows>
- Manual run ref selection:
  <https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow>
- Re-run actor and attempt semantics:
  <https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs>
- Secret scope and fork restrictions:
  <https://docs.github.com/en/code-security/reference/secret-security/secret-types>
- Environment approval and deployment ref restrictions:
  <https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments>
- `pull_request_target` security warning:
  <https://docs.github.com/en/actions/reference/security/securely-using-pull_request_target>
