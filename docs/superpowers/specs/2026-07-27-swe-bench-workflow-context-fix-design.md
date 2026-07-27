# SWE-bench Workflow Context Fix Design

Date: 2026-07-27
Status: approved approach; implementation pending

## Problem

GitHub rejected `.github/workflows/swe-bench-oci-eval.yml` before creating any
job. The parser reported that `runner.temp` is unavailable at lines 78 and 79.
Those expressions currently appear in `jobs.instance.env`.

GitHub's context-availability contract does not expose the `runner` context to
`jobs.<job_id>.env`. The runner-provided `RUNNER_TEMP` environment variable
exists only after the job has been assigned to a runner. Therefore the workflow
must derive its temporary paths during step execution rather than while GitHub
is constructing the job.

The failed run had zero jobs, so it did not read model secrets, call a model, or
incur evaluation cost.

## Goals

- Make the manual SWE-bench workflow parse successfully on GitHub.
- Preserve runner-local temporary storage for RepoPilot state and the public
  Hugging Face cache.
- Preserve the manual-only trigger, fixed modes, concurrency bounds, cache
  allowlist, immutable action pins, and generation-step-only secret access.
- Add a regression contract that fails on unsupported job-level runner context.
- Obtain remote parser evidence without dispatching an evaluation.

## Non-goals

- Do not dispatch `checkpoint_5`, `baseline_50`, or any other evaluation.
- Do not read, modify, transmit, or validate model credentials.
- Do not change models, prompts, budgets, instance cohorts, scoring, or OCI
  behavior.
- Do not merge the pull request or begin Pilot-20 work.
- Do not modify or commit the user's existing `run_trace.py` change.

## Considered Approaches

### A. Runtime initialization through GITHUB_ENV

Remove `REPOPILOT_HOME` and `HF_HOME` from the job-level environment. Add one
early run step that appends exact values derived from `$RUNNER_TEMP` to
`$GITHUB_ENV`. Every subsequent step receives the same resolved paths.

This is the selected approach because it is portable, centralized, and keeps
the existing cache and execution paths unchanged.

### B. Repeat step-local environment expressions

Add `${{ runner.temp }}` expressions to every step that needs either path.
This is valid after runner assignment but duplicates configuration across
actions and shell steps, increasing drift and omission risk.

### C. Use fixed absolute paths

Use literal paths under `/tmp`. This would parse but would couple the workflow
to one runner layout and weaken the existing runner-temp lifecycle guarantee.

## Selected Design

The `instance` job retains only values whose contexts are valid at job
construction time: mode, instance ID, model endpoints and names, and escalation
controls.

Immediately after Python setup and before cache restoration or dependency
installation, add a step named `Configure temporary evaluation paths`. It
appends exactly these runtime-expanded assignments to `$GITHUB_ENV`:

- `REPOPILOT_HOME=$RUNNER_TEMP/repopilot-home`
- `HF_HOME=$RUNNER_TEMP/public-hf-cache`

The public cache action continues to reference `${{ runner.temp }}` in its
step-level `with.path`, where the runner context is valid. Existing shell
commands continue to use `$RUNNER_TEMP` directly.

No fallback path is added. If GitHub does not provide `RUNNER_TEMP` or
`GITHUB_ENV`, the initialization step fails and the evaluation remains
fail-closed.

## Regression Contract

Extend `tests/test_swe_bench_oci_workflow.py` with a focused helper and test
that enforce all of the following:

1. No `runner` expression appears inside any job-level `env` mapping.
2. The instance job contains exactly one `Configure temporary evaluation
   paths` step.
3. The step writes the two exact `RUNNER_TEMP`-derived assignments to
   `GITHUB_ENV`.
4. The initialization step precedes cache restoration and both installation
   steps.
5. A mutation that moves either runner expression back into job-level `env`
   is rejected.

The test must be observed failing against the current workflow before the
workflow is changed, then passing after the minimal fix.

## Verification

Local verification:

1. Run the new focused test in RED state.
2. Apply the workflow-only fix.
3. Run `tests/test_swe_bench_oci_workflow.py`.
4. Parse the YAML with the repository's existing Ruby validation command.
5. Run the focused OCI workflow/contract/runner/aggregate suite.
6. Run the full Python test suite, full Ruff command, and `git diff --check`.
7. Confirm `run_trace.py` remains unstaged with SHA-256
   `f721a313a68888a507608ea196b27e173d093f462065bf346c6b6a19f55b8eba`.

Remote verification:

1. Push the feature-branch commit to update PR #2.
2. Bind checks to the exact pushed head SHA.
3. Confirm the normal CI run contains exactly the expected eight jobs and all
   succeed.
4. Confirm GitHub no longer creates an invalid-workflow failure for the new
   head and recognizes the manual workflow.
5. Do not click `Run workflow`, rerun an evaluation, or otherwise dispatch
   SWE-bench.

## Completion Criteria

The repair is complete only when the regression test, local verification,
exact-head eight-job CI gate, and GitHub workflow-parser evidence all pass. The
result authorizes no paid evaluation, merge, or Pilot implementation.
