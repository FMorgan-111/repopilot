# SWE-bench OCI Matrix Evaluation Design

**Date:** 2026-07-22  
**Status:** Approved for planning  
**Branch:** `fix/release-readiness-20260717`

## Goal

Finish RepoPilot's success-first SWE-bench evaluation with reproducible runtime
evidence. Run the fixed five-instance checkpoint and the fixed ten-instance
baseline against SWE-bench Verified, give every RepoPilot test tool an immutable
instance-specific OCI environment, and use the official SWE-bench harness as the
only source of the final resolved metric.

This design closes the current gap in which macOS can generate and locally test
a candidate patch but cannot produce RepoPilot's required differential coverage
proof. It also avoids putting all ten large benchmark environments onto one
ordinary GitHub-hosted runner.

## Confirmed Decisions

1. Success rate is the primary objective; latency is secondary.
2. Gemini Flash remains the primary model and Claude Opus remains the one-way
   escalation model.
3. Evaluation uses the existing fixed instance lists in
   `eval/checkpoint_5_ids.txt` and `eval/baseline_10_ids.txt`.
4. Each GitHub Actions matrix job owns exactly one SWE-bench instance image.
5. At most two model jobs run concurrently to bound API load and cost.
6. The official remote `swebench/sweb.eval.x86_64.*` image is pulled, resolved
   to a local immutable `sha256:` image identity, and supplied through the
   operator-owned RepoPilot OCI environment variables.
7. RepoPilot's strict PatchGate and differential coverage requirements remain
   unchanged. Missing or unusable OCI infrastructure fails closed.
8. Official resolution is computed only by
   `swebench.harness.run_evaluation`; RepoPilot internal success is reported
   separately.
9. API credentials exist only as GitHub Actions encrypted secrets. They are not
   accepted as workflow inputs, command-line arguments, artifacts, cache keys,
   or repository files.
10. The five-instance checkpoint runs before the ten-instance baseline. The
    checkpoint is an infrastructure and model diagnostic gate, not a hidden
    redefinition of the requested ten-instance evaluation.

## Approaches Considered

### One GitHub Actions job for all instances

This is operationally simple but concentrates image storage, model runtime, and
the official scorer into one ordinary hosted runner. The official SWE-bench
documentation describes substantial storage requirements for general harness
runs, and a single long job is also more likely to exceed the hosted job time
limit. This approach is rejected.

### Local Docker on the Apple Silicon workstation

This would keep all state local, but Docker is not currently installed and the
official prebuilt evaluation images target Linux x86_64. Building ARM images
locally is supported experimentally but is slower and requires much more local
disk. This remains a diagnostic fallback, not the selected primary route.

### Per-instance GitHub Actions matrix

This is selected. Every job pulls only one prebuilt image, runs one model
sample, scores one prediction, uploads one bounded artifact bundle, and removes
the image. Failures are isolated per instance and successful jobs do not have
to be repeated when another repository fails.

## Architecture

```text
workflow_dispatch (checkpoint_5 or baseline_10)
                    |
                    v
          prepare matrix from tracked ID list
                    |
             max-parallel = 2
                    |
        +-----------+-----------+
        | one job per instance  |
        |                       |
        | load exact dataset row|
        | derive official image |
        | pull and pin sha256    |
        | capability preflight  |
        | run RepoPilot sample   |
        | run official scorer    |
        | sanitize artifacts     |
        +-----------+-----------+
                    |
                    v
              aggregate job
                    |
        combined internal + official report
```

The implementation has four boundaries:

- `eval/oci_instance.py`: resolves one allowlisted instance to its official
  image and validates the runtime configuration without exposing evaluator-only
  data to the agent.
- `eval/agent_v2_harness.py`: continues to run RepoPilot and write one safe
  result/prediction pair, now with an explicit single-instance entry path.
- `eval/aggregate_oci_eval.py`: validates and combines matrix artifacts in the
  exact tracked ID order, rejecting duplicates, missing instances, commit
  mismatches, malformed predictions, and untrusted extra fields.
- `.github/workflows/swe-bench-oci-eval.yml`: provides manual dispatch,
  per-instance isolation, secret injection, artifact retention, and the final
  aggregation job.

## Instance Image Resolution

The runner loads the official cached SWE-bench Verified row only inside the
evaluator boundary. It uses the installed official SWE-bench package to build a
`TestSpec` with namespace `swebench`. The resulting remote image must satisfy:

- instance ID exactly equals the requested tracked ID;
- architecture exactly equals `x86_64`;
- image name exactly follows the official TestSpec result;
- namespace exactly equals `swebench`;
- tag is fixed by the workflow implementation;
- Docker pull succeeds without building a repository-controlled Dockerfile.

After pull, Docker's local image identity must match `sha256:[0-9a-f]{64}`. The
agent receives that identity, not the mutable remote tag:

```text
REPOPILOT_TOOL_OCI_BACKEND=docker
REPOPILOT_TOOL_OCI_IMAGE=sha256:...
REPOPILOT_TOOL_PYTHON_EXECUTABLE=/opt/miniconda3/envs/testbed/bin/python
```

Before any model call, the runner invokes the same locked-down RepoPilot OCI
boundary used by production tools and proves that the configured Python can
import pytest. The container runs as a non-root user, with no network, a
read-only root filesystem, dropped capabilities, bounded CPU/memory/PIDs,
tmpfs-only writable locations, and one read-only disposable workspace mount.

If the official image cannot satisfy that boundary, the instance is recorded as
an infrastructure failure. The workflow never falls back to executing
repository code on the GitHub runner host.

## Model Execution

The job supplies these host-only settings:

```text
LLM_API_KEY                 <- GitHub secret
LLM_ESCALATION_API_KEY      <- GitHub secret
LLM_MODEL                   = gemini-3.5-flash:stable
LLM_ESCALATION_MODEL        = claude-opus-4-8:stable
REPOPILOT_ESCALATION_ENABLED=1
```

The credentials are available to the host model client only. RepoPilot's
minimal subprocess environment and OCI command builder must continue to exclude
them from every repository-controlled command and container. Workflow steps
must not enable shell tracing, echo environment variables, serialize the full
process environment, or upload RepoPilot home/cache directories.

Each sample keeps the current success-first limits:

- `max_retries=3`;
- `token_budget=100000`;
- Gemini primary with deterministic one-way Opus escalation;
- existing constrained tool and evidence limits;
- strict PatchGate authorization;
- strict differential coverage proof before internal success;
- safe result and prediction serialization.

The sample job always writes a result record even when clone, image, model, or
scorer infrastructure fails. It must not convert infrastructure failure into an
unresolved model verdict.

## Official Scoring

After RepoPilot writes one prediction, the job invokes the installed official
SWE-bench harness for that same instance with:

- dataset `SWE-bench/SWE-bench_Verified`;
- exactly one `--instance_ids` value;
- exactly one prediction JSONL;
- one worker;
- a deterministic run identifier containing the RepoPilot commit and instance
  identity;
- the official `swebench` image namespace;
- cleanup enabled after result extraction.

The scorer's report is reduced to a safe schema containing only instance ID,
submitted/completed/resolved booleans, and bounded infrastructure diagnostics.
Gold patches, test patches, expected test IDs, raw container logs, and evaluator
scripts are not copied into RepoPilot result artifacts or model traces.

An empty RepoPilot patch remains a valid submitted prediction for accounting but
cannot be labeled internally successful. The official harness owns whether the
instance is resolved.

## Workflow and Artifact Contract

The workflow is manual (`workflow_dispatch`) and accepts only a mode enum:

- `checkpoint_5`
- `baseline_10`

The mode maps to a tracked ID file. Arbitrary instance IDs are not accepted from
workflow input. The job matrix is created from the selected file and preserves
its exact order in the aggregate report.

GitHub exposes manual dispatch only after the workflow file exists on the
repository's default branch. Therefore implementation and review happen on the
feature branch, then the workflow must be merged through an authorized PR before
the live checkpoint can be dispatched. The feature branch never adds a
secret-bearing `pull_request` or arbitrary `push` trigger as a workaround.

Each instance artifact contains:

```text
result.json
prediction.jsonl
official_result.json
manifest.json
```

`manifest.json` binds the artifact to the instance ID, RepoPilot commit SHA,
dataset identifier, model names, OCI image SHA, and hashes of the other three
files. It contains no credentials or evaluator-only content.

The aggregate job downloads only artifacts from the current workflow run,
validates every manifest and file hash, and emits:

```text
eval/oci/<mode>/results.json
eval/oci/<mode>/predictions.jsonl
eval/oci/<mode>/official_results.json
eval/oci/<mode>/summary.md
```

The summary reports:

- requested, completed, internal-success, and official-resolved counts;
- infrastructure-failure count;
- failure taxonomy by decisive class;
- primary/escalation model invocation totals;
- per-instance internal and official status;
- RepoPilot commit and workflow run identity.

The workflow artifact is the authoritative eval deliverable. Generated eval
outputs are not committed to Git.

## Checkpoint and Baseline Sequencing

The five-instance run must prove:

1. all five matrix jobs start from the requested exact base commits;
2. all usable instances have a digest-pinned OCI configuration;
3. RepoPilot safe result and prediction files exist for all five;
4. the official scorer reports a terminal status for every non-infrastructure
   instance;
5. aggregation finds no duplicate, missing, mismatched, or unhashed artifact.

The ten-instance run starts only after these infrastructure properties pass.
The five-instance model success rate is diagnostic and is not a threshold that
can silently cancel the requested baseline. A genuine shared infrastructure
failure stops the baseline until fixed; an individual model failure does not.

## Failure Handling and Recovery

- **Dataset or TestSpec failure:** record `dataset_infra`; do not guess an image.
- **Image pull or digest failure:** record `oci_image_infra`; do not build or run
  on the host.
- **Capability/isolation failure:** record `oci_boundary_infra`; do not call the
  model for that instance.
- **Model/API failure:** retain bounded invocation diagnostics and record the
  existing model failure taxonomy.
- **Agent terminal failure:** export the safe empty or approved prediction and
  continue to official accounting.
- **Official scorer failure:** preserve internal result and record
  `official_scorer_infra`, never `official_resolved=false`.
- **Artifact validation failure:** fail aggregation loudly and identify only the
  instance and validation class.

GitHub Actions reruns may target failed matrix jobs. Artifact identities include
the commit and image SHA so results from different implementations cannot be
silently merged.

## Security Boundaries

1. Existing exposed API keys must be rotated before configuring GitHub secrets.
2. No secret value is placed in repository files, workflow inputs, step outputs,
   cache keys, artifact names, or CLI arguments.
3. Evaluator-only fields remain outside agent seeds, prompts, traces, result
   records, and summaries.
4. Repository-controlled code runs only through the verified OCI boundary.
5. Remote image tags are never persisted as the trusted execution identity;
   the local immutable SHA is required.
6. The workflow does not use third-party disk-cleanup actions or execute
   downloaded scripts outside pinned Python packages and official images.
7. Artifact aggregation accepts only fixed schemas and exact instance IDs.
8. Raw model responses, raw HTTP bodies, full test logs, and complete process
   environments are not uploaded.

## Testing Strategy

Implementation follows red-green-refactor cycles.

Unit tests must prove:

1. exact mode-to-ID mapping and rejection of arbitrary/duplicate IDs;
2. official TestSpec image-name validation and x86_64 enforcement;
3. mutable tags are converted to and persisted only as local SHA identities;
4. capability failure prevents the model call;
5. OCI configuration never includes model credentials;
6. one-sample runner emits a safe result for every infrastructure failure;
7. scorer parsing distinguishes unresolved from scorer infrastructure failure;
8. artifact manifests bind hashes, commit, instance, and image identity;
9. aggregation rejects missing, duplicate, extra, mismatched, malformed, or
   cross-commit artifacts;
10. summary counts and failure taxonomy are deterministic;
11. evaluator-only and credential-shaped sentinels cannot enter artifacts;
12. workflow structure uses a manual enum, matrix jobs, `max-parallel: 2`,
    encrypted secrets, and no secret-bearing command arguments.

Integration tests use mocked Docker and scorer boundaries for deterministic CI.
The existing real Docker isolation test remains the lower-level security gate.
A live workflow preflight on one tracked checkpoint instance is required before
the full five-instance run.

## Acceptance Criteria

Implementation is accepted only when:

1. focused tests, the complete project suite, Ruff, and package build pass;
2. security review finds no credential, evaluator, host-execution, or mutable
   image trust bypass;
3. a one-instance live preflight proves the official image supports RepoPilot's
   locked-down OCI execution;
4. the fixed five-instance workflow completes and produces a validated aggregate
   artifact;
5. the fixed ten-instance workflow completes and produces a validated aggregate
   artifact;
6. every requested instance has an internal terminal result and either an
   official terminal result or an explicit scorer infrastructure failure;
7. final reported resolution counts come from the official harness, not
   RepoPilot heuristics;
8. the implementation branch and workflow commits are pushed to the requested
   remote branch.

## Out of Scope

- Training or fine-tuning on SWE-bench evaluator data.
- Passing gold patches, test patches, or expected test IDs to RepoPilot.
- Replacing RepoPilot's strict differential coverage gate with the official
  scorer.
- General arbitrary-dataset workflow inputs.
- Supporting arm64 official images in this first workflow.
- Automatically purchasing or configuring GitHub larger runners.
- Treating GitHub Actions internal success as the official benchmark score.
