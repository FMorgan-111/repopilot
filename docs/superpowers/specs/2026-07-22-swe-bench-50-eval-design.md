# SWE-bench 50-Instance Evaluation Design

**Date:** 2026-07-22  
**Status:** Approved design; written-spec review  
**Branch:** `fix/release-readiness-20260717`

## Goal

Extend the existing immutable OCI SWE-bench pipeline from its ten-instance
baseline to one fixed, reproducible 50-instance SWE-bench Verified evaluation.
The official harness resolution rate is the primary RepoPilot score. Runtime
reliability, internal-verdict calibration, and speed remain visible secondary
metrics and must never be blended in a way that hides the raw official score.

This document amends, rather than replaces,
`2026-07-22-swe-bench-oci-matrix-eval-design.md`. All security, model,
isolation, artifact, and official-scoring requirements in that design remain
binding except where this amendment explicitly changes the public baseline mode
from ten instances to 50.

## Confirmed Decisions

1. Public evaluation modes are exactly `checkpoint_5` and `baseline_50`.
   `baseline_10` is retired as a dispatchable mode so the requested run cannot
   accidentally report a ten-instance denominator.
2. `eval/baseline_10_ids.txt` remains as historical selection provenance, but
   is not mapped by `EvalMode` or accepted by either CLI.
3. `eval/baseline_50_ids.txt` is the authoritative ordered allowlist. It
   contains exactly 50 unique IDs, starts with the complete historical
   ten-instance list in its original order, and therefore also contains all
   five checkpoint IDs.
4. The five-instance checkpoint is run first as an infrastructure diagnostic.
   It is a subset of the 50, not five additional scored instances. The final
   benchmark denominator is exactly 50.
5. Success rate remains the primary objective. Gemini Flash remains primary,
   Claude Opus remains the one-way escalation model, generation keeps three
   retries and a 100,000-token budget, and at most two model jobs run at once.
6. Official image pulls gain bounded transient retry. There is no fallback to
   a mutable image identity, repository Dockerfile, or host execution.
7. The aggregator rejects every unexpected top-level artifact entry, including
   a directory without a manifest. It continues to require exactly one valid,
   hash-bound bundle for every tracked ID.
8. Only public dataset caches may cross jobs. Cache keys and payloads contain no
   model credentials, generated patches, model responses, RepoPilot home data,
   target checkouts, or evaluator run logs.
9. The workflow must exist on the repository's actual default branch,
   `master`, before manual dispatch. The feature branch is still the evaluation
   ref after the workflow definition is merged through a user-authorized PR.

## Fixed Sample Construction

The frozen selection is derived from the locally cached official
`SWE-bench/SWE-bench_Verified` test split at revision `main`. The selection
source contained 500 unique instances across 12 repositories and had SHA-256
`f61cd55ceb35b61ad592f645abcbfc8ea4d294c6c9f3c8f15e83211a8e8db98c`.

Construction is deterministic:

1. Copy `eval/baseline_10_ids.txt` as the fixed prefix without reordering it.
2. Exclude those ten IDs from the 500-row source.
3. Use `random.Random(17)` and the project's existing shuffled repository
   queues to produce a stable repository priority.
4. Fill each repository up to four instances where capacity permits.
5. In seeded repository-priority order, raise the first seven repositories
   with sufficient capacity to five instances.
6. Append selected IDs in that stable round-robin order until the tracked file
   contains exactly 50 IDs.

The committed ID file, not a runtime random-selection function, is the
execution contract. Tests validate its size, uniqueness, historical prefix,
checkpoint inclusion, and membership in the cached official dataset when that
cache is available. Runtime jobs accept no arbitrary instance input outside
the selected mode's tracked file.

The target repository distribution is:

| Repository | Instances |
| --- | ---: |
| `astropy/astropy` | 5 |
| `django/django` | 5 |
| `matplotlib/matplotlib` | 5 |
| `mwaskom/seaborn` | 2 |
| `pallets/flask` | 1 |
| `psf/requests` | 5 |
| `pydata/xarray` | 4 |
| `pylint-dev/pylint` | 5 |
| `pytest-dev/pytest` | 4 |
| `scikit-learn/scikit-learn` | 5 |
| `sphinx-doc/sphinx` | 5 |
| `sympy/sympy` | 4 |

## Runtime Reliability

Image preparation still derives the official
`swebench/sweb.eval.x86_64.*:latest` name from the official TestSpec and trusts
only the post-pull local `sha256:` identity. A failed pull is retried at most
twice after the initial attempt, with bounded delays of 5 and 20 seconds, only
when diagnostics indicate a transient registry or transport condition such as
HTTP 429/500/502/503/504, timeout, connection reset, temporary DNS failure,
TLS handshake timeout, or unexpected EOF. Authentication, not-found, malformed
image, and other permanent failures fail immediately.

After a successful pull, digest inspection and the locked OCI capability
preflight remain mandatory. Exhausted retry is recorded as
`oci_image_infra`; it is not an unresolved model verdict.

The workflow may cache only the public Hugging Face/dataset download state,
using a versioned key derived from dataset identity and revision. It must not
cache Docker's mutable tag as trusted state. A cache miss or cache service
failure falls back to the normal official dataset download and does not weaken
the dataset or agent-data boundary.

## Workflow and Aggregation

`workflow_dispatch` offers exactly the two fixed modes. The checkpoint and
baseline both use one instance per `ubuntu-latest` matrix job,
`fail-fast: false`, `max-parallel: 2`, and the existing 360-minute per-instance
timeout. A workflow-level concurrency group prevents two manually dispatched
evaluation runs from multiplying the model concurrency limit; queued runs are
not cancelled.

Every job still uploads only:

- `result.json`
- `prediction.jsonl`
- `official_result.json`
- `manifest.json`

Aggregation enumerates the artifact root before reading manifests. Every
top-level entry must be a real directory containing exactly the safe bundle;
symlinks, root files, nested extras, missing manifests, duplicate instance IDs,
unknown IDs, mode mismatches, commit mismatches, and hash mismatches fail the
aggregate job. Output order is exactly `eval/baseline_50_ids.txt` order.

## Scoring and Final Report

The primary benchmark score is always displayed first:

```text
official_score = 100 * official_resolved / 50
```

Infrastructure failures remain in the denominator and are also reported by
class. The report additionally shows:

- official terminal coverage: terminal official verdicts divided by 50;
- internal success count and agreement with terminal official verdicts;
- primary and escalation invocation totals;
- per-instance and aggregate elapsed time and token usage when safely present;
- checkpoint outcome separately from the 50-instance baseline.

For the requested product-style grade, a transparent secondary engineering
score may be reported, never in place of the official score:

```text
engineering_score =
    80 * official_resolved / 50
  + 10 * non_infrastructure_instances / 50
  +  5 * internal_official_agreements / official_terminal_instances
  +  5 * within_time_and_token_budget_instances / 50
```

If there are no terminal official instances, the agreement component is zero.
`within_time_and_token_budget_instances` requires both a completed job within
360 minutes and generation within the 100,000-token budget. The final handoff
shows all four numerators and denominators so the composite remains auditable.

## Testing and Acceptance

Implementation follows red-green-refactor TDD. Focused tests must prove:

1. only `checkpoint_5` and `baseline_50` are accepted publicly;
2. the 50-ID file is exact-size, unique, ordered, preserves the historical ten
   prefix, and contains every checkpoint ID;
3. runner and aggregate CLIs accept `baseline_50` and reject `baseline_10`;
4. transient pull failures retry exactly within the 5/20-second schedule,
   permanent failures do not retry, and successful retry is digest-pinned;
5. aggregation rejects a top-level file or directory without a manifest as
   well as its existing malformed, missing, duplicate, extra, mode, commit,
   hash, and unsafe-content cases;
6. workflow inputs are exact, global and per-run model concurrency stay
   bounded, secrets remain generation-step-only, and caches cannot include
   secret or generated-output paths;
7. the aggregate summary keeps official, internal, infrastructure, calibration,
   and budget metrics separate and deterministic.

Acceptance requires focused tests, the complete project suite, Ruff, and a
no-isolation package build to pass locally. Live acceptance then requires the
workflow definition on `master`, a successful five-instance checkpoint, and a
complete 50-instance aggregate in which every requested ID has either an
official terminal verdict or an explicit infrastructure result. Only the
official harness determines resolution.

## Out of Scope

- Arbitrary instance IDs or a runtime sample-count input.
- Treating the checkpoint as extra scored data or changing the denominator
  after seeing outcomes.
- Caching credentials, generated patches, model responses, target checkouts,
  Docker mutable-tag trust, or evaluator logs.
- Raising model concurrency above two to optimize speed.
- Replacing official scoring with RepoPilot's internal PatchGate verdict.
- Fixing unrelated GitHub API cache behavior in the same implementation plan.
