# SWE-bench Integration and Resilient API Cache Design

**Date:** 2026-07-20

## Goal

Make RepoPilot evaluation reproducible and independent of live GitHub Issue
availability by integrating SWE-bench Verified first, then make the normal
GitHub API path tolerate short service outages through bounded stale-cache
fallbacks and visible cache diagnostics.

## Approaches Considered

### Evaluation data

1. Continue enriching `data/samples/issues_fixes.jsonl` from live GitHub APIs.
   This preserves the current schema but remains vulnerable to GitHub outages
   and still lacks reliable historical checkout information.
2. Update every existing sample to the latest default branch. This is easy but
   invalid: the historical issues have already been fixed, so the agent would
   be evaluated against post-fix code.
3. Integrate SWE-bench Verified and honor each instance's `base_commit`
   (selected). It supplies a problem statement, immutable pre-fix commit,
   reference patch, test patch, and explicit fail-to-pass/pass-to-pass tests.

### Repository reuse

1. Refresh one mutable per-repository checkout to the latest default branch.
   This is appropriate for normal live runs but cannot reproduce historical
   benchmark instances.
2. Create an independent full clone for every instance. This is correct but
   wastes network, disk, and setup time.
3. Reuse cached Git objects while creating commit-keyed benchmark work trees
   (selected). Normal live runs retain latest-branch semantics; benchmark runs
   always use their exact `base_commit`.

### Final scoring

1. Treat RepoPilot's guessed local pytest command as the benchmark verdict.
   This is fast but not comparable or reliably reproducible.
2. Export the generated patch in SWE-bench prediction format and use the
   official containerized harness for the final resolved metric (selected).
   RepoPilot's local test loop remains useful agent feedback, but it is not the
   benchmark ground truth.

## Phase 1: SWE-bench Verified Integration

### Dataset acquisition and normalization

SWE-bench support is an optional evaluation capability, not a production
runtime dependency. An `eval` package extra provides the official Hugging Face
`datasets` client. The importer loads `SWE-bench/SWE-bench_Verified`, records
the dataset identifier and revision, and normalizes selected rows into a local
JSONL cache under `${REPOPILOT_HOME}/eval/datasets/`. Downloaded dataset data is
not committed to Git.

The normalized sample contract contains:

- `instance_id`
- `repo` (`owner/name`)
- `issue_id`, `issue_url`, and `problem_statement`
- `base_commit`
- `patch` and `test_patch`
- `FAIL_TO_PASS` and `PASS_TO_PASS`
- `version`, `created_at`, and difficulty when supplied

The agent may receive only the repository identity, issue metadata,
`problem_statement`, and `base_commit`. The reference `patch`, `test_patch`,
and expected test lists remain evaluator-only to prevent answer leakage.

The importer supports an explicit sample count, deterministic seed, and
repository-diverse selection. A ten-sample run should not simply take the first
ten rows; it should distribute samples across repositories where possible and
record the selected instance IDs in the result artifact.

### Agent seed and graph entry

`AgentState` gains an optional immutable `repo_ref`. A non-oracle benchmark
seed populates issue fields, `repo_ref`, and a prepared local `repo_path`, then
starts the graph at `LOCATE`. A seed containing pre-hydrated relevant files
retains the current oracle behavior and starts at `PLAN`.

This separates two meanings that are currently conflated by `seed`:

- issue seed: skips only `UNDERSTAND`, preserving end-to-end code location;
- oracle-file seed: skips `UNDERSTAND` and `LOCATE` and is labeled
  `oracle_files`.

SWE-bench Verified runs use the issue-seed path and remain labeled
`end_to_end`.

### Commit-aware repository cache

Repository preparation has two explicit modes:

- live mode, without `repo_ref`: refresh the cached default branch before
  creating or resetting its work tree;
- benchmark mode, with a 40-character hexadecimal `repo_ref`: fetch that exact
  commit when absent and create/reuse a work tree keyed by repository and full
  commit identity.

Benchmark work trees must verify that `HEAD == repo_ref` before reuse. A
missing, corrupt, empty, or mismatched cache is discarded and rebuilt. Patch
application and tests run only in the work tree, never in the shared Git-object
cache. Cache logs report mode, hit/miss/rebuild, repository, and abbreviated
commit without printing credentials.

### Historical local code location

For a state with `repo_path`, LOCATE must use that checkout for both searching
and reading. It enumerates tracked files with Git, searches bounded source-file
content and filenames for the existing issue-derived terms, excludes the same
documentation paths as the current locator, ranks candidates with the existing
BM25 machinery, and hydrates files directly from disk.

It must not combine local historical content with GitHub code search results
from the latest default branch. The current GitHub API locator remains the
fallback for normal states without a prepared local checkout.

### Prediction export and official evaluation

Each completed inference result records the SWE-bench `instance_id`,
`base_commit`, model, RepoPilot commit SHA, generated patch, token use, timing,
and infrastructure/model failure classification. A separate prediction JSONL
contains the official fields `instance_id`, `model_name_or_path`, and
`model_patch`.

The official SWE-bench Docker harness consumes that prediction file and owns
the final `resolved` result. Gold patches and test patches are never included
in model prompts or RepoPilot traces. If Docker or an instance image cannot be
prepared, the result is classified as infrastructure failure rather than
unresolved model output.

## Phase 2: Resilient GitHub API Cache

### Fresh and stale windows

The existing successful-response file cache remains read-through. Its default
fresh TTL stays 600 seconds. Entries are no longer deleted immediately when
they leave the fresh window; they remain eligible for a bounded stale window,
defaulting to 86,400 seconds and configurable through
`REPOPILOT_CACHE_STALE_TTL`.

On a fresh hit, the wrapped function returns without network access. On a
miss, it calls the API and stores successful results. When a stale entry exists,
the wrapper first attempts a live refresh and returns stale data only if that
refresh fails with a retryable GitHub condition: HTTP 429/502/503/504,
network error, or timeout. Authentication errors, permission errors, 404s,
schema errors, and other application failures are re-raised and must not be
hidden by stale data.

Stale fallback never fabricates a cache entry. If no previous successful value
exists, the original error is returned after the normal retry policy.

### Diagnostics

API cache operations emit concise structured diagnostics for `hit`, `miss`,
`refresh`, `stale_fallback`, `expired`, and `save_failed`. Logs include the
function name, cache age, and cache key prefix, but never cached response bodies,
request headers, authorization tokens, or tokenized repository URLs.

Repository cache operations emit corresponding `worktree_hit`, `object_hit`,
`remote_fetch`, `rebuild`, and `ref_mismatch` events. These diagnostics make it
possible to distinguish GitHub Issue failures from clone behavior without
inspecting internal directories.

## Compatibility and Migration

- Existing `data/samples/issues_fixes.jsonl` remains supported.
- Existing callers that do not provide `repo_ref` keep live default-branch
  behavior.
- Existing cache files remain readable; timestamps older than the stale window
  expire normally.
- `REPOPILOT_DISABLE_CACHE` continues to bypass both fresh and stale API cache
  reads.
- Historical eval result files are not rewritten.
- The existing `--seed-gold-files` flag remains available and continues to mean
  oracle-file evaluation.

## Testing and Acceptance

Implementation follows red-green-refactor cycles. Acceptance requires:

1. Dataset-adapter tests prove field mapping, deterministic diverse sampling,
   local persistence, and that evaluator-only fields never enter the agent
   seed.
2. Seed-routing tests prove issue-only seeds start at LOCATE while oracle-file
   seeds start at PLAN.
3. Repository tests prove exact-commit cache population, healthy reuse,
   mismatched-HEAD rebuild, live-branch refresh, and credential redaction.
4. Locator tests prove a historical local checkout is searched/read without
   GitHub API calls and that its results are BM25-ranked.
5. Prediction tests prove correct SWE-bench JSONL output and no gold/test data
   leakage into traces or model payloads.
6. Cache tests prove fresh hit, miss, successful stale refresh, retryable stale
   fallback, stale-window expiry, non-retryable error propagation, disabled
   cache behavior, and safe diagnostics.
7. Focused suites, the complete project test suite, Ruff, and package build all
   pass.
8. A ten-instance Gemini run uses SWE-bench Verified issue seeds and exact
   `base_commit` checkouts. Final resolved counts come from the official
   SWE-bench harness; infrastructure failures are reported separately.

## Out of Scope

- Training or fine-tuning a model on SWE-bench gold patches.
- Supporting SWE-bench Multimodal or Multilingual in this change.
- Replacing the official SWE-bench Docker harness with RepoPilot's local test
  heuristics.
- Bulk-mutating the existing custom dataset to invent missing base commits.
- Serving cached GitHub data beyond the configured stale window.
