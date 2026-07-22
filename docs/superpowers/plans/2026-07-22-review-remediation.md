# Release Review Remediation Plan

**Date:** 2026-07-22
**Branch:** `fix/release-readiness-20260717`
**Scope:** Resolve the Critical and Important findings from the final branch,
SWE-bench/OCI, and runtime-security reviews before push or evaluation.

## Non-negotiable outcomes

- A process holding model credentials never executes target-repository build or
  test code on the host during the OCI evaluation.
- Official scoring, public dataset input, and uploaded artifacts are bound to
  immutable identities and bounded inputs.
- Missing telemetry never receives engineering-score credit, and every score
  component is reproducible from the aggregate report.
- API/run persistence/provider boundaries fail closed on untrusted identifiers,
  endpoints, repositories, and unauthenticated requests.
- Existing user changes in `run_trace.py` remain untouched.

## Task 1: Harden the SWE-bench evaluation boundary

Files: `eval/swe_bench.py`, `eval/oci_contract.py`, `eval/oci_runner.py`,
`eval/oci_aggregate.py`, `.github/workflows/swe-bench-oci-eval.yml`, `README.md`,
and their focused tests.

1. Pin `SWE-bench_Verified` to immutable Hugging Face revision
   `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`.
2. Validate cached dataset metadata, row schema, row count, and content SHA-256;
   bind the selected row digest into runtime and artifact manifests, and key the
   public workflow cache by the revision.
3. Treat bounded-process pull timeouts as transient and preserve the exact
   three-attempt, 5/20-second retry schedule.
4. Tag the already-inspected image SHA with a run-local scorer tag, verify the
   alias before and after official scoring, and never score through `latest`.
5. Enforce per-file and total artifact byte limits before and during reads.
6. Award agreement/budget points only for explicit valid internal verdict and
   completed, present model telemetry. Emit all four score fractions and
   per-instance token/elapsed usage; correct README scoring semantics.

TDD evidence: timeout retry, mutable-cache rejection, row-digest mismatch,
retargeted-`latest` isolation, oversized regular-file rejection, missing
telemetry zero-credit, and complete report fields.

## Task 2: Route repository execution through OCI and isolate clones

Files: `src/nodes/execute.py`, `src/tool_policy.py`, and focused execution,
clone-cache, policy, OCI, and integration tests.

1. When an OCI tool sandbox is configured, skip host venv creation/editable
   installation entirely. Build an exact approved disposable snapshot and run
   only a fixed allowlisted pytest argv through `run_oci_process`.
2. Default to fail-closed when no sandbox is available. Retain legacy host
   execution only behind the explicit operator opt-in
   `REPOPILOT_UNSAFE_ALLOW_HOST_EXECUTION=1`; the evaluation workflow never sets
   it. Even opt-in subprocesses receive a minimal credential-free environment.
3. Prove with hostile build/test sentinels that OCI mode never invokes host
   install/test functions and never exposes credential or Actions environment.
4. Keep a shared immutable object cache, but derive each mutable checkout and
   sibling venv from a safe hash of the run trace. Serialize cache population
   and refresh per cache key, check reset/clean results, and verify a clean tree.

## Task 3: Close runtime persistence, provider, HTTP, and commit gaps

Files: `src/run_store.py`, `src/model_provider.py`, `src/http_client.py`,
`src/nodes/commit.py`, `src/cache.py`, and focused tests/docs.

1. Enforce a short safe run-ID grammar, containment checks, regular-file/no-
   follow reads, and atomic writes.
2. Pair generic `LLM_BASE_URL` only with `LLM_API_KEY`; remove cross-vendor key
   fallbacks, require HTTPS except explicit loopback development endpoints, and
   keep primary/escalation credentials independent.
3. Bound JSON/SSE response bytes, events, choices, content, and tool arguments.
4. Use collision-resistant per-run repair branches. On any ref collision,
   require the remote head to equal the approved base, and verify the final
   remote diff exactly matches the approved manifest before PR creation.
5. Partition authenticated GitHub cache entries by a non-secret credential
   fingerprint so private responses do not cross token rotations.

## Task 4: Put a fail-closed API boundary around ambient authority

Files: `src/main.py`, `.env.example`, README, and API tests.

1. Keep health checks public, but require a configured bearer token for agent,
   resume, inspect, and replay endpoints.
2. Enforce hard retry/token/request limits, bounded concurrency, and an explicit
   repository allowlist before a request can use ambient GitHub/LLM authority.
3. Treat missing API auth or repository authorization as a rejection before
   cloning, model calls, or writes.

## Explicit approved-design exception

Generated differential tests remain eligible for the internal
`generated_verified` status because the previously approved success-first spec
requires it. They do not affect the authoritative official `resolved` score;
the report exposes internal/official agreement so self-certifying or unrelated
tests remain visible. Changing this policy requires a separate user decision.

## Final gate

Run all focused suites while implementing, then run:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src/ tests/ eval/ --select=E,F,I --ignore=E501
git diff --check origin/master...HEAD
```

Build an archive of committed `HEAD` into a temporary directory, produce both
sdist and wheel, re-run the fixed-50 identity and credential scans, then obtain
a fresh read-only final review over the complete merge-base range. Push and PR
creation remain blocked until no Critical or Important finding remains.
