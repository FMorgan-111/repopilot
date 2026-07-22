# Task 2 Report — OCI repository execution and trace-isolated clones

## Status

Completed from baseline `99c6366a34a019432112c726ac0b441e092debdf`.
The scoped commit is recorded in the parent-agent handoff. No push was made.
The user's existing `run_trace.py` modification was not edited or staged.

## Implemented contract

- `src/tool_policy.py` now selects repository execution mode before clone or
  patch mutation. A configured OCI sandbox always wins; otherwise the process
  fails closed unless the operator value is exactly
  `REPOPILOT_UNSAFE_ALLOW_HOST_EXECUTION=1`.
- `src/nodes/execute.py` requires an exact 40-hex `repo_ref`, matching
  PatchGate approval, patch digest, and manifest digest before OCI execution.
  OCI mode skips host venv creation, editable installation, pytest
  installation, and `run_pytest` completely.
- OCI execution creates a fresh exact-base approved snapshot through the
  runtime-local `tool_router.disposable_test_snapshot` import and calls
  `run_oci_process` with a fixed argv. A valid planner test command reaches OCI
  only after `fixed_test_argv` approval. Invalid or hostile text is ignored in
  favor of `[sandbox_python, -P, -m, pytest, -q]`; no planner string or shell is
  passed to a subprocess.
- Explicit unsafe host subprocesses now receive only `minimal_subprocess_env`.
  Model, GitHub, cloud, CI, and GitHub Actions environment sentinels are absent.
- Mutable checkout names use only SHA-256 of the non-empty trace identity;
  sibling venv names inherit the same scope. The SWE-bench pre-clone and final
  agent now share one generated 12-hex trace identity.
- Shared cache inspection, population, refresh, and local clone are serialized
  under a per-cache advisory lock with safe directory/file ownership, mode,
  link-count, and no-follow checks. Cache state is rechecked under the lock.
- Credential-free remote restoration is checked and fails closed. Reused and
  new mutable checkouts require the approved HEAD and an empty status after
  checked hard reset and `git clean -fdx`, including ignored build outputs.
- The OCI evaluation workflow remains free of the unsafe host opt-in.

## Files changed

- Production: `src/nodes/execute.py`, `src/tool_policy.py`,
  `src/safe_subprocess.py`, `src/new_agent.py`, `eval/agent_v2_harness.py`.
- Tests: added `tests/test_execute_security.py`; updated focused execution,
  install, clone-cache, eval, preflight, and convergence tests to make legacy
  host execution an explicit test-only opt-in and to assert the new contracts.
- Report: `.superpowers/sdd/task-2-report.md`.

## TDD RED evidence

Initial execution-security command:

```text
.venv/bin/python -m pytest -q tests/test_execute_security.py
```

Result before implementation:

```text
14 failed, 3 passed in 9.82s
```

The failures demonstrated the missing mode helper, clone-before-fail-close,
host venv/test invocation in OCI mode, missing fixed-argv OCI route, inherited
host credentials/Actions environment, shared mutable worktree naming, absent
cache serialization, and unchecked reset/dirty status.

Trace-identity command before implementation:

```text
.venv/bin/python -m pytest -q \
  tests/test_agent_v2_eval.py::test_swe_bench_preclone_and_agent_share_one_trace_identity \
  tests/test_new_agent.py::test_seeded_trace_identity_is_used_by_final_agent
```

Result: `2 failed in 0.35s`; the eval seed lacked a trace and the final agent
generated a different identity.

The safe-lock mode test was then added and observed RED independently:

```text
1 failed in 0.31s
```

It proved a world-writable cache lock directory was accepted before the safety
checks were implemented.

## GREEN evidence

Focused new boundary:

```text
.venv/bin/python -m pytest -q tests/test_execute_security.py \
  tests/test_agent_v2_eval.py::test_swe_bench_preclone_and_agent_share_one_trace_identity \
  tests/test_new_agent.py::test_seeded_trace_identity_is_used_by_final_agent
19 passed in 1.11s
```

Complete focused Task 2 set after final cleanup hardening:

```text
.venv/bin/python -m pytest -q tests/test_execute_install.py \
  tests/test_execute_security.py tests/test_patch_preflight.py \
  tests/test_new_agent.py tests/test_convergence_diversity.py \
  tests/test_tool_policy.py tests/test_tool_router.py \
  tests/test_agent_v2_eval.py tests/test_swe_bench_oci_workflow.py \
  tests/test_safe_subprocess.py
334 passed in 11.66s
```

Full project suite:

```text
.venv/bin/python -m pytest -q
1185 passed, 2 skipped, 1 warning in 37.40s
```

The warning is the existing sqlite-vec/NumPy fallback warning from
`tests/test_error_episodes.py`.

Static and diff gates:

```text
.venv/bin/python -m ruff check src/ tests/ eval/ --select=E,F,I --ignore=E501
All checks passed!

git diff --check
(no output; exit 0)

git diff --check origin/master...HEAD
(no output; exit 0)
```

## Self-review

- Mode selection and OCI approval validation occur before `git_clone` and
  before either patch application path. Tests make clone/patch helpers fail if
  reached in a fail-closed state.
- The only planner-command conversion in OCI execution is the existing
  `fixed_test_argv` policy. Both shell-metacharacter and non-allowlisted cases
  fall back to the policy-owned full pytest argv.
- The exact snapshot rechecks patch and manifest binding independently; any
  mismatch or inability to construct the snapshot is an infrastructure error,
  never a host fallback.
- No targeted-test, PatchGate, generated-test, differential-coverage, token,
  or scoring policy was weakened.
- Per-module legacy execution fixtures use the exact unsafe value solely to
  retain coverage of the compatibility path. Default fail-close tests delete
  or vary that environment explicitly.
- Cache locking covers the lock-internal recheck and the entire refresh,
  population, and local-clone sequence. Two concurrent trace identities share
  one cache population but receive distinct mutable paths.
- The staging plan explicitly excludes `run_trace.py`; status and staged diff
  are rechecked before commit.

## Concerns and caveats

- The real Docker/Podman isolation integration remains environment-gated and
  was one of the two skipped tests. OCI orchestration is covered with the
  existing mocked boundary and exact-snapshot tests; no live image was run in
  this workstation verification.
- Advisory locking uses POSIX `flock`, matching the Linux OCI runner and macOS
  development environments. Native Windows execution is not part of this OCI
  evaluation contract.
