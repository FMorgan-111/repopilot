# Final Release Security Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` to implement this plan task-by-task,
> with a fresh implementer and an independent review after every task.

**Goal:** Remove every remaining Critical and Important release finding before
RepoPilot is pushed, opened as a PR, or used for a live SWE-bench evaluation.

**Architecture:** Keep repository acquisition, resumable state, model execution,
and evaluation artifacts as separate fail-closed trust boundaries. Git transports
receive credentials only through command-scoped URL rewriting; OCI work is
adapted to asyncio through cancellation-drained workers; resume claims are
single-use durable leases; legacy host evaluation requires an explicit unsafe
opt-in; scoring accepts only strict, cross-file-consistent records; CI dependencies
and actions are immutable before a secret-bearing step starts.

**Tech Stack:** Python 3.11, asyncio, Pydantic v2, Git, Docker/OCI CLI, pytest,
pytest-asyncio, uv/pip hash-locked requirements, GitHub Actions.

## Global constraints

- Preserve the user's unstaged `run_trace.py` change exactly and never include it
  in a task commit.
- Keep the already-completed immutable evaluation-input fix at commit `0583604`.
- Optimize for successful, trustworthy resolutions before execution speed.
- Use red-green-refactor TDD. Every task begins with a focused failing test, ends
  with its focused suite green, and is committed separately.
- Use `gpt-5.6-sol` for concurrency, cancellation, security, and review work.
- Never place a real model or GitHub credential in a command, test fixture, log,
  generated lockfile, artifact, commit, or review package.
- Keep synchronous `run_oci_process` for the evaluation CLI. Async agent paths
  must call a cancellation-drained adapter and must not block the event loop.
- Target-repository code never executes on the host by default. The legacy path
  requires both an explicit CLI/API choice and the exact environment value
  `REPOPILOT_UNSAFE_ALLOW_HOST_EXECUTION=1`.
- A cancelled or crashed resume is consumed fail-closed; it is not replayed and
  does not duplicate remote effects.
- Push/PR creation happens only after the final review has zero Critical and
  Important findings. Merging remains a separate explicit user authorization.

---

### Task 1: Make clone and Git cache cancellation credential-safe

**Files:**
- Modify: `src/nodes/execute.py`
- Test: `tests/test_execute_security.py`

**Interfaces and behavior:**
- Add a command-scoped Git transport helper that invokes Git with
  `-c url.<authenticated-url>.insteadOf=<safe-url>` while every persisted
  `remote.origin.url` remains the credential-free HTTPS URL.
- Validate an existing cache's origin against the exact safe URL. Remove and
  rebuild legacy or mismatched caches even when their `HEAD` is otherwise valid.
- Catch `BaseException`, not only `Exception`, around live-cache refresh,
  exact-ref population, live-cache population, and local mutable-checkout
  preparation. Remove a partially mutated cache before re-raising. Remove a
  partial worktree while preserving a fully verified immutable cache.
- Keep the existing per-cache lock and checked cleanup behavior. Cleanup failure
  must not be silently ignored or replace the original exception without
  chaining.

- [ ] Write failing tests covering cancellation during live refresh, exact-ref
  population, live clone, and local checkout/reset; safe persisted origins;
  rejection of a tokenized legacy cache; and cleanup failure chaining.
- [ ] Run
  `.venv/bin/python -m pytest tests/test_execute_security.py -q` and confirm the
  new tests fail against `0583604`.
- [ ] Implement the smallest shared URL-rewrite/origin-validation and
  `BaseException` cleanup helpers; update older tests that asserted tokenized
  clone arguments or a post-clone `remote set-url` call.
- [ ] Re-run the focused test file and require it to pass.
- [ ] Commit only the task files with
  `fix(execute): make clone cancellation credential safe`.

---

### Task 2: Drain OCI work before propagating async cancellation

**Files:**
- Modify: `src/safe_subprocess.py`
- Modify: `src/nodes/execute.py`
- Modify: `src/tool_policy.py`
- Modify: `eval/agent_v2_harness.py`
- Test: `tests/test_safe_subprocess.py`
- Test: `tests/test_execute_security.py`
- Test: `tests/test_tool_policy.py`
- Test: `tests/test_agent_v2_eval.py`

**Interfaces and behavior:**
- Retain synchronous `run_oci_process(...)` for the evaluator.
- Add `ProcessCancellationRequested` and an optional `threading.Event` to
  `run_bounded_process(...)`, checking it before spawn and during polling.
- Ensure process-group termination, stdout/stderr reader joins, stdin writer
  completion, and OCI container removal run in `finally` paths that cover every
  `BaseException`. `_force_remove_container` itself must remain uncancellable.
- Add `async run_oci_process_async(...)` using `asyncio.to_thread`, a cancellation
  event, and a shielded worker. When its caller is cancelled, signal the worker
  and continue draining through repeated cancellations; propagate the original
  `CancelledError` only after cleanup. Chain `IsolationCleanupError` if cleanup
  fails.
- Await this adapter in execute pytest, constrained targeted tools, and generated
  coverage. Keep the approved repository snapshot alive until the worker drains.

- [ ] Write failing tests for process-group reaping on cancellation and arbitrary
  `BaseException`, OCI removal on `BaseException`, event-loop heartbeat, delayed
  cancellation until cleanup, repeated cancellation, cleanup-error chaining,
  and execute/router/coverage snapshot lifetime.
- [ ] Run the four focused test files and confirm the new contract fails.
- [ ] Implement the cancellation token, `finally` cleanup, async adapter, and
  async call-site changes without duplicating the existing bounded-output logic.
- [ ] Re-run the four focused test files and require them to pass.
- [ ] Commit only the task files with
  `fix(oci): drain sandbox processes on cancellation`.

---

### Task 3: Disable mutable, credential-bearing legacy host evaluation

**Files:**
- Modify: `eval/harness.py`
- Test: `tests/test_eval_harness.py`
- Test: `tests/test_packaging_contract.py`

**Interfaces and behavior:**
- Make the safe Agent V2 evaluator the explicit supported path. The legacy
  evaluator must fail before cloning or executing unless the caller explicitly
  selects it and `REPOPILOT_UNSAFE_ALLOW_HOST_EXECUTION` is exactly `1`.
- Require every legacy sample to carry an exact 40-lowercase-hex base commit.
  Clone by initializing a repository, fetching that exact commit, and checking
  it out detached; never evaluate a mutable default branch or target `HEAD`.
- Pass the production minimal subprocess environment to every legacy Git,
  install, test, and patch subprocess so hostile target code cannot observe LLM,
  GitHub, Actions, or unrelated ambient secrets.
- Do not load `.env` merely by importing or choosing the default evaluator.

- [ ] Write failing tests proving the default never runs host code, both unsafe
  opt-ins are required, a missing/malformed commit fails before clone, detached
  exact-commit fetch is used, and hostile subprocesses receive no sentinels.
- [ ] Run
  `.venv/bin/python -m pytest tests/test_eval_harness.py tests/test_packaging_contract.py -q`
  and confirm the new tests fail.
- [ ] Implement the fail-closed CLI/function boundary, immutable fetch, and
  minimal environment plumbing.
- [ ] Re-run the focused tests and require them to pass.
- [ ] Commit only the task files with
  `fix(eval): fail closed on legacy host execution`.

---

### Task 4: Claim each resumed run exactly once across processes

**Files:**
- Modify: `src/state.py`
- Modify: `src/run_store.py`
- Modify: `src/new_agent.py`
- Modify: `src/main.py`
- Test: `tests/test_run_store.py`
- Test: `tests/test_new_agent.py`
- Test: `tests/test_main.py`
- Test: `tests/test_api_security.py`

**Interfaces and behavior:**
- Add a backward-compatible persisted `resume_in_progress: bool = False` field.
- Add a per-run cross-process lock and make both saves and claims use it.
- Implement `claim_run_for_resume(run_id, expected_state, *, root_dir=None)`:
  under the lock, re-read the durable state, compare the exact expected state,
  require `WAITING_FOR_USER` plus a pending request and no lease, then atomically
  persist a deep-copied claimed state with `resume_in_progress=True`, phase
  `PLAN`, and cleared pending/request fields.
- `resume_agent_v2` claims before injecting the answer or invoking the graph.
  Every normal terminal or newly paused outcome clears the lease and persists.
  Cancellation or an unhandled crash leaves the lease durable and consumed so
  replay cannot duplicate side effects.
- Raise a typed `ResumeConflictError`; the HTTP boundary maps it to a stable 409.

- [ ] Write failing tests for a stable serial second-resume 409, two simultaneous
  resumes executing the graph exactly once, two different run IDs proceeding
  concurrently, stale expected-state rejection, and cancellation leaving a
  durable consumed lease.
- [ ] Run the four focused test files and confirm the new tests fail.
- [ ] Implement descriptor-safe per-run locking, atomic claim/CAS, lifecycle
  persistence, typed conflict handling, and legacy-state compatibility.
- [ ] Re-run the focused tests and require them to pass.
- [ ] Commit only the task files with
  `fix(resume): make run continuation single use`.

---

### Task 5: Reject malformed or impossible evaluation telemetry

**Files:**
- Modify: `eval/oci_contract.py`
- Modify: `eval/oci_aggregate.py`
- Test: `tests/test_oci_contract.py`
- Test: `tests/test_oci_aggregate.py`

**Interfaces and behavior:**
- Define `extra="forbid"` Pydantic models for the complete result record and each
  model invocation. Invocation status is exactly `ok`, `invalid_response`, or
  `error`; provider/model pairs must match the configured primary or escalation
  model; elapsed time and token counts are finite and nonnegative.
- Define a closed failure taxonomy covering successful, agent/test failures, and
  all infrastructure classes, including `infra`, `model_gateway_infra`, and
  `coverage_infra`.
- Validate result, runtime manifest, official result, patch, and invocation data
  together. A non-ready runtime cannot claim terminal official completion,
  non-empty patches, or model invocations. Unknown, missing, mixed-validity, or
  impossible telemetry rejects the bundle rather than being filtered.
- Award non-infrastructure and budget credit only when runtime is ready,
  official scoring completed, telemetry is complete and valid, and all existing
  caps are met. Every infrastructure class receives zero non-infrastructure
  credit.

- [ ] Write failing tests for a mixed valid/malformed invocation list, garbage
  status, missing/unknown failure class, every infrastructure class, provider /
  model mismatch, and non-ready runtime paired with terminal official output.
- [ ] Run the two focused test files and confirm the new tests fail.
- [ ] Implement strict schemas, cross-file validation, and fail-closed scoring;
  remove permissive filtering/defaulting.
- [ ] Re-run the focused tests and require them to pass.
- [ ] Commit only the task files with
  `fix(eval): validate scoring telemetry strictly`.

---

### Task 6: Make the secret-bearing workflow supply chain immutable

**Files:**
- Create: `requirements-eval.in`
- Create: `requirements-eval.lock`
- Modify: `.github/workflows/swe-bench-oci-eval.yml`
- Modify: `tests/test_swe_bench_oci_workflow.py`
- Modify: `tests/test_packaging_contract.py`

**Interfaces and behavior:**
- Pin workflow actions to these audited full commits, retaining tag comments:
  - checkout: `11d5960a326750d5838078e36cf38b85af677262` (`v4`)
  - setup-python: `a26af69be951a213d495a4c3e4e4022e16d87065` (`v5`)
  - cache: `0057852bfaa89a56745cba8c7296529d2fc39830` (`v4`)
  - upload-artifact: `ea165f8d65b6e75b540449e92b4886f43607fa02` (`v4`)
  - download-artifact: `d3f86a106a0bac45b974a628896c90dbdf5c8093` (`v4`)
- Generate `requirements-eval.lock` from the tracked input for CPython 3.11 on
  x86_64 Linux, with hashes for every package/file accepted by pip. The input
  covers runtime, memory, SWE-bench evaluation, and build tooling dependencies.
- In every workflow job, install the lock with
  `python -m pip install --require-hashes -r requirements-eval.lock`, then install
  the project with `--no-deps --no-build-isolation -e .`. Complete all dependency
  installation before the generation step receives model secrets.
- No mutable action tag or unconstrained dependency install may remain.

- [ ] Write failing workflow tests that require 40-hex action references, a
  hash-checked lock install before generation, no secret on install steps, and no
  plain or extras-based editable install.
- [ ] Run the two focused test files and confirm the new tests fail.
- [ ] Generate the Linux/Python 3.11 lock with `uv pip compile --generate-hashes`
  and apply the immutable workflow changes.
- [ ] Validate the lock by creating a clean temporary virtual environment and
  running pip's hash-checked install plus the no-deps/no-build-isolation project
  install. Network/sandbox inability is infrastructure failure, not permission
  to weaken the lock.
- [ ] Re-run the focused tests and require them to pass.
- [ ] Commit only the task files with
  `build(eval): pin workflow supply chain`.

---

### Task 7: Fresh release gate, whole-branch review, push, and PR

**Files:** No implementation changes unless the final review identifies a
Critical or Important defect. Record deferred Minor findings in the review
ledger: general cache symlink/unbounded-I/O hardening and official scorer report
ingestion/cleanup hardening.

- [ ] Run the complete suite from the worktree:
  `.venv/bin/python -m pytest -q`.
- [ ] Run lint and patch checks:
  `.venv/bin/python -m ruff check src/ tests/ eval/ --select=E,F,I --ignore=E501`
  and `git diff --check origin/master...HEAD`.
- [ ] Export committed `HEAD` with `git archive`, run the complete suite there,
  and build both sdist and wheel without relying on ignored/untracked inputs.
- [ ] Re-check fixed-50 uniqueness/order/subset contracts and scan the committed
  tree plus reachable branch history for credential material without printing
  secret values.
- [ ] Produce a fresh whole-branch review package from the merge base through
  `HEAD`; use an independent `gpt-5.6-sol` reviewer and fix/re-review every
  Critical or Important finding.
- [ ] Confirm `git status --short` contains only the preserved `run_trace.py`
  modification, push `fix/release-readiness-20260717`, and create or update a PR
  targeting `master` with test/security evidence.
- [ ] Stop before merge and request explicit user authorization. After an
  authorized merge, dispatch `checkpoint_5`; only if its infrastructure and
  scoring artifacts are valid, dispatch `baseline_50` and report resolved rate,
  failure taxonomy, token budget, and elapsed-time metrics.

## Plan acceptance checklist

- Every Critical and Important review finding maps to exactly one task.
- Task interfaces agree: synchronous OCI stays available; async paths await the
  adapter; resume claims occur before side effects; scoring consumes strict
  records; workflow installation consumes the tracked lock.
- No task changes the selected Gemini primary / Opus escalation policy.
- No task includes `run_trace.py`, a credential value, or live evaluation output.
- The final gate uses committed clean-checkout evidence, not only the dirty
  development worktree.
