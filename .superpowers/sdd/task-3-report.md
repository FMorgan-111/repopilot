# Task 3 Report: Runtime Persistence, Provider, HTTP, Commit, and Cache Boundaries

## Scope

Implemented Task 3 from
`docs/superpowers/plans/2026-07-22-review-remediation.md` on baseline
`ebf8450fab79a2ff5b368a9e70fdfd7bda36b972`.

The existing user edit in `run_trace.py` was not modified or staged.

## Implementation

### Run persistence

- Enforced the exact run ID grammar
  `^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$` before constructing a run path.
- Validated the runs directory as a non-symlink directory and used directory
  file descriptors for read/write operations.
- Opened run files with `O_NOFOLLOW` where supported and `O_NONBLOCK`, then
  required a regular file with `fstat` and capped reads at 8 MiB.
- Replaced direct writes with same-directory exclusive temporary files,
  `fsync`, `os.replace`, directory `fsync`, and mode `0600`; failed replaces
  preserve the previous file and clean the temporary file.
- Ignored symlink/non-regular entries while listing runs.

### Provider configuration

- Made `LLM_BASE_URL` the primary endpoint variable paired only with
  `LLM_API_KEY`.
- Retained `OPENAI_BASE_URL` only as a deprecated, unambiguous alias; conflicting
  values fail closed.
- Removed `LINOAPI_API_KEY` and `DEEPSEEK_API_KEY` fallbacks from the production
  provider boundary.
- Kept escalation credentials independent and rejected blank escalation keys.
- Required HTTPS except exact `localhost`, `127.0.0.1`, and `::1` HTTP
  development endpoints; rejected userinfo, query, fragment, invalid schemes,
  missing hosts, and lookalike loopback hosts.
- Changed dotenv loading to `override=False` and separated non-secret model-name
  lookup from request configuration so infrastructure-failure reports do not
  require credentials while actual primary requests still fail before client
  construction if `LLM_API_KEY` is absent.
- Updated `.env.example`, README, and documentation contracts.

### LLM HTTP response bounds

- Replaced unbounded `aread()`/`aiter_lines()` consumption with a single bounded
  `aiter_bytes()` pipeline for JSON, SSE, and HTTP error bodies.
- Enforced inclusive limits of 8 MiB response bytes, 50,000 SSE events, eight
  choices, 4 MiB individual and aggregate content, 64 tool calls, and 2 MiB per
  tool argument string.
- Counted response bytes and events after `[DONE]` until stream completion, so a
  provider cannot hide an oversized tail.
- Preserved fragmented UTF-8 SSE parsing and returned only sanitized, fixed
  `LLMResponseError` messages for bounded-parser failures.

### Commit boundary

- Generated per-run branches as a safe deterministic issue/trace prefix plus a
  cryptographic 16-hex suffix.
- After create-ref (including 422), resolved the exact full branch ref and
  required its head to equal the approved base before any Contents API write.
- Fetched exact base/head Git commits and complete recursive trees, reconstructed
  the expected head tree from the approved live binding, and rejected truncated,
  malformed, extra, missing, wrong-mode, or wrong-blob trees.
- Fetched every approved remote blob and verified Git object identity, bytes,
  size, and SHA-256 content digest.
- Revalidated the base ref, terminal coverage binding, remote tree, and blobs
  immediately before PR creation.

### GitHub cache

- Partitioned cache keys into `github-anonymous` or
  `github-auth-<sha256-token-fingerprint>` namespaces.
- Switched call-argument digests to SHA-256.
- Token rotation now misses the previous private cache partition, while the same
  token still hits; raw tokens never enter keys, cache files, or diagnostics.

## TDD evidence

Observed RED before each implementation wave:

- Run ID/path wave: four representative unsafe IDs failed with `DID NOT RAISE`;
  the existing FIFO read blocked and the test was terminated before adding
  non-blocking/fstat handling.
- Provider wave: eight representative tests failed, including ignored
  `LLM_BASE_URL`, missing deprecation warning, no conflict rejection, accepted
  cross-vendor keys, and unsafe URLs.
- HTTP wave: the inclusive JSON limit test failed because production called the
  test double's forbidden unbounded `aread()`.
- Commit wave: two tests failed because the secure branch generator was absent
  and a malicious colliding branch reached the write function.
- Cache wave: three tests failed because token rotation and anonymous/authenticated
  calls shared the same key, and the key lacked an authenticated partition.
- Edge wave: three tests failed for post-`[DONE]` bytes, a non-directory runs
  path, and a whitespace-only escalation key.
- First full-suite run: `1 failed, 1273 passed, 2 skipped, 1 warning`; the failure
  showed infrastructure-report metadata calling credential-requiring
  `get_model_config`. The focused reproduction passed after introducing
  non-secret `get_model_name` lookup.

Final GREEN commands and exact results:

```text
.venv/bin/python -m pytest -q tests/test_run_store.py tests/test_model_provider.py tests/test_http_client.py tests/test_commit.py tests/test_cache.py tests/test_tools.py tests/test_documentation_contract.py tests/test_agent_v2_eval.py tests/test_llm.py tests/test_model_escalation_integration.py tests/test_new_agent.py tests/test_main.py tests/test_oci_runner.py
375 passed in 5.66s

.venv/bin/python -m pytest -q
1274 passed, 2 skipped, 1 warning in 42.29s

.venv/bin/python -m ruff check src/ tests/ eval/ --select=E,F,I --ignore=E501
All checks passed!

git diff --check
(no output; exit 0)
```

The one full-suite warning is the pre-existing macOS `sqlite-vec` unavailable
fallback in `tests/test_error_episodes.py`; the store intentionally falls back
to NumPy cosine search.

## Self-review

- Confirmed all remote-commit decisions still originate from
  `validate_terminal_coverage_binding`; no PatchGate or coverage checks were
  weakened.
- Confirmed a 422 branch collision cannot reach a file write unless the exact
  named ref currently equals the approved base.
- Confirmed remote verification compares the complete leaf tree against the
  approved base plus manifest and validates target bytes before PR creation.
- Confirmed response limits are inclusive and apply to success and error bodies,
  fragmented streams, post-`[DONE]` tails, JSON/SSE content, choices, tool calls,
  and tool arguments.
- Confirmed credential fingerprints are one-way SHA-256 values and no raw token
  is logged or persisted by the cache key path.
- Confirmed `run_trace.py` remains the only unrelated worktree modification.

## Caveats

- GitHub's recursive tree endpoint is required to return `truncated: false`;
  very large repositories that GitHub truncates fail closed before PR creation.
- Approved deletions remain unsupported by the existing commit boundary and are
  still rejected before GitHub mutation.

## Runtime-Boundary Re-review Follow-up (2026-07-22)

This follow-up remediates the complete Important/Minor re-review set against
baseline `b1da2e2bc150526a813e82465c256763de1efd59`. The pre-existing user edit in
`run_trace.py` was neither modified nor staged.

### Directory-FD anchored run persistence

- Anchored both the configured run-store root and its `runs` child with
  `O_DIRECTORY | O_NOFOLLOW` file descriptors, comparing `lstat`, `fstat`, and
  stable post-open identities before use.
- Created `runs` relative to the anchored root descriptor and fsynced that
  directory entry. All resolve/stat/open/fstat failure paths close acquired
  descriptors.
- Distinguished an absent store (`[]`) from a dangling or non-directory
  `runs` symlink (fail closed), while retaining no-follow regular-file reads and
  atomic descriptor-relative writes.

TDD evidence: six new adversarial cases were RED before implementation (root
symlink, stat/open swap, dangling `runs`, and injected cleanup failures). The
focused run-store suite then passed with `46 passed`.

### One provider path for production and eval

- `eval/harness.py` now imports the shared primary provider resolver and
  delegates requests to the bounded production HTTP client. Its independent
  `httpx` client and Lino/DeepSeek credential fallbacks were removed.
- Both runtime entrypoints load dotenv with `override=False`; the authoritative
  workflow exports `LLM_BASE_URL`, not `OPENAI_BASE_URL`.

TDD evidence: documentation contracts first failed on both override/workflow
requirements; the provider/eval focused selections subsequently passed with
`4 passed` and `2 passed`.

### Canonical JSON/SSE validation

- Every choice, message, delta, tool-call fragment, index, finish reason, and
  retained usage counter is validated; a mixed valid/malformed response is
  rejected as a whole.
- Choice/tool indexes require exact non-negative integers (booleans and numeric
  strings are rejected), duplicate event indexes are rejected, and repeated
  role/ID/type/name/finish metadata cannot conflict.
- JSON and SSE return only canonical validated fields. Aggregate tool argument
  bytes are bounded across all choices/calls in addition to the existing
  per-call, content, event, choice, and response bounds.

TDD evidence: all 17 new strict-shape cases were RED before implementation;
the complete HTTP suite passed with `92 passed`.

### Contents paths, complete Git trees, and exact ancestry

- GitHub repository owner/name segments and Contents API paths are percent
  encoded without collapsing path separators; the unused unsafe file-SHA
  helper was removed.
- Recursive tree payloads now validate canonical paths, exact mode/type pairs,
  non-boolean non-negative blob sizes, parent-directory presence, and no empty
  injected tree objects. Every nested tree SHA and the root SHA are recomputed
  from Git object bytes before the complete expected DAG is compared.
- Each Contents API write must return one commit whose sole parent is the exact
  previously verified commit. The chain begins at the approved base, contains
  exactly one commit per approved target, and is re-fetched and re-proven before
  PR creation; merge commits, unrelated parents, and older common ancestors
  fail closed.

TDD evidence: the nested-tree/path/mode/size set produced eight RED failures;
the ancestry cases then proved exact-parent rejection.

### Ref-stable PR creation

- The verified repair head and exact write chain are passed from push to PR
  creation instead of being re-derived. Base/head refs are checked at both ends
  of remote verification.
- PR POST responses and a subsequent GET are bound to exact number, repository,
  base/head SHA and ref, and canonical API/HTML URLs. Refs are read again after
  confirmation. A known newly-created PR is closed if any identity check or
  ref-stability check fails.

TDD evidence: four PR binding/race tests were RED before implementation, and a
separate wrong-resource URL test was RED until URL identity became exact. The
complete commit suite passed with `27 passed`.

### Canonical cache-call identity

- Cache keys now include the unwrapped function module and qualname plus
  signature-bound, default-expanded arguments.
- Argument encoding carries explicit types for scalar and container values;
  positional/keyword equivalents share a key, while `1`/`"1"`, list/tuple, and
  same-named functions in different modules do not. Unsupported objects fail
  instead of falling through `default=str`.

TDD evidence: three canonical-identity cases were RED before implementation;
the cache/tools covering run passed with `23 passed`.

### Follow-up verification

An independent read-only review then found six remaining edge cases. The same
TDD wave closed all six before commit:

- SSE finalization now rejects any incomplete choice instead of filtering it
  from an otherwise usable mixed response.
- A failed PR validation followed by a failed close raises `PRCleanupError`,
  preserving the PR number plus both the validation and cleanup exceptions.
- The canonical repository owner/name is derived from the verified Repository
  API response during push, revalidated before PR creation, and used for POST
  and GET response identity checks; input URL casing is not treated as
  canonical GitHub casing.
- Missing nested run-store roots are created through a no-follow descriptor
  walk from the filesystem anchor. Each `mkdirat` is followed by an fsync of
  its anchored parent directory, and symlinked intermediate components fail
  closed.
- `list_runs` and `inspect_run` derive timestamps from the safely opened run
  file's `fstat`, never a later path-based stat.
- Remote blob sizes require an exact non-boolean, non-negative integer.

The new regression cases were observed RED (`1` SSE, `2` run-store, and `3`
commit-boundary failures), then the expanded focused suites passed with
`140 passed` and `80 passed`.

Fresh completion evidence after all follow-up changes:

```text
.venv/bin/python -m pytest -q
1324 passed, 2 skipped, 1 warning in 41.09s

.venv/bin/python -m ruff check src/ tests/ eval/ --select=E,F,I --ignore=E501
All checks passed!

git diff --check
(no output; exit 0)
```

The warning remains the expected macOS sqlite-vec-unavailable fallback to NumPy
in `tests/test_error_episodes.py`. No push or live evaluation was performed.

### Cancellation-safe PR transaction reconciliation

A final review found that a cancellation or indeterminate network result during
PR creation could leave a remote PR orphaned: the previous cleanup path only
knew how to close a PR after a valid numeric identifier had been returned.

- Before POST, the client now snapshots the exact set of already-open PRs whose
  repository, head/base refs, head/base SHAs, and canonical resource identities
  match the approved transaction.
- The POST, confirmation GET, and final ref checks run as one shielded task. On
  cancellation the task is drained before cleanup, so a remotely successful
  POST cannot outlive the local cleanup decision.
- When POST has an unknown outcome or returns a malformed response without a
  number, cleanup performs bounded eventual-consistency reconciliation and
  closes only newly observed exact matches. Pre-existing or merely similar PRs
  are never closed.
- Cleanup failures remain explicit through `PRCleanupError`. Cancellation keeps
  `CancelledError` semantics; if cleanup also fails,
  `PRCancellationCleanupError` carries the PR number when known and the cleanup
  exception.

Seven cancellation and reconciliation cases were first RED, covering
cancellation during POST, confirmation, and final-ref checks; failed close
during cancellation; missing-number success; unknown POST outcome; and failed
reconciliation. An eighth boundary test proves retries never close a
pre-existing or unrelated PR.

Fresh final evidence after this wave:

```text
.venv/bin/python -m pytest -q tests/test_commit.py
39 passed in 3.92s

.venv/bin/python -m pytest -q tests/test_run_store.py tests/test_http_client.py tests/test_commit.py
181 passed in 5.48s

.venv/bin/python -m pytest -q
1332 passed, 2 skipped, 1 warning in 42.19s

.venv/bin/ruff check src/cache.py src/http_client.py src/main.py src/nodes/commit.py src/run_store.py eval/harness.py tests/test_agent_v2_eval.py tests/test_cache.py tests/test_commit.py tests/test_documentation_contract.py tests/test_http_client.py tests/test_run_store.py tests/test_tools.py
All checks passed!

git diff --check
(no output; exit 0)
```

The warning remains the expected macOS sqlite-vec-unavailable fallback to NumPy
in `tests/test_error_episodes.py`. The user-owned `run_trace.py` remained
untouched and unstaged; no push or live evaluation was performed.

### Final timeout-composition and deterministic-rejection review

The next fresh review identified one diagnostic regression and one production
composition gap, plus an inaccurate deterministic-rejection classification:

- The original `TimeoutError` is now used only for bounded cause extraction.
  Persisted node diagnostics again receive an empty standard `TimeoutError`, so
  a node that directly raises a timeout containing credentials cannot leak that
  message. Two direct-timeout tests first reproduced the leak in both graph
  runners and now prove the persisted value remains exactly `TimeoutError`.
- The fallback graph no longer registers `_wrap_node` wrappers beneath
  `FallbackCompiledGraph`. The fallback runner is the single owner of timeout,
  progress, routing, and diagnostic behavior; the LangGraph path keeps its
  wrapper. A real `build_agent_graph(StateGraph=None)` test first reproduced the
  double-timeout race and missing cleanup cause, then proved redacted PR cleanup
  evidence survives with exactly one `START`/`TIMEOUT` progress pair. A forced
  fallback saved-state routing regression also remains green.
- A GitHub POST that returns a definitive 4xx rejection is recorded separately
  from an indeterminate transport/server outcome. It re-raises the original
  HTTP error without claiming unresolved cleanup or performing post-failure
  reconciliation; 5xx/transport-unknown paths retain the conservative exact
  reconciliation behavior.

Fresh completion evidence after these fixes:

```text
.venv/bin/python -m pytest -q tests/test_commit.py tests/test_new_agent.py
104 passed in 5.80s

.venv/bin/python -m pytest -q tests/test_model_escalation_integration.py -k graph_replays_an_already_escalated_saved_state
1 passed, 32 deselected in 0.23s

.venv/bin/python -m pytest -q
1340 passed, 2 skipped, 1 warning in 45.47s

.venv/bin/ruff check src/graph.py src/new_agent.py src/nodes/commit.py src/timeout_diagnostics.py tests/test_new_agent.py tests/test_commit.py
All checks passed!
```

The warning remains the expected macOS sqlite-vec-unavailable fallback to NumPy
in `tests/test_error_episodes.py`. The user-owned `run_trace.py` remained
untouched and unstaged; no push or live evaluation was performed.

### Final PR-cleanup and timeout-evidence hardening

Two independent final reviews identified three more transaction edge cases,
all closed in a final TDD wave:

- An unknown POST outcome that remains absent through bounded reconciliation is
  now an explicit cleanup failure. The caller therefore retains both the
  original network/cancellation condition and the unresolved cleanup state,
  rather than implying that no orphan can exist.
- A number returned by POST is trusted for direct cleanup only after the POST
  payload passes the complete PR identity check. An unvalidated number always
  goes through exact pre/post reconciliation, so it cannot close an unrelated
  existing PR.
- Additional cancellation during an already-running cleanup is delayed until
  cleanup completes. A successful cleanup is followed by the original
  cancellation; only an actual cleanup-task exception becomes
  `PRCancellationCleanupError`.

Three focused regressions were observed RED before this implementation, then
passed together: unknown outcome with no observed exact PR, an unvalidated
response number naming an unrelated PR, and a second cancellation during
successful cleanup.

The reviews also found that production phase timeouts could hide cleanup
evidence: `asyncio.wait_for` converts the cancellation subclass into a
`TimeoutError`. A new independent `timeout_diagnostics` helper walks a bounded
cause chain without importing the commit node, recognizes the structured
cleanup protocol, and emits only a bounded, credential-redacted summary. Both
the fallback graph runner and LangGraph node wrapper retain the timeout status
while adding the cleanup error type, summary, and PR number to the failure
reason and node diagnostic. Ordinary timeout output remains unchanged.

Two real `asyncio.wait_for` integration tests (fallback graph and wrapped node)
were observed RED before implementation. They now prove cleanup evidence is
visible while credential-shaped sentinels are replaced with `[REDACTED]`.

Fresh final evidence after the review fixes:

```text
.venv/bin/python -m pytest -q tests/test_commit.py tests/test_new_agent.py
100 passed in 5.50s

.venv/bin/python -m pytest -q
1336 passed, 2 skipped, 1 warning in 43.51s

.venv/bin/ruff check src/nodes/commit.py src/graph.py src/new_agent.py src/timeout_diagnostics.py tests/test_commit.py tests/test_new_agent.py
All checks passed!
```

The warning remains the expected macOS sqlite-vec-unavailable fallback to NumPy
in `tests/test_error_episodes.py`. The user-owned `run_trace.py` remained
untouched and unstaged; no push or live evaluation was performed.
