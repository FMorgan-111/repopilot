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
