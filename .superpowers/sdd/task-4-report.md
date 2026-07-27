# Task 4 implementation report

## Scope

Implemented the fail-closed HTTP API boundary for all ambient-authority and
saved-run routes. The only public route is `GET /health`. Added static bearer
authentication, exact repository authorization, bounded body/model inputs, a
non-queueing concurrency gate, authorized saved-run snapshots, safe error
responses, API documentation, and focused regression coverage.

The user's existing `run_trace.py` change was neither edited nor staged.

## TDD evidence

### Baseline

```text
.venv/bin/python -m pytest tests/test_main.py tests/test_run_store.py tests/test_new_agent.py -q
127 passed in 1.22s
```

### RED: authentication and repository boundary

```text
.venv/bin/python -m pytest tests/test_api_security.py::test_missing_or_blank_api_token_fails_closed_before_handler -x -vv
FAILED ... AssertionError: authority helper must not run
1 failed in 0.48s
```

The pre-boundary endpoint invoked the monkeypatched authority helper when API
configuration was missing. After the minimal middleware and repository parser:

```text
.venv/bin/python -m pytest tests/test_api_security.py -q
31 passed in 0.30s
```

### RED: body and Pydantic limits

```text
.venv/bin/python -m pytest \
  tests/test_api_security.py::test_declared_request_body_limit_rejects_one_over \
  tests/test_api_security.py::test_request_model_string_limits_and_extra_fields \
  tests/test_api_security.py::test_validation_errors_are_stable_and_do_not_echo_input -q
3 failed in 7.47s
```

The old API accepted a 65,537-byte body, accepted overlong/extra fields, and
returned unsanitized default validation output. After bounded ASGI pre-read and
strict models:

```text
.venv/bin/python -m pytest tests/test_api_security.py -q
47 passed in 0.30s
```

### RED: saved-run authorization and inspect/replay

```text
.venv/bin/python -m pytest \
  tests/test_api_security.py::test_resume_checks_allowlist_configuration_before_loading_run \
  tests/test_api_security.py::test_resume_authorizes_stored_run_before_invocation \
  tests/test_api_security.py::test_inspect_returns_safe_summary_only_after_stored_repo_authorization \
  tests/test_api_security.py::test_replay_format_is_limited_to_json_or_markdown -q
4 failed in 0.34s
```

The failures showed that the API had no authorized `load_run` boundary, no
inspect route, and no replay-format constraint. After using one safely loaded,
repository-authorized snapshot for inspect/replay:

```text
.venv/bin/python -m pytest tests/test_api_security.py -q
57 passed in 0.31s
```

### RED: non-queueing concurrency

```text
.venv/bin/python -m pytest tests/test_api_security.py::test_protected_request_concurrency_rejects_third_without_queueing -q
FAILED ... AssertionError: the third request must be rejected, not queued
1 failed in 0.29s
```

After the loop-independent atomic two-slot gate, including exception and
cancellation release tests:

```text
.venv/bin/python -m pytest tests/test_api_security.py -q
60 passed in 0.29s
```

### RED: authorized resume snapshot, safe errors, strict names/types

```text
.venv/bin/python -m pytest tests/test_new_agent.py::test_resume_agent_v2_uses_preloaded_authorized_state -q
FAILED ... TypeError: resume_agent_v2() got an unexpected keyword argument 'state'
1 failed in 0.28s

.venv/bin/python -m pytest \
  tests/test_api_security.py::test_resume_authorizes_stored_run_before_invocation \
  tests/test_api_security.py::test_agent_error_responses_do_not_echo_ambient_secrets \
  tests/test_api_security.py::test_missing_or_malformed_allowlist_fails_closed_before_agent -q
7 failed, 10 passed in 0.57s

.venv/bin/python -m pytest tests/test_api_security.py::test_request_model_numeric_limits -q
5 failed in 0.41s
```

The failures captured the second run-file read, raw agent error leakage,
GitHub-owner grammar gaps, and Pydantic numeric coercion. The implementation
now passes the already-authorized `AgentState` into `resume_agent_v2`, returns
stable generic errors, validates GitHub owner/repository names, and uses strict
bounded integer fields.

### RED: pathological Content-Length

```text
.venv/bin/python -m pytest tests/test_api_security.py::test_extremely_large_declared_content_length_is_rejected_without_parsing -q
FAILED ... ValueError: Exceeds the limit (4300 digits) for integer string conversion
1 failed in 0.42s
```

The final parser compares a normalized decimal length lexically, avoiding an
unbounded integer conversion. Incoming chunks are length-checked before being
copied into the bounded buffer.

### Independent-review fix wave

The read-only pre-commit reviewer identified three boundary bypasses and one
strict-URL differential: the legacy intelligent endpoint passed `max_turns=10`
as ten retries, saved states could resume with out-of-range execution limits,
non-ASCII bearer bytes raised `TypeError`, and `urlsplit()` stripped raw control
characters before authorization while the original URL reached the agent.

```text
.venv/bin/python -m pytest \
  tests/test_api_security.py::test_intelligent_agent_never_exceeds_agent_v2_retry_cap \
  tests/test_api_security.py::test_non_ascii_authorization_header_gets_uniform_401 \
  tests/test_api_security.py::test_resume_rejects_saved_state_outside_hard_execution_limits -q
6 failed in 1.22s

# after bounded retry mapping, saved-state limit revalidation, and byte comparison
6 passed in 0.29s
```

The controller additionally required saved filename/state identity binding and
containment of authority-helper exceptions so provider exception messages do
not reach server error logging. Those checks and the control-character URL
regression failed before the fix:

```text
.venv/bin/python -m pytest \
  tests/test_api_security.py::test_resume_rejects_run_id_that_does_not_match_loaded_state \
  tests/test_api_security.py::test_concurrency_slot_is_released_after_handler_error \
  'tests/test_api_security.py::test_strict_issue_url_rejection_precedes_agent[https://github.com/ac\nme/widget/issues/42]' -q
3 failed in 0.78s

# after run-id binding, control rejection, and generic exception containment
23 passed in 0.24s
```

## Final verification

```text
.venv/bin/python -m pytest tests/test_api_security.py tests/test_main.py tests/test_new_agent.py tests/test_run_store.py -q
220 passed in 1.94s

.venv/bin/python -m pytest -q
1433 passed, 2 skipped, 1 warning in 44.13s

.venv/bin/python -m ruff check src/ tests/ eval/ --select=E,F,I --ignore=E501
All checks passed!

git diff --check
(no output; exit 0)
```

The full-suite warning is the pre-existing `sqlite-vec` unavailable fallback
in `tests/test_error_episodes.py`; the test intentionally selects the NumPy
backend and still passes.

## Self-review

- `GET /health` bypasses token/config/body/concurrency checks; every other
  current route crosses the bearer middleware and public schema routes are
  disabled.
- Authentication runs before body receive. Declared and actual bodies are
  capped at exactly 65,536 bytes, including chunked requests.
- Repository allowlist validation and strict issue-URL parsing finish before a
  process-local slot or any GitHub/model/clone/write helper is used.
- The concurrency gate uses a synchronous lock for atomic non-waiting claims,
  has exactly two slots, and releases them in `finally` on return, exception,
  or cancellation.
- Resume/inspect/replay authorize a safely loaded state and never return or act
  on an unauthorized repository. Resume consumes the same authorized snapshot
  rather than reopening the run file; its `trace_id`, retry limit, and token
  budget must match the requested run and current API bounds.
- Validation and agent/storage errors return fixed bodies without request
  input, configured secrets, allowlist contents, stored unauthorized repository
  names, or provider exception strings. Agent/helper exceptions are contained
  before the ASGI server can log their messages; cancellation still propagates.
- `.env.example` and README contain names/placeholders only and document exact
  allowlist syntax, bearer use, limits, concurrency, public health, and
  fail-closed behavior.

## Caveats

- Tests exercise the real ASGI middleware with `httpx.ASGITransport`; no live
  network listener was started.
- The known sqlite-vector fallback warning remains outside Task 4 scope.

## Independent review

Fresh read-only re-review after the fix wave returned **Ready**:

- Critical: none
- Important: none
- Minor: the known pre-existing sqlite-vec fallback warning only
- Spec verdict: pass
- Code-quality verdict: pass with the recorded warning debt

The reviewer independently reported 25 targeted security/concurrency tests,
220 focused tests, 1,433 full-suite passes with two skips and the one warning,
Ruff success, and a clean diff check. The reviewer made no filesystem or Git
changes.
