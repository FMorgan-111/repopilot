# Task 7: Python 3.10 trusted-pytest isolation report

## Outcome

Implemented version-stable trusted pytest startup with Python isolated mode (`-I`) at every affected product boundary. The existing `PYTEST_BOOTSTRAP` remains unchanged: it imports installed pytest before adding `os.getcwd()` to `sys.path`, then runs pytest with the original suffix arguments.

Base commit: `dab92e056658f28287f969d9ef8ff0e98bb6b55b`

Pre-existing user change: `run_trace.py` remained unstaged and byte-for-byte unchanged by this task. Its initial and final Git blob hash is `00606b3710fa0ca8f9f798ea3d01e818acc42eea`.

The approved cancellation commits `8a95a3c`, `81a347a`, and `dab92e0` were not modified.

## `-P` audit and classification

The audit was completed before production edits.

| Occurrence | Classification | Action and reason |
| --- | --- | --- |
| `src/tool_policy.py` in `fixed_pytest_argv` | Product trusted-pytest launcher | Changed to `-I`; Python 3.10 must be able to execute the shared bootstrap. |
| `src/safe_subprocess.py` in the OCI capability probe | Product trusted pytest capability boundary | Changed to `python -I -m pytest --version`; the probe does not need checkout imports and must not fail on Python 3.10 before execution. |
| `eval/oci_runner.py` in `_preflight_image` | Product image-dependent testbed preflight | Changed to `-I`; image Python versions may be 3.10. |
| `eval/harness.py` in `run_tests` | Product legacy host test launcher | Changed to `python -I -c PYTEST_BOOTSTRAP ...`; this imports installed pytest first and restores checkout imports only afterward. |
| Exact contracts in tool policy, execute fallback, coverage, router, OCI probe, evaluator preflight, and legacy harness tests | Tests of affected product launchers | Updated to require `-I` and, where practical, strengthened from prefix/string checks to exact argv checks. |
| Remaining `-P` values in `tests/test_safe_subprocess.py` | Direct input fixtures supplied to the generic OCI subprocess boundary | Retained. They verify immutable command passthrough, cleanup, timeout, overflow, cancellation, and arbitrary `BaseException` behavior. RepoPilot does not construct these fixture commands, so rewriting them would conflate the capability-probe fix with caller-input behavior. |
| Two remaining `-P` values in `tests/test_oci_integration.py` | Direct scripts in the pinned Python 3.12 integration image | Retained. One runs a sibling boundary-check script and one exercises timeout cleanup; neither is a product pytest launcher, and changing their path semantics was explicitly out of scope. |

Post-change audit: there are no `-P` occurrences under `src/` or `eval/`. All residual occurrences are the intentionally retained test inputs above.

## Root cause and hypothesis

`-P` was added in Python 3.11. The existing launcher therefore exited in Python argument parsing on Python 3.10, before the trusted bootstrap could import pytest. The real regression reproduced exit code 2 and `Unknown option: -P`.

Hypothesis: replacing only the affected launcher flags with `-I` would retain the security property and restore Python 3.10 support. `-I` excludes the current working directory, user site-packages, and `PYTHON*` environment influence during the initial import. The unchanged bootstrap then imports installed pytest and only afterward inserts the checkout working directory at `sys.path[0]`, preserving target package imports during collection.

## TDD evidence

### RED 1: real Python 3.10 incompatibility on the original production code

Command:

```text
/tmp/repopilot-compat-310/bin/python -m pytest tests/test_tool_policy.py::test_fixed_pytest_argv_imports_trusted_pytest_before_workspace -q
```

Result: `1 failed`; the child interpreter returned 2 with:

```text
Unknown option: -P
usage: /tmp/repopilot-compat-310/bin/python [option] ...
```

This regression uses a real hostile checkout-level `pytest.py` and a real test importing `src.widget` from the checkout.

### RED 2: exact argv contracts before production edits

After changing tests only, command:

```text
.venv/bin/python -m pytest tests/test_tool_policy.py tests/test_execute_security.py tests/test_coverage_gate.py tests/test_tool_router.py tests/test_safe_subprocess.py tests/test_oci_runner.py tests/test_eval_harness.py -q --basetemp=/tmp/repopilot-task7-red-311
```

Result: `11 failed, 271 passed in 20.40s`. Every failure was the expected `-P` versus `-I` contract mismatch across:

- tool policy (2)
- execute fallback (2)
- coverage (1)
- router (1)
- OCI capability probe (3 parametrized outcomes)
- evaluator preflight (1)
- legacy harness (1)

No production file had been edited at this point.

### GREEN: minimal production changes

The same seven focused files passed on all supported versions, with separate basetemp paths:

```text
/tmp/repopilot-compat-310/bin/python -m pytest tests/test_tool_policy.py tests/test_execute_security.py tests/test_coverage_gate.py tests/test_tool_router.py tests/test_safe_subprocess.py tests/test_oci_runner.py tests/test_eval_harness.py -q --basetemp=/tmp/repopilot-task7-green-310
282 passed in 24.24s

.venv/bin/python -m pytest tests/test_tool_policy.py tests/test_execute_security.py tests/test_coverage_gate.py tests/test_tool_router.py tests/test_safe_subprocess.py tests/test_oci_runner.py tests/test_eval_harness.py -q --basetemp=/tmp/repopilot-task7-green-311
282 passed in 23.93s

/tmp/repopilot-compat-312/bin/python -m pytest tests/test_tool_policy.py tests/test_execute_security.py tests/test_coverage_gate.py tests/test_tool_router.py tests/test_safe_subprocess.py tests/test_oci_runner.py tests/test_eval_harness.py -q --basetemp=/tmp/repopilot-task7-green-312
282 passed in 24.20s
```

The hostile `pytest.py` regression is in `tests/test_tool_policy.py`, so it ran and passed in each of these Python 3.10, 3.11, and 3.12 suites.

## Implementation and security self-review

- `fixed_pytest_argv(python_executable, args)` now returns exactly `[python_executable, "-I", "-c", PYTEST_BOOTSTRAP, *args]`.
- `PYTEST_BOOTSTRAP` itself is unchanged. Its statement ordering remains `import os`, `import sys`, `import pytest`, `sys.path.insert(0, os.getcwd())`, then `pytest.main(sys.argv[1:])`.
- `-I` prevents a hostile checkout `pytest.py` from participating in the initial pytest import. Only after pytest is resident in `sys.modules` is the checkout inserted for collection imports.
- Selector and pytest option suffixes remain distinct argv tokens and retain order. No shell parsing was introduced.
- The OCI capability probe is exactly `python -I -m pytest --version`. It intentionally does not expose the checkout because it is only a capability check.
- The evaluator preflight remains an import-only check and changes only its isolation flag.
- The legacy harness reuses the shared bootstrap constant rather than using `-I -m pytest`, preserving checkout module imports while still importing trusted pytest first.
- The legacy harness retains its credential-minimal environment, checkout `cwd`, timeout, captured output, and pytest arguments (`-x -q --tb=short`).
- Changes are limited to the four named product boundaries, their exact contract tests, and this report. No approved cancellation implementation was changed.

## Static and repository verification

Ruff command:

```text
.venv/bin/python -m ruff check --select E,F,I src/tool_policy.py src/safe_subprocess.py eval/oci_runner.py eval/harness.py tests/test_tool_policy.py tests/test_execute_security.py tests/test_coverage_gate.py tests/test_tool_router.py tests/test_safe_subprocess.py tests/test_oci_runner.py tests/test_eval_harness.py
```

Result: the repository's existing line-length debt is reported as exactly `100 E501 line-too-long` findings. Inspection of the zero-context task diff confirms none is on an added or modified task line. The same command with `--ignore E501` reports `All checks passed!`, so there are no other E/F/I findings in changed files. These unrelated historical line-length violations were not reformatted.

`git diff --check` completed with no output.

## Review and commit

Independent read-only review reported no Critical, Important, or Minor findings and returned `READY`. The reviewer specifically confirmed the exact `-I` bootstrap argv, import-before-cwd ordering, OCI probe/preflight behavior, legacy harness reuse, and residual `-P` classifications. No files were edited by the reviewer.

Commit message: `fix: support pytest isolation on Python 3.10`
