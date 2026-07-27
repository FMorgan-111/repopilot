# Final Review Fix Report

Date: 2026-07-20

Implementation commit: `014d3c3` (`fix: address final review findings`)

## Summary

All seven verified final-review findings were addressed in the isolated Gemini
worktree. The changes preserve the configured Gemini/LinoAPI defaults, the
production API-key precedence, opt-in episode memory, and the sqlite-vec to
NumPy warning fallback. No live evaluation was run, no push was performed, and
no credential values were recorded in this report.

## Finding 1: Optional memory imports NumPy while disabled

Implementation:

- Made both sqlite-vec and NumPy backend imports lazy in
  `src/memory/vector_backend.py`.
- Kept sqlite-vec as the first choice and retained the existing one-time warning
  before falling back to NumPy.
- Added a subprocess packaging smoke test that unsets
  `REPOPILOT_ENABLE_EPISODES`, blocks optional backend imports, imports the
  episode-store path, calls `get_episode_store()`, and verifies NumPy was not
  loaded.

TDD evidence:

- RED: `.venv/bin/python -m pytest tests/test_packaging_contract.py::test_base_install_can_use_disabled_episode_memory_without_numpy -q`
  failed (`1 failed`) because importing `error_episode_store` reached the
  eager NumPy backend import.
- GREEN: the same command passed (`1 passed in 0.07s`).
- Covering memory/package run passed: `16 passed, 1 skipped, 1 warning`.

## Finding 2: Legacy eval key resolution differs from production

Implementation:

- Removed the import-time eval key snapshot.
- Added `_get_llm_api_key()` in `eval/harness.py` with exact request-time
  precedence: `LINOAPI_API_KEY`, `LLM_API_KEY`, `DEEPSEEK_API_KEY`, then empty.
- Made `llm_request()` resolve the key for every request.
- Added tests for all three individual variables, multi-key precedence, empty
  fallback, and request-time header construction. Tests use synthetic values
  only and do not print authorization headers.

TDD evidence:

- RED: `.venv/bin/python -m pytest tests/test_agent_v2_eval.py -q -k 'legacy_harness and (api_key or request_time)'`
  failed all six selected tests: five because the resolver did not exist and
  one because the stale import-time key was used.
- GREEN: the same command passed (`6 passed, 16 deselected in 0.11s`).

## Finding 3: Per-mode eval reporting is incomplete

Implementation:

- Reused `eval.failure_taxonomy.summarize()` for each evaluation mode.
- Each `by_evaluation_mode` entry now includes samples, successes, the existing
  success-rate field, a resolved-rate alias, decisive taxonomy, sorted non-empty
  model names, and sorted non-empty commit SHAs.
- Added a dedicated Markdown mode table that keeps mode, samples, models,
  commits, resolved rate, and decisive taxonomy on the same row.
- Renamed the mixed aggregate rate to
  `agent_v2_combined_all_modes_resolved_rate` so it cannot be read as
  end-to-end performance.
- Preserved missing/unknown-mode compatibility as `end_to_end`.
- Preserved both package imports and direct `python eval/report.py` execution.

TDD evidence:

- RED: `.venv/bin/python -m pytest tests/test_eval_report.py -q` produced
  `2 failed, 2 passed`; the expected metadata/taxonomy fields and clearly
  labeled presentation were absent.
- GREEN: the focused report suite passed (`4 passed in 0.02s`).
- Self-review found a direct-script import regression. A new subprocess test
  first failed (`1 failed`) from the eval directory; after using the appropriate
  relative/top-level import for each execution mode, the suite passed
  (`5 passed in 0.05s`) and `python eval/report.py` exited successfully with the
  expected missing-results message.
- Mixed-mode coverage includes `test_failed` and `wrong_file_path` decisive
  categories plus resolved samples, two models, and two commits per mode.

## Finding 4: Contributor guide has stale model default

Implementation:

- Updated `CLAUDE.md` to name `gemini-3.5-flash:stable`.
- Extended the documentation contract to require that default and reject the
  stale contributor-guide default.

TDD evidence:

- RED (combined documentation/trace selection):
  `.venv/bin/python -m pytest tests/test_documentation_contract.py -q -k 'contributor_guide or trace_utility'`
  produced `2 failed, 5 deselected`.
- GREEN: the same selection passed (`2 passed, 5 deselected in 0.01s`).

## Finding 5: Malformed provider indexes leak raw exceptions

Implementation:

- Added typed provider-index conversion so malformed choice and tool-call
  indexes raise `LLMResponseError` rather than raw conversion errors.
- Added shared tool-call structure validation for streamed and ordinary JSON
  completions.
- Streamed tool-call state no longer invents a function type; incomplete IDs,
  types, function names, or argument fields are rejected.
- Added focused tests for malformed choice/tool indexes and incomplete streamed
  and JSON tool calls, while retaining valid tool-call-only coverage.

TDD evidence:

- RED: `.venv/bin/python -m pytest tests/test_http_client.py -q -k 'malformed_provider_indexes or incomplete_streamed_tool_call or incomplete_json_tool_call'`
  produced `4 failed, 34 deselected`; raw `ValueError`/`TypeError` escaped and
  incomplete tool calls were accepted.
- GREEN (including the valid-tool-call regression): the focused selection
  passed (`5 passed, 33 deselected in 0.09s`).

## Finding 6: NumPy persistence test does not cross a restart boundary

Implementation/test hardening:

- Replaced the in-memory/same-connection test with a temporary SQLite file.
- The first connection now writes, commits, and closes; a second connection and
  index instance reopen the file and prove nearest-neighbor search survives.

Evidence:

- The focused strengthened test passed:
  `.venv/bin/python -m pytest tests/test_error_episodes.py::test_numpy_vector_index_cosine_knn_and_persistence -q`
  -> `1 passed in 0.23s`.
- This finding corrected the regression test boundary rather than production
  behavior, so there was no legitimate production RED state to create; the
  pre-change defect was structural (one connection to `:memory:` cannot prove
  persistence across process/store restart).

## Finding 7: Trace utility forces DeepSeek

Implementation:

- Removed the `LLM_MODEL` override from `run_trace.py`; it now uses the shared
  runtime resolver/default.
- Added a documentation contract that rejects the stale trace model and any
  utility-local `LLM_MODEL` default override.

TDD evidence:

- RED/GREEN evidence is the combined documentation selection recorded under
  Finding 4: two failures before the guide/trace changes, then two passes.

## Covering and final verification

Covering suites after the focused cycles:

```text
.venv/bin/python -m pytest tests/test_packaging_contract.py tests/test_error_episodes.py tests/test_agent_v2_eval.py tests/test_eval_report.py tests/test_documentation_contract.py tests/test_http_client.py tests/test_llm.py -q
93 passed, 1 skipped, 1 warning in 0.73s
```

Required final commands and fresh outputs:

```text
.venv/bin/python -m pytest tests/ -q
421 passed, 1 skipped, 1 warning in 1.64s

.venv/bin/python -m ruff check src/ tests/ eval/
All checks passed!

git diff --check
exit 0, no output
```

The one skip is the existing sqlite-vec extension-loading check on a Python
build without `Connection.enable_load_extension`. The warning is the expected,
tested sqlite-vec-to-NumPy fallback warning.

## Files changed

- `CLAUDE.md`
- `eval/harness.py`
- `eval/report.py`
- `run_trace.py`
- `src/http_client.py`
- `src/memory/vector_backend.py`
- `tests/test_agent_v2_eval.py`
- `tests/test_documentation_contract.py`
- `tests/test_error_episodes.py`
- `tests/test_eval_report.py`
- `tests/test_http_client.py`
- `tests/test_packaging_contract.py`
- `.superpowers/sdd/final-review-fix-report.md` (this report)

## Self-review

- Rechecked each binding requirement against the final diff.
- Confirmed disabled episode memory imports without either optional vector
  backend, while enabled memory still tries sqlite-vec first and warns before
  the lazy NumPy fallback.
- Compared eval key precedence directly with `src/http_client.py`.
- Confirmed mode metadata is filtered to non-empty values and sorted, and legacy
  missing-mode records remain end-to-end.
- Confirmed mixed rates are explicitly labeled and the per-mode summary keeps
  performance and provenance together.
- Confirmed malformed completion paths now raise `LLMResponseError` and valid
  streamed tool calls remain accepted.
- Confirmed the persistence test closes and reopens a file-backed database.
- Searched active guide/trace/report code for stale forced model defaults.
- Reviewed the staged diff and ran the whitespace check before committing.

## Concerns

- No functional concerns remain.
- The expected sqlite-vec skip/warning noted above is environment-specific and
  demonstrates the required NumPy fallback path rather than a failure.
- The worktree has a pre-existing untracked `.venv` link; it was not modified or
  committed.

## Re-review Follow-up (2026-07-20)

Two additional Important findings from the final re-review were addressed.

### Exact CI import-order failure

Implementation:

- Removed the extra blank line between the first-party import and module
  constant in `tests/test_eval_report.py`, matching Ruff/isort's exact expected
  import-block layout.

Evidence:

- RED: `.venv/bin/python -m ruff check src/ tests/ eval/ --select=E,F,I --ignore=E501`
  reported one `I001` at `tests/test_eval_report.py:1` and exited 1.
- GREEN: the exact same CI command reported `All checks passed!` and exited 0.

### Ambiguous terminal summary for mixed evaluation modes

Implementation:

- Extended the existing mixed-mode report test to capture terminal output.
- Replaced the generic combined terminal line with an explicitly labeled
  `all modes combined` resolved rate.
- Added one terminal line for each non-empty mode containing the mode name,
  sample count, sorted models, sorted commits, resolved count/rate, and decisive
  failure taxonomy together.
- Terminal presentation consumes the existing `by_evaluation_mode` metrics and
  shares taxonomy formatting with Markdown; it does not reclassify samples.

TDD evidence:

- RED: `.venv/bin/python -m pytest tests/test_eval_report.py::test_agent_v2_metrics_and_reports_separate_evaluation_modes -q`
  failed (`1 failed`) because output still contained the generic
  `agent_v2: 2/4 success` line and no per-mode terminal summaries.
- GREEN: the same focused test passed (`1 passed in 0.02s`).
- Covering suite: `.venv/bin/python -m pytest tests/test_eval_report.py -q`
  passed (`5 passed in 0.04s`).

### Re-review final verification

```text
.venv/bin/python -m pytest tests/ -q
421 passed, 1 skipped, 1 warning in 1.64s

.venv/bin/python -m ruff check src/ tests/ eval/ --select=E,F,I --ignore=E501
All checks passed!

git diff --check
exit 0, no output
```

The skip and warning are the same expected sqlite-vec environment behavior
documented above.

Files changed in this follow-up:

- `eval/report.py`
- `tests/test_eval_report.py`
- `.superpowers/sdd/final-review-fix-report.md`

Follow-up self-review:

- Confirmed the combined terminal rate is explicitly labeled as spanning all
  modes.
- Confirmed empty modes are omitted and both populated modes print all required
  provenance/performance/taxonomy fields on a single line.
- Confirmed model and commit ordering comes from the already-sorted metric
  fields.
- Confirmed terminal rendering reuses computed failure taxonomy and does not
  duplicate classification logic.
- Confirmed the exact CI Ruff command, not the narrower default Ruff selection,
  is clean.

Follow-up concerns:

- No new functional concerns.
- The pre-existing untracked `.venv` link remains untouched.
