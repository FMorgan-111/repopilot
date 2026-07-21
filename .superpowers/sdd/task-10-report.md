# Task 10 Report: Integrate policy, tools, and failure semantics into PLAN/REFLECT

## Status

Implementation and independent review complete; all review findings are fixed.

## TDD RED evidence

Tests were added before production wiring in:

- `tests/test_model_escalation_integration.py`
- `tests/test_context_loop_brake.py`
- `tests/test_convergence_diversity.py`
- `tests/test_failure_taxonomy.py`

Required RED command:

```text
.venv/bin/python -m pytest tests/test_model_escalation_integration.py tests/test_context_loop_brake.py tests/test_convergence_diversity.py tests/test_failure_taxonomy.py -q
```

Observed RED: `8 failed, 47 passed`.

Expected missing behavior represented by the failures:

1. Opus repair failures still returned a generic `ValueError` failure instead of enforcing two no-progress rounds and `opus_no_progress_limit`.
2. PLAN ignored `tool_intent`, so it neither routed tools nor re-prompted with new evidence.
3. Duplicate tools were not recorded as no-progress and the eight-request round cap was not enforced.
4. Two repeated Gemini invalid anchors did not activate deterministic one-way escalation.
5. REFLECT did not require a different target symbol after an unchanged assertion.
6. VERIFY routed syntax/import failures through REFLECT rather than direct patch correction.
7. VERIFY did not track unchanged assertion signatures or terminate the second unchanged repeat.

Two self-review regression tests were also observed RED before their fixes: mixed tool/legacy-patch variants were not rejected, and a REFLECT tool-triggered escalation reused the primary prompt.

The independent review first exposed a collection RED because the strict response
classifier did not yet exist. After adding only that importable seam, the review
regressions produced the intended behavioral RED: `10 failed, 47 passed`. Those
failures covered assertion streak persistence through PLAN, deterministic target
diversity, VERIFY classification order, exclusive response variants, exhausted
Opus PatchGate rounds, and quoted/unquoted `gold_patch` filtering. A further
cross-outcome-field regression was observed RED (`1 failed, 3 passed`) before the
explicit outcome field allowlists were completed.

## Implementation

- Added `src/reasoning_loop.py` as the shared PLAN/REFLECT mechanism for discriminated tool responses, a hard eight-request reasoning-round cap, safe tool diagnostics, Opus-local no-progress limits, and re-prompting with only the immediately new evidence IDs.
- Kept legacy PLAN/REFLECT responses without a `kind` discriminator compatible while accepting explicit `tool`, outcome, and stop variants.
- Kept escalation deterministic and one-way: model output never changes providers directly; only `ModelPolicy` applies escalation.
- Kept Gemini on the existing plan path. Escalated PLAN continues to use the existing `RepairPlan` to `VerifiedEditBatch` two-stage flow and PatchGate.
- Connected invalid anchors, repeated edits, duplicate tools, evidence progress, pending edit signatures, and test-failure signatures to existing progress/no-progress state.
- Extended the semantic plan transaction signature with the current pending patch/edit transaction so a genuinely different invalid anchor is progress.
- Added VERIFY routing for infrastructure errors without retry consumption, syntax/import direct PLAN correction, assertion REFLECT, one required different-target-symbol round, and stable repeated-assertion termination.
- Preserved an assertion-specific failure signature and streak across intervening PLAN progress, and made PLAN reject a required-diversity repair that reuses any prior failed target.
- Classified infrastructure, syntax/import, and assertion failures before the generic same-patch replay brake while retaining the existing replay brake for ordinary patch failures.
- Made explicitly tagged PLAN/REFLECT responses exclusive field allowlists; untagged historical outcomes still pass through a separate compatibility adapter.
- Counted exhausted Opus PatchGate rounds toward the two-round `opus_no_progress_limit`, independently of issue fingerprint changes.
- Added evaluator-payload boundaries to safe evidence normalization; mixed tool/outcome payloads are rejected. The legacy PLAN `patch` response remains at its pre-existing compatibility boundary.
- Verified saved already-escalated state replays without provider downgrade or phase renaming.

No phase was renamed and COVERAGE was not added. No network/API, clone, checkout, push, PR, scoring, or arbitrary shell tool request is used by the new tests.

## Verification

Required focused suite:

```text
.venv/bin/python -m pytest tests/test_model_escalation_integration.py tests/test_decision_frame.py tests/test_context_loop_brake.py tests/test_convergence_diversity.py tests/test_patch_retry.py tests/test_new_agent.py tests/test_failure_taxonomy.py -q
```

Result after review fixes: `178 passed in 1.94s`.

Required Ruff command:

```text
.venv/bin/python -m ruff check src/nodes/plan.py src/nodes/reflect.py src/nodes/verify.py src/nodes/failure.py src/graph.py src/new_agent.py tests/test_model_escalation_integration.py
```

Result: `All checks passed!`.

Additional Ruff over new/shared files and modified companion tests: `All checks passed!`.

Full suite:

```text
.venv/bin/python -m pytest -q
```

Result: `785 passed, 2 skipped, 1 warning in 22.61s`. The warning is the existing sqlite-vec unavailable fallback to NumPy in `test_error_episodes.py`.

`git diff --check`: passed.

## Safety and compatibility review

- Tool diagnostics persist action, status, bounded counters, and evidence ID only; raw arguments and model rationales are not persisted there.
- Duplicate/rejected/error tool calls add no fabricated evidence IDs.
- Evidence re-prompt selection is ID-based and bounded by the existing `EvidenceStore` renderer.
- Evaluator-only payload markers are truncated before safe evidence reaches state or prompts.
- No credential or API-key value was added to code, tests, or this report.
- Existing public entry points, saved-state defaults, provider fields, phases, and legacy structured outcomes remain compatible.

## Concerns

- None known. The full suite has one pre-existing optional sqlite-vec fallback warning, unrelated to this task.
