# Task 13B PatchGate Stable Identity Report

## Status

DONE

## Root cause

`PatchGate._fingerprint()` serialized every `PatchEdit` field. PLAN and EXECUTE
populate `resolved_target_symbol` after approval for reasoning and diversity,
even though `PatchEdit` documents that value as metadata-only and exact search
application continues to use the verbatim `search` anchor. The metadata update
therefore changed only the approval fingerprint and caused revalidation to reject
an otherwise unchanged exact patch.

## TDD evidence

Baseline before the regression:

```text
.venv/bin/python -m pytest tests/test_patch_gate.py -q
26 passed in 3.87s
```

RED command after adding the real exact-search regression:

```text
.venv/bin/python -m pytest tests/test_patch_gate.py::test_gate_approval_survives_resolved_search_target_metadata -q
```

RED result: exit 1, `1 failed in 0.26s`. The failure occurred at
`revalidate_approved_patch(state)` after setting
`resolved_target_symbol = "Nominal._setup"` and raised the expected
`ValueError: PatchGate approval fingerprint or canonical fields changed`.

GREEN command after the minimal implementation:

```text
.venv/bin/python -m pytest tests/test_patch_gate.py::test_gate_approval_survives_resolved_search_target_metadata -q
```

GREEN result: exit 0, `1 passed in 0.38s`.

## Implementation

- Added an explicit PatchGate approval serializer for ordered exact edits.
- Bound the serializer to `file_path`, `search`, `node_target`, `replace`,
  `replace_all`, `expected_content_sha256`, and `exact_only`.
- Excluded only `resolved_target_symbol`, which cannot affect the exact mutation.
- Kept plan, ordered edit list, result manifest, patch SHA-256, and base ref in
  the existing fingerprint payload.
- Retained the negative post-approval edit-tampering tests, including replacement
  mutation and edit-order mutation, which continue to fail closed.

## Verification

Focused PatchGate suite:

```text
.venv/bin/python -m pytest tests/test_patch_gate.py -q
27 passed in 3.53s
```

Relevant PLAN metadata, escalation integration, and EXECUTE preflight suites:

```text
.venv/bin/python -m pytest tests/test_model_escalation_integration.py tests/test_convergence_diversity.py tests/test_patch_preflight.py -q
80 passed in 0.81s
```

Full local regression suite:

```text
.venv/bin/python -m pytest -q
1050 passed, 2 skipped, 1 warning in 36.28s
```

The warning is the established optional sqlite-vec fallback to NumPy in
`tests/test_error_episodes.py`.

Changed-file Ruff:

```text
.venv/bin/python -m ruff check src/patch_gate.py tests/test_patch_gate.py
All checks passed!
```

Diff validation:

```text
git diff --check
(no output; exit 0)
```

## Self-review

- The exact application and snapshot rebuild paths are unchanged.
- Path canonicalization/confinement, no-follow and symlink checks, preimage
  binding, exact-only enforcement, atomic application, and live-worktree
  validation are unchanged.
- Every field used to locate or produce the mutation remains fingerprint-bound,
  and list serialization preserves edit ordering.
- The regression approves and applies a real Python exact-search edit after the
  same metadata mutation performed by PLAN.
- No unrelated refactor or dependency change was made.

## Files changed

- `src/patch_gate.py`
- `tests/test_patch_gate.py`
- `.superpowers/sdd/task-13b-patchgate-stable-identity-report.md`

## Boundaries and concerns

No network/API calls, live eval, credential access, attachments, or evaluator
gold fields were used. No known concerns.
