# Task 13C Escalation Repair Context Report

## Starting point

- Required starting HEAD: `dc12dae283b2a30d32ac72a32872760e1cdb1d79`
- Worktree was clean before task edits.
- No network, credentials, attachments, live eval, or evaluator gold artifacts were accessed.

## TDD evidence

### RED 1: hydrated relevant-file evidence is absent from the first RepairPlan packet

Command:

```text
.venv/bin/python -m pytest -q tests/test_model_escalation_integration.py::test_escalated_plan_seeds_bounded_safe_relevant_file_evidence
```

Result: exit 1, `1 failed`. The integration test reached escalated PLAN and failed on the intended assertion:

```text
assert [item.file_path for item in packet.evidence] == [
    "src/widget.py",
    "src/transport.py",
]
E   AssertionError: assert [] == ['src/widget.py', 'src/transport.py']
```

The earlier plain `pytest` invocation was not accepted as RED because it used the system interpreter and failed during collection on Pydantic v1. The repository-prescribed `.venv/bin/python -m pytest` runtime uses Pydantic 2.13.4 and pytest 9.1.1.

### GREEN 1: bounded safe source evidence reaches the first RepairPlan packet

Command:

```text
.venv/bin/python -m pytest -q tests/test_model_escalation_integration.py::test_escalated_plan_seeds_bounded_safe_relevant_file_evidence
```

Result: exit 0, `1 passed in 0.16s`.

The passing assertions prove that the packet contains only eligible files from the planner's top three, includes the issue-relevant fix area below a long header, caps every source item at 6,000 characters, remains within the 70,000-character packet render cap, and excludes credential-shaped text, evaluator fields, raw HTTP, generated tests, the fourth relevant file, conversation history, tool-call payloads, and full failed-patch content.

### RED 2: RepairContextError retry repeats the same prompt

Command:

```text
.venv/bin/python -m pytest -q tests/test_model_escalation_integration.py::test_repair_context_error_adds_only_bounded_safe_target_correction
```

Result: exit 1, `1 failed`. The first repair call raised a `RepairContextError`; the second returned a valid repair, so PLAN reached the later prompt assertion and failed for the intended reason:

```text
assert prompts[1] != prompts[0]
E   assert '<same bounded EscalationPacket>' != '<same bounded EscalationPacket>'
```

### GREEN 2: bounded classified correction changes only RepairContextError retries

Command:

```text
.venv/bin/python -m pytest -q tests/test_model_escalation_integration.py::test_repair_context_error_adds_only_bounded_safe_target_correction tests/test_model_escalation_integration.py::test_two_opus_no_progress_rounds_stop_with_stable_reason
```

Result: exit 0, `2 passed in 0.16s`.

The first test proves that a target-symbol `RepairContextError` changes the second first-stage prompt, adds no more than 500 characters of fixed classified feedback, tells the model to choose a different valid evidence-backed target, does not copy the rejected model-authored target, and reaches EXECUTE. The second proves that generic unexpected errors retain identical retry prompts and stop after exactly two calls with `opus_no_progress_limit`.

### RED 3: explicit tool evidence drops mandatory seeded source evidence

Command:

```text
.venv/bin/python -m pytest -q tests/test_model_escalation_integration.py::test_explicit_tool_evidence_precedes_but_does_not_drop_seeded_source
```

Result: exit 1, `1 failed`. The packet contained only the explicit fresh-tool evidence ID; the expected seeded `planner_relevant_file` ID was absent.

### GREEN 3: fresh explicit evidence is first and seeded source remains present

Command:

```text
.venv/bin/python -m pytest -q tests/test_model_escalation_integration.py::test_explicit_tool_evidence_precedes_but_does_not_drop_seeded_source
```

Result: exit 0, `1 passed in 0.16s`. The explicit tool evidence is selected first, then the seeded source window; unrelated historical evidence is excluded from the explicit-delta packet.

### RED 4: raw HTTP was packet-safe but remained in persisted seeded evidence

Command:

```text
.venv/bin/python -m pytest -q tests/test_model_escalation_integration.py::test_escalated_plan_seeds_bounded_safe_relevant_file_evidence
```

Result: exit 1, `1 failed`. The strengthened safety assertion found `raw-http-third-sentinel` in the persisted `planner_relevant_file` evidence even though final packet validation removed it. This demonstrated that sanitization needed to happen before EvidenceStore insertion as well as at packet rendering.

### GREEN 4: seeded evidence is sanitized before persistence

Command:

```text
.venv/bin/python -m pytest -q tests/test_model_escalation_integration.py::test_escalated_plan_seeds_bounded_safe_relevant_file_evidence
```

Result: exit 0, `1 passed in 0.17s`; changed-file Ruff and `git diff --check` also exited 0. The source window now passes through the existing escalation sanitizer before EvidenceStore insertion, so secret, evaluator, raw-HTTP, and generated-path boundaries are applied to persisted seed evidence as well as the final packet.

### RED 5: full evidence capacity evicts the explicit fresh-tool item

Command:

```text
.venv/bin/python -m pytest -q tests/test_model_escalation_integration.py::test_explicit_tool_evidence_precedes_but_does_not_drop_seeded_source
```

Result: exit 1, `1 failed`. With 29 stale items plus the fresh explicit item already filling EvidenceStore's 30-item cap, source seeding retained the oldest 29 and evicted the fresh explicit item before packet selection.

### GREEN 5: explicit evidence is protected during bounded compaction

Command:

```text
.venv/bin/python -m pytest -q tests/test_model_escalation_integration.py::test_explicit_tool_evidence_precedes_but_does_not_drop_seeded_source
```

Result: exit 0, `1 passed in 0.15s`. At full 30-item state capacity, explicit IDs are protected first, stale items are compacted, and the packet selects fresh explicit evidence before seeded source evidence.

### RED 6: the real inner RepairPlan tool reprompt bypasses source merging

Command:

```text
.venv/bin/python -m pytest -q tests/test_model_escalation_integration.py::test_inner_repairplan_tool_reprompt_keeps_fresh_then_seeded_evidence
```

Result: exit 1, `1 failed`. A real `plan_fix()` → `generate_opus_repair()` integration run requested one approved tool, then produced a valid RepairPlan and verified edit. The second RepairPlan prompt contained only `search_text`; `planner_relevant_file` was missing. This confirmed the read-only review finding that `repair_flow.py`'s internal reprompt callback bypassed PLAN's evidence merge.

### GREEN 6: the real inner tool reprompt retains fresh and seeded evidence

Command:

```text
.venv/bin/python -m pytest -q tests/test_model_escalation_integration.py::test_inner_repairplan_tool_reprompt_keeps_fresh_then_seeded_evidence
```

Result: exit 0, `1 passed in 0.46s`. The real two-stage flow reached EXECUTE; its second RepairPlan prompt contained the fresh `search_text` evidence first, the seeded `planner_relevant_file` evidence second, and no unrelated stale evidence.

## Implementation

- Added PLAN-local source seeding from only `state.relevant_files[:3]`; no filesystem, checkout, network, attachment, credential, or eval-artifact read was introduced.
- Reused `_relevance_window()` with the existing 6,000-character planner/evidence cap.
- Passed each source window through the existing escalation sanitizer and an isolated `EvidenceStore`, excluding approved generated-test paths before insertion.
- Replaced colliding persisted entries with the freshly derived hydrated source evidence, bounded persisted evidence to 30 items, and kept source IDs stable across retries.
- Prioritized seeded source in normal escalation packets. For explicit tool-delta packets, prioritized and protected the fresh explicit IDs first, then included seeded source without reintroducing unrelated historical evidence.
- Added a fixed 500-character maximum correction section for `RepairContextError` only. The correction selects an allowlisted constraint class from the exception and never copies exception text or the rejected model-authored target.
- Added a backwards-compatible optional first-stage reprompt callback to `generate_opus_repair()`. PLAN supplies its explicit-first-plus-seeded-source builder; direct repair-flow callers retain fresh delta evidence first and only source evidence explicitly supplied in the initial packet.
- Left generic unexpected-error prompting unchanged and retained the two-round `record_opus_no_progress()` terminal path and `opus_no_progress_limit` reason.
- Did not modify RepairPlan, exact target snapshot, target evidence, PatchGate, or checkout validation.

## Final verification

Focused escalation, packet, evidence, relevance-window, PLAN retry, two-stage repair, and convergence suites:

```text
.venv/bin/python -m pytest -q tests/test_model_escalation_integration.py tests/test_escalation_packet.py tests/test_evidence.py tests/test_relevance_window.py tests/test_patch_retry.py tests/test_repair_flow.py tests/test_convergence_diversity.py
```

Result after the review fix: exit 0, `127 passed in 8.45s`.

Changed-file lint:

```text
.venv/bin/python -m ruff check src/nodes/plan.py src/repair_flow.py tests/test_model_escalation_integration.py
```

Result: exit 0, `All checks passed!`.

Whitespace/error check:

```text
git diff --check
```

Result: exit 0 with no output.

## Self-review

- Source provenance: every seeded content byte originates from a pre-hydrated `FileInfo.content`; the implementation performs no source read.
- Bounds: at most the first three PLAN files are considered; each persisted source item is at most 6,000 characters; EvidenceStore selection, 12-item/24,000-character packet evidence bounds, and the 70,000-character rendered packet bound remain enforced.
- Safety: credentials are redacted and evaluator/raw-HTTP/generated-test boundaries are truncated before persistence and again at packet construction. Tests prove conversation history, tool-call payloads, evaluator fields, generated tests, and failed-patch content do not enter the prompt.
- Retry behavior: only `RepairContextError` receives classified correction text; generic exceptions receive the unchanged prompt; seeded evidence deduplicates; inner RepairPlan tool rounds retain fresh evidence first and seeded source second; exactly two consecutive Opus no-progress calls remain the terminal maximum.
- Validation: no RepairPlan, snapshot, evidence, PatchGate, checkout, or exact-edit validation was weakened.

## Initial review status

`DONE`

Concerns: none.

## Independent review follow-up

### RED 7: source seeding leaves no capacity for the real tool router

Command:

```text
.venv/bin/python -m pytest -q tests/test_model_escalation_integration.py::test_source_seeding_reserves_capacity_across_real_tool_reprompts
```

Result: exit 1, `1 failed`. Starting with 29 stale items, the first source-seeded prompt filled the 30-item EvidenceStore. A real approved local `search_text` routed through `route_tool_intent()` then returned `status='error'` instead of `status='ok'` because EvidenceStore had no remaining slot.

### GREEN 7: every prompt build reserves one real tool-evidence slot

Command:

```text
.venv/bin/python -m pytest -q tests/test_model_escalation_integration.py::test_source_seeding_reserves_capacity_across_real_tool_reprompts
```

Result: exit 0, `1 passed in 0.85s`. The initial source-seeded prompt leaves 29 persisted items, the real router adds fresh evidence as item 30, the explicit-evidence reprompt compacts back to 29 while retaining explicit-first/source-second packet ordering, and a second real tool round succeeds without exceeding the 30-item cap.

### RED 8: an inner tool round drops the outer RepairContextError correction

Command:

```text
.venv/bin/python -m pytest -q tests/test_model_escalation_integration.py::test_repair_context_correction_survives_inner_tool_reprompt
```

Result: exit 1, `1 failed`. The combined integration reached EXECUTE after outer `RepairContextError`, a corrected second attempt, an inner approved tool request, a valid RepairPlan, and a verified edit. The corrected attempt's first prompt contained the fixed correction, but its tool reprompt contained zero `REPAIR TARGET CORRECTION` sections.

### GREEN 8: classified correction survives every inner RepairPlan tool round

Command:

```text
.venv/bin/python -m pytest -q tests/test_model_escalation_integration.py::test_repair_context_correction_survives_inner_tool_reprompt tests/test_model_escalation_integration.py::test_repair_context_error_adds_only_bounded_safe_target_correction tests/test_model_escalation_integration.py::test_two_opus_no_progress_rounds_stop_with_stable_reason
```

Result: exit 0, `3 passed in 0.37s`. Both corrected RepairPlan prompts contain exactly one identical bounded classification/instruction suffix and exclude secret, evaluator, raw-HTTP, and raw exception values. The run reaches EXECUTE; the standalone correction and generic two-round terminal regressions remain green.

### RED 9: direct repair-flow default reprompt drops supplied source evidence

Command:

```text
.venv/bin/python -m pytest -q tests/test_repair_flow.py::test_default_repairplan_tool_reprompt_retains_initial_source_evidence
```

Result: exit 1, `1 failed`. Direct `generate_opus_repair()` received an initial packet containing one `planner_relevant_file` source item plus unrelated old evidence. After a successful tool round, the next RepairPlan prompt contained only `search_text`; the supplied source item was absent.

### GREEN 9: direct default reprompt merges fresh-first with supplied source

Command:

```text
.venv/bin/python -m pytest -q tests/test_repair_flow.py::test_default_repairplan_tool_reprompt_retains_initial_source_evidence tests/test_repair_flow.py::test_opus_inner_repair_tool_uses_delta_evidence_and_pre_call_policy
```

Result: exit 0, `2 passed in 0.23s`. The direct default reprompt contains fresh `search_text` evidence first and the initial packet's `planner_relevant_file` evidence second, each exactly once, excludes unrelated old evidence, and remains within the existing item/content/total/render bounds. The existing no-source direct caller stays delta-only.

## Independent review fixes

- PLAN source seeding now reserves one evidence slot on every prompt build, so a real tool round can add its delta while the state remains capped at 30 items.
- Protected explicit evidence and seeded source survive reprompt compaction; stale unprotected evidence is evicted first.
- Repair-context feedback is held as a fixed bounded suffix for the active outer retry and is reattached to every inner RepairPlan tool reprompt.
- The direct repair-flow default revalidates a packet made from current fields, fresh explicit evidence, and only `planner_relevant_file` evidence from the initial packet, with ID/fingerprint deduplication.
- No source is synthesized when `relevant_files` and the initial packet have none.

## Independent review verification

Covering suites:

```text
.venv/bin/python -m pytest -q tests/test_model_escalation_integration.py tests/test_repair_flow.py tests/test_evidence.py tests/test_escalation_packet.py tests/test_convergence_diversity.py
```

Result: exit 0, `113 passed in 7.67s`.

Changed-file lint:

```text
.venv/bin/python -m ruff check src/nodes/plan.py src/repair_flow.py tests/test_model_escalation_integration.py tests/test_repair_flow.py
```

Result: exit 0, `All checks passed!`.

Whitespace/error check:

```text
git diff --check
```

Result: exit 0 with no output.

## Final self-review

- Capacity remains capped at 30; reserving one slot only evicts stale unprotected evidence and does not enlarge any bound.
- Every PLAN prompt/reprompt retains deterministic source evidence, while immediate tool evidence stays first.
- RepairContextError feedback remains fixed, classified, sanitized, at most 500 characters, and present through inner tool rounds; generic errors remain unchanged.
- Direct repair-flow calls preserve only source actually supplied in the initial packet and do not infer or read unavailable source.
- RepairPlan, target snapshot/evidence, PatchGate, checkout, and exact-edit validation remain unchanged.

## Final status

`DONE`

Concerns: none.

## Direct repair-boundary review follow-up

### Starting point

- Follow-up starting HEAD: `21907729fac03f749edfd00dd8e47cac13991d65`.
- The worktree was clean before follow-up edits.
- No network, credentials, attachments, live eval, or evaluator gold artifacts were accessed.

### RED 10: direct repair calls do not hydrate or reserve source context

Command:

```text
.venv/bin/python -m pytest -q tests/test_repair_flow.py::test_direct_repair_hydrates_source_for_first_and_inner_plan_prompts tests/test_repair_flow.py::test_direct_repair_truncates_hydrated_source_candidates_to_three tests/test_repair_flow.py::test_direct_repair_reserves_state_capacity_for_real_tool_evidence
```

Result: exit 1, `3 failed in 0.52s`.

- The direct call's first RepairPlan prompt had no `planner_relevant_file` evidence despite hydrated `state.relevant_files`.
- The inner RepairPlan reprompt likewise had no derived source evidence.
- Four hydrated source candidates produced zero source items instead of the bounded first three.
- Starting from a full 30-item evidence state left no capacity for the real `search_text` route, so its evidence was absent from the reprompt.

Representative intended failures:

```text
E   AssertionError: assert [] == ['planner_relevant_file']
E   AssertionError: assert [] == ['src/widget.py', 'src/second.py', 'src/third.py']
E   AssertionError: assert [] == ['search_text', 'planner_relevant_file']
```

### GREEN 10: the direct repair boundary hydrates every RepairPlan prompt

Command:

```text
.venv/bin/python -m pytest -q tests/test_repair_flow.py::test_direct_repair_hydrates_source_for_first_and_inner_plan_prompts tests/test_repair_flow.py::test_direct_repair_truncates_hydrated_source_candidates_to_three tests/test_repair_flow.py::test_direct_repair_reserves_state_capacity_for_real_tool_evidence
```

Result: exit 0, `3 passed in 0.89s`.

The first direct RepairPlan prompt now contains a relevance-centered source window derived from hydrated state. After a tool request, the inner RepairPlan reprompt contains fresh `search_text` evidence first and the deterministic source window second. Four hydrated candidates stop at three, each source item remains within 6,000 characters, and a full state is compacted to reserve the real router's next evidence slot without exceeding 30 persisted items.

### Empty-relevant-files fallback verification

Command:

```text
.venv/bin/python -m pytest -q tests/test_repair_flow.py::test_default_repairplan_tool_reprompt_retains_initial_source_evidence tests/test_repair_flow.py::test_opus_inner_repair_tool_uses_delta_evidence_and_pre_call_policy
```

Result: exit 0, `2 passed in 0.25s`.

With empty `state.relevant_files`, direct repair retains only the first three source items supplied in the initial packet. With neither hydrated nor initially supplied source, it synthesizes none and keeps the existing delta-only reprompt behavior.

## Direct-boundary implementation

- Moved the single relevance-window implementation into dependency-safe `src/escalation.py`; both the ordinary PLAN prompt and escalation source preparation import it.
- Added one shared `prepare_repair_plan_packet()` boundary used by PLAN and direct repair flow. It derives stable source evidence from only `state.relevant_files[:3]`, sanitizes before persistence, excludes approved generated-test paths, and uses the existing evidence/packet validators for per-item, total-evidence, and rendered-packet bounds.
- The helper compacts persisted evidence to at most 29 items before a real tool route, protecting explicit delta evidence and deterministic source while reserving one slot under the existing 30-item state cap.
- Initial RepairPlan calls hydrate their packet inside `generate_opus_repair()` itself. Default inner reprompts rebuild through the same helper with explicit tool evidence first and source second.
- When hydrated files are empty, the helper retains at most three already-supplied `planner_relevant_file` items from the initial packet and never fabricates source.
- Custom PLAN prompt suffixes, bounded RepairContextError correction retention, and the existing second-stage edit flow remain unchanged.

## Direct-boundary verification

Required suites:

```text
.venv/bin/python -m pytest -q tests/test_model_escalation_integration.py tests/test_repair_flow.py tests/test_evidence.py tests/test_escalation_packet.py tests/test_relevance_window.py tests/test_convergence_diversity.py
```

Result: exit 0, `121 passed in 7.72s`.

Changed-file lint:

```text
.venv/bin/python -m ruff check src/escalation.py src/nodes/plan.py src/repair_flow.py tests/test_repair_flow.py tests/test_relevance_window.py
```

Result: exit 0, `All checks passed!`.

Whitespace/error check:

```text
git diff --check
```

Result: exit 0 with no output.

## Direct-boundary self-review

- Direct ownership: `generate_opus_repair()` prepares its own first-stage packet before any model call; its default inner reprompt uses the same preparation boundary.
- Source selection: hydrated source always supersedes packet-supplied source, takes only the first three relevant candidates, applies the planner's shared relevance window, and persists no item over 6,000 characters.
- Safety: the existing secret/evaluator/raw-HTTP sanitizer runs before persistence and packet validation; generated-test paths are excluded. No new filesystem or external data read was introduced.
- Ordering and bounds: explicit tool evidence precedes source on reprompts; existing 12-item/24,000-character evidence and 70,000-character packet bounds remain active; state is capped at 30 and reserves one pre-route slot.
- Compatibility: empty hydrated state keeps at most three initially supplied source items and otherwise adds none. PLAN correction suffixes and RepairPlan/PatchGate/checkout/exact-edit validation remain intact.

## Single-packet prompt follow-up

### RED 11: independently prepared custom prompt can duplicate the packet

Command:

```text
.venv/bin/pytest -q tests/test_repair_flow.py::test_generate_opus_repair_renders_one_bounded_packet_with_suffix
```

Result before implementation: exit 1 because `generate_opus_repair()` did not
accept a structured first-stage suffix. The production caller instead supplied a
complete packet-bearing prompt. If packet preparation reordered or compacted
evidence, the repair boundary could not find that exact original packet and
prepended a second complete packet, bypassing the effective evidence bounds.

### GREEN 11: render one prepared packet and append only bounded context

The repair boundary now distinguishes an exact full prompt override from a
first-stage suffix. The normal PLAN path passes only its rolling-summary and
sanitized RepairContext correction suffix; `generate_opus_repair()` prepares
and renders the packet once, then appends that suffix. Its default inner tool
reprompt repeats the same operation with fresh explicit evidence first. A full
override and suffix are mutually exclusive, so there is no ambiguous merge or
fallback prepend path.

The regression fixture starts with three large hydrated source files, fifteen
stale evidence items, and a rolling-summary suffix. It decodes exactly one JSON
packet, verifies the suffix occurs after that packet, checks at most twelve
evidence items and three source items, and rechecks the 24,000-character evidence
budget.

Focused verification:

```text
.venv/bin/pytest -q tests/test_repair_flow.py tests/test_model_escalation_integration.py
```

Result: exit 0, `57 passed in 2.73s`.

Changed-file lint and whitespace verification:

```text
.venv/bin/ruff check src/repair_flow.py src/nodes/plan.py tests/test_repair_flow.py tests/test_model_escalation_integration.py
git diff --check
```

Result: both exit 0; Ruff reported `All checks passed!` and the diff check had no
output.

Full regression suite:

```text
.venv/bin/python -m pytest -q
```

Result: exit 0, `1061 passed, 2 skipped, 1 known sqlite-vec fallback warning in
38.02s`.

### RED 12: full prompt override discards hydrated source

An independent re-review confirmed the duplicate-packet path was gone, then
found that the compatibility `first_stage_prompt` override still selected its
old packet verbatim. With one hydrated relevant file, the model therefore saw
one packet but zero `planner_relevant_file` items.

Command:

```text
.venv/bin/pytest -q tests/test_repair_flow.py::test_custom_first_stage_prompt_retains_hydrated_source
```

Result before implementation: exit 1; expected `src/widget.py` source evidence,
observed none.

### GREEN 12: overrides contribute suffixes, never packets

The compatibility path now parses and validates the caller's leading
`EscalationPacket`, discards that stale packet, retains only its trailing suffix,
and rebuilds the prompt from the currently prepared packet. Invalid arbitrary
overrides fail closed. Custom inner reprompts use the same rule, so explicit tool
evidence and deterministic hydrated source cannot be displaced by a caller's
old packet.

Focused verification:

```text
.venv/bin/pytest -q tests/test_repair_flow.py::test_custom_first_stage_prompt_retains_hydrated_source tests/test_repair_flow.py::test_direct_repair_hydrates_source_for_first_and_inner_plan_prompts
```

Result: exit 0, `2 passed in 0.32s`. The first test asserts one JSON packet and
one hydrated source item; the second asserts that a custom inner reprompt still
contains fresh tool evidence first, hydrated source second, one packet, and one
suffix.

Post-fix required suites: `123 passed in 8.29s`. Post-fix full regression suite:
`1062 passed, 2 skipped, 1 known sqlite-vec fallback warning in 37.06s`.

### RED 13: a second packet can masquerade as a suffix

The next independent re-review supplied `valid_packet + valid_packet`. Parsing
only the leading packet allowed the second structured packet to survive as an
unclassified suffix. The same gap existed for custom inner reprompts and for
arbitrary text after a valid packet.

Command:

```text
.venv/bin/pytest -q tests/test_repair_flow.py::test_custom_first_stage_prompt_rejects_unclassified_tail tests/test_repair_flow.py::test_custom_inner_reprompt_rejects_second_packet_tail
```

Result before implementation: exit 1, `3 failed`; both full-prompt cases reached
the model, and the inner case repeated until the tool-round limit.

### GREEN 13: suffixes have a strict, sanitized grammar

First-stage suffixes now accept only the exact rolling-summary section (maximum
200 characters), the exact RepairContext correction section (maximum 500
characters including its header), or both in that order. Section bodies must
pass the summary safety sanitizer unchanged, cannot repeat section headers, and
cannot contain structured-object delimiters. Empty, arbitrary, duplicated,
credential-shaped, raw-HTTP, evaluator-shaped, or second-packet tails fail
closed before another model call. This validation applies equally to the direct
suffix API, compatibility full prompts, and custom inner reprompts.

Focused result: `6 passed in 1.22s`; Ruff passed.

Post-grammar required suites: `126 passed in 8.01s`. Post-grammar full regression
suite: `1065 passed, 2 skipped, 1 known sqlite-vec fallback warning in 36.55s`.
