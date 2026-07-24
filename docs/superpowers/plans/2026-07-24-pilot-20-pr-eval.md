# Pilot-20 Pre-Merge Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` to implement this plan task-by-task,
> with a fresh implementer and an independent spec/code review after every task.

**Goal:** Run one immutable, officially scored 20-instance SWE-bench Verified
pilot against PR 2, prove 20 determined outcomes and zero infrastructure
failures, then merge exactly the evaluated head/base pair to `master`.

**Architecture:** Freeze selection in source control; bind each runtime and
artifact to strict GitHub provenance; publish pilot aggregation atomically with
a hash-bound completion manifest; independently reconstruct and verify the run;
keep paid credentials behind a protected GitHub environment; make a non-skipped
required check and strict server-side branch rules the merge authority.

**Tech Stack:** Python 3.10-3.12, Pydantic v2, pytest, SWE-bench Verified,
Docker/OCI on GitHub-hosted Linux x86_64, GitHub Actions, GitHub Environments,
GitHub REST API / `gh` CLI.

## Global constraints

- Implement the approved design in
  `docs/superpowers/specs/2026-07-24-pilot-20-pr-eval-design.md` only after the
  final-gate telemetry/cancellation plan is green and independently reviewed.
- Preserve the user's unstaged `run_trace.py` modification byte-for-byte and
  never stage or commit it.
- Use red-green-refactor TDD. Every repository task begins with a focused failing
  test and ends with its own commit and independent review.
- Success comes only from official SWE-bench `status == "resolved"`; internal
  success, engineering score, and model confidence are diagnostic only.
- A valid `0/20` run passes when all 20 outcomes are determined and
  infrastructure failures are zero. Never add a score threshold after results
  are visible.
- Never re-run jobs, reuse prior instances, change the cohort, or tune prompts
  based on score. A repaired systemic failure starts a complete new run with a
  new authorization event and provenance.
- Never place model credentials in repository secrets, commands, logs, files,
  artifacts, workflow job-level environments, or tool-call output. When secret
  values are needed, pause for the owner to enter fresh values interactively.
- Do not authorize paid inference, configure merge rules, or merge until the
  preceding local and review gates are complete. Owner-approved exception
  `A,A` permits a feature-branch push after final-gate code findings are fixed
  and freshly reviewed, solely to obtain the existing Python 3.10/3.11/3.12 CI
  matrix evidence when no local Python 3.10 exists. If CI exposes a defect,
  reviewed fix follow-up pushes remain limited to that same evidence loop. This
  does not permit Pilot implementation, secrets, paid evaluation, any
  environment action, merge, prompt changes, or cohort changes. Pilot work
  remains blocked until the exact-head CI run has overall `success`, including
  all six Ubuntu and macOS Python 3.10/3.11/3.12 test jobs, `lint`, and
  `oci-integration`.
- Stage files explicitly for each commit; never use `git add -A` or `git add .`.

---

### Task 1: Freeze the exact pilot cohort and public mode

**Files:**

- Create: `eval/pilot_20_ids.txt`
- Modify: `eval/oci_contract.py`
- Modify: `eval/oci_runner.py`
- Modify: `eval/oci_aggregate.py`
- Modify: `tests/test_oci_contract.py`
- Modify: `tests/test_oci_runner.py`
- Modify: `tests/test_oci_aggregate.py`
- Modify: `tests/test_packaging_contract.py`

**Exact cohort:**

```text
pydata__xarray-7233
astropy__astropy-7336
scikit-learn__scikit-learn-12682
pytest-dev__pytest-10081
django__django-11815
mwaskom__seaborn-3069
psf__requests-1724
pallets__flask-5014
sympy__sympy-24562
pylint-dev__pylint-6903
django__django-12858
scikit-learn__scikit-learn-13142
astropy__astropy-13398
psf__requests-6028
matplotlib__matplotlib-23476
pylint-dev__pylint-7277
sphinx-doc__sphinx-7454
mwaskom__seaborn-3187
sympy__sympy-21379
pydata__xarray-6461
```

The exact selection SHA-256, including the final newline, is
`a5bd81ef063e951b589eb803dce3bafec3b89e84ba17b8c2df83c04459b93709`.

**Interfaces and behavior:**

- Extend `EvalMode` and `_MODE_FILES` with `pilot_20`.
- `load_mode_instance_ids("pilot_20")` must require exactly 20 unique IDs and
  exact ordered equality with `load_mode_instance_ids("baseline_50")[:20]`.
  Explicitly reject the baseline's twenty-first ID.
- Tests recompute and require the exact selection hash above; implementation
  must not hard-code a different serialized ordering.
- Extend runner and aggregate CLI choices to
  `checkpoint_5`, `pilot_20`, and `baseline_50` with no free-form mode.
- Keep legacy mode behavior unchanged; provenance and the release gate arrive in
  later tasks.
- Track the new file in packaging and ensure ignore rules cannot omit it.

- [ ] Write failing loader tests for exact equality, reorder, duplicate,
  substitution, extra row, too few rows, and admission of baseline row 21.
- [ ] Write failing runner/aggregate CLI tests that accept `pilot_20` and still
  reject unknown/retired modes.
- [ ] Extend packaging tests to require all three tracked selection files.
- [ ] Run
  `.venv/bin/python -m pytest tests/test_oci_contract.py tests/test_oci_runner.py tests/test_oci_aggregate.py tests/test_packaging_contract.py -q`
  and confirm failures are limited to the absent mode/file.
- [ ] Add the exact file and smallest mode/CLI extensions.
- [ ] Re-run the focused files and require them to pass.
- [ ] Commit only the eight task files with
  `feat(eval): freeze pilot 20 cohort`.

---

### Task 2: Bind every pilot runtime and instance artifact to workflow provenance

**Files:**

- Modify: `eval/oci_contract.py`
- Modify: `eval/oci_runner.py`
- Modify: `tests/test_oci_contract.py`
- Modify: `tests/test_oci_runner.py`
- Modify: `tests/test_oci_aggregate.py`

**New strict model:**

```python
LowerHexCommit = Annotated[
    str,
    Field(strict=True, min_length=40, max_length=40, pattern=r"^[0-9a-f]{40}$"),
]
EmptyOrLowerHexCommit = Annotated[
    str,
    Field(strict=True, max_length=40, pattern=r"^(?:|[0-9a-f]{40})$"),
]
BoundedWorkflowRef = Annotated[
    str,
    Field(strict=True, min_length=1, max_length=512),
]
StrictPositiveInt = Annotated[int, Field(strict=True, ge=1)]
StrictRunAttempt = Annotated[int, Field(strict=True, ge=1, le=1)]
StrictPRNumber = Annotated[int, Field(strict=True, ge=2, le=2)]


class WorkflowProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    repository: Literal["FMorgan-111/repopilot"]
    event_name: Literal["pull_request", "workflow_dispatch"]
    event_ref: BoundedWorkflowRef
    workflow_ref: BoundedWorkflowRef
    workflow_sha: LowerHexCommit
    run_id: StrictPositiveInt
    run_attempt: StrictRunAttempt
    candidate_sha: LowerHexCommit
    pr_number: StrictPRNumber | None
    pr_base_ref: Literal["master", ""]
    pr_base_sha: EmptyOrLowerHexCommit
    pr_head_repository: Literal["FMorgan-111/repopilot", ""]
    pr_head_ref: Literal["fix/release-readiness-20260717", ""]
    pr_head_sha: EmptyOrLowerHexCommit
    authorization_label: Literal["repopilot-pilot-20-approved", ""]
```

**Cross-field rules:**

- PR: exact merge ref `refs/pull/2/merge`; all PR fields populated;
  `candidate_sha == pr_head_sha`; base `master`; exact repository and branch;
  exact approval label; workflow ref exactly
  `FMorgan-111/repopilot/.github/workflows/swe-bench-oci-eval.yml@refs/pull/2/merge`.
- Dispatch: exact ref `refs/heads/master`; all PR fields empty/null;
  `candidate_sha == workflow_sha`; authorization label empty; workflow ref
  exactly
  `FMorgan-111/repopilot/.github/workflows/swe-bench-oci-eval.yml@refs/heads/master`.
- Bound all strings before comparing. The strict integer aliases reject bool,
  float, and string coercions for run ID, run attempt, and PR number; require
  lowercase 40-hex SHAs and `run_attempt == 1`.

**Runtime/package plumbing:**

- Add `provenance: WorkflowProvenance | None` to `RuntimeRecord` and
  `InstanceManifest`.
- Each model's own after-validator requires provenance for `pilot_20` and, when
  provenance is present, requires `commit_sha == provenance.candidate_sha`.
  Legacy modes may omit it. `package_instance()` parses the runtime, validates
  its commit/provenance, and copies that exact provenance into the manifest;
  callers cannot supply a second manifest value. The safe artifact bundle does
  not contain `runtime.json`, so aggregate compares manifest provenance only to
  its independently loaded expected provenance. A single-model validator does
  not attempt a cross-object comparison.
- Add `load_workflow_provenance(path: Path) -> WorkflowProvenance` in the
  contract module. It accepts at most 16 KiB of UTF-8 JSON from one no-follow
  regular file and is the only provenance file loader used by runner and
  aggregate CLIs.
- The loader opens with `O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK` where
  available, verifies one regular file with `fstat`, rejects `st_size > 16_384`,
  reads at most 16,385 bytes, rejects size/identity drift, decodes strict UTF-8,
  and validates one JSON object with no trailing second value.
- Add `--provenance-file PATH` to `oci_runner prepare`. Read a small regular
  no-follow JSON file, parse exactly one `WorkflowProvenance`, and pass it into
  `prepare_instance`. Pilot prepare fails before dataset/image work if it is
  absent, oversized, symlinked, malformed, or contradictory.
- `package_instance()` copies the validated runtime provenance into the
  manifest; callers cannot provide a second value.

- [ ] Add failing model tests for valid PR and dispatch shapes and one mutation
  of every field, including bool/float/string forms of all three integer fields,
  uppercase SHAs, wrong ref, missing PR field, populated dispatch PR field,
  candidate/head mismatch, and extra key.
- [ ] Add failing runtime/manifest tests proving provenance is mandatory only
  for pilot and copied exactly.
- [ ] Add prepare CLI/function tests for missing, symlinked, oversized,
  malformed, and valid provenance; prove invalid provenance fails before the
  dataset loader or OCI runner is called.
- [ ] Add package tests proving runtime provenance is copied exactly and cannot
  be overridden, plus aggregate tests proving manifest-to-expected provenance
  drift is rejected. Do not add `runtime.json` to the safe artifact set.
- [ ] Run
  `.venv/bin/python -m pytest tests/test_oci_contract.py tests/test_oci_runner.py tests/test_oci_aggregate.py -q`
  and confirm the new provenance assertions fail.
- [ ] Implement the strict model and single-source plumbing without accepting a
  caller-supplied manifest override.
- [ ] Re-run the focused tests and require them to pass.
- [ ] Commit only the five task files with
  `feat(eval): bind pilot artifacts to workflow`.

---

### Task 3: Gate and atomically publish one complete pilot aggregate

**Files:**

- Modify: `eval/oci_contract.py`
- Modify: `eval/oci_aggregate.py`
- Test: `tests/test_oci_contract.py`
- Test: `tests/test_oci_aggregate.py`

**New strict model:**

Add these exact aliases and models (all Pydantic integer fields also reject
bools in a before-validator):

```python
LowerHexSha256 = Annotated[
    str,
    Field(strict=True, min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"),
]
StrictPilotCount = Annotated[int, Field(strict=True, ge=0, le=20)]
StrictRate = Annotated[
    float,
    Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
]


class PilotRunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal[1]
    mode: Literal["pilot_20"]
    evaluated_commit: LowerHexCommit
    provenance: WorkflowProvenance
    instance_ids: list[Annotated[str, Field(strict=True, min_length=1, max_length=500)]] = Field(
        strict=True, min_length=20, max_length=20
    )
    selection_sha256: LowerHexSha256
    requested: Literal[20]
    determined: Literal[20]
    official_resolved: StrictPilotCount
    official_unresolved: StrictPilotCount
    empty_patch: StrictPilotCount
    infrastructure_failures: Literal[0]
    official_resolved_rate: StrictRate
    files: dict[
        Literal[
            "results.json",
            "predictions.jsonl",
            "official_results.json",
            "manifests.json",
            "summary.md",
        ],
        LowerHexSha256,
    ]

    @field_validator(
        "schema_version",
        "requested",
        "determined",
        "infrastructure_failures",
        mode="before",
    )
    @classmethod
    def _require_exact_int_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("pilot literal counters must be strict integers")
        return value

    @field_validator("official_resolved_rate", mode="before")
    @classmethod
    def _require_exact_float_type(cls, value: object) -> object:
        if type(value) is not float:
            raise ValueError("official resolved rate must be a strict float")
        return value


@dataclass(frozen=True)
class VerifiedBundle:
    manifest: InstanceManifest
    payload: VerifiedPayload


@dataclass(frozen=True)
class PilotGateMetrics:
    requested: int
    determined: int
    official_resolved: int
    official_unresolved: int
    empty_patch: int
    infrastructure_failures: int
    official_resolved_rate: float
    gate_passed: bool
```

The manifest hash map contains exactly:

```text
results.json
predictions.jsonl
official_results.json
manifests.json
summary.md
```

Its after-validator enforces:

- candidate SHA equals evaluated commit;
- IDs exactly equal the tracked cohort and are unique;
- selection hash equals SHA-256 of `"\n".join(ids) + "\n"`;
- `resolved + unresolved + empty_patch == determined == requested == 20`;
- `infrastructure_failures == 0`;
- finite rate has the exact same binary representation as
  `official_resolved / 20`, checked with
  `official_resolved_rate.hex() == (official_resolved / 20).hex()`;
- `files` contains exactly the five names above and lowercase SHA-256 values.

**Gate semantics:**

For each verified bundle compute one infrastructure boolean:

```python
instance_infra = (
    bundle.manifest.runtime_status != "ready"
    or bundle.payload.official.status == "scorer_infra"
    or bundle.payload.result.failure_class in INFRASTRUCTURE_FAILURE_CLASSES
)
```

Count `resolved`, `unresolved`, and `empty_patch` only from
`bundle.payload.official.status`.
Gate success requires exactly 20 determined outcomes and zero instances whose
single infrastructure boolean is true. Twenty empty patches must publish a
valid `0/20` marker; one empty patch plus infrastructure must fail.

**Atomic publication:**

- Preserve `aggregate_artifacts() -> Path` and its current signature/behavior
  for legacy modes. Add the exact pilot interface
  `aggregate_pilot_artifacts(artifacts_dir: Path, output_dir: Path, *,
  expected_commit: str, expected_provenance: WorkflowProvenance,
  repo_root: Path = REPO_ROOT) -> Path`; it returns the published
  `run_manifest.json` path.

- Promote these exact side-effect-free units inside the aggregate producer:

  - `verify_artifact_bundle(snapshot: _BundleSnapshot, *,
    expected_instance_id: str, expected_commit: str, mode: EvalMode,
    expected_provenance: WorkflowProvenance | None = None) -> VerifiedBundle`;
  - `compute_pilot_gate(ordered: Sequence[VerifiedBundle]) -> PilotGateMetrics`;
  - `render_aggregate_outputs(mode: EvalMode,
    ordered: Sequence[VerifiedBundle], expected_commit: str, *,
    include_manifests: bool) -> dict[str, bytes]`.

  These helpers consume already bounded snapshots and return values/bytes; they
  never create, rename, or delete output directories. The independent verifier
  in Task 4 must not import or call them.
- The pilot publisher requires independently loaded `expected_provenance` and
  rejects any manifest difference.
- Require an existing regular, non-symlink parent and an absent output path.
  Create a unique sibling staging directory with `tempfile.mkdtemp`, open each
  of the five files with exclusive creation, write tracked-order bytes, flush
  and `os.fsync` each descriptor before close, and verify the exact five-name
  set. Hash those closed files, exclusively create/write/fsync
  `run_manifest.json` last, verify the exact six-name set, fsync the staging
  directory descriptor, recheck that output is absent, call `os.rename` once,
  then fsync the parent directory descriptor.
- On gate failure, stale output, payload/marker write failure, or rename failure,
  publish no output directory and remove only the exact owned staging directory.
- Add `--expected-provenance-file` to the aggregate CLI; require it for pilot and
  reject it for legacy modes to catch operator mistakes. CLI `main()` dispatches
  pilot to `aggregate_pilot_artifacts()` and legacy modes to the unchanged
  `aggregate_artifacts()`.

- [ ] Add failing `PilotRunManifest` tests for every count/rate/hash/cohort/
  provenance mutation and for extra/missing fields.
- [ ] Add aggregate tests for a valid mixed 20-outcome run, a valid 20-empty
  `0/20` run, and each infrastructure source. Assert one infra instance counts
  once even when multiple flags are present.
- [ ] Add failure tests for missing/extra/duplicate/mixed-run bundles, wrong
  expected provenance, and cross-commit/cross-mode artifacts.
- [ ] Add atomicity tests for existing output, marker-write failure, earlier
  file-write failure, gate failure, rename failure, unexpected staging file, and
  success. Every failure must leave neither output nor an owned staging remnant.
- [ ] Assert successful output contains exactly the six approved files and that
  `manifests.json` preserves all 20 strict manifests in tracked order.
- [ ] Run
  `.venv/bin/python -m pytest tests/test_oci_contract.py tests/test_oci_aggregate.py -q`
  and confirm the new gate/atomicity tests fail.
- [ ] Implement the pilot-only gate and publication path without changing
  diagnostic aggregation for the other modes.
- [ ] Re-run the two focused files and require them to pass.
- [ ] Commit only the four task files with
  `feat(eval): publish atomic pilot gate`.

---

### Task 4: Independently verify aggregate and all 20 instance artifacts

**Files:**

- Create: `eval/oci_verify.py`
- Create: `tests/test_oci_verify.py`
- Modify: `tests/test_packaging_contract.py`

**Public interface:**

```bash
python -m eval.oci_verify \
  --aggregate-dir "$PILOT_DOWNLOAD_DIR/swe-bench-oci-pilot_20" \
  --instance-artifact-dir "$PILOT_DOWNLOAD_DIR/swe-bench-oci-instance-pydata__xarray-7233" \
  --instance-artifact-dir "$PILOT_DOWNLOAD_DIR/swe-bench-oci-instance-astropy__astropy-7336" \
  --instance-artifact-dir "$PILOT_DOWNLOAD_DIR/swe-bench-oci-instance-scikit-learn__scikit-learn-12682" \
  --instance-artifact-dir "$PILOT_DOWNLOAD_DIR/swe-bench-oci-instance-pytest-dev__pytest-10081" \
  --instance-artifact-dir "$PILOT_DOWNLOAD_DIR/swe-bench-oci-instance-django__django-11815" \
  --instance-artifact-dir "$PILOT_DOWNLOAD_DIR/swe-bench-oci-instance-mwaskom__seaborn-3069" \
  --instance-artifact-dir "$PILOT_DOWNLOAD_DIR/swe-bench-oci-instance-psf__requests-1724" \
  --instance-artifact-dir "$PILOT_DOWNLOAD_DIR/swe-bench-oci-instance-pallets__flask-5014" \
  --instance-artifact-dir "$PILOT_DOWNLOAD_DIR/swe-bench-oci-instance-sympy__sympy-24562" \
  --instance-artifact-dir "$PILOT_DOWNLOAD_DIR/swe-bench-oci-instance-pylint-dev__pylint-6903" \
  --instance-artifact-dir "$PILOT_DOWNLOAD_DIR/swe-bench-oci-instance-django__django-12858" \
  --instance-artifact-dir "$PILOT_DOWNLOAD_DIR/swe-bench-oci-instance-scikit-learn__scikit-learn-13142" \
  --instance-artifact-dir "$PILOT_DOWNLOAD_DIR/swe-bench-oci-instance-astropy__astropy-13398" \
  --instance-artifact-dir "$PILOT_DOWNLOAD_DIR/swe-bench-oci-instance-psf__requests-6028" \
  --instance-artifact-dir "$PILOT_DOWNLOAD_DIR/swe-bench-oci-instance-matplotlib__matplotlib-23476" \
  --instance-artifact-dir "$PILOT_DOWNLOAD_DIR/swe-bench-oci-instance-pylint-dev__pylint-7277" \
  --instance-artifact-dir "$PILOT_DOWNLOAD_DIR/swe-bench-oci-instance-sphinx-doc__sphinx-7454" \
  --instance-artifact-dir "$PILOT_DOWNLOAD_DIR/swe-bench-oci-instance-mwaskom__seaborn-3187" \
  --instance-artifact-dir "$PILOT_DOWNLOAD_DIR/swe-bench-oci-instance-sympy__sympy-21379" \
  --instance-artifact-dir "$PILOT_DOWNLOAD_DIR/swe-bench-oci-instance-pydata__xarray-6461" \
  --expected-commit "$PILOT_EXPECTED_HEAD" \
  --expected-run-id "$PILOT_RUN_ID"
```

The Python interface is exactly
`verify_pilot_artifacts(aggregate_dir: Path,
instance_artifact_dirs: Sequence[Path], *, expected_commit: str,
expected_run_id: int, repo_root: Path = REPO_ROOT) -> PilotRunManifest`.

**Verification behavior:**

- Read only; never repair, rewrite, or delete downloaded evidence.
- At entry require `type(expected_run_id) is int and expected_run_id >= 1`, a
  lowercase 40-hex expected commit, and exactly 20 directory arguments before
  opening any artifact.
- Open aggregate and instance roots as exact directories. Reject symlink roots,
  symlink files, hard-link/identity replacement, FIFOs/devices, oversized
  files, cumulative byte overflow, missing/extra files, and path escape.
- Parse all 20 instance manifests and safe payloads independently from the
  producer's aggregate function. Recompute payload hashes, runtime/result/
  prediction/official consistency, fixed order, candidate, full provenance,
  official counts, infrastructure booleans, selection hash, and gate result.
- Parse `run_manifest.json`, hash each of the five aggregate files, compare all
  aggregate payloads byte/record-wise to the recomputed instance projection,
  and require its run ID and commit to equal the two command arguments.
- Do not trust `summary.md`; recompute the canonical expected summary and compare
  it to the file and manifest hash.
- Exit 0 only for one exact, gate-passing bundle. Emit bounded class/reason text
  on failure; never print payload bodies, environment data, or credentials.
- The verifier may reuse only strict Pydantic models, fixed filenames/limits,
  and cryptographic hash constants. It owns separate no-follow snapshot,
  cross-file consistency, gate, summary, and canonical output-rendering code; it
  must not import/call either publisher or the producer's bundle/gate/render
  helpers, and it never trusts producer counts or hashes. The CLI requires
  exactly 20 repeatable instance-directory arguments; accepting an ambient root
  with extra undisclosed artifacts is not allowed.

- [ ] Build one valid 20-bundle fixture and prove verification succeeds without
  changing any mtime/content.
- [ ] Add a parameterized mutation matrix for every instance file, aggregate
  file, manifest hash, schema field, count, rate, order, selection, commit,
  run ID, attempt, workflow SHA/ref, base, PR/head, and verdict.
- [ ] Add filesystem attack tests for root/file symlinks, FIFO, extra file,
  replacement during read, oversized file, and cumulative overflow.
- [ ] Add a test monkeypatching the producer aggregator to prove the verifier
  never calls it. Also monkeypatch the producer's bundle, gate, and render
  helpers to return forged success/bytes and prove the independent verifier still
  rejects mutated evidence.
- [ ] Run
  `.venv/bin/python -m pytest tests/test_oci_verify.py tests/test_packaging_contract.py -q`
  and confirm the module/contract tests fail before implementation.
- [ ] Implement the smallest independent bounded reader and recomputation path.
- [ ] Re-run the focused files and require them to pass.
- [ ] Commit only the three task files with
  `feat(eval): independently verify pilot artifacts`.

---

### Task 5: Add fail-closed workflow authorization and immutable checkout

**Files:**

- Modify: `.github/workflows/swe-bench-oci-eval.yml`
- Modify: `tests/test_swe_bench_oci_workflow.py`

**Trigger and authorization contract:**

- Triggers are exactly manual dispatch and:

  ```yaml
  pull_request:
    types: [labeled]
    branches: [master]
  ```

  Never add `pull_request_target`, push, schedule, issue-comment, or arbitrary
  repository-dispatch triggers.
- Manual choices are exactly `checkpoint_5`, `pilot_20`, `baseline_50`.
- Add the first `authorization` job with `permissions: {}`, no checkout, no
  action other than already-pinned platform actions if unavoidable, no secrets,
  and no repository code. Its validation step ID is `authorize`; the job exposes
  exactly four outputs: `mode`, lowercase 40-hex `candidate_sha`, one-line
  `provenance_json`, and lowercase 64-hex `provenance_sha256`.
- Canonical provenance bytes are UTF-8 of
  `json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))`
  with no trailing newline. The authorization step hashes those exact bytes,
  rejects newline/control characters and output size above 16 KiB, and appends
  the four allowlisted one-line values to `GITHUB_OUTPUT` only after every guard
  passes.
- PR admission requires every exact repository/event/ref/PR/open/label/author/
  actor/triggering-actor/base/head/branch/candidate-variable/attempt predicate
  from the approved spec. PR mode is forced to `pilot_20`.
- Dispatch admission requires the trusted owner identities, attempt 1, exact
  branch ref `refs/heads/master`, workflow ref/SHA from master, and an allowlisted
  mode. Candidate is `github.sha`; all PR provenance fields are empty.
- An absent/malformed candidate or provenance output is a hard failure, never an
  implicit checkout of the triggering ref.

**Execution identity:**

- `prepare`, `instance`, and `aggregate` depend on successful authorization,
  validate candidate format before checkout, use the pinned checkout action
  with `ref: ${{ needs.authorization.outputs.candidate_sha }}` and
  `persist-credentials: false`, then require `git rev-parse HEAD` to equal it.
- Job permissions are exact: authorization `{}`, prepare `contents: read`,
  instance `contents: read` plus `pull-requests: read`, aggregate the same, and
  the final gate `{}`.
- Set root workflow permissions to `{}`; each read-bearing job declares only its
  own exact permissions.
- Preserve immutable action pins, locked installs, fixed public dataset cache,
  max-parallel 2, and non-canceling serialized runs.
- Final job order is exactly `authorization`, `prepare`, `instance`, `aggregate`,
  `pilot_release_gate`. The existing audited action counts stay checkout/setup
  three each, cache one, upload two, and download one; no new action is needed.

- [ ] Replace the existing manual-only test with strict trigger and manual-choice
  assertions plus mutations adding every forbidden trigger.
- [ ] Add static contract helpers and mutation tests for every PR/dispatch guard,
  exact authorization step/job outputs and canonical digest, no checkout/code/
  secrets in authorization, exact job dependencies/permissions, explicit
  candidate checkout, and HEAD equality.
- [ ] Add tests proving authorization failure cannot satisfy a checkout or
  aggregate condition and empty outputs cannot become fallback refs.
- [ ] Run
  `.venv/bin/python -m pytest tests/test_swe_bench_oci_workflow.py -q`
  and confirm the new authorization tests fail against the manual workflow.
- [ ] Implement the no-checkout authorization job and candidate-bound execution
  skeleton while retaining all existing safe installation/action constraints.
- [ ] Re-run the workflow tests and require them to pass.
- [ ] Commit only the two task files with
  `ci(eval): authorize immutable pilot candidate`.

---

### Task 6: Protect inference and publication with live admission checks

**Files:**

- Modify: `.github/workflows/swe-bench-oci-eval.yml`
- Modify: `tests/test_swe_bench_oci_workflow.py`

**Secret and environment boundary:**

- Attach only the secret-bearing matrix `instance` job to the exact environment
  `repopilot-expensive-eval`.
- Reference `secrets.LLM_API_KEY` and `secrets.LLM_ESCALATION_API_KEY` only in
  the `Generate patch` step's `env`. No other step/job receives them.
- Repeat exact owner, triggering actor, and `run_attempt == 1` guards at both the
  job and generation step. A GitHub re-run must fail before environment access.
- Write canonical provenance from the authorization output into a bounded file
  with platform Python, then pass it to `oci_runner prepare`. Repository code
  cannot choose or rewrite the expected provenance.

**Live admission:**

- Immediately before generation, use the read-only GitHub token/API to fetch PR
  2 and require open state, author, base ref/SHA, head repository/ref/SHA, and
  continued exact approval label. Compare every value with authorization
  provenance; fail before the model-secret step on any difference.
- Immediately before pilot aggregation, repeat the complete PR check. For a
  dispatch, require live `master` to still equal candidate.
- Aggregate runs only under the approved `always() && authorization succeeded &&
  prepare succeeded && valid candidate && attempt 1 && trusted triggering actor`
  condition. An authorized aggregate may diagnose failed matrix artifacts, but
  cannot publish a marker without all 20 valid bundles.
- Only score/package/upload steps inside an already authorized job use
  unconditional `always()`.
- Branch aggregate execution explicitly on the authorized mode. For
  `pilot_20`, pass expected provenance, call `aggregate_pilot_artifacts()`, run
  `eval.oci_verify` against the just-published aggregate and all downloaded
  per-instance artifacts, publish the six outputs, then upload. For
  `checkpoint_5` and `baseline_50`, call the unchanged
  `aggregate_artifacts()`, perform the existing diagnostic upload, never invoke
  the pilot verifier/publisher, and leave all six aggregate job outputs empty.
- Give the independent verifier step ID `verify_pilot`. The following standard-
  library-only step has ID `publish_gate_outputs` and runs only after
  `verify_pilot` succeeds. It strictly parses `run_manifest.json`, recomputes the
  canonical provenance bytes/digest and exact file SHA-256, validates all output
  formats, then writes exactly:

  ```text
  marker_verified=true
  marker_mode=pilot_20
  marker_candidate_sha=<40 lowercase hex from evaluated_commit>
  marker_run_id=<strict positive decimal from provenance.run_id>
  marker_provenance_sha256=<64 lowercase hex>
  marker_sha256=<64 lowercase hex of exact run_manifest.json bytes>
  ```

- The aggregate job exposes those same six names from
  `steps.publish_gate_outputs.outputs`. If aggregation, independent verification,
  strict parsing, or output publication fails, the step does not emit a complete
  set and the final gate must fail.

- [ ] Add static and mutation tests for exact environment name, no repository
  secret fallback, generation-only model secret references, repeated attempt/
  actor guards, provenance-file construction, and both live admission checks.
- [ ] Mutate/remove each live field check independently and require the static
  contract to fail.
- [ ] Test aggregate conditions, ordering (live check before aggregation;
  aggregation before independent verification; verification before exact
  gate-output publication; output publication before upload), the six exact
  output formats/digests, and absence of `always()` at unauthorized boundaries.
- [ ] Add a three-mode truth table: only `pilot_20` can reach expected-
  provenance loading, atomic publication, `verify_pilot`, or
  `publish_gate_outputs`; both legacy modes still aggregate/upload and expose
  six empty marker outputs.
- [ ] Run
  `.venv/bin/python -m pytest tests/test_swe_bench_oci_workflow.py -q`
  and confirm the new environment/live-check tests fail.
- [ ] Implement the secret boundary, provenance plumbing, full API comparisons,
  aggregate verifier, and bounded outputs.
- [ ] Re-run the workflow tests and require them to pass.
- [ ] Commit only the two task files with
  `ci(eval): protect pilot inference and publication`.

---

### Task 7: Make `pilot_release_gate` non-skippable for PR-label runs

**Files:**

- Modify: `.github/workflows/swe-bench-oci-eval.yml`
- Modify: `tests/test_swe_bench_oci_workflow.py`

**Interfaces and behavior:**

- Add a final job whose job/check name is exactly `pilot_release_gate`.
- Its job condition contains `always()` and allows it to start for every
  `pull_request/labeled` run, including the wrong label and failed/cancelled/
  skipped dependencies. Manual workflows may omit/skip this PR-only gate.
- Give it `permissions: {}`, no environment, no checkout, no repository code,
  no artifacts, and no secrets.
- The job explicitly exits nonzero unless all are true: authorized PR event;
  exact label/mode/attempt/owner identities; authorization, prepare, instance,
  and aggregate results all `success`; aggregate reports a published and
  independently verified marker; marker candidate/run/provenance equals the
  authorization outputs.
- Concretely compare
  `marker_verified == "true"`, `marker_mode == "pilot_20" == authorization.mode`,
  `marker_candidate_sha == authorization.candidate_sha`, decimal
  `marker_run_id == github.run_id`, and
  `marker_provenance_sha256 == authorization.provenance_sha256`; also require
  `marker_sha256` to match exactly 64 lowercase hex. Empty or partial job outputs
  fail these checks. The job does not parse an artifact or call repository code.
- Never encode authorization in a job-level condition that can mark the required
  PR check `skipped`. Wrong labels and dependency failure must produce a failing
  check conclusion.

- [ ] Add a workflow test proving the job exists once with exact name,
  `permissions: {}`, `always()`, PR-only start behavior, and no checkout/code/
  secrets/environment.
- [ ] Add mutation tests for wrong-label, failed/skipped/cancelled dependency,
  attempt, actor, mode, marker, candidate, run ID, and provenance checks.
- [ ] Mutate each of the six aggregate outputs to empty, malformed, or mismatched
  and prove the truth table returns failure.
- [ ] Add a semantic truth-table helper: the only success row is a fully admitted
  PR pilot with all dependencies successful and matching verified marker.
- [ ] Run
  `.venv/bin/python -m pytest tests/test_swe_bench_oci_workflow.py -q`
  and confirm the final-gate tests fail before implementation.
- [ ] Implement the smallest no-checkout final job and exact failure checks.
- [ ] Re-run the workflow tests and require them to pass.
- [ ] Commit only the two task files with
  `ci(eval): add non-skippable pilot release gate`.

---

### Task 8: Document the operator contract and run a clean release gate

**Files:**

- Modify: `README.md`
- Modify: `tests/test_documentation_contract.py`
- Verify: `tests/test_swe_bench_oci_workflow.py`
- Verify: `tests/test_packaging_contract.py`

**Documentation:**

- Replace checkpoint/baseline-first release instructions with the approved
  fixed pilot flow while retaining legacy modes as diagnostics.
- Document the exact cohort origin, official `R/20`, valid determined statuses,
  zero-infrastructure gate, soft token estimate semantics, no reruns, protected
  environment, external candidate variable, label authorization, six aggregate
  files, independent verifier command, and exact merge provenance.
- State that missing environment/ruleset capabilities block the paid run/merge
  and trigger the already-approved trusted-dispatcher fallback design; repository
  secrets are never a fallback.

**Verification:**

- [ ] Run every Task 8 command fence, in order, inside one newly allocated PTY
  shell. Initialize fail-fast mode and an exact private handoff directory first;
  any nonzero command ends the task and no later fence may run:

  ```bash
  set -euo pipefail
  umask 077
  PILOT_RELEASE_ROOT=/private/tmp/repopilot-release-20260724
  test ! -e "$PILOT_RELEASE_ROOT"
  mkdir -m 700 "$PILOT_RELEASE_ROOT"
  PILOT_STATE_FILE="$PILOT_RELEASE_ROOT/state.json"
  ```

- [ ] Add/extend README assertions in `tests/test_documentation_contract.py`
  before editing prose; keep YAML structure assertions in the workflow test.
- [ ] Run the complete affected evaluation suite:

  ```bash
  .venv/bin/python -m pytest \
    tests/test_oci_contract.py tests/test_oci_runner.py \
    tests/test_oci_aggregate.py tests/test_oci_verify.py \
    tests/test_swe_bench_oci_workflow.py tests/test_packaging_contract.py \
    tests/test_documentation_contract.py -q
  ```

- [ ] Use these exact local identities and a private temporary release root.
  The explicit fallback Git binary is required on this workstation because
  `/usr/bin/git` is blocked by the missing Xcode license helper:

  ```bash
  PILOT_GIT=/Users/morgan/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/fallback/git
  PILOT_PY311=/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11
  PILOT_PY312=/Users/morgan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3.12
  PILOT_WORKTREE_ROOT="$PWD"
  test "$("$PILOT_GIT" rev-parse --show-toplevel)" = "$PILOT_WORKTREE_ROOT"
  CONDA_PKGS_DIRS="$PILOT_RELEASE_ROOT/conda-pkgs" \
    /Users/morgan/anaconda3/bin/conda create --yes \
    --prefix "$PILOT_RELEASE_ROOT/python-3.10" python=3.10 pip
  PILOT_PY310="$PILOT_RELEASE_ROOT/python-3.10/bin/python"
  test "$("$PILOT_PY310" -c 'import sys; print("%d.%d" % sys.version_info[:2])')" = 3.10
  test "$("$PILOT_PY311" -c 'import sys; print("%d.%d" % sys.version_info[:2])')" = 3.11
  test "$("$PILOT_PY312" -c 'import sys; print("%d.%d" % sys.version_info[:2])')" = 3.12
  ```

- [ ] Run the complete project suite independently on Python 3.10, 3.11, and
  3.12. Each interpreter receives a new environment and installs the declared
  project extras; require zero failures and record the exact pass/skip counts:

  ```bash
  for PILOT_PY in "$PILOT_PY310" "$PILOT_PY311" "$PILOT_PY312"; do
    PILOT_TAG="$("$PILOT_PY" -c 'import sys; print("%d%d" % sys.version_info[:2])')"
    "$PILOT_PY" -m venv "$PILOT_RELEASE_ROOT/venv-$PILOT_TAG"
    "$PILOT_RELEASE_ROOT/venv-$PILOT_TAG/bin/python" -m pip install \
      --disable-pip-version-check -e '.[memory,dev]'
    "$PILOT_RELEASE_ROOT/venv-$PILOT_TAG/bin/python" -m pytest -q
  done
  ```

- [ ] Run the primary local full suite, Ruff with the repository's CI flags,
  and patch checks against the committed candidate:

  ```bash
  .venv/bin/python -m pytest -q
  .venv/bin/python -m ruff check src tests eval --select=E,F,I --ignore=E501
  "$PILOT_GIT" diff --check origin/master...HEAD
  "$PILOT_GIT" diff --cached --check
  ```

- [ ] Scan every reachable commit without printing a matching value. The only
  permitted output on failure is `commit:path:line`; an empty result is the
  passing result:

  ```bash
  PILOT_CREDENTIAL_PATTERN='sk-[A-Za-z0-9_-]{40,}|gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}'
  PILOT_CREDENTIAL_HITS="$(
    "$PILOT_GIT" rev-list --all |
      while IFS= read -r PILOT_COMMIT; do
        "$PILOT_GIT" grep -n -I -E "$PILOT_CREDENTIAL_PATTERN" \
          "$PILOT_COMMIT" -- . 2>/dev/null || test $? -eq 1
      done |
      awk -F: '{print $1 ":" $2 ":" $3}' |
      sort -u
  )"
  if test -n "$PILOT_CREDENTIAL_HITS"; then
    printf '%s\n' "$PILOT_CREDENTIAL_HITS"
    exit 1
  fi
  ```

- [ ] Export only committed `HEAD`, install it from the tracked evaluation
  lock plus declared test/build tools, prove imports resolve inside the archive,
  run the full suite and contracts, then build and inspect exactly one sdist and
  wheel. Retain the temporary root until review evidence has been recorded:

  ```bash
  PILOT_ARCHIVE_DIR="$PILOT_RELEASE_ROOT/archive"
  mkdir "$PILOT_ARCHIVE_DIR"
  "$PILOT_GIT" archive --format=tar HEAD | tar -xf - -C "$PILOT_ARCHIVE_DIR"
  "$PILOT_PY311" -m venv "$PILOT_RELEASE_ROOT/archive-venv"
  "$PILOT_RELEASE_ROOT/archive-venv/bin/python" -m pip install \
    --disable-pip-version-check --require-hashes \
    -r "$PILOT_ARCHIVE_DIR/requirements-eval.lock"
  "$PILOT_RELEASE_ROOT/archive-venv/bin/python" -m pip install \
    --disable-pip-version-check 'pytest>=9,<10' 'pytest-asyncio>=1,<2' \
    'ruff>=0.12,<1' 'build>=1,<2'
  (
    cd "$PILOT_ARCHIVE_DIR"
    "$PILOT_RELEASE_ROOT/archive-venv/bin/python" -m pip install \
      --no-deps --no-build-isolation -e .
    "$PILOT_RELEASE_ROOT/archive-venv/bin/python" -c \
      'from pathlib import Path; import src; assert Path(src.__file__).resolve().is_relative_to(Path.cwd().resolve())'
    "$PILOT_RELEASE_ROOT/archive-venv/bin/python" -m pytest -q
    "$PILOT_RELEASE_ROOT/archive-venv/bin/python" -m pytest \
      tests/test_swe_bench_oci_workflow.py tests/test_packaging_contract.py -q
    "$PILOT_RELEASE_ROOT/archive-venv/bin/python" -m ruff check \
      src tests eval --select=E,F,I --ignore=E501
    "$PILOT_RELEASE_ROOT/archive-venv/bin/python" -m build --no-isolation
    test "$(find dist -maxdepth 1 -type f -name '*.tar.gz' | wc -l | tr -d ' ')" = 1
    test "$(find dist -maxdepth 1 -type f -name '*.whl' | wc -l | tr -d ' ')" = 1
    PILOT_SDIST="$(find dist -maxdepth 1 -type f -name '*.tar.gz')"
    PILOT_WHEEL="$(find dist -maxdepth 1 -type f -name '*.whl')"
    tar -tzf "$PILOT_SDIST" | sort
    unzip -l "$PILOT_WHEEL"
  )
  cd "$PILOT_WORKTREE_ROOT"
  ```

- [ ] Persist only bounded, non-secret handoff paths for fresh Tasks 9–11. Use
  an atomic same-directory rename and mode `0600`; never put credentials,
  environment values, model output, or GitHub tokens in this file:

  ```bash
  jq -n \
    --arg release_root "$PILOT_RELEASE_ROOT" \
    --arg worktree_root "$PILOT_WORKTREE_ROOT" \
    --arg git "$PILOT_GIT" \
    '{
      schema_version: 1,
      release_root: $release_root,
      worktree_root: $worktree_root,
      git: $git
    }' > "$PILOT_RELEASE_ROOT/state.next"
  chmod 600 "$PILOT_RELEASE_ROOT/state.next"
  mv "$PILOT_RELEASE_ROOT/state.next" "$PILOT_STATE_FILE"
  test "$(stat -f '%Lp' "$PILOT_STATE_FILE")" = 600
  ```

- [ ] Obtain a fresh independent whole-branch review from merge base through
  `HEAD`. Fix and re-review every Critical or Important finding.
- [ ] Commit only `README.md` and `tests/test_documentation_contract.py` with
  `docs(eval): document pilot 20 release gate`, plus separate TDD commits for any
  real verification fix.
- [ ] Confirm final worktree status contains only the user's existing
  `run_trace.py` modification before any push.

---

### Task 9: Push and configure the protected GitHub environment

**External state:** GitHub repository `FMorgan-111/repopilot`. No repository
files change in this task.

- [ ] Run every Task 9 fence in one fresh PTY shell. Re-establish the bounded
  handoff, fail-fast behavior, and exact worktree without trusting inherited
  shell variables:

  ```bash
  set -euo pipefail
  umask 077
  PILOT_RELEASE_ROOT=/private/tmp/repopilot-release-20260724
  PILOT_STATE_FILE="$PILOT_RELEASE_ROOT/state.json"
  PILOT_WORKTREE_ROOT=/Users/morgan/Documents/Codex/2026-07-17/https-github-com-fmorgan-111-repopilot/work/repopilot/.worktrees/gemini-default-model
  PILOT_GIT=/Users/morgan/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/fallback/git
  test -d "$PILOT_RELEASE_ROOT"
  test ! -L "$PILOT_RELEASE_ROOT"
  test -f "$PILOT_STATE_FILE"
  test ! -L "$PILOT_STATE_FILE"
  test "$(stat -f '%l' "$PILOT_STATE_FILE")" = 1
  test "$(stat -f '%u' "$PILOT_STATE_FILE")" = "$(id -u)"
  test "$(stat -f '%Lp' "$PILOT_STATE_FILE")" = 600
  test "$(stat -f '%z' "$PILOT_STATE_FILE")" -le 4096
  jq -e --arg root "$PILOT_RELEASE_ROOT" --arg worktree "$PILOT_WORKTREE_ROOT" \
    --arg git "$PILOT_GIT" '
      .schema_version == 1 and
      .release_root == $root and
      .worktree_root == $worktree and
      .git == $git and
      (keys | sort) == ["git", "release_root", "schema_version", "worktree_root"]
    ' "$PILOT_STATE_FILE"
  cd "$PILOT_WORKTREE_ROOT"
  ```

- [ ] Establish the exact repository, candidate, and API version, and require
  that the index is clean and `run_trace.py` is the only worktree difference:

  ```bash
  PILOT_REPO=FMorgan-111/repopilot
  PILOT_BRANCH=fix/release-readiness-20260717
  PILOT_API_VERSION=2026-03-10
  gh auth status
  test "$(gh repo view "$PILOT_REPO" --json nameWithOwner --jq .nameWithOwner)" = "$PILOT_REPO"
  test "$(gh repo view "$PILOT_REPO" --json defaultBranchRef --jq .defaultBranchRef.name)" = master
  PILOT_EXPECTED_HEAD="$("$PILOT_GIT" rev-parse HEAD)"
  test "$(printf '%s' "$PILOT_EXPECTED_HEAD" | wc -c | tr -d ' ')" = 40
  "$PILOT_GIT" diff --cached --quiet
  "$PILOT_GIT" diff --quiet HEAD -- . ':(exclude)run_trace.py'
  test -z "$("$PILOT_GIT" ls-files --others --exclude-standard)"
  test "$("$PILOT_GIT" status --porcelain --untracked-files=all)" = ' M run_trace.py'
  test "$(gh pr view 2 --repo "$PILOT_REPO" --json state --jq .state)" = OPEN
  test "$(gh pr view 2 --repo "$PILOT_REPO" --json baseRefName --jq .baseRefName)" = master
  test "$(gh pr view 2 --repo "$PILOT_REPO" --json headRefName --jq .headRefName)" = "$PILOT_BRANCH"
  test "$(gh pr view 2 --repo "$PILOT_REPO" --json headRepositoryOwner \
    --jq .headRepositoryOwner.login)" = FMorgan-111
  jq --arg head "$PILOT_EXPECTED_HEAD" '. + {candidate_sha: $head}' \
    "$PILOT_STATE_FILE" > "$PILOT_RELEASE_ROOT/state.next"
  chmod 600 "$PILOT_RELEASE_ROOT/state.next"
  mv "$PILOT_RELEASE_ROOT/state.next" "$PILOT_STATE_FILE"
  ```

- [ ] Create/update the environment with one exact owner reviewer, self-review
  permitted for this owner-triggered run, and custom branch policies. Use only
  fields documented by the environment REST schema; never send an unsupported
  admin-bypass field:

  ```bash
  PILOT_OWNER_ID="$(gh api -H "X-GitHub-Api-Version: $PILOT_API_VERSION" \
    users/FMorgan-111 --jq .id)"
  jq -n --argjson owner_id "$PILOT_OWNER_ID" '{
    wait_timer: 0,
    prevent_self_review: false,
    reviewers: [{type: "User", id: $owner_id}],
    deployment_branch_policy: {
      protected_branches: false,
      custom_branch_policies: true
    }
  }' > "$PILOT_RELEASE_ROOT/environment.json"
  gh api --method PUT -H "X-GitHub-Api-Version: $PILOT_API_VERSION" \
    repos/FMorgan-111/repopilot/environments/repopilot-expensive-eval \
    --input "$PILOT_RELEASE_ROOT/environment.json"
  gh api -H "X-GitHub-Api-Version: $PILOT_API_VERSION" \
    repos/FMorgan-111/repopilot/environments/repopilot-expensive-eval \
    > "$PILOT_RELEASE_ROOT/environment-readback.json"
  jq -e --argjson owner_id "$PILOT_OWNER_ID" '
    .name == "repopilot-expensive-eval" and
    .deployment_branch_policy == {
      protected_branches:false, custom_branch_policies:true
    } and
    ([.protection_rules[] | select(.type == "required_reviewers") |
      .prevent_self_review] == [false]) and
    ([.protection_rules[] | select(.type == "required_reviewers") |
      .reviewers[].reviewer.id] == [$owner_id])
  ' "$PILOT_RELEASE_ROOT/environment-readback.json"
  ```

- [ ] Apply the approved admin-bypass capability gate before writing any secret
  or pushing the candidate. The documented REST update schema has no setter for
  this control. Continue Approach A only if the authoritative environment
  readback nevertheless exposes `can_admins_bypass` as the strict boolean
  `false`, proving that the repository already has the required external
  setting. If the field is missing or true, stop here and write/approve the full
  trusted-default-branch dispatcher spec and plan (Approach B); do not prompt
  for secrets, push, set a candidate variable, or add the label:

  ```bash
  jq -e 'has("can_admins_bypass") and (.can_admins_bypass | type) == "boolean" and
    .can_admins_bypass == false' "$PILOT_RELEASE_ROOT/environment-readback.json"
  ```

- [ ] Create only the exact deployment branch policies and read back the exact
  set. If pre-existing policies add any third pattern, stop for owner review;
  do not delete or broaden external policy silently:

  ```bash
  gh api -H "X-GitHub-Api-Version: $PILOT_API_VERSION" \
    repos/FMorgan-111/repopilot/environments/repopilot-expensive-eval/deployment-branch-policies \
    > "$PILOT_RELEASE_ROOT/environment-policies-before.json"
  if ! jq -e '.branch_policies | any(.name == "master")' \
    "$PILOT_RELEASE_ROOT/environment-policies-before.json" >/dev/null; then
    gh api --method POST -H "X-GitHub-Api-Version: $PILOT_API_VERSION" \
      repos/FMorgan-111/repopilot/environments/repopilot-expensive-eval/deployment-branch-policies \
      -f name=master -f type=branch
  fi
  if ! jq -e '.branch_policies | any(.name == "refs/pull/2/merge")' \
    "$PILOT_RELEASE_ROOT/environment-policies-before.json" >/dev/null; then
    gh api --method POST -H "X-GitHub-Api-Version: $PILOT_API_VERSION" \
      repos/FMorgan-111/repopilot/environments/repopilot-expensive-eval/deployment-branch-policies \
      -f name=refs/pull/2/merge -f type=branch
  fi
  gh api -H "X-GitHub-Api-Version: $PILOT_API_VERSION" \
    repos/FMorgan-111/repopilot/environments/repopilot-expensive-eval/deployment-branch-policies \
    > "$PILOT_RELEASE_ROOT/environment-policies.json"
  jq -e '
    .total_count == 2 and
    ([.branch_policies[].name] | sort) == ["master", "refs/pull/2/merge"]
  ' "$PILOT_RELEASE_ROOT/environment-policies.json"
  ```

  A `422` or inability to read back the exact merge ref also stops before
  secrets and requires the complete Approach B design/plan. It is never
  replaced by `refs/pull/*/merge`, a branch wildcard, or repository secrets.

- [ ] Ask the owner to enter fresh values interactively with:

  ```bash
  gh secret set LLM_API_KEY --env repopilot-expensive-eval --repo FMorgan-111/repopilot
  gh secret set LLM_ESCALATION_API_KEY --env repopilot-expensive-eval --repo FMorgan-111/repopilot
  ```

  Never paste, read, echo, log, or recover their values.
- [ ] Verify exact environment secret names, remove only same-named repository
  copies after that proof, and verify repository scope is clear. These commands
  expose names only:

  ```bash
  gh secret list --env repopilot-expensive-eval --repo "$PILOT_REPO" \
    --json name --jq 'map(.name) | sort' \
    > "$PILOT_RELEASE_ROOT/environment-secret-names.json"
  jq -e '. == ["LLM_API_KEY", "LLM_ESCALATION_API_KEY"]' \
    "$PILOT_RELEASE_ROOT/environment-secret-names.json"
  gh secret list --repo "$PILOT_REPO" --json name \
    > "$PILOT_RELEASE_ROOT/repository-secret-names-before.json"
  for PILOT_SECRET_NAME in LLM_API_KEY LLM_ESCALATION_API_KEY; do
    if jq -e --arg name "$PILOT_SECRET_NAME" \
      'any(.[]; .name == $name)' \
      "$PILOT_RELEASE_ROOT/repository-secret-names-before.json" >/dev/null; then
      gh secret delete "$PILOT_SECRET_NAME" --repo "$PILOT_REPO"
    fi
  done
  gh secret list --repo "$PILOT_REPO" --json name \
    > "$PILOT_RELEASE_ROOT/repository-secret-names-after.json"
  jq -e '[.[] | select(
    .name == "LLM_API_KEY" or .name == "LLM_ESCALATION_API_KEY"
  )] == []' "$PILOT_RELEASE_ROOT/repository-secret-names-after.json"
  ```

- [ ] Only after the environment, deployment refs, secret names, admin-bypass
  gate, and repository-secret removal are proven, push without force. Poll for
  at most 50 seconds for one synchronized ordinary CI run; zero or multiple
  matches abort without triggering another run. Validate candidate identity via
  the run's PR projection rather than the synthetic merge `head_sha`:

  ```bash
  PILOT_PUSH_NOT_BEFORE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$PILOT_GIT" push origin HEAD:refs/heads/fix/release-readiness-20260717
  PILOT_REMOTE_HEAD="$(gh api -H "X-GitHub-Api-Version: $PILOT_API_VERSION" \
    repos/FMorgan-111/repopilot/git/ref/heads/fix/release-readiness-20260717 \
    --jq .object.sha)"
  test "$PILOT_REMOTE_HEAD" = "$PILOT_EXPECTED_HEAD"
  test "$(gh pr view 2 --repo "$PILOT_REPO" --json state --jq .state)" = OPEN
  test "$(gh pr view 2 --repo "$PILOT_REPO" --json baseRefName --jq .baseRefName)" = master
  test "$(gh pr view 2 --repo "$PILOT_REPO" --json headRefName --jq .headRefName)" = "$PILOT_BRANCH"
  test "$(gh pr view 2 --repo "$PILOT_REPO" --json headRefOid --jq .headRefOid)" = "$PILOT_EXPECTED_HEAD"
  PILOT_CI_RUN_ID=''
  for PILOT_POLL in 1 2 3 4 5; do
    gh run list --repo "$PILOT_REPO" --workflow CI \
      --event pull_request --branch "$PILOT_BRANCH" --limit 20 \
      --json databaseId,createdAt \
      > "$PILOT_RELEASE_ROOT/ci-run-candidates.json"
    PILOT_CI_COUNT="$(jq --arg not_before "$PILOT_PUSH_NOT_BEFORE" \
      '[.[] | select(.createdAt >= $not_before)] | length' \
      "$PILOT_RELEASE_ROOT/ci-run-candidates.json")"
    if test "$PILOT_CI_COUNT" -eq 1; then
      PILOT_CI_RUN_ID="$(jq -r --arg not_before "$PILOT_PUSH_NOT_BEFORE" \
        '.[] | select(.createdAt >= $not_before) | .databaseId' \
        "$PILOT_RELEASE_ROOT/ci-run-candidates.json")"
      break
    fi
    test "$PILOT_CI_COUNT" -eq 0
    sleep 10
  done
  test -n "$PILOT_CI_RUN_ID"
  test "$PILOT_CI_RUN_ID" -ge 1
  gh run watch "$PILOT_CI_RUN_ID" --repo "$PILOT_REPO" --exit-status
  gh api -H "X-GitHub-Api-Version: $PILOT_API_VERSION" \
    "repos/FMorgan-111/repopilot/actions/runs/$PILOT_CI_RUN_ID" \
    > "$PILOT_RELEASE_ROOT/ci-run.json"
  gh run view "$PILOT_CI_RUN_ID" --repo "$PILOT_REPO" --json jobs \
    > "$PILOT_RELEASE_ROOT/ci-jobs.json"
  PILOT_CI_WORKFLOW_ID="$(gh api -H "X-GitHub-Api-Version: $PILOT_API_VERSION" \
    repos/FMorgan-111/repopilot/actions/workflows/ci.yml --jq .id)"
  jq -e --arg branch "$PILOT_BRANCH" --arg head "$PILOT_EXPECTED_HEAD" \
    --argjson workflow_id "$PILOT_CI_WORKFLOW_ID" '
    .event == "pull_request" and
    .workflow_id == $workflow_id and
    .run_attempt == 1 and
    .conclusion == "success" and
    .head_branch == $branch and
    (.pull_requests | length) == 1 and
    .pull_requests[0].number == 2 and
    .pull_requests[0].head.ref == $branch and
    .pull_requests[0].head.sha == $head and
    .pull_requests[0].base.ref == "master"
  ' "$PILOT_RELEASE_ROOT/ci-run.json"
  jq -e '
    all(.jobs[]; .conclusion == "success") and
    ([.jobs[].name] | sort) == ([
      "lint",
      "oci-integration",
      "test (macos-latest, 3.10)",
      "test (macos-latest, 3.11)",
      "test (macos-latest, 3.12)",
      "test (ubuntu-latest, 3.10)",
      "test (ubuntu-latest, 3.11)",
      "test (ubuntu-latest, 3.12)"
    ] | sort)
  ' "$PILOT_RELEASE_ROOT/ci-jobs.json"
  ```

- [ ] Set/read the exact candidate variable and create, but never overwrite, the
  authorization label:

  ```bash
  gh variable set REPOPILOT_EVAL_CANDIDATE_SHA \
    --body "$PILOT_EXPECTED_HEAD" --repo "$PILOT_REPO"
  test "$(gh variable get REPOPILOT_EVAL_CANDIDATE_SHA --repo "$PILOT_REPO")" = \
    "$PILOT_EXPECTED_HEAD"
  gh label list --repo "$PILOT_REPO" --limit 1000 --json name \
    > "$PILOT_RELEASE_ROOT/repository-labels.json"
  if ! jq -e 'any(.[]; .name == "repopilot-pilot-20-approved")' \
    "$PILOT_RELEASE_ROOT/repository-labels.json" >/dev/null; then
    gh label create repopilot-pilot-20-approved --repo "$PILOT_REPO" \
      --color B60205 --description 'Authorize one immutable Pilot-20 run'
  fi
  test "$(gh label view repopilot-pilot-20-approved --repo "$PILOT_REPO" \
    --json name --jq .name)" = repopilot-pilot-20-approved
  jq '. + {task9_complete: true}' "$PILOT_STATE_FILE" \
    > "$PILOT_RELEASE_ROOT/state.next"
  chmod 600 "$PILOT_RELEASE_ROOT/state.next"
  mv "$PILOT_RELEASE_ROOT/state.next" "$PILOT_STATE_FILE"
  ```

- [ ] Record sanitized API evidence: environment name, protection settings,
  allowed refs, secret names, repository-secret absence, variable value, PR/base/
  head identity. Do not record tokens or secret values.

---

### Task 10: Authorize, monitor, download, and independently verify the pilot

**External state:** One paid GitHub Actions run and downloaded evidence outside
the repository.

- [ ] Run every Task 10 fence in one fresh PTY shell with fail-fast mode. Load
  only the completed, bounded Task 9 handoff and re-derive all other values:

  ```bash
  set -euo pipefail
  umask 077
  PILOT_RELEASE_ROOT=/private/tmp/repopilot-release-20260724
  PILOT_STATE_FILE="$PILOT_RELEASE_ROOT/state.json"
  PILOT_WORKTREE_ROOT=/Users/morgan/Documents/Codex/2026-07-17/https-github-com-fmorgan-111-repopilot/work/repopilot/.worktrees/gemini-default-model
  PILOT_GIT=/Users/morgan/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/fallback/git
  PILOT_REPO=FMorgan-111/repopilot
  PILOT_BRANCH=fix/release-readiness-20260717
  PILOT_API_VERSION=2026-03-10
  test -f "$PILOT_STATE_FILE"
  test ! -L "$PILOT_STATE_FILE"
  test "$(stat -f '%l' "$PILOT_STATE_FILE")" = 1
  test "$(stat -f '%u' "$PILOT_STATE_FILE")" = "$(id -u)"
  test "$(stat -f '%Lp' "$PILOT_STATE_FILE")" = 600
  test "$(stat -f '%z' "$PILOT_STATE_FILE")" -le 4096
  PILOT_TASK10_PHASE="$(jq -er \
    --arg root "$PILOT_RELEASE_ROOT" \
    --arg worktree "$PILOT_WORKTREE_ROOT" \
    --arg git "$PILOT_GIT" '
      def exact_keys($wanted): (keys | sort) == ($wanted | sort);
      def common:
        .schema_version == 1 and
        .release_root == $root and
        .worktree_root == $worktree and
        .git == $git and
        .task9_complete == true and
        (.candidate_sha | test("^[0-9a-f]{40}$"));
      if common and exact_keys([
        "candidate_sha", "git", "release_root", "schema_version",
        "task9_complete", "worktree_root"
      ]) then
        "fresh"
      elif common and exact_keys([
        "candidate_sha", "git", "label_event_state", "label_not_before",
        "release_root", "schema_version", "task9_complete", "worktree_root"
      ]) and
        (.label_event_state == "armed" or .label_event_state == "triggered") and
        (.label_not_before | test(
          "^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
        )) then
        "label_event"
      elif common and exact_keys([
        "candidate_sha", "git", "label_event_state", "label_not_before",
        "release_root", "run_id", "schema_version", "task9_complete",
        "worktree_root"
      ]) and
        (.label_event_state == "armed" or .label_event_state == "triggered") and
        (.label_not_before | test(
          "^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
        )) and
        (.run_id | type) == "number" and .run_id >= 1 and
        (.run_id | floor) == .run_id then
        "run_bound"
      else
        error("invalid Task 10 handoff state")
      end
    ' "$PILOT_STATE_FILE")"
  PILOT_EXPECTED_HEAD="$(jq -er .candidate_sha "$PILOT_STATE_FILE")"
  cd "$PILOT_WORKTREE_ROOT"
  gh auth status
  ```

- [ ] Re-read gateway balance/spend/rate settings with the owner. Explain that
  the 100,000-token value is an Agent soft threshold, not a billing ceiling.
- [ ] Recheck exact PR/candidate identity, remove the authorization label only
  if it is present, then add it exactly once. Record UTC time immediately before
  the add; no rerun API or workflow-dispatch command is permitted:

  ```bash
  gh api -H "X-GitHub-Api-Version: $PILOT_API_VERSION" \
    repos/FMorgan-111/repopilot/pulls/2 \
    > "$PILOT_RELEASE_ROOT/preauthorization-pr.json"
  jq -e --arg branch "$PILOT_BRANCH" --arg head "$PILOT_EXPECTED_HEAD" '
    .state == "open" and
    .user.login == "FMorgan-111" and
    .base.ref == "master" and
    .head.repo.full_name == "FMorgan-111/repopilot" and
    .head.ref == $branch and
    .head.sha == $head
  ' "$PILOT_RELEASE_ROOT/preauthorization-pr.json"
  test "$(gh variable get REPOPILOT_EVAL_CANDIDATE_SHA --repo "$PILOT_REPO")" = "$PILOT_EXPECTED_HEAD"
  if test "$PILOT_TASK10_PHASE" = fresh; then
    if jq -e 'any(.labels[]; .name == "repopilot-pilot-20-approved")' \
      "$PILOT_RELEASE_ROOT/preauthorization-pr.json" >/dev/null; then
      gh pr edit 2 --repo "$PILOT_REPO" \
        --remove-label repopilot-pilot-20-approved
    fi
    PILOT_LABEL_NOT_BEFORE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    jq --arg not_before "$PILOT_LABEL_NOT_BEFORE" \
      '. + {label_event_state: "armed", label_not_before: $not_before}' \
      "$PILOT_STATE_FILE" > "$PILOT_RELEASE_ROOT/state.next"
    chmod 600 "$PILOT_RELEASE_ROOT/state.next"
    mv "$PILOT_RELEASE_ROOT/state.next" "$PILOT_STATE_FILE"
    gh pr edit 2 --repo "$PILOT_REPO" \
      --add-label repopilot-pilot-20-approved
    jq '. + {label_event_state: "triggered"}' "$PILOT_STATE_FILE" \
      > "$PILOT_RELEASE_ROOT/state.next"
    chmod 600 "$PILOT_RELEASE_ROOT/state.next"
    mv "$PILOT_RELEASE_ROOT/state.next" "$PILOT_STATE_FILE"
    PILOT_TASK10_PHASE=label_event
  else
    PILOT_LABEL_NOT_BEFORE="$(jq -er .label_not_before "$PILOT_STATE_FILE")"
  fi
  ```

- [ ] Identify exactly one newly created pull-request run, then read the run
  object and require attempt 1, the expected actor, branch, event, workflow, and
  candidate before approving its pending environment deployment. Run lookup is
  bounded to 50 seconds; environment lookup is bounded to five minutes because
  `prepare` precedes the protected matrix job. Keep the PTY session yielding and
  send a status update at least once per minute; timeout stops without another
  label event:

  ```bash
  if test "$PILOT_TASK10_PHASE" = run_bound; then
    PILOT_RUN_ID="$(jq -er .run_id "$PILOT_STATE_FILE")"
  else
    PILOT_RUN_ID=''
    for PILOT_POLL in 1 2 3 4 5; do
      gh run list --repo "$PILOT_REPO" --workflow swe-bench-oci-eval.yml \
        --event pull_request --branch "$PILOT_BRANCH" --limit 20 \
        --json databaseId,createdAt \
        > "$PILOT_RELEASE_ROOT/pilot-run-candidates.json"
      PILOT_RUN_COUNT="$(jq --arg not_before "$PILOT_LABEL_NOT_BEFORE" \
        '[.[] | select(.createdAt >= $not_before)] | length' \
        "$PILOT_RELEASE_ROOT/pilot-run-candidates.json")"
      if test "$PILOT_RUN_COUNT" -eq 1; then
        PILOT_RUN_ID="$(jq -r --arg not_before "$PILOT_LABEL_NOT_BEFORE" \
          '.[] | select(.createdAt >= $not_before) | .databaseId' \
          "$PILOT_RELEASE_ROOT/pilot-run-candidates.json")"
        break
      fi
      test "$PILOT_RUN_COUNT" -eq 0
      sleep 10
    done
  fi
  test -n "$PILOT_RUN_ID"
  test "$PILOT_RUN_ID" -ge 1
  gh api -H "X-GitHub-Api-Version: $PILOT_API_VERSION" \
    "repos/FMorgan-111/repopilot/actions/runs/$PILOT_RUN_ID" \
    > "$PILOT_RELEASE_ROOT/pilot-run.json"
  PILOT_WORKFLOW_ID="$(gh api -H "X-GitHub-Api-Version: $PILOT_API_VERSION" \
    repos/FMorgan-111/repopilot/actions/workflows/swe-bench-oci-eval.yml --jq .id)"
  jq -e --arg branch "$PILOT_BRANCH" --arg head "$PILOT_EXPECTED_HEAD" \
    --argjson workflow_id "$PILOT_WORKFLOW_ID" '
    .event == "pull_request" and
    .workflow_id == $workflow_id and
    .run_attempt == 1 and
    .actor.login == "FMorgan-111" and
    .triggering_actor.login == "FMorgan-111" and
    .head_branch == $branch and
    (.pull_requests | length) == 1 and
    .pull_requests[0].number == 2 and
    .pull_requests[0].head.ref == $branch and
    .pull_requests[0].head.sha == $head and
    .pull_requests[0].base.ref == "master" and
    .path == ".github/workflows/swe-bench-oci-eval.yml"
  ' "$PILOT_RELEASE_ROOT/pilot-run.json"
  gh api -H "X-GitHub-Api-Version: $PILOT_API_VERSION" \
    repos/FMorgan-111/repopilot/pulls/2 \
    > "$PILOT_RELEASE_ROOT/preapproval-pr.json"
  jq -e --arg branch "$PILOT_BRANCH" --arg head "$PILOT_EXPECTED_HEAD" '
    .state == "open" and
    .user.login == "FMorgan-111" and
    .base.ref == "master" and
    .head.repo.full_name == "FMorgan-111/repopilot" and
    .head.ref == $branch and
    .head.sha == $head and
    ([.labels[].name | select(. == "repopilot-pilot-20-approved")] | length) == 1
  ' "$PILOT_RELEASE_ROOT/preapproval-pr.json"
  test "$(gh variable get REPOPILOT_EVAL_CANDIDATE_SHA --repo "$PILOT_REPO")" = "$PILOT_EXPECTED_HEAD"
  jq --argjson run_id "$PILOT_RUN_ID" \
    '. + {label_event_state: "triggered", run_id: $run_id}' \
    "$PILOT_STATE_FILE" > "$PILOT_RELEASE_ROOT/state.next"
  chmod 600 "$PILOT_RELEASE_ROOT/state.next"
  mv "$PILOT_RELEASE_ROOT/state.next" "$PILOT_STATE_FILE"
  PILOT_TASK10_PHASE=run_bound
  PILOT_ENVIRONMENT_ID=''
  for PILOT_POLL in {1..30}; do
    gh api -H "X-GitHub-Api-Version: $PILOT_API_VERSION" \
      "repos/FMorgan-111/repopilot/actions/runs/$PILOT_RUN_ID/pending_deployments" \
      > "$PILOT_RELEASE_ROOT/pending-deployments.json"
    PILOT_PENDING_COUNT="$(jq '[.[] | select(
      .environment.name == "repopilot-expensive-eval"
    )] | length' "$PILOT_RELEASE_ROOT/pending-deployments.json")"
    if test "$PILOT_PENDING_COUNT" -eq 1; then
      test "$(jq 'length' "$PILOT_RELEASE_ROOT/pending-deployments.json")" -eq 1
      PILOT_ENVIRONMENT_ID="$(jq -er '
        .[] | select(.environment.name == "repopilot-expensive-eval") |
        .environment.id
      ' "$PILOT_RELEASE_ROOT/pending-deployments.json")"
      break
    fi
    test "$PILOT_PENDING_COUNT" -eq 0
    test "$(jq 'length' "$PILOT_RELEASE_ROOT/pending-deployments.json")" -eq 0
    sleep 10
  done
  test -n "$PILOT_ENVIRONMENT_ID"
  test "$PILOT_ENVIRONMENT_ID" -ge 1
  jq -n --argjson environment_id "$PILOT_ENVIRONMENT_ID" '{
    environment_ids: [$environment_id],
    state: "approved",
    comment: "Approved immutable RepoPilot Pilot-20 candidate"
  }' > "$PILOT_RELEASE_ROOT/deployment-review.json"
  gh api --method POST -H "X-GitHub-Api-Version: $PILOT_API_VERSION" \
    "repos/FMorgan-111/repopilot/actions/runs/$PILOT_RUN_ID/pending_deployments" \
    --input "$PILOT_RELEASE_ROOT/deployment-review.json"
  ```

  The startup validator is the recovery state machine: `fresh` may execute the
  label mutation once; `label_event` reuses the stored timestamp and performs
  only bounded read-only run discovery; `run_bound` uses the stored run ID and
  revalidates the full run/live-PR identity before looking for its deployment.
  If an `armed` event has no unique run, stop as indeterminate. If a recovered
  run has no pending deployment because the approval POST may already have
  crossed the boundary, stop for owner audit rather than re-approving or
  creating another label event.

- [ ] Monitor without rerunning, download into one run-ID-specific directory,
  and prove the directory set contains exactly the aggregate plus the 20 tracked
  instance artifacts:

  ```bash
  gh run watch "$PILOT_RUN_ID" --repo "$PILOT_REPO" --exit-status
  PILOT_DOWNLOAD_ROOT="$(mktemp -d /private/tmp/repopilot-pilot-20.XXXXXX)"
  PILOT_DOWNLOAD_DIR="$PILOT_DOWNLOAD_ROOT/run-$PILOT_RUN_ID"
  mkdir "$PILOT_DOWNLOAD_DIR"
  gh run download "$PILOT_RUN_ID" --repo "$PILOT_REPO" \
    --dir "$PILOT_DOWNLOAD_DIR"
  {
    printf '%s\n' swe-bench-oci-pilot_20
    sed 's#^#swe-bench-oci-instance-#' eval/pilot_20_ids.txt
  } | sort > "$PILOT_RELEASE_ROOT/expected-artifacts.txt"
  find "$PILOT_DOWNLOAD_DIR" -mindepth 1 -maxdepth 1 -type d \
    -exec basename '{}' \; | sort > "$PILOT_RELEASE_ROOT/downloaded-artifacts.txt"
  diff -u "$PILOT_RELEASE_ROOT/expected-artifacts.txt" \
    "$PILOT_RELEASE_ROOT/downloaded-artifacts.txt"
  ```

- [ ] Run the exact 20-directory verifier command from Task 4 with the recorded
  `PILOT_DOWNLOAD_DIR`, `PILOT_EXPECTED_HEAD`, and `PILOT_RUN_ID`. No aggregate-
  only shortcut is acceptable.

- [ ] Require exactly 20 determined outcomes, zero infrastructure failures,
  strict provenance/hash agreement, and a valid marker. Record the official
  `R/20` even if R is zero.

  ```bash
  PILOT_RUN_MANIFEST="$PILOT_DOWNLOAD_DIR/swe-bench-oci-pilot_20/run_manifest.json"
  jq -e --arg head "$PILOT_EXPECTED_HEAD" --argjson run_id "$PILOT_RUN_ID" '
    .mode == "pilot_20" and
    .evaluated_commit == $head and
    .provenance.run_id == $run_id and
    .provenance.run_attempt == 1 and
    .requested == 20 and
    .determined == 20 and
    .infrastructure_failures == 0 and
    (.official_resolved + .official_unresolved + .empty_patch) == 20
  ' "$PILOT_RUN_MANIFEST"
  PILOT_RUN_MANIFEST_SHA256="$(shasum -a 256 "$PILOT_RUN_MANIFEST" | awk '{print $1}')"
  test "$(printf '%s' "$PILOT_RUN_MANIFEST_SHA256" | wc -c | tr -d ' ')" = 64
  jq --argjson run_id "$PILOT_RUN_ID" \
    --arg download_dir "$PILOT_DOWNLOAD_DIR" \
    --arg manifest_sha "$PILOT_RUN_MANIFEST_SHA256" '
      . + {
        run_id: $run_id,
        download_dir: $download_dir,
        run_manifest_sha256: $manifest_sha,
        task10_complete: true
      }
    ' "$PILOT_STATE_FILE" > "$PILOT_RELEASE_ROOT/state.next"
  chmod 600 "$PILOT_RELEASE_ROOT/state.next"
  mv "$PILOT_RELEASE_ROOT/state.next" "$PILOT_STATE_FILE"
  ```

- [ ] If the run is invalid because of infrastructure/system code, diagnose with
  `superpowers:systematic-debugging`, add a focused TDD fix, rerun all local
  gates/review, push a new candidate, update the variable, and authorize one
  complete new label event. Never rerun or reuse instance jobs.
- [ ] Once one valid run exists, freeze candidate and base. No repository commit,
  force push, base update, or score-driven retry is permitted.

---

### Task 11: Configure strict merge rules and merge the evaluated pair

**External state:** Active `master` ruleset, PR 2 merge, final provenance record.

- [ ] Run every Task 11 fence in one fresh PTY shell with fail-fast mode. Load
  only a completed Task 10 handoff and revalidate its bounded paths and values:

  ```bash
  set -euo pipefail
  umask 077
  PILOT_RELEASE_ROOT=/private/tmp/repopilot-release-20260724
  PILOT_STATE_FILE="$PILOT_RELEASE_ROOT/state.json"
  PILOT_WORKTREE_ROOT=/Users/morgan/Documents/Codex/2026-07-17/https-github-com-fmorgan-111-repopilot/work/repopilot/.worktrees/gemini-default-model
  PILOT_GIT=/Users/morgan/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/fallback/git
  PILOT_REPO=FMorgan-111/repopilot
  PILOT_BRANCH=fix/release-readiness-20260717
  PILOT_API_VERSION=2026-03-10
  test -f "$PILOT_STATE_FILE"
  test ! -L "$PILOT_STATE_FILE"
  test "$(stat -f '%l' "$PILOT_STATE_FILE")" = 1
  test "$(stat -f '%u' "$PILOT_STATE_FILE")" = "$(id -u)"
  test "$(stat -f '%Lp' "$PILOT_STATE_FILE")" = 600
  test "$(stat -f '%z' "$PILOT_STATE_FILE")" -le 4096
  jq -e --arg root "$PILOT_RELEASE_ROOT" --arg worktree "$PILOT_WORKTREE_ROOT" \
    --arg git "$PILOT_GIT" '
    .schema_version == 1 and
    .release_root == $root and
    .worktree_root == $worktree and
    .git == $git and
    .task9_complete == true and
    .task10_complete == true and
    .label_event_state == "triggered" and
    (.candidate_sha | test("^[0-9a-f]{40}$")) and
    (.run_manifest_sha256 | test("^[0-9a-f]{64}$")) and
    (.run_id | type) == "number" and .run_id >= 1 and
    (.run_id | floor) == .run_id and
    (.download_dir | type) == "string" and
    (.download_dir | startswith("/private/tmp/repopilot-pilot-20.")) and
    (keys | sort) == [
      "candidate_sha", "download_dir", "git", "label_event_state",
      "label_not_before", "release_root", "run_id", "run_manifest_sha256",
      "schema_version", "task10_complete", "task9_complete", "worktree_root"
    ]
  ' "$PILOT_STATE_FILE"
  PILOT_EXPECTED_HEAD="$(jq -er .candidate_sha "$PILOT_STATE_FILE")"
  PILOT_RUN_ID="$(jq -er .run_id "$PILOT_STATE_FILE")"
  PILOT_DOWNLOAD_DIR="$(jq -er .download_dir "$PILOT_STATE_FILE")"
  PILOT_RUN_MANIFEST_SHA256="$(jq -er .run_manifest_sha256 "$PILOT_STATE_FILE")"
  test -d "$PILOT_DOWNLOAD_DIR"
  test ! -L "$PILOT_DOWNLOAD_DIR"
  cd "$PILOT_WORKTREE_ROOT"
  gh auth status
  ```

- [ ] Before any ruleset mutation, run the exact 20-directory verifier command
  from Task 4 again with the reloaded `PILOT_DOWNLOAD_DIR`,
  `PILOT_EXPECTED_HEAD`, and `PILOT_RUN_ID`. Then require the current manifest
  bytes to match the Task 10 handoff hash:

  ```bash
  PILOT_RUN_MANIFEST="$PILOT_DOWNLOAD_DIR/swe-bench-oci-pilot_20/run_manifest.json"
  test -f "$PILOT_RUN_MANIFEST"
  test ! -L "$PILOT_RUN_MANIFEST"
  test "$(shasum -a 256 "$PILOT_RUN_MANIFEST" | awk '{print $1}')" = \
    "$PILOT_RUN_MANIFEST_SHA256"
  ```

- [ ] Identify exactly one successful `pilot_release_gate` job in the frozen
  run, follow its API-provided `check_run_url`, and capture the GitHub Actions
  app integration ID. Do not assume a PR check is attached directly to the head
  commit; GitHub may attach it to the synthetic merge ref:

  ```bash
  gh api -H "X-GitHub-Api-Version: $PILOT_API_VERSION" \
    "repos/FMorgan-111/repopilot/actions/runs/$PILOT_RUN_ID/jobs?filter=latest&per_page=100" \
    > "$PILOT_RELEASE_ROOT/pilot-jobs.json"
  PILOT_GATE_CHECK_URL="$(jq -er '
    [.jobs[] | select(
      .name == "pilot_release_gate" and
      .status == "completed" and
      .conclusion == "success"
    )] |
    if length == 1 then .[0].check_run_url else error("expected one passing gate job") end
  ' "$PILOT_RELEASE_ROOT/pilot-jobs.json")"
  case "$PILOT_GATE_CHECK_URL" in
    https://api.github.com/repos/FMorgan-111/repopilot/check-runs/[0-9]*) ;;
    *) exit 1 ;;
  esac
  PILOT_GATE_CHECK_ID="${PILOT_GATE_CHECK_URL##*/}"
  printf '%s\n' "$PILOT_GATE_CHECK_ID" | grep -Eq '^[0-9]+$'
  gh api -H "X-GitHub-Api-Version: $PILOT_API_VERSION" \
    "repos/FMorgan-111/repopilot/check-runs/$PILOT_GATE_CHECK_ID" \
    > "$PILOT_RELEASE_ROOT/pilot-check-run.json"
  PILOT_ACTIONS_APP_ID="$(jq -er --arg run_id "$PILOT_RUN_ID" '
    select(
      .name == "pilot_release_gate" and
      .status == "completed" and
      .conclusion == "success" and
      .app.slug == "github-actions" and
      (.details_url | contains("/actions/runs/" + $run_id + "/job/"))
    ) | .app.id
  ' "$PILOT_RELEASE_ROOT/pilot-check-run.json")"
  test "$PILOT_ACTIONS_APP_ID" -ge 1
  ```

- [ ] Build this exact dedicated ruleset payload. It preserves every unrelated
  existing ruleset, has no bypass actor, permits merge commits only, requires a
  PR, disallows deletion/force-push, and app-binds the strict status check:

  ```bash
  jq -n --argjson app_id "$PILOT_ACTIONS_APP_ID" '{
    name: "RepoPilot pilot release gate",
    target: "branch",
    enforcement: "active",
    bypass_actors: [],
    conditions: {
      ref_name: {include: ["refs/heads/master"], exclude: []}
    },
    rules: [
      {type: "deletion"},
      {type: "non_fast_forward"},
      {type: "pull_request", parameters: {
        allowed_merge_methods: ["merge"],
        dismiss_stale_reviews_on_push: false,
        require_code_owner_review: false,
        require_last_push_approval: false,
        required_approving_review_count: 0,
        required_review_thread_resolution: false
      }},
      {type: "required_status_checks", parameters: {
        do_not_enforce_on_create: false,
        required_status_checks: [{
          context: "pilot_release_gate",
          integration_id: $app_id
        }],
        strict_required_status_checks_policy: true
      }}
    ]
  }' > "$PILOT_RELEASE_ROOT/ruleset-request.json"
  PILOT_RULESET_IDS="$(gh api -H "X-GitHub-Api-Version: $PILOT_API_VERSION" \
    repos/FMorgan-111/repopilot/rulesets \
    --jq '.[] | select(.name == "RepoPilot pilot release gate") | .id')"
  test "$(printf '%s\n' "$PILOT_RULESET_IDS" | sed '/^$/d' | wc -l | tr -d ' ')" -le 1
  if test -z "$PILOT_RULESET_IDS"; then
    PILOT_RULESET_ID="$(gh api --method POST \
      -H "X-GitHub-Api-Version: $PILOT_API_VERSION" \
      repos/FMorgan-111/repopilot/rulesets \
      --input "$PILOT_RELEASE_ROOT/ruleset-request.json" --jq .id)"
  else
    PILOT_RULESET_ID="$PILOT_RULESET_IDS"
    gh api --method PUT -H "X-GitHub-Api-Version: $PILOT_API_VERSION" \
      "repos/FMorgan-111/repopilot/rulesets/$PILOT_RULESET_ID" \
      --input "$PILOT_RELEASE_ROOT/ruleset-request.json"
  fi
  ```

- [ ] Read back and structurally compare the complete authoritative ruleset,
  then read the effective rules for `master`. The documented bypass authority
  is the exact empty `bypass_actors` array; do not depend on an unsupported
  `current_user_can_bypass` response field. Any rejected parameter, extra bypass
  actor, missing effective rule, or mismatch blocks merge:

  ```bash
  gh api -H "X-GitHub-Api-Version: $PILOT_API_VERSION" \
    "repos/FMorgan-111/repopilot/rulesets/$PILOT_RULESET_ID" \
    > "$PILOT_RELEASE_ROOT/ruleset-readback.json"
  jq -e --slurpfile wanted "$PILOT_RELEASE_ROOT/ruleset-request.json" '
    .name == $wanted[0].name and
    .target == $wanted[0].target and
    .enforcement == $wanted[0].enforcement and
    .bypass_actors == [] and
    .conditions == $wanted[0].conditions and
    ([.rules[] | {type, parameters}] | sort_by(.type)) ==
      ([$wanted[0].rules[] | {type, parameters}] | sort_by(.type))
  ' "$PILOT_RELEASE_ROOT/ruleset-readback.json"
  gh api -H "X-GitHub-Api-Version: $PILOT_API_VERSION" \
    repos/FMorgan-111/repopilot/rules/branches/master \
    > "$PILOT_RELEASE_ROOT/master-effective-rules.json"
  jq -e --argjson ruleset_id "$PILOT_RULESET_ID" '
    ([.[] | select(.ruleset_id == $ruleset_id) | .type] | sort) ==
      (["deletion", "non_fast_forward", "pull_request", "required_status_checks"] | sort)
  ' "$PILOT_RELEASE_ROOT/master-effective-rules.json"
  ```

- [ ] Re-read PR 2 immediately before merge and require open state, exact label,
  live admission fields, head equal to `run_manifest.evaluated_commit`, base SHA
  equal to provenance, and all standard/required checks successful.

  ```bash
  PILOT_RUN_MANIFEST="$PILOT_DOWNLOAD_DIR/swe-bench-oci-pilot_20/run_manifest.json"
  PILOT_EXPECTED_BASE="$(jq -er .provenance.pr_base_sha "$PILOT_RUN_MANIFEST")"
  test "$(jq -er .evaluated_commit "$PILOT_RUN_MANIFEST")" = "$PILOT_EXPECTED_HEAD"
  gh api -H "X-GitHub-Api-Version: $PILOT_API_VERSION" \
    repos/FMorgan-111/repopilot/pulls/2 \
    > "$PILOT_RELEASE_ROOT/premerge-pr.json"
  jq -e --arg base "$PILOT_EXPECTED_BASE" --arg head "$PILOT_EXPECTED_HEAD" '
    .state == "open" and
    .user.login == "FMorgan-111" and
    .base.ref == "master" and
    .base.sha == $base and
    .head.repo.full_name == "FMorgan-111/repopilot" and
    .head.ref == "fix/release-readiness-20260717" and
    .head.sha == $head and
    ([.labels[].name | select(. == "repopilot-pilot-20-approved")] | length) == 1
  ' "$PILOT_RELEASE_ROOT/premerge-pr.json"
  test "$(gh api -H "X-GitHub-Api-Version: $PILOT_API_VERSION" \
    repos/FMorgan-111/repopilot/git/ref/heads/master --jq .object.sha)" = "$PILOT_EXPECTED_BASE"
  gh pr checks 2 --repo "$PILOT_REPO"
  gh pr checks 2 --repo "$PILOT_REPO" --required
  ```

- [ ] Merge only after the readback checks pass. Use the REST merge endpoint so
  a failed check or concurrent base change returns a terminal non-merge result;
  never invoke `gh pr merge`, whose pending-check behavior may enable auto-
  merge. The exact request has only the expected head SHA and merge method:

  ```bash
  jq -n --arg sha "$PILOT_EXPECTED_HEAD" '{
    sha: $sha,
    merge_method: "merge"
  }' > "$PILOT_RELEASE_ROOT/merge-request.json"
  gh api --method PUT -H "X-GitHub-Api-Version: $PILOT_API_VERSION" \
    repos/FMorgan-111/repopilot/pulls/2/merge \
    --input "$PILOT_RELEASE_ROOT/merge-request.json" \
    > "$PILOT_RELEASE_ROOT/merge-response.json"
  jq -e '.merged == true and (.sha | test("^[0-9a-f]{40}$"))' \
    "$PILOT_RELEASE_ROOT/merge-response.json"
  PILOT_MERGE_SHA="$(jq -er .sha "$PILOT_RELEASE_ROOT/merge-response.json")"
  gh api -H "X-GitHub-Api-Version: $PILOT_API_VERSION" \
    "repos/FMorgan-111/repopilot/commits/$PILOT_MERGE_SHA" \
    > "$PILOT_RELEASE_ROOT/merge-commit.json"
  jq -e --arg base "$PILOT_EXPECTED_BASE" --arg head "$PILOT_EXPECTED_HEAD" '
    [.parents[].sha] == [$base, $head]
  ' "$PILOT_RELEASE_ROOT/merge-commit.json"
  test "$(gh api -H "X-GitHub-Api-Version: $PILOT_API_VERSION" \
    repos/FMorgan-111/repopilot/git/ref/heads/master --jq .object.sha)" = "$PILOT_MERGE_SHA"
  ```

- [ ] Record run ID, workflow ref/SHA, run-manifest SHA-256, selection SHA-256,
  expected base, evaluated head, merge commit, direct parents, and official
  `R/20` in the final release handoff.
- [ ] Confirm PR 2 is merged, the required rules remain active, no secret value
  entered logs/artifacts, and the project is complete.

## Plan acceptance checklist

- Every approved design section maps to one task and one explicit verification.
- The fixed denominator is 20; success is official-only; zero is a valid score.
- Runtime, instance, aggregate, verifier, workflow, and merge use one exact
  candidate/run/attempt/base/provenance chain.
- The final PR check cannot become a successful skipped check.
- Paid credentials remain outside repository-controlled secret scope.
- No task changes model policy, cohort, concurrency, retry policy, or score gate
  after observing results.
- The clean committed tree, not the dirty worktree, supplies release evidence.
