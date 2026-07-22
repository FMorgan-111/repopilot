# SWE-bench 50-Instance Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend RepoPilot's immutable OCI evaluation from a fixed ten-instance baseline to a reproducible 50-instance SWE-bench Verified run with bounded pull retry, strict artifact discovery, and auditable scoring.

**Architecture:** The existing contract remains the single source of allowed modes and tracked IDs, while a new committed `baseline_50_ids.txt` freezes selection order. The per-instance runner adds transient-only Docker pull retry without weakening digest pinning, the aggregator rejects every untracked root entry and derives transparent metrics from already-sanitized bundles, and the manual workflow exposes only the checkpoint and 50-instance baseline with global concurrency and public-dataset-only caching.

**Tech Stack:** Python 3.10+, Pydantic v2, pytest/pytest-asyncio, Docker CLI, official `swebench>=4.1,<5`, GitHub Actions, Ruff, Python build.

## Global Constraints

- Public modes are exactly `checkpoint_5` and `baseline_50`; `baseline_10` remains only as historical selection provenance and is not dispatchable.
- `eval/baseline_50_ids.txt` contains exactly 50 unique IDs, starts with `eval/baseline_10_ids.txt` unchanged, and contains every `eval/checkpoint_5_ids.txt` ID.
- Dataset is exactly `SWE-bench/SWE-bench_Verified`, split `test`, revision `main`; the selection source SHA-256 is `f61cd55ceb35b61ad592f645abcbfc8ea4d294c6c9f3c8f15e83211a8e8db98c` and the selection seed is `17`.
- Primary model is `gemini-3.5-flash:stable`; one-way escalation model is `claude-opus-4-8:stable`; generation uses `max_retries=3` and `token_budget=100000`.
- Per-run matrix `max-parallel` is `2`, and workflow-level concurrency prevents overlapping manual eval runs without cancelling the running run.
- Each official image is pulled as `linux/amd64`, inspected, and trusted only as a local `sha256:<64 lowercase hex>` identity; there is no build, mutable-tag trust, or host-execution fallback.
- Transient image pull retry has exactly three total attempts and delays of `5` then `20` seconds; permanent errors fail immediately.
- Only public dataset download state may be cached. Credentials, generated patches, model responses, RepoPilot home data other than the public dataset directory, target checkouts, Docker tag state, and evaluator logs must not enter cache keys or paths.
- Model secrets exist only in the generation workflow step and never enter subprocesses, containers, scorer steps, cache keys, logs, manifests, or artifacts.
- Exactly four files are accepted per instance bundle: `result.json`, `prediction.jsonl`, `official_result.json`, and `manifest.json`; every top-level artifact entry must be one real bundle directory.
- `swebench.harness.run_evaluation` remains the only source of official resolution. The raw official score is displayed before the secondary engineering score and is never replaced by RepoPilot's internal verdict.
- The workflow must first be merged to the actual default branch, `master`, through a user-authorized PR; the live evaluation ref remains the feature branch commit being scored.
- Use red-green-refactor TDD and commit after every independently reviewable task. Preserve unrelated existing edits, especially `run_trace.py`.

---

## File Structure

- Create `eval/baseline_50_ids.txt`: immutable ordered 50-instance allowlist.
- Modify `eval/oci_contract.py`: replace the public `baseline_10` mode mapping with `baseline_50`.
- Modify `eval/oci_runner.py`: accept the new mode and add bounded transient Docker pull retry.
- Modify `eval/oci_aggregate.py`: accept the new mode, strictly enumerate bundle roots, and report auditable scoring metrics.
- Modify `.github/workflows/swe-bench-oci-eval.yml`: expose the 50 mode, serialize eval runs, and cache only public dataset downloads.
- Modify `README.md`: document `master`, the 50-instance denominator, artifact names, and safe dispatch sequence.
- Modify `tests/test_oci_contract.py`, `tests/test_oci_runner.py`, `tests/test_oci_aggregate.py`, and `tests/test_swe_bench_oci_workflow.py`: prove each contract through red-green cycles.

---

### Task 1: Freeze the 50-Instance Allowlist and Public Mode

**Files:**
- Create: `eval/baseline_50_ids.txt`
- Modify: `eval/oci_contract.py`
- Modify: `eval/oci_runner.py`
- Modify: `eval/oci_aggregate.py`
- Test: `tests/test_oci_contract.py`
- Test: `tests/test_oci_runner.py`
- Test: `tests/test_oci_aggregate.py`

**Interfaces:**
- Produces: `EvalMode = Literal["checkpoint_5", "baseline_50"]`
- Produces: `_MODE_FILES["baseline_50"] == "baseline_50_ids.txt"`
- Preserves: `load_mode_instance_ids(mode, repo_root) -> tuple[str, ...]`
- Preserves: `require_mode_instance(mode, instance_id, repo_root) -> None`
- Consumed by: runner CLI, aggregator CLI, workflow matrix construction, `RuntimeRecord`, and `InstanceManifest`.

- [ ] **Step 1: Write failing contract and CLI tests**

Add real-repository contract coverage to `tests/test_oci_contract.py`:

```python
def test_baseline_50_is_fixed_unique_and_preserves_historical_sets() -> None:
    baseline_50 = load_mode_instance_ids("baseline_50")
    baseline_10 = tuple(
        line.strip()
        for line in (REPO_ROOT / "eval" / "baseline_10_ids.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    )
    checkpoint_5 = load_mode_instance_ids("checkpoint_5")

    assert len(baseline_50) == 50
    assert len(set(baseline_50)) == 50
    assert baseline_50[:10] == baseline_10
    assert set(checkpoint_5) <= set(baseline_50)


def test_retired_baseline_10_is_not_a_public_mode() -> None:
    with pytest.raises(ValueError, match="unsupported evaluation mode"):
        load_mode_instance_ids("baseline_10")
```

Import `REPO_ROOT` in that test. Change the duplicate fixture test to write
`baseline_50_ids.txt` and call `load_mode_instance_ids("baseline_50", tmp_path)`.

Add parser coverage in `tests/test_oci_runner.py` and
`tests/test_oci_aggregate.py` that invokes each CLI with `baseline_10` and
asserts `pytest.raises(SystemExit)`, then invokes `baseline_50` through a
monkeypatched implementation and asserts the forwarded mode is
`"baseline_50"`.

Use this runner shape (the existing checkpoint test may be parameterized rather
than duplicated):

```python
@pytest.mark.parametrize("mode", ["checkpoint_5", "baseline_50"])
def test_prepare_cli_accepts_public_modes(monkeypatch, tmp_path: Path, mode: str) -> None:
    seen: dict[str, object] = {}

    def fake_prepare(selected_mode, instance_id, output_dir):
        seen.update(mode=selected_mode, instance_id=instance_id, output_dir=output_dir)

    monkeypatch.setattr(oci_runner, "prepare_instance", fake_prepare)

    assert oci_runner.main([
        "prepare", "--mode", mode, "--instance-id", INSTANCE_ID,
        "--output-dir", str(tmp_path),
    ]) == 0
    assert seen["mode"] == mode


def test_prepare_cli_rejects_retired_baseline_10(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        oci_runner.main([
            "prepare", "--mode", "baseline_10", "--instance-id", INSTANCE_ID,
            "--output-dir", str(tmp_path),
        ])
```

Use this aggregate CLI coverage, importing `eval.oci_aggregate` as
`oci_aggregate`:

```python
def test_aggregate_cli_accepts_baseline_50(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake_aggregate(mode, artifacts_dir, output_dir, *, expected_commit):
        seen.update(
            mode=mode,
            artifacts_dir=artifacts_dir,
            output_dir=output_dir,
            expected_commit=expected_commit,
        )

    monkeypatch.setattr(oci_aggregate, "aggregate_artifacts", fake_aggregate)

    assert oci_aggregate.main([
        "--mode", "baseline_50",
        "--artifacts-dir", str(tmp_path / "artifacts"),
        "--output-dir", str(tmp_path / "combined"),
        "--expected-commit", COMMIT_SHA,
    ]) == 0
    assert seen["mode"] == "baseline_50"


def test_aggregate_cli_rejects_retired_baseline_10(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        oci_aggregate.main([
            "--mode", "baseline_10",
            "--artifacts-dir", str(tmp_path / "artifacts"),
            "--output-dir", str(tmp_path / "combined"),
            "--expected-commit", COMMIT_SHA,
        ])
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_oci_contract.py tests/test_oci_runner.py tests/test_oci_aggregate.py -q
```

Expected: FAIL because `baseline_50` is unsupported and its tracked file does
not exist; the old CLI still accepts `baseline_10`.

- [ ] **Step 3: Create the exact ordered allowlist**

Create `eval/baseline_50_ids.txt` with exactly these lines and this order:

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
pytest-dev__pytest-7205
django__django-16315
scikit-learn__scikit-learn-13124
astropy__astropy-12907
psf__requests-1921
matplotlib__matplotlib-20826
pylint-dev__pylint-8898
sphinx-doc__sphinx-7462
sympy__sympy-13615
pydata__xarray-7393
pytest-dev__pytest-7982
django__django-11292
scikit-learn__scikit-learn-14087
astropy__astropy-14369
psf__requests-1142
matplotlib__matplotlib-24870
pylint-dev__pylint-4661
sphinx-doc__sphinx-7985
sympy__sympy-23824
pydata__xarray-3095
pytest-dev__pytest-7324
django__django-16136
scikit-learn__scikit-learn-14629
astropy__astropy-14508
psf__requests-2931
matplotlib__matplotlib-22865
pylint-dev__pylint-4551
sphinx-doc__sphinx-9711
matplotlib__matplotlib-23299
sphinx-doc__sphinx-10323
```

- [ ] **Step 4: Replace the public mode mapping and CLI choices**

In `eval/oci_contract.py`, use:

```python
EvalMode = Literal["checkpoint_5", "baseline_50"]

_MODE_FILES: dict[str, str] = {
    "checkpoint_5": "checkpoint_5_ids.txt",
    "baseline_50": "baseline_50_ids.txt",
}
```

In both `eval/oci_runner.py` and `eval/oci_aggregate.py`, replace the parser
choices with:

```python
choices=("checkpoint_5", "baseline_50")
```

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_oci_contract.py tests/test_oci_runner.py tests/test_oci_aggregate.py -q
```

Expected: PASS, including exact-size, prefix, inclusion, and retired-mode
assertions.

- [ ] **Step 6: Commit the fixed public contract**

```bash
git add eval/baseline_50_ids.txt eval/oci_contract.py eval/oci_runner.py eval/oci_aggregate.py tests/test_oci_contract.py tests/test_oci_runner.py tests/test_oci_aggregate.py
git commit -m "feat(eval): freeze SWE-bench 50-instance baseline"
```

---

### Task 2: Retry Only Transient Official Image Pull Failures

**Files:**
- Modify: `eval/oci_runner.py`
- Test: `tests/test_oci_runner.py`

**Interfaces:**
- Preserves: `_pull_and_pin_image(image: str, command_runner: CommandRunner) -> str`
- Adds: `_PULL_RETRY_DELAYS = (5.0, 20.0)`
- Adds: `_is_transient_pull_failure(result: BoundedProcessResult) -> bool`
- Requires: successful return remains a validated local `sha256:` identity.

- [ ] **Step 1: Write failing transient, permanent, and exhaustion tests**

Import `_pull_and_pin_image` in `tests/test_oci_runner.py`. Add tests using a
fake command runner and `monkeypatch.setattr(oci_runner.time, "sleep", sleeps.append)`:

```python
def test_pull_retries_transient_failures_with_bounded_schedule(monkeypatch) -> None:
    pulls = 0
    sleeps: list[float] = []

    def command_runner(argv, **kwargs):
        nonlocal pulls
        if argv[1] == "pull":
            pulls += 1
            if pulls < 3:
                return BoundedProcessResult(list(argv), 1, "", "HTTP 503 unavailable")
            return BoundedProcessResult(list(argv), 0, "pulled", "")
        return BoundedProcessResult(list(argv), 0, IMAGE_SHA, "")

    monkeypatch.setattr(oci_runner.time, "sleep", sleeps.append)

    assert _pull_and_pin_image(OFFICIAL_IMAGE, command_runner) == IMAGE_SHA
    assert pulls == 3
    assert sleeps == [5.0, 20.0]
```

Add a permanent `unauthorized: authentication required` case asserting one
pull and no sleeps. Add an always-`429 Too Many Requests` case asserting three
pulls, `[5.0, 20.0]`, no inspect, and
`OciImageInfrastructureError`.

- [ ] **Step 2: Run retry tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_oci_runner.py -q
```

Expected: FAIL because `_pull_and_pin_image` performs only one pull and
`eval.oci_runner` has no `time` import or transient classifier.

- [ ] **Step 3: Implement bounded transient-only retry**

In `eval/oci_runner.py`, import `time`, define the exact delays, and classify
only bounded transport/registry diagnostics from `stdout` plus `stderr`:

```python
_PULL_RETRY_DELAYS = (5.0, 20.0)
_TRANSIENT_PULL_RE = re.compile(
    r"(?:\b(?:429|500|502|503|504)\b|"
    r"timed?\s*out|timeout|connection reset|temporary failure|"
    r"tls handshake timeout|unexpected eof|network is unreachable|i/o timeout)",
    re.IGNORECASE,
)


def _is_transient_pull_failure(result: BoundedProcessResult) -> bool:
    diagnostic = f"{result.stdout}\n{result.stderr}"
    return bool(_TRANSIENT_PULL_RE.search(diagnostic))
```

Change the pull section of `_pull_and_pin_image` to attempt once plus the two
delays. Sleep only when another attempt is permitted and the prior failure is
transient. Raise `OciImageInfrastructureError("official image pull failed")`
for the first permanent failure or the third transient failure. Inspect the
image exactly once and only after a successful pull; retain the existing digest
regex validation.

Use this loop before the unchanged inspect block:

```python
    for attempt in range(len(_PULL_RETRY_DELAYS) + 1):
        pulled = command_runner(
            ["docker", "pull", "--platform=linux/amd64", image],
            cwd=REPO_ROOT,
            timeout=1_800,
            max_output_bytes=32_000,
            decode_errors="strict",
        )
        if not pulled.returncode:
            break
        if (
            attempt == len(_PULL_RETRY_DELAYS)
            or not _is_transient_pull_failure(pulled)
        ):
            raise OciImageInfrastructureError("official image pull failed")
        time.sleep(_PULL_RETRY_DELAYS[attempt])
```

- [ ] **Step 4: Run runner tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_oci_runner.py -q
```

Expected: PASS; successful first pulls remain one pull plus one inspect,
transient recovery uses exactly three attempts, and permanent failures do not
sleep.

- [ ] **Step 5: Commit pull resilience**

```bash
git add eval/oci_runner.py tests/test_oci_runner.py
git commit -m "fix(eval): retry transient OCI image pulls"
```

---

### Task 3: Reject Unmanifested Roots and Emit Auditable Scores

**Files:**
- Modify: `eval/oci_aggregate.py`
- Test: `tests/test_oci_aggregate.py`

**Interfaces:**
- Adds: `_discover_manifest_paths(artifacts_dir: Path) -> list[Path]`
- Preserves: `aggregate_artifacts(...) -> Path`
- Extends: `_summary(...) -> str` with official score, terminal coverage,
  agreement, model-token/elapsed totals, and transparent engineering score.

- [ ] **Step 1: Write failing strict-root discovery tests**

Extend the invalid-artifact mutation parameterization with
`"root_file"`, `"root_dir_without_manifest"`, and `"root_symlink"`:

```python
elif mutation == "root_file":
    (artifacts / "unexpected.txt").write_text("unsafe", encoding="utf-8")
elif mutation == "root_dir_without_manifest":
    (artifacts / "unmanifested").mkdir()
elif mutation == "root_symlink":
    (artifacts / "linked-bundle").symlink_to(bundle, target_is_directory=True)
```

Each case must raise `ArtifactContractError`. Skip only the symlink case on an
OS that denies symlink creation; do not weaken production validation.

- [ ] **Step 2: Write failing scoring-summary tests**

Build two synthetic bundles, one resolved/internal-success and one
unresolved/internal-failure, with safe model invocation token and elapsed
fields. Assert the generated summary contains exact, separate metrics:

```python
assert "| Official score | 50.00/100 |" in summary
assert "| Official terminal coverage | 2/2 |" in summary
assert "| Internal/official agreement | 2/2 |" in summary
assert "| Model tokens | 30 |" in summary
assert "| Model elapsed seconds | 2.000 |" in summary
assert "| Engineering score | 100.00/100 |" in summary
```

Add a scorer-infrastructure case proving it remains in the requested
denominator, is excluded from terminal/agreement denominators, and lowers the
infrastructure component rather than becoming an unresolved verdict.

Extend `_completed_output` with optional `official_status`, `agent_success`,
and invocation values, then construct the infrastructure case as:

```python
official = OfficialResult(
    instance_id=instance_id,
    status="scorer_infra",
    submitted=False,
    completed=False,
    resolved=False,
    error_class="DockerUnavailable",
)
```

For a two-bundle aggregate containing one resolved bundle and this scorer
failure, assert:

```python
assert "| Official score | 50.00/100 |" in summary
assert "| Official terminal coverage | 1/2 |" in summary
assert "| Internal/official agreement | 1/1 |" in summary
assert "| Infrastructure failure | 1 |" in summary
```

- [ ] **Step 3: Run aggregate tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_oci_aggregate.py -q
```

Expected: FAIL because glob discovery ignores root entries without manifests
and the summary does not contain the new metrics.

- [ ] **Step 4: Implement strict top-level discovery**

Add:

```python
def _discover_manifest_paths(artifacts_dir: Path) -> list[Path]:
    root = Path(artifacts_dir)
    try:
        entries = list(root.iterdir())
    except OSError as exc:
        raise ArtifactContractError("artifact root unavailable") from exc
    manifests: list[Path] = []
    for entry in entries:
        if entry.is_symlink() or not entry.is_dir():
            raise ArtifactContractError("unexpected artifact root entry")
        manifest = entry / "manifest.json"
        if manifest.is_symlink() or not manifest.is_file():
            raise ArtifactContractError("artifact bundle manifest missing")
        manifests.append(manifest)
    return manifests
```

Replace `glob("*/manifest.json")` with this helper. Leave `_verify_bundle` as
the second boundary that enforces the exact four-file set and hashes.

- [ ] **Step 5: Implement deterministic scoring metrics**

Within `_summary`, compute:

```python
official_terminal = sum(
    payload.official.status != "scorer_infra" for _manifest, payload in ordered
)
non_infrastructure = sum(
    manifest.runtime_status == "ready"
    and payload.official.status != "scorer_infra"
    and payload.result.get("failure_class") != "infra"
    for manifest, payload in ordered
)
agreements = sum(
    payload.official.status != "scorer_infra"
    and (payload.result.get("agent_success") is True) == payload.official.resolved
    for _manifest, payload in ordered
)
```

Import `math` and add this bounded projection rather than trusting arbitrary
numeric shapes:

```python
def _model_usage(result: dict[str, Any]) -> tuple[int, float]:
    tokens = 0
    elapsed = 0.0
    invocations = result.get("model_invocations", [])
    if not isinstance(invocations, list):
        return tokens, elapsed
    for invocation in invocations:
        if not isinstance(invocation, dict):
            continue
        for key in ("input_tokens", "output_tokens"):
            value = invocation.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                tokens += value
        duration = invocation.get("elapsed_seconds")
        if (
            isinstance(duration, (int, float))
            and not isinstance(duration, bool)
            and math.isfinite(float(duration))
            and duration > 0
        ):
            elapsed += float(duration)
    return tokens, elapsed
```

Sum `_model_usage(payload.result)` across bundles. A complete bundle proves the
job reached package within its 360-minute workflow timeout; count it within
budget when its summed tokens are at most `100_000`. Compute with explicit
components:

```python
official_score = 100.0 * official_resolved / requested
resolution_component = 80.0 * official_resolved / requested
infrastructure_component = 10.0 * non_infrastructure / requested
agreement_component = (
    5.0 * agreements / official_terminal if official_terminal else 0.0
)
budget_component = 5.0 * within_budget / requested
engineering_score = (
    resolution_component
    + infrastructure_component
    + agreement_component
    + budget_component
)
```

Format scores with two decimals and elapsed seconds with three decimals. Keep
counts and the existing per-instance table. The raw official score must appear
before the engineering score.

- [ ] **Step 6: Run aggregate tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_oci_aggregate.py -q
```

Expected: PASS with strict root rejection and deterministic score strings.

- [ ] **Step 7: Commit aggregation hardening and scoring**

```bash
git add eval/oci_aggregate.py tests/test_oci_aggregate.py
git commit -m "feat(eval): harden and score OCI aggregates"
```

---

### Task 4: Expose the 50-Instance Workflow and Safe Dataset Cache

**Files:**
- Modify: `.github/workflows/swe-bench-oci-eval.yml`
- Modify: `README.md`
- Test: `tests/test_swe_bench_oci_workflow.py`

**Interfaces:**
- Workflow input enum: exactly `checkpoint_5`, `baseline_50`.
- Workflow concurrency group: `swe-bench-oci-evaluation`,
  `cancel-in-progress: false`.
- Public cache key: `swe-bench-verified-main-v1`.
- Public cache paths: `${{ runner.temp }}/public-hf-cache` and
  `${{ runner.temp }}/repopilot-home/eval/datasets` only.

- [ ] **Step 1: Write failing workflow contract tests**

Change the fixed-mode test to require `baseline_50` and reject
`baseline_10`. Add:

```python
def test_workflow_serializes_eval_runs_without_cancelling() -> None:
    text = _workflow_text()

    assert "group: swe-bench-oci-evaluation" in text
    assert "cancel-in-progress: false" in text


def test_workflow_cache_is_public_dataset_only() -> None:
    text = _workflow_text()
    cache = _named_step(text, "Restore public SWE-bench dataset cache")

    assert "actions/cache@v4" in cache
    assert "swe-bench-verified-main-v1" in cache
    assert "public-hf-cache" in cache
    assert "repopilot-home/eval/datasets" in cache
    for forbidden in (
        "llm_api_key",
        "llm_escalation_api_key",
        "prediction",
        "result.json",
        "target checkout",
        "docker",
    ):
        assert forbidden not in cache.casefold()
```

If `_named_step` currently stops before a `uses:` body, adjust it so a named
step returns all indented fields until the next step. Keep the secret-isolation
test intact.

- [ ] **Step 2: Run workflow tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_swe_bench_oci_workflow.py -q
```

Expected: FAIL because the workflow still exposes `baseline_10` and has no
concurrency group or public dataset cache step.

- [ ] **Step 3: Update the manual workflow**

At workflow scope add:

```yaml
concurrency:
  group: swe-bench-oci-evaluation
  cancel-in-progress: false
```

Replace the second input choice with `baseline_50`. In the instance job add:

```yaml
      HF_HOME: ${{ runner.temp }}/public-hf-cache
```

After Python setup and before dependency installation, add:

```yaml
      - name: Restore public SWE-bench dataset cache
        uses: actions/cache@v4
        with:
          path: |
            ${{ runner.temp }}/public-hf-cache
            ${{ runner.temp }}/repopilot-home/eval/datasets
          key: swe-bench-verified-main-v1
```

Do not cache the rest of `REPOPILOT_HOME`, `$RUNNER_TEMP/instance`, upload
directories, Docker state, credentials, or model outputs.

- [ ] **Step 4: Update the existing README runbook without losing local edits**

Preserve the already-written 100,000-token and OCI safety material. Change:

- default branch instructions and dispatch references from `main` to `master`;
- all public `baseline_10` examples and artifact paths to `baseline_50`;
- “ten-instance baseline” wording to “fixed 50-instance baseline”;
- checkpoint sequencing text to say the five IDs are a subset and the final
  score denominator is exactly 50;
- scoring text to identify official score first and engineering score as the
  auditable secondary metric.

Do not stage or modify the unrelated `run_trace.py` edit.

- [ ] **Step 5: Run workflow tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_swe_bench_oci_workflow.py -q
```

Expected: PASS; secret values remain confined to `Generate patch`, the cache
contains only public dataset paths, manual triggers remain the only triggers,
and both concurrency bounds are present.

- [ ] **Step 6: Run the complete focused OCI suite**

Run:

```bash
.venv/bin/python -m pytest tests/test_oci_contract.py tests/test_oci_runner.py tests/test_oci_aggregate.py tests/test_swe_bench_oci_workflow.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit workflow and runbook changes**

```bash
git add .github/workflows/swe-bench-oci-eval.yml README.md tests/test_swe_bench_oci_workflow.py
git commit -m "ci(eval): dispatch fixed SWE-bench 50 baseline"
```

---

### Task 5: Verify the Branch Before Remote Handoff

**Files:**
- Verify only; do not stage `run_trace.py`.

**Interfaces:**
- Consumes all implementation commits.
- Produces test, lint, and build evidence for review and PR handoff.

- [ ] **Step 1: Run the complete test suite**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: all tests pass; the two environment-gated skips may remain.

- [ ] **Step 2: Run Ruff on the CI scopes**

Run:

```bash
.venv/bin/python -m ruff check src/ tests/ eval/ --select=E,F,I --ignore=E501
```

Expected: `All checks passed!`

- [ ] **Step 3: Build without network isolation**

Run:

```bash
.venv/bin/python -m build --no-isolation
```

Expected: source distribution and wheel are built successfully using installed
dependencies, without attempting to download a build environment.

- [ ] **Step 4: Audit the final diff and secret boundary**

Run:

```bash
git status --short --branch
git diff --check origin/master...HEAD
git diff --name-only origin/master...HEAD
```

Confirm the 50-ID file has 50 unique lines, no credential value or
credential-shaped literal appears in tracked changes, `run_trace.py` remains
unstaged, and the workflow still has no `push` or `pull_request` trigger.

- [ ] **Step 5: Review, push, and create the PR**

After task-level and whole-branch review are clean, push
`fix/release-readiness-20260717`, create a PR targeting `master`, and verify only
the existence of repository secrets named `LLM_API_KEY` and
`LLM_ESCALATION_API_KEY`; never read their values. Do not merge without the
user's explicit authorization.

- [ ] **Step 6: Run live evaluation after authorized merge**

After the workflow definition exists on `master`, dispatch
`checkpoint_5` against the exact feature-branch commit. If its infrastructure
contract passes, dispatch `baseline_50` against the same commit. Download the
aggregate, verify all 50 tracked IDs and hashes, then report the raw official
score and transparent engineering components. Do not mark the goal complete
until every ID has an official terminal verdict or explicit infrastructure
result and the requested score is delivered.
