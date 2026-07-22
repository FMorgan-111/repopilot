# SWE-bench OCI Matrix Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run RepoPilot's fixed five- and ten-instance SWE-bench Verified evaluations with immutable per-instance OCI coverage and official resolved scoring.

**Architecture:** A pure contract module owns fixed-mode validation and safe artifact schemas. A single-instance runner pins the official SWE-bench image, preflights RepoPilot's existing OCI boundary, runs generation with credentials only in that process, and runs the official scorer without model credentials. A manifest-verified aggregator combines isolated matrix artifacts in tracked order, while a manual GitHub Actions workflow limits concurrent model calls to two.

**Tech Stack:** Python 3.10+, Pydantic v2, asyncio, Docker CLI, official `swebench>=4.1,<5`, Hugging Face `datasets`, pytest/pytest-asyncio, GitHub Actions.

## Global Constraints

- Optimize for successful resolution first; evaluate speed only after correctness.
- Modes are exactly `checkpoint_5` and `baseline_10`; IDs come only from `eval/checkpoint_5_ids.txt` and `eval/baseline_10_ids.txt`.
- Dataset is exactly `SWE-bench/SWE-bench_Verified`, split `test`, revision `main`.
- Primary model is `gemini-3.5-flash:stable`; one-way escalation model is `claude-opus-4-8:stable`.
- Generation uses `max_retries=3` and `token_budget=100000`; matrix `max-parallel` is `2`.
- Every usable instance uses the official `swebench/sweb.eval.x86_64.*` image pinned locally to `sha256:<64 lowercase hex>`.
- OCI execution uses `/opt/miniconda3/envs/testbed/bin/python`, user `65532:65532`, memory `4g`, CPUs `2.0`, PIDs `256`, no network, read-only root, dropped capabilities, tmpfs-only writable paths, and a read-only disposable workspace mount.
- Missing dataset, image, Docker, boundary, model, or scorer infrastructure is explicit infrastructure failure; no host execution fallback is allowed.
- Model secrets exist only in the generation workflow step and must not reach subprocesses, containers, scorer steps, cache keys, logs, manifests, or artifacts.
- `swebench.harness.run_evaluation` is the only source of official resolved status; RepoPilot internal success is reported separately.
- Only four safe files are uploaded per instance: `result.json`, `prediction.jsonl`, `official_result.json`, and `manifest.json`.
- The workflow is manual-only and must be merged to the default branch through a user-authorized PR before live dispatch.
- Use red-green-refactor TDD and commit after every independently reviewable task.

---

## File Structure

- Create `eval/oci_contract.py`: fixed-mode lookup, safe Pydantic records, hashing, and cross-field validation.
- Create `eval/oci_runner.py`: CLI and prepare/generate/score/package boundaries for exactly one instance.
- Create `eval/oci_aggregate.py`: verify downloaded bundles and write deterministic combined outputs.
- Create `.github/workflows/swe-bench-oci-eval.yml`: manual prepare, matrix, and aggregate jobs.
- Modify `eval/swe_bench.py`: public exact-row loaders.
- Modify `eval/agent_v2_harness.py`: public safe failure result and exact-instance entry point.
- Modify `src/safe_subprocess.py`: operator-owned resource limits.
- Modify `README.md`: credential-safe runbook and result semantics.
- Create focused tests in `tests/test_oci_contract.py`, `tests/test_oci_runner.py`, `tests/test_oci_aggregate.py`, and `tests/test_swe_bench_oci_workflow.py`; extend existing adapter, harness, and subprocess tests.

---

### Task 1: Fixed Evaluation Contract and Exact Dataset Lookup

**Files:**
- Create: `eval/oci_contract.py`
- Modify: `eval/swe_bench.py`
- Test: `tests/test_oci_contract.py`
- Test: `tests/test_swe_bench.py`

**Interfaces:**
- Produces: `load_mode_instance_ids(mode: EvalMode, repo_root: Path = REPO_ROOT) -> tuple[str, ...]`
- Produces: `require_mode_instance(mode: EvalMode, instance_id: str) -> None`
- Produces: `load_verified_rows(*, dataset_loader=load_dataset) -> list[Mapping[str, Any]]`
- Produces: `load_verified_instance(instance_id: str, *, dataset_loader=load_dataset) -> Mapping[str, Any]`
- Produces: `RuntimeRecord`, `OfficialResult`, `InstanceManifest`, `sha256_file(path: Path) -> str`, and `write_model(path: Path, model: BaseModel) -> None`

- [x] **Step 1: Write contract tests that fail before the module exists**

```python
def test_mode_ids_preserve_tracked_order(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "checkpoint_5_ids.txt").write_text("b\na\n", encoding="utf-8")
    assert load_mode_instance_ids("checkpoint_5", tmp_path) == ("b", "a")

def test_manifest_requires_image_sha_only_for_ready_runtime() -> None:
    with pytest.raises(ValidationError, match="image_sha"):
        InstanceManifest(runtime_status="ready", image_sha="", **MANIFEST_BASE)
    manifest = InstanceManifest(
        runtime_status="oci_image_infra", image_sha="", **MANIFEST_BASE
    )
    assert manifest.image_sha == ""

def test_manifest_rejects_evaluator_only_fields() -> None:
    with pytest.raises(ValidationError):
        InstanceManifest.model_validate({**VALID_MANIFEST, "gold_patch": "secret"})
```

- [x] **Step 2: Run the new tests and confirm import/behavior failures**

Run: `pytest tests/test_oci_contract.py tests/test_swe_bench.py -q`

Expected: FAIL because `eval.oci_contract` and public exact-row loaders do not exist.

- [x] **Step 3: Implement the minimal strict schemas and mode loader**

```python
EvalMode = Literal["checkpoint_5", "baseline_10"]
RuntimeStatus = Literal[
    "ready", "dataset_infra", "oci_image_infra", "oci_boundary_infra"
]

class RuntimeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    mode: EvalMode
    instance_id: str
    dataset_name: Literal["SWE-bench/SWE-bench_Verified"]
    dataset_revision: Literal["main"]
    commit_sha: str
    status: RuntimeStatus
    remote_image: str = ""
    image_sha: str = ""
    python_executable: Literal["/opt/miniconda3/envs/testbed/bin/python"]
    error_class: str = ""

class InstanceManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    mode: EvalMode
    instance_id: str
    commit_sha: str
    runtime_status: RuntimeStatus
    image_sha: str
    primary_model: Literal["gemini-3.5-flash:stable"]
    escalation_model: Literal["claude-opus-4-8:stable"]
    files: dict[Literal["result.json", "prediction.jsonl", "official_result.json"], str]

    @model_validator(mode="after")
    def validate_image_identity(self) -> "InstanceManifest":
        if self.runtime_status == "ready" and not IMAGE_SHA.fullmatch(self.image_sha):
            raise ValueError("ready runtime requires immutable image_sha")
        if self.runtime_status != "ready" and self.image_sha:
            raise ValueError("infrastructure runtime cannot claim image_sha")
        return self
```

Also reject blank/duplicate mode IDs, use an allowlist rather than arbitrary filenames, hash files by streaming bytes, write JSON atomically through a sibling temporary file, and expose exact dataset row lookup without leaking `test_patch` into agent inputs.

- [x] **Step 4: Run focused tests and make them pass**

Run: `pytest tests/test_oci_contract.py tests/test_swe_bench.py -q`

Expected: PASS.

- [x] **Step 5: Commit the contract**

```bash
git add eval/oci_contract.py eval/swe_bench.py tests/test_oci_contract.py tests/test_swe_bench.py
git commit -m "feat(eval): define fixed OCI evaluation contract"
```

### Task 2: Immutable Official Image Preparation

**Files:**
- Create: `eval/oci_runner.py`
- Modify: `src/safe_subprocess.py`
- Test: `tests/test_oci_runner.py`
- Test: `tests/test_safe_subprocess.py`

**Interfaces:**
- Consumes: Task 1 mode, dataset, and runtime interfaces.
- Produces: `prepare_instance(mode: EvalMode, instance_id: str, output_dir: Path, *, command_runner=run_bounded_process) -> RuntimeRecord`
- Produces: CLI `python -m eval.oci_runner prepare --mode MODE --instance-id ID --output-dir DIR`

- [x] **Step 1: Write failing tests for image derivation, digest pinning, limits, and fail-closed behavior**

```python
def test_prepare_pulls_official_image_and_preflights_digest(monkeypatch, tmp_path):
    calls = install_fake_dataset_image_and_docker(monkeypatch)
    record = prepare_instance("checkpoint_5", "pytest-dev__pytest-10081", tmp_path)
    assert record.status == "ready"
    assert record.image_sha == "sha256:" + "a" * 64
    assert calls[0] == ["docker", "pull", "--platform=linux/amd64", OFFICIAL_IMAGE]
    assert calls[1][:3] == ["docker", "image", "inspect"]
    assert calls.preflight.image == record.image_sha

def test_prepare_records_infra_and_never_calls_model(monkeypatch, tmp_path):
    monkeypatch.setattr("eval.oci_runner.load_verified_instance", raise_dataset_error)
    record = prepare_instance("checkpoint_5", "pytest-dev__pytest-10081", tmp_path)
    assert record.status == "dataset_infra"
    assert record.image_sha == ""
    assert (tmp_path / "runtime.json").exists()

def test_resource_limits_are_operator_owned(monkeypatch):
    monkeypatch.setenv("REPOPILOT_TOOL_MEMORY", "4g")
    monkeypatch.setenv("REPOPILOT_TOOL_CPUS", "2.0")
    monkeypatch.setenv("REPOPILOT_TOOL_PIDS_LIMIT", "256")
    config = tool_sandbox_config_from_env()
    assert (config.memory, config.cpus, config.pids_limit) == ("4g", 2.0, 256)
```

- [x] **Step 2: Run the focused tests and confirm failures**

Run: `pytest tests/test_oci_runner.py tests/test_safe_subprocess.py -q`

Expected: FAIL because prepare logic and resource variables are absent.

- [x] **Step 3: Implement official image preparation and boundary preflight**

```python
def _official_image(row: Mapping[str, Any]) -> str:
    image = make_test_spec(row, namespace="swebench").instance_image_key
    if not image.startswith("swebench/sweb.eval.x86_64."):
        raise ValueError("official x86_64 image required")
    return image

def _inspect_sha(image: str, command_runner: CommandRunner) -> str:
    result = command_runner(
        ["docker", "image", "inspect", "--format={{.Id}}", image],
        timeout=60,
    )
    sha = result.stdout.strip()
    if result.returncode or not IMAGE_SHA.fullmatch(sha):
        raise OciImageInfrastructureError("image digest unavailable")
    return sha
```

Extend `tool_sandbox_config_from_env()` to parse bounded memory/CPU/PID values. Build the preflight `ToolSandboxConfig` with the exact global constraints and call the existing `run_oci_process()` to run the testbed Python import check. Catch only classified infrastructure exceptions, write `runtime.json`, return exit status zero for recorded infrastructure outcomes, and never invoke host pytest or repository code.

- [x] **Step 4: Run focused tests and make them pass**

Run: `pytest tests/test_oci_runner.py tests/test_safe_subprocess.py -q`

Expected: PASS.

- [x] **Step 5: Commit image preparation**

```bash
git add eval/oci_runner.py src/safe_subprocess.py tests/test_oci_runner.py tests/test_safe_subprocess.py
git commit -m "feat(eval): prepare immutable SWE-bench OCI runtime"
```

### Task 3: Credential-Separated Generation and Official Scoring

**Files:**
- Modify: `eval/agent_v2_harness.py`
- Modify: `eval/oci_runner.py`
- Test: `tests/test_agent_v2_eval.py`
- Test: `tests/test_oci_runner.py`

**Interfaces:**
- Produces: `safe_failed_sample_result(sample: Mapping[str, Any], error_class: str, *, commit_sha: str) -> dict[str, Any]`
- Produces: `run_exact_verified_instance(instance_id: str, *, output_dir: Path, max_retries: int = 3, token_budget: int = 100000) -> dict[str, Any]`
- Produces: `generate_instance(runtime_path: Path, output_dir: Path) -> None`
- Produces: `score_instance(runtime_path: Path, output_dir: Path, *, scorer=run_evaluation_main) -> OfficialResult`

- [x] **Step 1: Write failing generation/scorer isolation tests**

```python
@pytest.mark.asyncio
async def test_generate_uses_exact_success_first_limits(monkeypatch, ready_runtime, tmp_path):
    seen = install_fake_exact_agent(monkeypatch)
    await generate_instance(ready_runtime, tmp_path)
    assert seen == {
        "instance_id": "pytest-dev__pytest-10081",
        "max_retries": 3,
        "token_budget": 100000,
    }

def test_scorer_rejects_model_credentials(monkeypatch, ready_runtime, tmp_path):
    monkeypatch.setenv("LLM_API_KEY", "must-not-cross-boundary")
    with pytest.raises(CredentialIsolationError):
        score_instance(ready_runtime, tmp_path)

def test_runtime_infra_skips_official_scorer(monkeypatch, infra_runtime, tmp_path):
    scorer = Mock(side_effect=AssertionError("must not run"))
    result = score_instance(infra_runtime, tmp_path, scorer=scorer)
    assert result.status == "scorer_infra"
    scorer.assert_not_called()
```

- [x] **Step 2: Run tests and confirm the public entry points are missing**

Run: `pytest tests/test_agent_v2_eval.py tests/test_oci_runner.py -q`

Expected: FAIL on missing functions and credential boundary behavior.

- [x] **Step 3: Implement exact generation and scorer projection**

```python
SCORER_FORBIDDEN_ENV = frozenset({
    "LLM_API_KEY", "LLM_ESCALATION_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"
})

def _require_credential_free_scorer_env() -> None:
    present = sorted(name for name in SCORER_FORBIDDEN_ENV if os.environ.get(name))
    if present:
        raise CredentialIsolationError("model credentials present in scorer environment")

def _project_official_report(report: Mapping[str, Any], instance_id: str) -> OfficialResult:
    return OfficialResult(
        instance_id=instance_id,
        status=derive_terminal_status(report, instance_id),
        submitted=instance_id in report.get("submitted_ids", []),
        completed=instance_id in report.get("completed_ids", []),
        resolved=instance_id in report.get("resolved_ids", []),
        error_class="",
    )
```

Promote the safe failure helper without weakening its allowlist. For ready runtimes, generation calls the exact-instance harness with retries `3` and token budget `100000`; for infrastructure runtimes it writes a safe failure result and empty-patch prediction without a model call. The scorer runs from a temporary working directory and calls official `swebench.harness.run_evaluation.main()` for exactly one ID, dataset, prediction file, worker, namespace, and 1800-second timeout. It projects only the safe booleans/status/error class into `official_result.json`; raw logs and evaluator material remain outside the upload bundle.

- [x] **Step 4: Run focused tests and make them pass**

Run: `pytest tests/test_agent_v2_eval.py tests/test_oci_runner.py -q`

Expected: PASS.

- [x] **Step 5: Commit generation and scoring**

```bash
git add eval/agent_v2_harness.py eval/oci_runner.py tests/test_agent_v2_eval.py tests/test_oci_runner.py
git commit -m "feat(eval): run and score one OCI benchmark instance"
```

### Task 4: Hash-Bound Packaging and Deterministic Aggregation

**Files:**
- Create: `eval/oci_aggregate.py`
- Modify: `eval/oci_runner.py`
- Test: `tests/test_oci_aggregate.py`
- Test: `tests/test_oci_runner.py`

**Interfaces:**
- Produces: `package_instance(runtime_path: Path, output_dir: Path, artifact_dir: Path) -> InstanceManifest`
- Produces: `aggregate_artifacts(mode: EvalMode, artifacts_dir: Path, output_dir: Path, *, expected_commit: str) -> Path`
- Produces: CLI `package` and `python -m eval.oci_aggregate --mode MODE --artifacts-dir DIR --output-dir DIR --expected-commit SHA`

- [x] **Step 1: Write failing tests for safe packaging and hostile artifact rejection**

```python
def test_package_copies_only_safe_files(tmp_path, completed_instance):
    artifact = tmp_path / "upload"
    manifest = package_instance(
        completed_instance.runtime, completed_instance.output, artifact
    )
    assert sorted(path.name for path in artifact.iterdir()) == [
        "manifest.json", "official_result.json", "prediction.jsonl", "result.json"
    ]
    assert manifest.files["result.json"] == sha256_file(artifact / "result.json")

@pytest.mark.parametrize("mutation", [
    "missing", "duplicate", "extra_instance", "commit", "hash", "model", "unsafe_field"
])
def test_aggregate_rejects_invalid_bundle(tmp_path, valid_bundles, mutation):
    corrupt_bundle(valid_bundles, mutation)
    with pytest.raises(ArtifactContractError):
        aggregate_artifacts("checkpoint_5", valid_bundles, tmp_path / "out", expected_commit=SHA)
```

- [x] **Step 2: Run tests and confirm packaging/aggregation failures**

Run: `pytest tests/test_oci_runner.py tests/test_oci_aggregate.py -q`

Expected: FAIL because package and aggregate interfaces are absent.

- [x] **Step 3: Implement safe bundle validation and ordered outputs**

```python
SAFE_ARTIFACT_FILES = (
    "result.json", "prediction.jsonl", "official_result.json", "manifest.json"
)

def _verify_bundle(bundle: Path, expected_id: str, expected_commit: str) -> VerifiedBundle:
    names = {path.name for path in bundle.iterdir() if path.is_file()}
    if names != set(SAFE_ARTIFACT_FILES):
        raise ArtifactContractError("artifact file set mismatch")
    manifest = InstanceManifest.model_validate_json((bundle / "manifest.json").read_text())
    if manifest.instance_id != expected_id or manifest.commit_sha != expected_commit:
        raise ArtifactContractError("artifact identity mismatch")
    for name, digest in manifest.files.items():
        if not hmac.compare_digest(sha256_file(bundle / name), digest):
            raise ArtifactContractError(f"artifact hash mismatch: {name}")
    return parse_and_scan_safe_files(bundle, manifest)
```

Packaging parses every source file through its strict schema, scans allowed strings for secret/evaluator-only markers, copies only the three safe payloads, hashes copied bytes, and writes the manifest last. Aggregation walks the tracked ID order, rejects missing/duplicate/extra/cross-commit/cross-model/hash-mismatched bundles, and writes `results.json`, `predictions.jsonl`, `official_results.json`, and `summary.md` atomically. Infrastructure failures count separately from official unresolved outcomes.

- [x] **Step 4: Run focused tests and make them pass**

Run: `pytest tests/test_oci_runner.py tests/test_oci_aggregate.py -q`

Expected: PASS.

- [x] **Step 5: Commit artifact handling**

```bash
git add eval/oci_runner.py eval/oci_aggregate.py tests/test_oci_runner.py tests/test_oci_aggregate.py
git commit -m "feat(eval): aggregate hash-bound OCI results"
```

### Task 5: Manual GitHub Actions Matrix Workflow

**Files:**
- Create: `.github/workflows/swe-bench-oci-eval.yml`
- Create: `tests/test_swe_bench_oci_workflow.py`

**Interfaces:**
- Consumes: Tasks 1-4 CLIs.
- Produces: manual workflow `swe-bench-oci-eval.yml` with modes `checkpoint_5` and `baseline_10`.

- [x] **Step 1: Write failing textual workflow contract tests**

```python
def test_workflow_is_manual_only_and_bounded() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text
    assert "pull_request:" not in text
    assert "\npush:" not in text
    assert "max-parallel: 2" in text

def test_secrets_exist_only_in_generation_step() -> None:
    steps = split_workflow_steps(WORKFLOW.read_text(encoding="utf-8"))
    secret_steps = [name for name, body in steps if "secrets.LLM_" in body]
    assert secret_steps == ["Generate patch"]

def test_upload_uses_sanitized_bundle_only() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "--artifact-dir $RUNNER_TEMP/upload" in text
    assert "path: ${{ runner.temp }}/upload" in text
```

- [x] **Step 2: Run the workflow tests and confirm the file is missing**

Run: `pytest tests/test_swe_bench_oci_workflow.py -q`

Expected: FAIL because the workflow does not exist.

- [x] **Step 3: Implement the exact manual prepare/matrix/aggregate workflow**

```yaml
on:
  workflow_dispatch:
    inputs:
      mode:
        type: choice
        required: true
        options: [checkpoint_5, baseline_10]

jobs:
  instance:
    strategy:
      fail-fast: false
      max-parallel: 2
      matrix: ${{ fromJSON(needs.prepare.outputs.matrix) }}
    steps:
      - name: Prepare OCI runtime
        run: python -m eval.oci_runner prepare --mode "$MODE" --instance-id "$INSTANCE_ID" --output-dir "$RUNNER_TEMP/instance"
      - name: Generate patch
        env:
          LLM_API_KEY: ${{ secrets.LLM_API_KEY }}
          LLM_ESCALATION_API_KEY: ${{ secrets.LLM_ESCALATION_API_KEY }}
        run: python -m eval.oci_runner generate --runtime "$RUNNER_TEMP/instance/runtime.json" --output-dir "$RUNNER_TEMP/instance"
      - name: Score prediction without model credentials
        run: python -m eval.oci_runner score --runtime "$RUNNER_TEMP/instance/runtime.json" --output-dir "$RUNNER_TEMP/instance"
      - name: Package safe artifact
        run: python -m eval.oci_runner package --runtime "$RUNNER_TEMP/instance/runtime.json" --output-dir "$RUNNER_TEMP/instance" --artifact-dir "$RUNNER_TEMP/upload"
```

The prepare job emits the exact tracked ID list as JSON. The instance job installs pinned project/eval dependencies, uses `ubuntu-latest`, sets exact model/resource variables outside secret-bearing steps, uploads only `$RUNNER_TEMP/upload`, and uses `if: always()` for score/package/upload so classified failures still yield artifacts. The aggregate job downloads only this run's named artifacts, validates the current commit, writes combined outputs, and uploads the final bundle. No cache contains credentials or generated repository content.

- [x] **Step 4: Run workflow tests and YAML syntax validation**

Run: `pytest tests/test_swe_bench_oci_workflow.py -q`

Expected: PASS.

Run: `ruby -e 'require "yaml"; YAML.load_file(".github/workflows/swe-bench-oci-eval.yml", aliases: true)'`

Expected: exit 0.

- [x] **Step 5: Commit the workflow**

```bash
git add .github/workflows/swe-bench-oci-eval.yml tests/test_swe_bench_oci_workflow.py
git commit -m "ci(eval): add SWE-bench OCI matrix workflow"
```

### Task 6: Runbook, Verification, Review, and Live Evaluation

**Files:**
- Modify: `README.md`
- Verify: all files from Tasks 1-5
- Output (not committed): downloaded workflow artifacts under a temporary directory.

**Interfaces:**
- Consumes: completed manual workflow and `gh` authentication.
- Produces: reviewed feature branch plus official checkpoint-5 and baseline-10 result artifacts.

- [ ] **Step 1: Document the credential-safe operator sequence**

Add exact commands and semantics:

```bash
gh workflow run swe-bench-oci-eval.yml -f mode=checkpoint_5
gh run list --workflow=swe-bench-oci-eval.yml --limit 5
gh run watch RUN_ID --exit-status
gh run download RUN_ID --name swe-bench-oci-checkpoint_5 --dir ARTIFACT_DIR
```

Document that the two previously exposed API keys must be revoked, only rotated keys may be stored as repository secrets named `LLM_API_KEY` and `LLM_ESCALATION_API_KEY`, internal success is not official resolution, infrastructure failure is not unresolved, and the 10-instance run begins only after checkpoint artifact integrity passes.

- [ ] **Step 2: Run focused and full verification**

Run: `pytest tests/test_oci_contract.py tests/test_oci_runner.py tests/test_oci_aggregate.py tests/test_swe_bench_oci_workflow.py tests/test_swe_bench.py tests/test_agent_v2_eval.py tests/test_safe_subprocess.py -q`

Expected: PASS.

Run: `pytest -q`

Expected: all tests pass with only documented skips.

Run: `ruff check .`

Expected: `All checks passed!`

Run: `python -m build`

Expected: wheel and source distribution build successfully.

Run: `git diff --check && git status --short`

Expected: no whitespace errors; only intended documentation changes remain before commit.

- [ ] **Step 3: Commit documentation**

```bash
git add README.md
git commit -m "docs(eval): document OCI matrix evaluation"
```

- [ ] **Step 4: Perform security and correctness review**

Inspect the complete feature diff for secret references outside the generation step, host fallbacks, unpinned image use after preparation, evaluator-only fields, raw-log uploads, arbitrary workflow IDs, unclassified infrastructure failures, and official/internal metric conflation. Re-run any focused test affected by review fixes and commit each fix separately.

- [ ] **Step 5: Push the reviewed feature branch**

Run: `git push origin fix/release-readiness-20260717`

Expected: remote branch advances to the verified local commit.

- [ ] **Step 6: Obtain explicit authorization, merge to default, and configure rotated secrets**

Create or update a PR only with user approval. Do not merge until the user explicitly authorizes it. After merge, ask the user to configure rotated repository secrets through GitHub's secret UI or `gh secret set`; never request secret values in chat or command arguments.

- [ ] **Step 7: Dispatch and audit the five-instance checkpoint**

Run the checkpoint workflow, watch it to a terminal conclusion, download the aggregate artifact, verify its hashes/order/commit and all five infrastructure terminal states, and record the official resolved count plus internal-success count. If infrastructure properties fail, diagnose and fix the implementation before any ten-instance run.

- [ ] **Step 8: Dispatch and audit the ten-instance baseline**

After checkpoint acceptance, run the baseline workflow to terminal conclusion, download the aggregate artifact, verify all ten requested IDs, and report official resolved count, internal-success count, infrastructure failures, escalation totals, decisive failure taxonomy, commit SHA, workflow run IDs, and artifact paths.

- [ ] **Step 9: Final completion audit**

Confirm both runs are terminal, all 15 requested matrix records are represented, the official harness owns every resolved verdict, no credentials/evaluator material appear in artifacts, and the pushed branch contains every verified implementation commit. Only then mark the eval goal complete.
