# SWE-bench Workflow Context Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the manual SWE-bench workflow valid on GitHub by deriving runner-local paths only after a runner starts, with a regression contract that prevents the invalid job-level context from returning.

**Architecture:** Keep the workflow manual-only and retain the existing runner-temp path layout. Remove runner-derived values from `jobs.instance.env`, then export their resolved `$RUNNER_TEMP` values once through `$GITHUB_ENV` before cache restoration and dependency installation.

**Tech Stack:** GitHub Actions YAML, Python 3.11, pytest, Ruff, Ruby YAML parser, GitHub pull-request checks.

## Global Constraints

- Do not dispatch `checkpoint_5`, `baseline_50`, or any SWE-bench evaluation.
- Do not read, modify, transmit, or validate model credentials.
- Do not change models, prompts, budgets, instance cohorts, scoring, OCI behavior, or evaluation triggers.
- Do not merge PR #2 or begin Pilot-20 work.
- Preserve `run_trace.py` as an unstaged user change with SHA-256 `f721a313a68888a507608ea196b27e173d093f462065bf346c6b6a19f55b8eba`.
- The remote acceptance gate binds to the exact feature-branch head and requires exactly the existing eight CI jobs to succeed.

---

### Task 1: Move Runner-Local Paths to Step Runtime

**Files:**
- Modify: `tests/test_swe_bench_oci_workflow.py:179-231`
- Modify: `.github/workflows/swe-bench-oci-eval.yml:69-99`

**Interfaces:**
- Consumes: `_workflow_text() -> str`, `_mapping_block(text: str, key: str, indent: int) -> str`, `_job_block(text: str, name: str) -> str`, and `_step_blocks(job: str) -> list[tuple[str, str]]`.
- Produces: `_assert_runner_temp_initialization_contract(text: str) -> None`, a positive workflow contract, and a mutation test that rejects `runner` expressions in job-level `env`.

- [ ] **Step 1: Add the failing workflow contract**

Add this helper after `_assert_public_dataset_cache_contract`:

```python
def _assert_runner_temp_initialization_contract(text: str) -> None:
    jobs = _mapping_block(text, "jobs", 0)
    job_envs = re.findall(
        r"(?ms)^    env:\n(?P<body>.*?)(?=^    \S|\Z)",
        jobs,
    )
    assert job_envs
    assert all("${{ runner." not in job_env for job_env in job_envs), (
        "runner context is unavailable in job-level env"
    )

    instance = _job_block(text, "instance")
    steps = _step_blocks(instance)
    names = [name for name, _block in steps]
    configure_name = "Configure temporary evaluation paths"
    assert names.count(configure_name) == 1

    configure_index = names.index(configure_name)
    configure = dict(steps)[configure_name]
    writes = re.findall(
        r'(?m)^          echo "([^"]+)" >> "\$GITHUB_ENV"$',
        configure,
    )
    assert writes == [
        "REPOPILOT_HOME=$RUNNER_TEMP/repopilot-home",
        "HF_HOME=$RUNNER_TEMP/public-hf-cache",
    ]

    for dependent_name in (
        "Restore public SWE-bench dataset cache",
        "Install locked dependencies",
        "Install RepoPilot without dependency resolution",
    ):
        assert configure_index < names.index(dependent_name)
```

Add these tests after `test_workflow_cache_is_public_dataset_only`:

```python
def test_workflow_initializes_runner_paths_after_runner_assignment() -> None:
    _assert_runner_temp_initialization_contract(_workflow_text())


def test_runner_path_contract_rejects_job_level_runner_context() -> None:
    text = _workflow_text()
    mutation = text.replace(
        '      REPOPILOT_ESCALATION_AFTER_NO_PROGRESS: "2"\n',
        '      REPOPILOT_ESCALATION_AFTER_NO_PROGRESS: "2"\n'
        "      FORBIDDEN_PATH: ${{ runner.temp }}/forbidden\n",
        1,
    )

    with pytest.raises(AssertionError, match="job-level env"):
        _assert_runner_temp_initialization_contract(mutation)
```

- [ ] **Step 2: Run the positive contract and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_swe_bench_oci_workflow.py::test_workflow_initializes_runner_paths_after_runner_assignment -q
```

Expected: FAIL with `runner context is unavailable in job-level env`. This
failure must occur before editing the workflow.

- [ ] **Step 3: Apply the minimal workflow fix**

Delete these two keys from the `instance` job-level `env` mapping:

```yaml
      REPOPILOT_HOME: ${{ runner.temp }}/repopilot-home
      HF_HOME: ${{ runner.temp }}/public-hf-cache
```

Immediately after the `Set up Python` step and before
`Restore public SWE-bench dataset cache`, add:

```yaml
      - name: Configure temporary evaluation paths
        run: |
          echo "REPOPILOT_HOME=$RUNNER_TEMP/repopilot-home" >> "$GITHUB_ENV"
          echo "HF_HOME=$RUNNER_TEMP/public-hf-cache" >> "$GITHUB_ENV"
```

Do not alter the cache paths, trigger, matrix, concurrency, secrets, models,
action pins, or command lines.

- [ ] **Step 4: Run both new contracts and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_swe_bench_oci_workflow.py::test_workflow_initializes_runner_paths_after_runner_assignment tests/test_swe_bench_oci_workflow.py::test_runner_path_contract_rejects_job_level_runner_context -q
```

Expected: `2 passed`.

- [ ] **Step 5: Run the complete workflow contract file**

Run:

```bash
.venv/bin/python -m pytest tests/test_swe_bench_oci_workflow.py -q
```

Expected: all tests pass; manual-only triggers, fixed modes, concurrency,
public-cache allowlist, immutable action pins, isolated dependency installs,
secret isolation, sanitized uploads, and the runtime-path contract remain
enforced.

- [ ] **Step 6: Parse the complete YAML document**

Run:

```bash
ruby -e 'require "yaml"; YAML.load_file(".github/workflows/swe-bench-oci-eval.yml", aliases: true)'
```

Expected: exit 0 with no output.

- [ ] **Step 7: Run the focused OCI evaluation suite**

Run:

```bash
.venv/bin/python -m pytest tests/test_oci_contract.py tests/test_oci_runner.py tests/test_oci_aggregate.py tests/test_swe_bench_oci_workflow.py -q
```

Expected: exit 0 with no failed tests.

- [ ] **Step 8: Run the complete Python test suite**

Run:

```bash
.venv/bin/python -m pytest tests/ -q
```

Expected: 1916 passed, 2 skipped, with only the approved sqlite-vec fallback
warning.

- [ ] **Step 9: Run lint, diff, and preserved-file gates**

Run:

```bash
.venv/bin/python -m ruff check src/ tests/ eval/ --select=E,F,I --ignore=E501
git diff --check
shasum -a 256 run_trace.py
git status --short
```

Expected:

- Ruff prints `All checks passed!`.
- `git diff --check` exits 0.
- `run_trace.py` prints SHA-256 `f721a313a68888a507608ea196b27e173d093f462065bf346c6b6a19f55b8eba`.
- Status contains the two intended modified files plus only the preserved,
  unstaged `run_trace.py`.

- [ ] **Step 10: Commit only the regression and workflow fix**

```bash
git add tests/test_swe_bench_oci_workflow.py .github/workflows/swe-bench-oci-eval.yml
git diff --cached --check
git status --short
git commit -m "fix(ci): initialize eval paths on runner"
```

Expected: the commit contains exactly the two intended files;
`run_trace.py` remains unstaged.

---

### Task 2: Obtain Exact-Head Remote Parser and CI Evidence

**Files:**
- Verify only: PR #2 and GitHub Actions
- Preserve: `run_trace.py`

**Interfaces:**
- Consumes: the Task 1 commit SHA and feature branch `fix/release-readiness-20260717`.
- Produces: exact-head evidence that GitHub accepts the manual workflow and the existing eight-job CI run succeeds.

- [ ] **Step 1: Recheck branch identity and local preservation**

Run:

```bash
git branch --show-current
git rev-parse HEAD
git status --short --branch
shasum -a 256 run_trace.py
```

Expected:

- Branch is `fix/release-readiness-20260717`.
- HEAD is the Task 1 implementation commit.
- The branch is ahead of origin only by the new spec/plan/fix commits.
- `run_trace.py` remains the only unstaged working-tree change at the required
  checksum.

- [ ] **Step 2: Push the feature branch without dispatching a workflow**

Run:

```bash
git push origin fix/release-readiness-20260717
```

Expected: the feature branch advances to the exact local HEAD. If sandbox DNS
still prevents the push, report the exact command for the owner to run and do
not use any alternate mutation path.

- [ ] **Step 3: Verify the exact-head GitHub parser result**

Read PR #2 checks and Actions for the pushed SHA.

Expected:

- No invalid-workflow failure for
  `.github/workflows/swe-bench-oci-eval.yml` is attached to the new SHA.
- GitHub recognizes the manual workflow.
- No SWE-bench job is dispatched and no model secret is read.

- [ ] **Step 4: Verify the existing CI matrix**

Read the CI run attached to the exact pushed SHA.

Expected exactly eight successful jobs:

1. `test (ubuntu-latest, 3.10)`
2. `test (ubuntu-latest, 3.11)`
3. `test (ubuntu-latest, 3.12)`
4. `test (macos-latest, 3.10)`
5. `test (macos-latest, 3.11)`
6. `test (macos-latest, 3.12)`
7. `lint`
8. `oci-integration`

Do not click `Run workflow`, rerun an evaluation, merge PR #2, or start
Pilot-20.
