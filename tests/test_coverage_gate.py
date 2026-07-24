from __future__ import annotations

import asyncio
import hashlib
import subprocess
from pathlib import Path

import pytest

from src.async_safety import CancellationDrainError
from src.coverage_gate import (
    CoverageCandidate,
    collect_changed_targets,
    is_allowed_test_path,
    normalize_test_result,
    temporary_base_checkout,
    validate_differential_coverage,
)
from src.safe_subprocess import BoundedProcessResult
from src.state import (
    AgentState,
    GeneratedTestApproval,
    SnapshotManifestEntry,
    TestRunFingerprint as RunFingerprint,
    ToolPatchApproval,
    ToolSandboxConfig,
    tool_manifest_fingerprint,
)
from src.tool_policy import PYTEST_BOOTSTRAP

_IMAGE = "registry.example/repopilot-tests@sha256:" + "2" * 64


@pytest.fixture(autouse=True)
def _mock_coverage_oci(monkeypatch):
    async def fake_oci(argv, *, sandbox, **_kwargs):
        fixed = "return 2" in (sandbox.workspace / "src" / "maths.py").read_text()
        if fixed:
            return BoundedProcessResult(list(argv), 0, "1 passed", "")
        return BoundedProcessResult(
            list(argv),
            1,
            "FAILED tests/test_maths.py::test_answer - AssertionError: expected 2 got 1",
            "",
        )

    monkeypatch.setattr(
        "src.coverage_gate.run_oci_process_async", fake_oci, raising=False
    )


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def differential_repo(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Test")
    (root / "src").mkdir()
    (root / "tests").mkdir()
    (root / "src" / "maths.py").write_text(
        "def answer():\n    return 1\n", encoding="utf-8"
    )
    (root / "tests" / "test_maths.py").write_text(
        "from src.maths import answer\n\n"
        "def test_answer():\n    assert answer() == 2\n",
        encoding="utf-8",
    )
    _git(root, "add", "--all")
    _git(root, "commit", "-qm", "base")
    base_ref = _git(root, "rev-parse", "HEAD")
    (root / "src" / "maths.py").write_text(
        "def answer():\n    return 2\n", encoding="utf-8"
    )
    state = AgentState(
        issue_url="https://github.com/a/b/issues/1",
        repo_path=str(root),
        repo_ref=base_ref,
        test_command="python -m pytest tests/test_maths.py -q",
        tool_sandbox_config=ToolSandboxConfig(
            backend="docker",
            image=_IMAGE,
            python_executable="/usr/bin/python3",
        ),
    )
    patch = subprocess.run(
        ["git", "-C", str(root), "diff", "--binary", base_ref, "--"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    content = (root / "src" / "maths.py").read_bytes()
    manifest = [
        SnapshotManifestEntry(
            path="src/maths.py",
            change="modified",
            mode="100644",
            content_sha256=hashlib.sha256(content).hexdigest(),
            size=len(content),
        )
    ]
    state.patch_content = patch
    state.tool_patch_approval = ToolPatchApproval(
        base_ref=base_ref,
        patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),
        patch_gate_fingerprint="a" * 64,
        changed_manifest=manifest,
        manifest_fingerprint=tool_manifest_fingerprint(manifest),
    )
    candidate = CoverageCandidate(
        test_files=["tests/test_maths.py"],
        argv=["python", "-m", "pytest", "tests/test_maths.py", "-q"],
        source="existing",
    )
    return root, base_ref, state, candidate


@pytest.mark.asyncio
async def test_differential_coverage_never_runs_repository_code_on_host(
    differential_repo,
    monkeypatch,
):
    root, _base_ref, state, candidate = differential_repo
    workspaces: list[Path] = []
    sentinel = "host-credential-must-not-leak"
    monkeypatch.setenv("COVERAGE_HOST_SENTINEL", sentinel)

    async def forbidden_host(*_args, **_kwargs):
        pytest.fail("host test runner was called")

    async def fake_oci(argv, *, sandbox, config, **_kwargs):
        assert config == state.tool_sandbox_config
        assert sandbox.workspace != root
        assert argv == [
            "/usr/bin/python3",
            "-I",
            "-c",
            PYTEST_BOOTSTRAP,
            "tests/test_maths.py",
            "-q",
        ]
        workspaces.append(sandbox.workspace)
        fixed = "return 2" in (sandbox.workspace / "src" / "maths.py").read_text()
        if fixed:
            return BoundedProcessResult(list(argv), 0, "1 passed", "")
        return BoundedProcessResult(
            list(argv),
            1,
            (
                "FAILED tests/test_maths.py::test_answer - "
                f"AssertionError: expected 2 got 1 {sentinel}"
            ),
            "",
        )

    monkeypatch.setattr("src.coverage_gate._run_candidate", forbidden_host)
    monkeypatch.setattr(
        "src.coverage_gate.run_oci_process_async", fake_oci, raising=False
    )
    decision = await validate_differential_coverage(state, candidate)

    assert decision.verified is True
    assert len(workspaces) == 4
    assert len({str(path) for path in workspaces}) == 4
    assert sentinel not in decision.model_dump_json()
    assert sentinel not in state.model_dump_json()


@pytest.mark.parametrize(
    ("recovery", "expected_calls"),
    [("fixed", 1), ("base", 3)],
)
@pytest.mark.asyncio
async def test_differential_coverage_reraises_cancellation_drain(
    differential_repo,
    monkeypatch,
    recovery,
    expected_calls,
):
    _root, _base_ref, state, candidate = differential_repo
    cancellation = asyncio.CancelledError(f"cancel {recovery} run")
    cleanup_error = RuntimeError(f"{recovery} cleanup failed")
    sentinel = CancellationDrainError(
        f"coverage {recovery}",
        cancellation,
        cleanup_error,
    )
    calls = 0

    async def run(_state, _candidate, *, apply_approved_changes):
        nonlocal calls
        calls += 1
        if recovery == "fixed" or not apply_approved_changes:
            raise sentinel
        return RunFingerprint(exit_code=0, outcome="pass", summary="pass")

    monkeypatch.setattr("src.coverage_gate._run_isolated_candidate", run)

    with pytest.raises(CancellationDrainError) as raised:
        await validate_differential_coverage(state, candidate)

    assert raised.value is sentinel
    assert calls == expected_calls
    assert state.coverage_proof is None


@pytest.mark.asyncio
async def test_generated_coverage_keeps_snapshot_alive_until_cancelled_worker_drains(
    differential_repo,
    monkeypatch,
):
    _root, _base_ref, state, candidate = differential_repo
    started = asyncio.Event()
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    captured: dict[str, Path] = {}

    async def draining_oci(argv, *, sandbox, **_kwargs):
        captured["workspace"] = sandbox.workspace
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            assert sandbox.workspace.is_dir()
            cleanup_started.set()
            await cleanup_release.wait()
            assert sandbox.workspace.is_dir()
            raise

    monkeypatch.setattr(
        "src.coverage_gate.run_oci_process_async", draining_oci, raising=False
    )
    task = asyncio.create_task(validate_differential_coverage(state, candidate))
    await asyncio.wait_for(started.wait(), timeout=3)
    task.cancel("cancel coverage OCI")
    await asyncio.wait_for(cleanup_started.wait(), timeout=3)

    workspace = captured["workspace"]
    assert workspace.is_dir()
    cleanup_release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()
    assert not workspace.exists()


@pytest.mark.asyncio
async def test_differential_coverage_rejects_snapshot_manifest_drift(
    differential_repo,
    monkeypatch,
):
    _root, _base_ref, state, candidate = differential_repo

    async def mutating_oci(argv, *, sandbox, **_kwargs):
        (sandbox.workspace / "tests" / "test_maths.py").write_text("mutated")
        return BoundedProcessResult(list(argv), 0, "1 passed", "")

    monkeypatch.setattr(
        "src.coverage_gate.run_oci_process_async", mutating_oci, raising=False
    )
    decision = await validate_differential_coverage(state, candidate)
    assert decision.verified is False
    assert decision.reason == "coverage_infra"
    assert state.coverage_proof is None


@pytest.mark.asyncio
async def test_differential_coverage_missing_oci_never_falls_back_to_host(
    differential_repo,
    monkeypatch,
):
    _root, _base_ref, state, candidate = differential_repo
    state.tool_sandbox_config = None

    async def forbidden_host(*_args, **_kwargs):
        pytest.fail("host fallback was called")

    monkeypatch.setattr("src.coverage_gate._run_candidate", forbidden_host)
    decision = await validate_differential_coverage(state, candidate)
    assert decision.verified is False
    assert decision.reason == "coverage_infra"


@pytest.mark.asyncio
async def test_generated_candidate_is_overlaid_on_base_snapshot_only(
    differential_repo,
    monkeypatch,
):
    root, _base_ref, state, _existing = differential_repo
    generated = root / "tests" / "test_generated.py"
    generated.write_text(
        "from src.maths import answer\n\n"
        "def test_generated():\n    assert answer() == 2\n",
        encoding="utf-8",
    )
    from src.nodes.execute import git_diff

    full_patch = git_diff(str(root))
    approval = state.tool_patch_approval
    assert approval is not None
    generated_bytes = generated.read_bytes()
    generated_entry = SnapshotManifestEntry(
        path="tests/test_generated.py",
        change="added",
        mode="100644",
        content_sha256=hashlib.sha256(generated_bytes).hexdigest(),
        size=len(generated_bytes),
    )
    manifest = [*approval.changed_manifest, generated_entry]
    combined_fingerprint = "7" * 64
    state.patch_content = full_patch
    state.tool_patch_approval = ToolPatchApproval(
        base_ref=state.repo_ref,
        patch_sha256=hashlib.sha256(full_patch.encode()).hexdigest(),
        patch_gate_fingerprint=combined_fingerprint,
        changed_manifest=manifest,
        manifest_fingerprint=tool_manifest_fingerprint(manifest),
    )
    state.coverage_test_files = ["tests/test_generated.py"]
    state.generated_test_approvals = [
        GeneratedTestApproval(
            path="tests/test_generated.py",
            content_sha256=generated_entry.content_sha256,
            patch_gate_fingerprint=combined_fingerprint,
        )
    ]
    candidate = CoverageCandidate(
        test_files=["tests/test_generated.py"],
        argv=["python", "-m", "pytest", "tests/test_generated.py", "-q"],
        source="generated",
    )

    async def generated_oci(argv, *, sandbox, **_kwargs):
        assert (sandbox.workspace / "tests" / "test_generated.py").is_file()
        fixed = "return 2" in (sandbox.workspace / "src" / "maths.py").read_text()
        if fixed:
            return BoundedProcessResult(list(argv), 0, "1 passed", "")
        return BoundedProcessResult(
            list(argv),
            1,
            "FAILED tests/test_generated.py::test_generated - AssertionError: expected 2 got 1",
            "",
        )

    monkeypatch.setattr(
        "src.coverage_gate.run_oci_process_async", generated_oci, raising=False
    )
    decision = await validate_differential_coverage(state, candidate)
    assert decision.verified is True
    assert decision.status == "generated_verified"


@pytest.mark.asyncio
async def test_modified_generated_test_is_bound_and_overlaid_on_exact_base(
    differential_repo,
    monkeypatch,
):
    root, base_ref, state, _existing = differential_repo
    generated = root / "tests" / "test_maths.py"
    base_test = generated.read_bytes()
    generated.write_text(
        "from src.maths import answer\n\n"
        "def test_answer():\n    assert answer() == 2, 'generated regression'\n",
        encoding="utf-8",
    )
    from src.nodes.execute import git_diff

    full_patch = git_diff(str(root))
    approval = state.tool_patch_approval
    assert approval is not None
    generated_bytes = generated.read_bytes()
    generated_entry = SnapshotManifestEntry(
        path="tests/test_maths.py",
        change="modified",
        mode="100644",
        content_sha256=hashlib.sha256(generated_bytes).hexdigest(),
        size=len(generated_bytes),
    )
    manifest = [*approval.changed_manifest, generated_entry]
    combined_fingerprint = "6" * 64
    state.patch_content = full_patch
    state.tool_patch_approval = ToolPatchApproval(
        base_ref=base_ref,
        patch_sha256=hashlib.sha256(full_patch.encode()).hexdigest(),
        patch_gate_fingerprint=combined_fingerprint,
        changed_manifest=manifest,
        manifest_fingerprint=tool_manifest_fingerprint(manifest),
    )
    state.coverage_test_files = ["tests/test_maths.py"]
    state.generated_test_approvals = [
        GeneratedTestApproval(
            path="tests/test_maths.py",
            change="modified",
            base_content_sha256=hashlib.sha256(base_test).hexdigest(),
            content_sha256=generated_entry.content_sha256,
            patch_gate_fingerprint=combined_fingerprint,
        )
    ]
    candidate = CoverageCandidate(
        test_files=["tests/test_maths.py"],
        argv=["python", "-m", "pytest", "tests/test_maths.py", "-q"],
        source="generated",
    )

    async def generated_oci(argv, *, sandbox, **_kwargs):
        assert "generated regression" in (
            sandbox.workspace / "tests" / "test_maths.py"
        ).read_text()
        fixed = "return 2" in (sandbox.workspace / "src" / "maths.py").read_text()
        if fixed:
            return BoundedProcessResult(list(argv), 0, "1 passed", "")
        return BoundedProcessResult(
            list(argv),
            1,
            "FAILED tests/test_maths.py::test_answer - AssertionError: expected 2 got 1",
            "",
        )

    monkeypatch.setattr(
        "src.coverage_gate.run_oci_process_async", generated_oci, raising=False
    )
    decision = await validate_differential_coverage(state, candidate)
    assert decision.verified is True
    assert decision.status == "generated_verified"


@pytest.mark.asyncio
async def test_differential_coverage_requires_stable_fixed_pass_base_failure(
    differential_repo,
):
    _root, _base_ref, state, candidate = differential_repo
    decision = await validate_differential_coverage(state, candidate)
    assert decision.verified is True
    assert decision.status == "existing_verified"
    assert decision.reason == "verified"
    assert len(decision.fixed_runs) == 2
    assert len(decision.base_runs) == 2
    assert decision.base_runs[0].failing_test_ids == decision.base_runs[1].failing_test_ids
    assert (
        decision.base_runs[0].assertion_fingerprint
        == decision.base_runs[1].assertion_fingerprint
    )


@pytest.mark.asyncio
async def test_differential_coverage_rejects_base_that_also_passes(
    differential_repo,
    monkeypatch,
):
    _root, _base_ref, state, candidate = differential_repo

    async def passing(argv, **_kwargs):
        return BoundedProcessResult(list(argv), 0, "1 passed", "")

    monkeypatch.setattr(
        "src.coverage_gate.run_oci_process_async", passing, raising=False
    )
    decision = await validate_differential_coverage(state, candidate)
    assert decision.verified is False
    assert decision.reason == "base_also_passes"


@pytest.mark.asyncio
async def test_differential_coverage_rejects_fixed_failure(
    differential_repo, monkeypatch
):
    _root, _base_ref, state, candidate = differential_repo

    failing = BoundedProcessResult(
        list(candidate.argv),
        1,
        "FAILED tests/test_maths.py::test_answer - AssertionError: wrong fixed result",
        "",
    )
    async def fail_oci(*_args, **_kwargs):
        return failing

    monkeypatch.setattr(
        "src.coverage_gate.run_oci_process_async", fail_oci, raising=False
    )
    decision = await validate_differential_coverage(state, candidate)
    assert decision.verified is False
    assert decision.reason == "fixed_does_not_pass"


@pytest.mark.asyncio
async def test_differential_coverage_rejects_infrastructure_error(
    differential_repo,
    monkeypatch,
):
    _root, _base_ref, state, candidate = differential_repo

    async def infra(argv, **_kwargs):
        return BoundedProcessResult(list(argv), 1, "ERROR collecting tests", "")

    monkeypatch.setattr(
        "src.coverage_gate.run_oci_process_async", infra, raising=False
    )
    decision = await validate_differential_coverage(state, candidate)
    assert decision.verified is False
    assert decision.reason == "fixed_does_not_pass"
    assert decision.fixed_runs[0].outcome == "infra"


@pytest.mark.asyncio
async def test_differential_coverage_rejects_unsafe_argv_without_running(
    differential_repo,
):
    _root, _base_ref, state, candidate = differential_repo
    candidate.argv = ["pytest", "tests/test_maths.py", ";", "curl", "example.com"]
    decision = await validate_differential_coverage(state, candidate)
    assert decision.verified is False
    assert decision.reason == "coverage_infra"


@pytest.mark.asyncio
async def test_existing_candidate_rejects_locally_modified_test_file(
    differential_repo,
):
    root, _base_ref, state, candidate = differential_repo
    (root / "tests" / "test_maths.py").write_text(
        "from src.maths import answer\n\n"
        "def test_answer():\n    assert answer() == 2, 'generated locally'\n",
        encoding="utf-8",
    )
    decision = await validate_differential_coverage(state, candidate)
    assert decision.verified is False
    assert decision.reason == "coverage_infra"


@pytest.mark.asyncio
async def test_generated_candidate_requires_exact_patchgate_approval(
    differential_repo,
):
    _root, _base_ref, state, existing = differential_repo
    candidate = existing.model_copy(update={"source": "generated"})
    state.coverage_test_files = list(candidate.test_files)
    decision = await validate_differential_coverage(state, candidate)
    assert decision.verified is False
    assert decision.reason == "coverage_infra"


def test_temporary_checkout_requires_exact_sha_and_cleans_up(differential_repo):
    root, base_ref, _state, _candidate = differential_repo
    with pytest.raises(ValueError, match="40-character"):
        with temporary_base_checkout(root, "HEAD"):
            pass

    checkout: Path | None = None
    with temporary_base_checkout(root, base_ref) as path:
        checkout = path
        assert _git(path, "rev-parse", "HEAD") == base_ref
        detached = subprocess.run(
            ["git", "-C", str(path), "symbolic-ref", "-q", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert detached.returncode == 1
    assert checkout is not None
    assert not checkout.exists()
    assert str(root.resolve()) in _git(root, "worktree", "list", "--porcelain")
    assert str(checkout) not in _git(root, "worktree", "list", "--porcelain")


def test_temporary_checkout_cleans_up_after_error(differential_repo):
    root, base_ref, _state, _candidate = differential_repo
    checkout: Path | None = None
    with pytest.raises(RuntimeError, match="boom"):
        with temporary_base_checkout(root, base_ref) as path:
            checkout = path
            raise RuntimeError("boom")
    assert checkout is not None
    assert not checkout.exists()
    assert str(checkout) not in _git(root, "worktree", "list", "--porcelain")


def test_test_paths_reject_traversal_symlink_and_non_test_files(tmp_path: Path):
    root = tmp_path / "repo"
    (root / "tests").mkdir(parents=True)
    (root / "src").mkdir()
    (root / "tests" / "test_ok.py").write_text("def test_ok(): pass\n")
    (root / "src" / "prod.py").write_text("value = 1\n")
    (root / "tests" / "test_link.py").symlink_to(root / "src" / "prod.py")

    assert is_allowed_test_path(root, "tests/test_ok.py")
    assert is_allowed_test_path(root, "tests/test_new.py")
    assert not is_allowed_test_path(root, "../tests/test_ok.py")
    assert not is_allowed_test_path(root, "tests/test_link.py")
    assert not is_allowed_test_path(root, "src/prod.py")
    assert not is_allowed_test_path(root, "tests/conftest.py")


def test_normalize_test_result_removes_paths_timings_lines_and_ids(tmp_path: Path):
    first = (
        f"FAILED {tmp_path}/tests/test_bug.py::test_bug - AssertionError: "
        "expected 2 got 1 at /tmp/run-a/mod.py:41 in 0.13s "
        "550e8400-e29b-41d4-a716-446655440000\n"
    )
    second = (
        f"FAILED {tmp_path}/tests/test_bug.py::test_bug - AssertionError: "
        "expected 2 got 1 at /private/tmp/run-b/mod.py:997 in 8.91s "
        "c56a4180-65aa-42ec-a945-5fd21dec0538\n"
    )
    one = normalize_test_result(1, first, tmp_path)
    two = normalize_test_result(1, second, tmp_path)
    assert one.outcome == "assertion_failure"
    assert one.failing_test_ids == two.failing_test_ids == [
        "tests/test_bug.py::test_bug"
    ]
    assert one.assertion_fingerprint == two.assertion_fingerprint
    assert str(tmp_path) not in one.summary
    assert "0.13" not in one.summary


def test_normalize_assertion_text_containing_not_found_is_not_infra(tmp_path: Path):
    result = normalize_test_result(
        1,
        "FAILED tests/test_bug.py::test_bug - AssertionError: expected 'not found' got 'ok'\n",
        tmp_path,
    )
    assert result.outcome == "assertion_failure"


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (
            "FAILED tests/test_bug.py::test_bug - AssertionError: expected 2 got 1",
            "assertion_failure",
        ),
        (
            "FAIL: test_bug (tests.test_bug.Case.test_bug)\nAssertionError: expected 2 got 1",
            "assertion_failure",
        ),
        (
            "ERROR tests/test_bug.py::test_bug - AssertionError during teardown",
            "infra",
        ),
        (
            "ERROR: test_bug (tests.test_bug.Case.test_bug)\nAssertionError in setUp",
            "infra",
        ),
        (
            "ERROR at setup of test_bug\nAssertionError: fixture exploded",
            "infra",
        ),
        (
            "ERROR collecting tests/test_bug.py\nAssertionError while importing",
            "infra",
        ),
    ],
)
def test_normalize_only_accepts_terminal_assertion_failures(
    tmp_path,
    output,
    expected,
):
    result = normalize_test_result(1, output, tmp_path)
    assert result.outcome == expected


def test_collect_changed_targets_uses_python_symbols(differential_repo):
    root, base_ref, _state, _candidate = differential_repo
    targets = collect_changed_targets(root, base_ref)
    assert targets
    assert targets[0].file_path == "src/maths.py"
    assert "answer" in targets[0].symbols
