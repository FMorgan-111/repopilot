from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.coverage_gate import (
    CoverageCandidate,
    collect_changed_targets,
    is_allowed_test_path,
    normalize_test_result,
    temporary_base_checkout,
    validate_differential_coverage,
)
from src.state import AgentState, TestRunFingerprint as RunFingerprint


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
    )
    candidate = CoverageCandidate(
        test_files=["tests/test_maths.py"],
        argv=["python", "-m", "pytest", "tests/test_maths.py", "-q"],
        source="existing",
    )
    return root, base_ref, state, candidate


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

    async def passing(*_):
        return RunFingerprint(exit_code=0, outcome="pass", summary="pass")

    monkeypatch.setattr("src.coverage_gate._run_candidate", passing)
    decision = await validate_differential_coverage(state, candidate)
    assert decision.verified is False
    assert decision.reason == "base_also_passes"


@pytest.mark.asyncio
async def test_differential_coverage_rejects_fixed_failure(differential_repo):
    root, _base_ref, state, candidate = differential_repo
    (root / "src" / "maths.py").write_text(
        "def answer():\n    return 3\n", encoding="utf-8"
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

    async def infra(*_):
        return RunFingerprint(exit_code=1, outcome="infra", summary="infra")

    monkeypatch.setattr("src.coverage_gate._run_candidate", infra)
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


def test_collect_changed_targets_uses_python_symbols(differential_repo):
    root, base_ref, _state, _candidate = differential_repo
    targets = collect_changed_targets(root, base_ref)
    assert targets
    assert targets[0].file_path == "src/maths.py"
    assert "answer" in targets[0].symbols
