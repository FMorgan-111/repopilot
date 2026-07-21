from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.coverage_gate import CoverageCandidate, CoverageDecision
from src.nodes.coverage import ensure_coverage
from src.state import (
    AgentState,
    Phase,
    TestRunFingerprint as RunFingerprint,
    VerifiedEditBatch,
)


def _state(tmp_path: Path) -> AgentState:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "config", "user.name", "Test"], check=True
    )
    (root / "src").mkdir()
    (root / "tests").mkdir()
    (root / "src" / "a.py").write_text("def value():\n    return 1\n")
    (root / "tests" / "test_a.py").write_text(
        "from src.a import value\ndef test_value(): assert value() == 1\n"
    )
    subprocess.run(["git", "-C", str(root), "add", "--all"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "base"], check=True)
    ref = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (root / "src" / "a.py").write_text("def value():\n    return 2\n")
    return AgentState(
        issue_url="https://github.com/a/b/issues/1",
        repo_path=str(root),
        repo_ref=ref,
        current_phase=Phase.COVERAGE,
        skip_commit=True,
    )


def _verified(candidate: CoverageCandidate) -> CoverageDecision:
    fixed = [RunFingerprint(exit_code=0, outcome="pass", summary="pass")] * 2
    base = [
        RunFingerprint(
            exit_code=1,
            outcome="assertion_failure",
            failing_test_ids=["tests/test_a.py::test_value"],
            assertion_fingerprint="a" * 64,
            summary="assertion_failure",
        )
    ] * 2
    return CoverageDecision(
        verified=True,
        status="existing_verified",
        reason="verified",
        candidate=candidate,
        fixed_runs=fixed,
        base_runs=base,
    )


@pytest.mark.asyncio
async def test_existing_candidate_only_succeeds_after_validator(
    tmp_path: Path,
    monkeypatch,
):
    state = _state(tmp_path)
    candidate = CoverageCandidate(
        test_files=["tests/test_a.py"],
        argv=["python", "-m", "pytest", "tests/test_a.py", "-q"],
        source="existing",
    )
    monkeypatch.setattr("src.nodes.coverage.collect_changed_targets", lambda *_: [])
    monkeypatch.setattr(
        "src.nodes.coverage.discover_coverage_candidates", lambda *_: [candidate]
    )

    async def validate(*_):
        return _verified(candidate)

    monkeypatch.setattr("src.nodes.coverage.validate_differential_coverage", validate)
    result = await ensure_coverage(state)
    assert result.current_phase == Phase.DONE
    assert result.coverage_status == "existing_verified"
    assert result.coverage_proof is not None
    assert result.coverage_proof.argv == candidate.argv


@pytest.mark.asyncio
async def test_two_invalid_generated_tests_fail_without_pr(
    tmp_path: Path,
    monkeypatch,
):
    state = _state(tmp_path)
    state.pr_url = None
    monkeypatch.setattr("src.nodes.coverage.collect_changed_targets", lambda *_: [])
    monkeypatch.setattr("src.nodes.coverage.discover_coverage_candidates", lambda *_: [])

    async def invalid(_state, _reason):
        return VerifiedEditBatch(
            edits=[
                {
                    "file_path": "src/a.py",
                    "search": "return 2",
                    "replace": "return 3",
                    "intent": "bad production edit",
                }
            ]
        )

    monkeypatch.setattr("src.test_generator.request_test_batch", invalid)
    result = await ensure_coverage(state)
    assert result.coverage_status == "failed"
    assert result.current_phase == Phase.FAILURE
    assert result.failure_reason.startswith("test_generation_failed:")
    assert result.pr_url is None
