from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from src.coverage_gate import CoverageCandidate, validate_differential_coverage
from src.new_agent import agent_payload_from_state
from src.nodes.commit import commit_fix
from src.patch_gate import apply_approved_patch, validate_patch_batch
from src.safe_subprocess import BoundedProcessResult
from src.state import (
    AgentState,
    CoverageProof,
    Phase,
    RepairPlan,
    ToolSandboxConfig,
    VerifiedEdit,
    VerifiedEditBatch,
)
from src.state import (
    TestRunFingerprint as RunFingerprint,
)

_IMAGE = "registry.example/repopilot-tests@sha256:" + "2" * 64


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _approved_state(tmp_path: Path) -> AgentState:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Test")
    (root / "src").mkdir()
    (root / "tests").mkdir()
    source = b"def answer():\n    return 1\n"
    (root / "src" / "maths.py").write_bytes(source)
    (root / "tests" / "test_maths.py").write_text(
        "from src.maths import answer\n\ndef test_answer():\n    assert answer() == 2\n",
        encoding="utf-8",
    )
    _git(root, "add", "--all")
    _git(root, "commit", "-qm", "base")
    ref = _git(root, "rev-parse", "HEAD")
    plan = RepairPlan(
        root_cause="answer returns the stale value",
        target_files=["src/maths.py"],
        target_symbols=[],
        required_behavior="answer returns two",
        regression_test_strategy="run the focused test",
    )
    edit = VerifiedEdit(
        file_path="src/maths.py",
        search="return 1",
        replace="return 2",
        intent="correct the return value",
    )
    edit._expected_content_sha256 = hashlib.sha256(source).hexdigest()
    edit._exact_only = True
    state = AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        issue_number=7,
        owner="acme",
        repo="widget",
        repo_path=str(root),
        repo_ref=ref,
        active_repair_plan=plan,
        tool_sandbox_config=ToolSandboxConfig(
            backend="docker",
            image=_IMAGE,
            python_executable="/usr/bin/python3",
        ),
    )
    result = validate_patch_batch(state, plan, VerifiedEditBatch(edits=[edit]))
    assert result.accepted
    apply_approved_patch(state)
    return state


def _attach_proof(state: AgentState) -> None:
    approval = state.tool_patch_approval
    assert approval is not None
    test_path = "tests/test_maths.py"
    fixed = [RunFingerprint(exit_code=0, outcome="pass", summary="pass")] * 2
    base = [
        RunFingerprint(
            exit_code=1,
            outcome="assertion_failure",
            failing_test_ids=[f"{test_path}::test_answer"],
            assertion_fingerprint="b" * 64,
            summary="assertion_failure",
        )
    ] * 2
    state.coverage_status = "existing_verified"
    state.coverage_test_files = [test_path]
    state.coverage_proof = CoverageProof(
        source="existing",
        status="existing_verified",
        test_files=[test_path],
        argv=["python", "-m", "pytest", test_path, "-q"],
        fixed_runs=fixed,
        base_runs=base,
        base_ref=state.repo_ref,
        patch_sha256=approval.patch_sha256,
        patch_gate_fingerprint=approval.patch_gate_fingerprint,
        manifest_fingerprint=approval.manifest_fingerprint,
        test_content_digests={
            test_path: hashlib.sha256(
                (Path(state.repo_path) / test_path).read_bytes()
            ).hexdigest()
        },
    )


@pytest.mark.asyncio
async def test_coverage_fails_closed_when_live_patch_drifts_between_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    state = _approved_state(tmp_path)
    root = Path(state.repo_path)
    candidate = CoverageCandidate(
        test_files=["tests/test_maths.py"],
        argv=["python", "-m", "pytest", "tests/test_maths.py", "-q"],
        source="existing",
    )
    calls = 0

    async def mutate_live_after_first_run(argv, *, sandbox, **_kwargs):
        nonlocal calls
        calls += 1
        fixed = "return 2" in (sandbox.workspace / "src" / "maths.py").read_text()
        if calls == 1:
            (root / "src" / "maths.py").write_text(
                "def answer():\n    return 999\n", encoding="utf-8"
            )
        if fixed:
            return BoundedProcessResult(list(argv), 0, "1 passed", "")
        return BoundedProcessResult(
            list(argv),
            1,
            "FAILED tests/test_maths.py::test_answer - AssertionError: expected 2 got 1",
            "",
        )

    monkeypatch.setattr(
        "src.coverage_gate.run_oci_process_async", mutate_live_after_first_run
    )
    decision = await validate_differential_coverage(state, candidate)

    assert decision.verified is False
    assert decision.reason == "coverage_infra"
    assert state.coverage_proof is None


def test_terminal_prediction_uses_only_the_approved_live_patch(tmp_path: Path):
    state = _approved_state(tmp_path)
    _attach_proof(state)
    state.current_phase = Phase.DONE
    approved_patch = state.patch_content

    valid_payload = agent_payload_from_state(state, turns_taken=1)
    assert valid_payload["success"] is True
    assert valid_payload["model_patch"] == approved_patch

    (Path(state.repo_path) / "unknown.py").write_text("UNAPPROVED = True\n")
    drifted_payload = agent_payload_from_state(state, turns_taken=1)

    assert drifted_payload["success"] is False
    assert drifted_payload["model_patch"] == ""


@pytest.mark.parametrize("terminal_phase", [Phase.DONE, Phase.COMMIT])
def test_terminal_prediction_rejects_valid_live_approval_without_proof(
    tmp_path: Path,
    terminal_phase: Phase,
):
    state = _approved_state(tmp_path)
    state.current_phase = terminal_phase

    payload = agent_payload_from_state(state, turns_taken=1)

    assert payload["success"] is False
    assert payload["model_patch"] == ""


@pytest.mark.parametrize("tamper", ["status", "fingerprint", "semantic"])
def test_terminal_prediction_rejects_mismatched_or_tampered_proof(
    tmp_path: Path,
    tamper: str,
):
    state = _approved_state(tmp_path)
    _attach_proof(state)
    state.current_phase = Phase.DONE
    if tamper == "status":
        state.coverage_status = "generated_verified"
    elif tamper == "fingerprint":
        proof = state.coverage_proof
        assert proof is not None
        state.coverage_proof = proof.model_copy(
            update={"patch_gate_fingerprint": "f" * 64}
        )
    else:
        proof = state.coverage_proof
        assert proof is not None
        object.__setattr__(
            state,
            "coverage_proof",
            proof.model_copy(update={"fixed_runs": proof.base_runs}),
        )

    payload = agent_payload_from_state(state, turns_taken=1)

    assert payload["success"] is False
    assert payload["model_patch"] == ""


@pytest.mark.asyncio
async def test_commit_rejects_live_drift_before_any_github_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    state = _approved_state(tmp_path)
    _attach_proof(state)
    state.current_phase = Phase.COMMIT
    (Path(state.repo_path) / "unknown.py").write_text("UNAPPROVED = True\n")
    calls: list[str] = []

    async def forbidden_repo_call(_state):
        calls.append("github")
        pytest.fail("GitHub API was reached before validating the live binding")

    monkeypatch.setattr("src.nodes.commit._github_get_repo", forbidden_repo_call)
    result = await commit_fix(state)

    assert calls == []
    assert result.current_phase == Phase.FAILURE
    assert result.pr_url is None
    assert result.coverage_proof is None


@pytest.mark.asyncio
async def test_commit_rejects_missing_terminal_proof_before_any_github_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    state = _approved_state(tmp_path)
    state.current_phase = Phase.COMMIT
    calls: list[str] = []

    async def forbidden_repo_call(_state):
        calls.append("github")
        pytest.fail("GitHub API was reached without a terminal coverage proof")

    monkeypatch.setattr("src.nodes.commit._github_get_repo", forbidden_repo_call)
    result = await commit_fix(state)

    assert calls == []
    assert result.current_phase == Phase.FAILURE
    assert result.pr_url is None
