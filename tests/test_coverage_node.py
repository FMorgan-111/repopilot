from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.coverage_gate import CoverageCandidate, CoverageDecision
from src.nodes.coverage import ensure_coverage
from src.state import (
    AgentState,
    CoverageProof,
    Phase,
    SnapshotManifestEntry,
    ToolPatchApproval,
    ToolSandboxConfig,
    VerifiedEditBatch,
    tool_manifest_fingerprint,
)
from src.state import (
    TestRunFingerprint as RunFingerprint,
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
    patch = subprocess.run(
        ["git", "-C", str(root), "diff", "--binary", ref, "--"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    content = (root / "src" / "a.py").read_bytes()
    manifest = [
        SnapshotManifestEntry(
            path="src/a.py",
            change="modified",
            mode="100644",
            content_sha256=hashlib.sha256(content).hexdigest(),
            size=len(content),
        )
    ]
    return AgentState(
        issue_url="https://github.com/a/b/issues/1",
        repo_path=str(root),
        repo_ref=ref,
        current_phase=Phase.COVERAGE,
        skip_commit=True,
        patch_content=patch,
        tool_patch_approval=ToolPatchApproval(
            base_ref=ref,
            patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),
            patch_gate_fingerprint="c" * 64,
            changed_manifest=manifest,
            manifest_fingerprint=tool_manifest_fingerprint(manifest),
        ),
        tool_sandbox_config=ToolSandboxConfig(
            backend="docker",
            image="registry.example/repopilot-tests@sha256:" + "3" * 64,
            python_executable="/usr/bin/python3",
        ),
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


def _valid_proof_payload() -> dict:
    candidate = CoverageCandidate(
        test_files=["tests/test_a.py"],
        argv=["python", "-m", "pytest", "tests/test_a.py", "-q"],
        source="existing",
    )
    decision = _verified(candidate)
    return {
        "source": "existing",
        "status": "existing_verified",
        "test_files": candidate.test_files,
        "argv": candidate.argv,
        "fixed_runs": decision.fixed_runs,
        "base_runs": decision.base_runs,
        "base_ref": "a" * 40,
        "patch_sha256": "b" * 64,
        "patch_gate_fingerprint": "c" * 64,
        "manifest_fingerprint": "d" * 64,
        "test_content_digests": {"tests/test_a.py": "e" * 64},
    }


def _proof_for_state(state: AgentState, candidate: CoverageCandidate) -> CoverageProof:
    approval = state.tool_patch_approval
    assert approval is not None
    decision = _verified(candidate)
    return CoverageProof(
        source=candidate.source,
        status="existing_verified",
        test_files=candidate.test_files,
        argv=candidate.argv,
        fixed_runs=decision.fixed_runs,
        base_runs=decision.base_runs,
        base_ref=state.repo_ref,
        patch_sha256=approval.patch_sha256,
        patch_gate_fingerprint=approval.patch_gate_fingerprint,
        manifest_fingerprint=approval.manifest_fingerprint,
        test_content_digests={
            path: hashlib.sha256(
                (Path(state.repo_path) / path).read_bytes()
            ).hexdigest()
            for path in candidate.test_files
        },
    )


def test_coverage_proof_requires_semantic_differential_evidence_and_bindings():
    valid = _valid_proof_payload()
    proof = CoverageProof.model_validate(valid)
    assert proof.base_ref == "a" * 40

    invalid_payloads = []
    base_pass = {**valid, "base_runs": valid["fixed_runs"]}
    invalid_payloads.append(base_pass)
    different_ids = {**valid, "base_runs": [*valid["base_runs"]]}
    different_ids["base_runs"] = [
        valid["base_runs"][0],
        valid["base_runs"][1].model_copy(
            update={"failing_test_ids": ["tests/test_a.py::other"]}
        ),
    ]
    invalid_payloads.append(different_ids)
    invalid_payloads.append({**valid, "argv": []})
    invalid_payloads.append({**valid, "base_ref": "A" * 40})
    invalid_payloads.append({**valid, "patch_sha256": "f" * 63})
    invalid_payloads.append({**valid, "test_content_digests": {}})
    invalid_payloads.append({**valid, "status": "generated_verified"})

    for payload in invalid_payloads:
        with pytest.raises(ValidationError):
            CoverageProof.model_validate(payload)


def test_test_run_fingerprint_removes_evaluator_fields_and_values():
    sentinel = "raw-evaluator-diff-7291"
    run = RunFingerprint(
        exit_code=1,
        outcome="infra",
        failing_test_ids=["instance_id=secret-id-9821"],
        summary=f"gold_patch={sentinel}",
    )
    rendered = run.model_dump_json()
    assert sentinel not in rendered
    assert "secret-id-9821" not in rendered
    assert "gold_patch" not in rendered.casefold()
    assert "instance_id" not in rendered.casefold()


@pytest.mark.asyncio
async def test_persisted_proof_is_revalidated_and_rerun_before_terminal_success(
    tmp_path,
    monkeypatch,
):
    state = _state(tmp_path)
    candidate = CoverageCandidate(
        test_files=["tests/test_a.py"],
        argv=["python", "-m", "pytest", "tests/test_a.py", "-q"],
        source="existing",
    )
    state.coverage_status = "existing_verified"
    state.coverage_proof = _proof_for_state(state, candidate)
    calls = 0

    async def rerun(_state, rerun_candidate):
        nonlocal calls
        calls += 1
        assert rerun_candidate == candidate
        return _verified(candidate)

    monkeypatch.setattr("src.nodes.coverage.validate_differential_coverage", rerun)
    result = await ensure_coverage(state)
    assert calls == 1
    assert result.current_phase == Phase.DONE


@pytest.mark.asyncio
async def test_tampered_persisted_proof_binding_never_fast_paths_to_done(
    tmp_path,
    monkeypatch,
):
    state = _state(tmp_path)
    candidate = CoverageCandidate(
        test_files=["tests/test_a.py"],
        argv=["python", "-m", "pytest", "tests/test_a.py", "-q"],
        source="existing",
    )
    state.coverage_status = "existing_verified"
    state.coverage_proof = _proof_for_state(state, candidate)
    state.patch_content += "tampered"
    monkeypatch.setattr("src.nodes.coverage.collect_changed_targets", lambda *_: [])
    monkeypatch.setattr("src.nodes.coverage.discover_coverage_candidates", lambda *_: [])

    async def failed_generation(*_):
        return CoverageDecision(verified=False, status="failed", reason="binding_mismatch")

    monkeypatch.setattr("src.nodes.coverage.run_test_generation_attempts", failed_generation)
    result = await ensure_coverage(state)
    assert result.current_phase == Phase.FAILURE
    assert result.coverage_proof is None
    assert result.pr_url is None


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
async def test_live_drift_before_proof_persistence_is_coverage_infra(
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

    def drifted(_state):
        raise ValueError("live binding drifted")

    monkeypatch.setattr("src.nodes.coverage.validate_differential_coverage", validate)
    monkeypatch.setattr("src.nodes.coverage.validate_live_coverage_binding", drifted)
    result = await ensure_coverage(state)

    assert result.current_phase == Phase.FAILURE
    assert result.coverage_status == "failed"
    assert result.coverage_failure_reason == "coverage_infra"
    assert result.coverage_proof is None
    assert result.pr_url is None


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
