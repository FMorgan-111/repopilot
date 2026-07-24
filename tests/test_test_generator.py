from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

import src.test_generator as test_generator
from src.async_safety import CancellationDrainError, wait_for_phase
from src.coverage_gate import ChangedTarget, CoverageDecision
from src.patch_gate import validate_patch_batch
from src.state import (
    AgentState,
    CoverageProof,
    PatchEdit,
    RepairPlan,
    SnapshotManifestEntry,
    ToolPatchApproval,
    VerifiedEdit,
    VerifiedEditBatch,
    TestRunFingerprint as RunFingerprint,
    _estimate_tokens,
    tool_manifest_fingerprint,
)
from src.test_generator import (
    _SYSTEM_PROMPT,
    is_allowed_test_path,
    request_test_batch,
    run_test_generation_attempts,
)
from src.timeout_diagnostics import extract_timeout_cleanup_evidence


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def generation_state(tmp_path: Path) -> AgentState:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Test")
    (root / "src").mkdir()
    (root / "tests").mkdir()
    (root / "src" / "calc.py").write_text(
        "def answer():\n    return 1\n", encoding="utf-8"
    )
    (root / "tests" / "test_smoke.py").write_text(
        "def test_smoke():\n    assert True\n", encoding="utf-8"
    )
    _git(root, "add", "--all")
    _git(root, "commit", "-qm", "base")
    ref = _git(root, "rev-parse", "HEAD")
    (root / "src" / "calc.py").write_text(
        "def answer():\n    return 2\n", encoding="utf-8"
    )
    patch = subprocess.run(
        ["git", "-C", str(root), "diff", "--binary", ref, "--"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    content = (root / "src" / "calc.py").read_bytes()
    manifest = [
        SnapshotManifestEntry(
            path="src/calc.py",
            change="modified",
            mode="100644",
            content_sha256=hashlib.sha256(content).hexdigest(),
            size=len(content),
        )
    ]
    production_plan = RepairPlan(
        root_cause="answer returned the wrong constant",
        target_files=["src/calc.py"],
        target_symbols=[],
        required_behavior="return two",
        regression_test_strategy="run targeted tests",
    )
    return AgentState(
        issue_url="https://github.com/a/b/issues/1",
        issue_title="Wrong answer",
        issue_body="answer should return two",
        repo_path=str(root),
        repo_ref=ref,
        active_provider="primary",
        active_model="gemini-3.5-flash:stable",
        patch_content=patch,
        patch_edits=[
            PatchEdit(file_path="src/calc.py", search="return 1", replace="return 2")
        ],
        active_repair_plan=production_plan,
        tool_patch_approval=ToolPatchApproval(
            base_ref=ref,
            patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),
            patch_gate_fingerprint="8" * 64,
            changed_manifest=manifest,
            manifest_fingerprint=tool_manifest_fingerprint(manifest),
        ),
    )


def _edit(path: str, *, search: str = "", replace: str) -> VerifiedEdit:
    return VerifiedEdit(
        file_path=path,
        search=search,
        replace=replace,
        intent="Add a targeted regression test.",
    )


def _bad_batch() -> VerifiedEditBatch:
    return VerifiedEditBatch(
        edits=[
            _edit(
                "src/calc.py",
                search="return 2",
                replace="return 3",
            )
        ]
    )


def _good_batch() -> VerifiedEditBatch:
    return VerifiedEditBatch(
        edits=[
            _edit(
                "tests/test_answer.py",
                replace=(
                    "from src.calc import answer\n\n"
                    "def test_answer():\n    assert answer() == 2\n"
                ),
            )
        ]
    )


class _UnknownModelError(Exception):
    pass


def _seed_coverage_snapshot(state: AgentState) -> CoverageProof:
    approval = state.tool_patch_approval
    assert approval is not None
    fixed = [RunFingerprint(exit_code=0, outcome="pass", summary="pass")] * 2
    base = [
        RunFingerprint(
            exit_code=1,
            outcome="assertion_failure",
            failing_test_ids=["tests/test_smoke.py::test_smoke"],
            assertion_fingerprint="a" * 64,
            summary="assertion_failure",
        )
    ] * 2
    proof = CoverageProof(
        source="existing",
        status="existing_verified",
        test_files=["tests/test_smoke.py"],
        argv=["python", "-m", "pytest", "tests/test_smoke.py", "-q"],
        fixed_runs=fixed,
        base_runs=base,
        base_ref=state.repo_ref,
        patch_sha256=approval.patch_sha256,
        patch_gate_fingerprint=approval.patch_gate_fingerprint,
        manifest_fingerprint=approval.manifest_fingerprint,
        test_content_digests={"tests/test_smoke.py": "b" * 64},
    )
    state.coverage_status = "existing_verified"
    state.coverage_test_files = ["tests/test_smoke.py"]
    state.coverage_test_command = "python -m pytest tests/test_smoke.py -q"
    state.coverage_failure_reason = "prior_coverage_failure"
    state.coverage_proof = proof
    return proof


def _assert_coverage_snapshot(state: AgentState, proof: CoverageProof) -> None:
    assert state.coverage_status == "existing_verified"
    assert state.coverage_test_files == ["tests/test_smoke.py"]
    assert state.coverage_test_command == "python -m pytest tests/test_smoke.py -q"
    assert state.coverage_failure_reason == "prior_coverage_failure"
    assert state.coverage_proof == proof


def test_patch_gate_test_only_rejects_production_even_when_plan_allows_it(
    generation_state,
):
    root = Path(generation_state.repo_path)
    before = (root / "src" / "calc.py").read_bytes()
    edit = _bad_batch().edits[0]
    edit._expected_content_sha256 = hashlib.sha256(before).hexdigest()
    plan = RepairPlan(
        root_cause="missing behavioral regression test",
        target_files=["src/calc.py"],
        target_symbols=[],
        required_behavior="prove the issue fails on base and passes on the fix",
        regression_test_strategy="targeted differential test",
    )
    generation_state.active_repair_plan = plan
    result = validate_patch_batch(
        generation_state,
        plan,
        VerifiedEditBatch(edits=[edit]),
        test_only=True,
    )
    assert result.accepted is False
    assert result.issues[0].code == "scope_violation"


@pytest.mark.asyncio
async def test_test_generator_is_test_only_and_escalates_second_attempt(
    generation_state,
    monkeypatch,
):
    responses = iter([_bad_batch(), _good_batch()])
    requests: list[tuple[str, str, dict[str, object]]] = []

    async def request(system, prompt, **_kwargs):
        raw = next(responses).model_dump()
        requests.append((system, prompt, raw))
        return raw

    async def verify(_state, candidate):
        return CoverageDecision(
            verified=True,
            status="generated_verified",
            reason="verified",
            candidate=candidate,
        )

    monkeypatch.setattr("src.test_generator.llm_call", request)
    monkeypatch.setattr("src.test_generator.validate_differential_coverage", verify)
    monkeypatch.setenv("REPOPILOT_ESCALATION_ENABLED", "1")
    monkeypatch.setenv("LLM_ESCALATION_API_KEY", "sentinel-not-for-prompts")

    result = await run_test_generation_attempts(generation_state, "base_also_passes")

    assert result.verified is True
    assert generation_state.test_generation_attempts == 2
    assert generation_state.active_provider == "escalation"
    assert [entry.provider for entry in generation_state.model_history] == [
        "primary",
        "escalation",
    ]
    assert [entry.model for entry in generation_state.model_history] == [
        "gemini-3.5-flash:stable",
        "claude-opus-4-8:stable",
    ]
    assert generation_state.token_usage == sum(
        _estimate_tokens(system, prompt)
        + _estimate_tokens(
            json.dumps(
                raw,
                ensure_ascii=False,
                sort_keys=True,
                default=lambda value: type(value).__name__,
            )
        )
        for system, prompt, raw in requests
    )
    assert all(
        is_allowed_test_path(Path(generation_state.repo_path), path)
        for path in generation_state.coverage_test_files
    )
    assert "tests/test_answer.py" in generation_state.patch_content


@pytest.mark.asyncio
async def test_missing_escalation_credentials_retries_primary(
    generation_state,
    monkeypatch,
):
    calls: list[str] = []

    async def invalid(*_args, **kwargs):
        calls.append(kwargs["provider"])
        return _bad_batch().model_dump()

    monkeypatch.setattr("src.test_generator.llm_call", invalid)
    result = await run_test_generation_attempts(generation_state, "no_candidate")
    assert result.verified is False
    assert calls == ["primary", "primary"]
    assert generation_state.test_generation_attempts == 2


@pytest.mark.asyncio
async def test_invalid_generation_restores_exact_production_authorization(
    generation_state,
    monkeypatch,
):
    original_plan = generation_state.active_repair_plan.model_copy(deep=True)
    original_edits = [edit.model_copy(deep=True) for edit in generation_state.patch_edits]
    original_approval = generation_state.tool_patch_approval.model_copy(deep=True)
    original_patch = generation_state.patch_content

    async def invalid(*_args, **_kwargs):
        return _bad_batch().model_dump()

    monkeypatch.setattr("src.test_generator.llm_call", invalid)
    result = await run_test_generation_attempts(generation_state, "no_candidate")
    assert result.verified is False
    assert generation_state.active_repair_plan == original_plan
    assert generation_state.patch_edits == original_edits
    assert generation_state.tool_patch_approval == original_approval
    assert generation_state.patch_content == original_patch
    assert generation_state.generated_test_approvals == []
    assert generation_state.coverage_test_files == []


@pytest.mark.asyncio
async def test_successful_generation_keeps_production_and_builds_combined_binding(
    generation_state,
    monkeypatch,
):
    original_plan = generation_state.active_repair_plan.model_copy(deep=True)
    original_edits = [edit.model_copy(deep=True) for edit in generation_state.patch_edits]
    original_manifest = list(
        generation_state.tool_patch_approval.changed_manifest
    )

    async def good(*_args, **_kwargs):
        return _good_batch().model_dump()

    async def verified(_state, candidate):
        return CoverageDecision(
            verified=True,
            status="generated_verified",
            reason="verified",
            candidate=candidate,
        )

    monkeypatch.setattr("src.test_generator.llm_call", good)
    monkeypatch.setattr("src.test_generator.validate_differential_coverage", verified)
    result = await run_test_generation_attempts(generation_state, "no_candidate")
    approval = generation_state.tool_patch_approval
    assert result.verified is True
    assert generation_state.active_repair_plan == original_plan
    assert generation_state.patch_edits == original_edits
    assert approval is not None
    assert {entry.path for entry in approval.changed_manifest} == {
        "src/calc.py",
        "tests/test_answer.py",
    }
    assert all(entry in approval.changed_manifest for entry in original_manifest)
    assert approval.patch_sha256 == hashlib.sha256(
        generation_state.patch_content.encode()
    ).hexdigest()
    assert generation_state.generated_test_approvals[0].patch_gate_fingerprint == (
        approval.patch_gate_fingerprint
    )


@pytest.mark.asyncio
async def test_rollback_failure_hard_stops_without_second_request_or_prediction_leak(
    generation_state,
    monkeypatch,
):
    original_patch = generation_state.patch_content
    requests = 0

    async def good(*_args, **_kwargs):
        nonlocal requests
        requests += 1
        return _good_batch().model_dump()

    async def rejected(_state, candidate):
        return CoverageDecision(
            verified=False,
            status="failed",
            reason="base_also_passes",
            candidate=candidate,
        )

    monkeypatch.setattr("src.test_generator.llm_call", good)
    monkeypatch.setattr("src.test_generator.validate_differential_coverage", rejected)
    monkeypatch.setattr(
        "src.test_generator._restore_test_files",
        lambda *_: (_ for _ in ()).throw(RuntimeError("rollback failed")),
    )
    result = await run_test_generation_attempts(generation_state, "no_candidate")
    assert requests == 1
    assert result.reason == "generated_test_rollback_failed"
    assert generation_state.coverage_proof is None
    assert generation_state.pr_url is None
    assert generation_state.patch_content == original_patch
    assert "tests/test_answer.py" not in generation_state.patch_content


@pytest.mark.asyncio
async def test_applied_test_cancellation_drain_rolls_back_before_reraise(
    generation_state,
    monkeypatch,
):
    root = Path(generation_state.repo_path)
    original_plan = generation_state.active_repair_plan.model_copy(deep=True)
    original_edits = [
        edit.model_copy(deep=True) for edit in generation_state.patch_edits
    ]
    original_approval = generation_state.tool_patch_approval.model_copy(deep=True)
    original_patch = generation_state.patch_content
    cancellation = asyncio.CancelledError("cancel generated coverage")
    cleanup_error = RuntimeError("coverage cleanup failed")
    sentinel = CancellationDrainError(
        "generated coverage",
        cancellation,
        cleanup_error,
    )

    async def good(*_args, **_kwargs):
        return _good_batch().model_dump()

    async def cancelled(_state, _candidate):
        raise sentinel

    monkeypatch.setattr("src.test_generator.llm_call", good)
    monkeypatch.setattr(
        "src.test_generator.validate_differential_coverage",
        cancelled,
    )

    with pytest.raises(CancellationDrainError) as raised:
        await run_test_generation_attempts(generation_state, "no_candidate")

    assert raised.value is sentinel
    assert raised.value.cancellation is cancellation
    assert raised.value.cleanup_error is cleanup_error
    assert not (root / "tests" / "test_answer.py").exists()
    assert generation_state.active_repair_plan == original_plan
    assert generation_state.patch_edits == original_edits
    assert generation_state.tool_patch_approval == original_approval
    assert generation_state.patch_content == original_patch
    assert generation_state.coverage_failure_reason == ""


@pytest.mark.asyncio
async def test_applied_test_direct_cancellation_rolls_back_before_reraise(
    generation_state,
    monkeypatch,
):
    root = Path(generation_state.repo_path)
    original_plan = generation_state.active_repair_plan.model_copy(deep=True)
    original_edits = [
        edit.model_copy(deep=True) for edit in generation_state.patch_edits
    ]
    original_approval = generation_state.tool_patch_approval.model_copy(deep=True)
    original_patch = generation_state.patch_content
    sentinel = asyncio.CancelledError("cancel generated coverage")

    async def good(*_args, **_kwargs):
        return _good_batch().model_dump()

    async def cancelled(_state, _candidate):
        raise sentinel

    monkeypatch.setattr("src.test_generator.llm_call", good)
    monkeypatch.setattr(
        "src.test_generator.validate_differential_coverage",
        cancelled,
    )

    with pytest.raises(asyncio.CancelledError) as raised:
        await run_test_generation_attempts(generation_state, "no_candidate")

    assert raised.value is sentinel
    assert not (root / "tests" / "test_answer.py").exists()
    assert generation_state.active_repair_plan == original_plan
    assert generation_state.patch_edits == original_edits
    assert generation_state.tool_patch_approval == original_approval
    assert generation_state.patch_content == original_patch
    assert generation_state.coverage_failure_reason == ""


@pytest.mark.asyncio
async def test_applied_test_rollback_failure_chains_under_pending_drain(
    generation_state,
    monkeypatch,
):
    original_plan = generation_state.active_repair_plan.model_copy(deep=True)
    original_edits = [
        edit.model_copy(deep=True) for edit in generation_state.patch_edits
    ]
    original_approval = generation_state.tool_patch_approval.model_copy(deep=True)
    original_patch = generation_state.patch_content
    cancellation = asyncio.CancelledError("cancel generated coverage")
    cleanup_error = RuntimeError("coverage cleanup failed")
    rollback_error = RuntimeError("generated test rollback failed")
    sentinel = CancellationDrainError(
        "generated coverage",
        cancellation,
        cleanup_error,
    )

    async def good(*_args, **_kwargs):
        return _good_batch().model_dump()

    async def cancelled(_state, _candidate):
        raise sentinel

    def failed_rollback(*_args):
        raise rollback_error

    monkeypatch.setattr("src.test_generator.llm_call", good)
    monkeypatch.setattr(
        "src.test_generator.validate_differential_coverage",
        cancelled,
    )
    monkeypatch.setattr(
        "src.test_generator._restore_test_files",
        failed_rollback,
    )

    with pytest.raises(CancellationDrainError) as raised:
        await run_test_generation_attempts(generation_state, "no_candidate")

    assert raised.value is sentinel
    assert raised.value.cancellation is cancellation
    assert raised.value.cleanup_error is cleanup_error
    assert raised.value.__cause__ is rollback_error
    assert generation_state.active_repair_plan == original_plan
    assert generation_state.patch_edits == original_edits
    assert generation_state.tool_patch_approval == original_approval
    assert generation_state.patch_content == original_patch
    assert generation_state.coverage_failure_reason == ""


@pytest.mark.asyncio
async def test_applied_test_rollback_failure_chains_under_direct_cancellation(
    generation_state,
    monkeypatch,
):
    original_plan = generation_state.active_repair_plan.model_copy(deep=True)
    original_edits = [
        edit.model_copy(deep=True) for edit in generation_state.patch_edits
    ]
    original_approval = generation_state.tool_patch_approval.model_copy(deep=True)
    original_patch = generation_state.patch_content
    sentinel = asyncio.CancelledError("cancel generated coverage")
    rollback_error = RuntimeError("generated test rollback failed")

    async def good(*_args, **_kwargs):
        return _good_batch().model_dump()

    async def cancelled(_state, _candidate):
        raise sentinel

    def failed_rollback(*_args):
        raise rollback_error

    monkeypatch.setattr("src.test_generator.llm_call", good)
    monkeypatch.setattr(
        "src.test_generator.validate_differential_coverage",
        cancelled,
    )
    monkeypatch.setattr(
        "src.test_generator._restore_test_files",
        failed_rollback,
    )

    with pytest.raises(CancellationDrainError) as raised:
        await run_test_generation_attempts(generation_state, "no_candidate")

    assert raised.value.operation == "generated test rollback"
    assert raised.value.cancellation is sentinel
    assert raised.value.cleanup_error is rollback_error
    assert raised.value.__cause__ is rollback_error
    assert generation_state.active_repair_plan == original_plan
    assert generation_state.patch_edits == original_edits
    assert generation_state.tool_patch_approval == original_approval
    assert generation_state.patch_content == original_patch
    assert generation_state.coverage_failure_reason == ""


@pytest.mark.asyncio
async def test_generation_timeout_retains_direct_cancellation_rollback_evidence(
    generation_state,
    monkeypatch,
):
    rollback_error = RuntimeError("generated test rollback failed")
    captured: dict[str, asyncio.CancelledError] = {}

    async def good(*_args, **_kwargs):
        return _good_batch().model_dump()

    async def in_flight_coverage(_state, _candidate):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as cancellation:
            captured["cancellation"] = cancellation
            raise

    def failed_rollback(*_args):
        raise rollback_error

    monkeypatch.setattr("src.test_generator.llm_call", good)
    monkeypatch.setattr(
        "src.test_generator.validate_differential_coverage",
        in_flight_coverage,
    )
    monkeypatch.setattr(
        "src.test_generator._restore_test_files",
        failed_rollback,
    )

    with pytest.raises(asyncio.TimeoutError) as raised:
        await wait_for_phase(
            run_test_generation_attempts(generation_state, "no_candidate"),
            timeout=0.01,
        )

    terminal = raised.value.__cause__
    assert isinstance(terminal, CancellationDrainError)
    assert terminal.operation == "generated test rollback"
    assert terminal.cancellation is captured["cancellation"]
    assert terminal.cleanup_error is rollback_error
    assert terminal.__cause__ is rollback_error
    evidence = extract_timeout_cleanup_evidence(raised.value)
    assert evidence is not None
    assert evidence.failure_kind == "generic_drain"
    assert evidence.operation == "generated test rollback"
    assert evidence.cause_type == "CancellationDrainError"
    assert evidence.cleanup_error_type == "RuntimeError"


@pytest.mark.asyncio
async def test_already_escalated_provider_never_downgrades(
    generation_state,
    monkeypatch,
):
    generation_state.active_provider = "escalation"
    generation_state.active_model = "claude-opus-4-8:stable"
    generation_state.escalated = True
    calls: list[str] = []

    async def invalid(*_args, **kwargs):
        calls.append(kwargs["provider"])
        return _bad_batch().model_dump()

    monkeypatch.setattr("src.test_generator.llm_call", invalid)
    await run_test_generation_attempts(generation_state, "no_candidate")
    assert calls == ["escalation", "escalation"]
    assert generation_state.active_model == "claude-opus-4-8:stable"


@pytest.mark.asyncio
async def test_successful_test_generation_records_the_complete_request(
    generation_state,
    monkeypatch,
):
    captured: dict[str, str] = {}
    raw = _good_batch().model_dump()

    async def good(system, prompt, **kwargs):
        captured.update(system=system, prompt=prompt, **kwargs)
        return raw

    monkeypatch.setattr("src.test_generator.llm_call", good)

    batch = await request_test_batch(generation_state, "no_candidate")

    response_text = json.dumps(
        raw,
        ensure_ascii=False,
        sort_keys=True,
        default=lambda value: type(value).__name__,
    )
    expected_input = _estimate_tokens(_SYSTEM_PROMPT, captured["prompt"])
    expected_output = _estimate_tokens(response_text)
    assert batch == _good_batch()
    assert generation_state.test_generation_attempts == 1
    assert generation_state.token_usage == expected_input + expected_output
    assert len(generation_state.model_history) == 1
    invocation = generation_state.model_history[0]
    assert invocation.model == "gemini-3.5-flash:stable"
    assert invocation.provider == "primary"
    assert invocation.node == "test_generation"
    assert invocation.status == "ok"
    assert invocation.elapsed_seconds >= 0
    assert invocation.input_tokens == expected_input
    assert invocation.output_tokens == expected_output
    assert invocation.error_class == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [[], {"edits": []}])
async def test_invalid_test_generation_responses_are_debited_and_recorded(
    generation_state,
    monkeypatch,
    raw,
):
    captured: dict[str, str] = {}

    async def invalid(_system, prompt, **_kwargs):
        captured["prompt"] = prompt
        return raw

    monkeypatch.setattr("src.test_generator.llm_call", invalid)

    with pytest.raises(ValueError):
        await request_test_batch(generation_state, "no_candidate")

    response_text = json.dumps(
        raw,
        ensure_ascii=False,
        sort_keys=True,
        default=lambda value: type(value).__name__,
    )
    expected_input = _estimate_tokens(_SYSTEM_PROMPT, captured["prompt"])
    expected_output = _estimate_tokens(response_text)
    assert generation_state.test_generation_attempts == 1
    assert generation_state.token_usage == expected_input + expected_output
    assert len(generation_state.model_history) == 1
    invocation = generation_state.model_history[0]
    assert invocation.status == "invalid_response"
    assert invocation.error_class == "ValueError"
    assert invocation.input_tokens == expected_input
    assert invocation.output_tokens == expected_output


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [OSError("gateway"), RuntimeError("gateway"), ValueError("gateway")],
)
async def test_model_call_failures_are_errors_and_store_only_their_class(
    generation_state,
    monkeypatch,
    error,
):
    captured: dict[str, str] = {}

    async def failed(_system, prompt, **_kwargs):
        captured["prompt"] = prompt
        raise error

    monkeypatch.setattr("src.test_generator.llm_call", failed)

    with pytest.raises(type(error)):
        await request_test_batch(generation_state, "no_candidate")

    expected_input = _estimate_tokens(_SYSTEM_PROMPT, captured["prompt"])
    assert generation_state.test_generation_attempts == 1
    assert generation_state.token_usage == expected_input
    assert len(generation_state.model_history) == 1
    invocation = generation_state.model_history[0]
    assert invocation.status == "error"
    assert invocation.error_class == type(error).__name__
    assert invocation.input_tokens == expected_input
    assert invocation.output_tokens == 0


@pytest.mark.asyncio
async def test_preflight_failure_terminates_without_a_model_request(
    generation_state,
    monkeypatch,
):
    calls = 0

    def preflight_failure(*_args):
        nonlocal calls
        calls += 1
        raise ValueError("oversized prompt")

    monkeypatch.setattr("src.test_generator._generation_prompt", preflight_failure)

    result = await run_test_generation_attempts(generation_state, "no_candidate")

    assert calls == 1
    assert result.reason == "test_generation_preflight_failed"
    assert generation_state.test_generation_attempts == 0
    assert generation_state.token_usage == 0
    assert generation_state.model_history == []


@pytest.mark.asyncio
async def test_budget_before_request_restores_production_state_without_generation(
    generation_state,
    monkeypatch,
):
    root = Path(generation_state.repo_path)
    original_plan = generation_state.active_repair_plan.model_copy(deep=True)
    original_edits = [
        edit.model_copy(deep=True) for edit in generation_state.patch_edits
    ]
    original_approval = generation_state.tool_patch_approval.model_copy(deep=True)
    original_patch = generation_state.patch_content
    calls = {
        "request": 0,
        "patch_gate": 0,
        "apply": 0,
        "coverage": 0,
        "escalation": 0,
    }
    generation_state.token_usage = generation_state.token_budget
    original_coverage_proof = _seed_coverage_snapshot(generation_state)

    async def request(*_args, **_kwargs):
        calls["request"] += 1
        return _good_batch().model_dump()

    def patch_gate(*args, **kwargs):
        calls["patch_gate"] += 1
        return original_patch_gate(*args, **kwargs)

    def apply(*args, **kwargs):
        calls["apply"] += 1
        return original_apply(*args, **kwargs)

    async def coverage(*_args, **_kwargs):
        calls["coverage"] += 1
        return CoverageDecision(verified=False, status="failed", reason="unexpected")

    def escalation(*args, **kwargs):
        calls["escalation"] += 1
        return original_escalation(*args, **kwargs)

    original_patch_gate = test_generator.validate_patch_batch
    original_apply = test_generator.apply_approved_patch
    original_escalation = test_generator.apply_escalation
    monkeypatch.setattr("src.test_generator.llm_call", request)
    monkeypatch.setattr("src.test_generator.validate_patch_batch", patch_gate)
    monkeypatch.setattr("src.test_generator.apply_approved_patch", apply)
    monkeypatch.setattr("src.test_generator.validate_differential_coverage", coverage)
    monkeypatch.setattr("src.test_generator.escalation_is_configured", lambda: True)
    monkeypatch.setattr("src.test_generator.apply_escalation", escalation)

    result = await run_test_generation_attempts(generation_state, "no_candidate")

    assert result.reason == "token_budget_exceeded"
    assert calls == {
        "request": 0,
        "patch_gate": 0,
        "apply": 0,
        "coverage": 0,
        "escalation": 0,
    }
    assert generation_state.active_repair_plan == original_plan
    assert generation_state.patch_edits == original_edits
    assert generation_state.tool_patch_approval == original_approval
    assert generation_state.patch_content == original_patch
    assert generation_state.generated_test_approvals == []
    assert not (root / "tests" / "test_answer.py").exists()
    _assert_coverage_snapshot(generation_state, original_coverage_proof)


@pytest.mark.asyncio
async def test_budget_after_request_restores_production_state_before_patch_gate(
    generation_state,
    monkeypatch,
):
    root = Path(generation_state.repo_path)
    original_plan = generation_state.active_repair_plan.model_copy(deep=True)
    original_edits = [
        edit.model_copy(deep=True) for edit in generation_state.patch_edits
    ]
    original_approval = generation_state.tool_patch_approval.model_copy(deep=True)
    original_patch = generation_state.patch_content
    calls = {
        "request": 0,
        "patch_gate": 0,
        "apply": 0,
        "coverage": 0,
        "escalation": 0,
    }
    generation_state.token_budget = 1
    original_coverage_proof = _seed_coverage_snapshot(generation_state)

    async def request(*_args, **_kwargs):
        calls["request"] += 1
        return _good_batch().model_dump()

    def patch_gate(*args, **kwargs):
        calls["patch_gate"] += 1
        return original_patch_gate(*args, **kwargs)

    def apply(*args, **kwargs):
        calls["apply"] += 1
        return original_apply(*args, **kwargs)

    async def coverage(*_args, **_kwargs):
        calls["coverage"] += 1
        return CoverageDecision(
            verified=True, status="generated_verified", reason="verified"
        )

    def escalation(*args, **kwargs):
        calls["escalation"] += 1
        return original_escalation(*args, **kwargs)

    original_patch_gate = test_generator.validate_patch_batch
    original_apply = test_generator.apply_approved_patch
    original_escalation = test_generator.apply_escalation
    monkeypatch.setattr("src.test_generator.llm_call", request)
    monkeypatch.setattr("src.test_generator.validate_patch_batch", patch_gate)
    monkeypatch.setattr("src.test_generator.apply_approved_patch", apply)
    monkeypatch.setattr("src.test_generator.validate_differential_coverage", coverage)
    monkeypatch.setattr("src.test_generator.escalation_is_configured", lambda: True)
    monkeypatch.setattr("src.test_generator.apply_escalation", escalation)

    result = await run_test_generation_attempts(generation_state, "no_candidate")

    assert result.reason == "token_budget_exceeded"
    assert calls == {
        "request": 1,
        "patch_gate": 0,
        "apply": 0,
        "coverage": 0,
        "escalation": 0,
    }
    assert generation_state.active_repair_plan == original_plan
    assert generation_state.patch_edits == original_edits
    assert generation_state.tool_patch_approval == original_approval
    assert generation_state.patch_content == original_patch
    assert generation_state.generated_test_approvals == []
    assert not (root / "tests" / "test_answer.py").exists()
    _assert_coverage_snapshot(generation_state, original_coverage_proof)


@pytest.mark.asyncio
async def test_budget_after_failed_request_stops_before_escalation_or_retry(
    generation_state,
    monkeypatch,
):
    root = Path(generation_state.repo_path)
    original_plan = generation_state.active_repair_plan.model_copy(deep=True)
    original_edits = [
        edit.model_copy(deep=True) for edit in generation_state.patch_edits
    ]
    original_approval = generation_state.tool_patch_approval.model_copy(deep=True)
    original_patch = generation_state.patch_content
    calls = {"request": 0, "escalation": 0}
    generation_state.token_budget = 1
    original_coverage_proof = _seed_coverage_snapshot(generation_state)

    async def request(*_args, **_kwargs):
        calls["request"] += 1
        raise RuntimeError("request failed")

    def escalation(*args, **kwargs):
        calls["escalation"] += 1
        return original_escalation(*args, **kwargs)

    original_escalation = test_generator.apply_escalation
    monkeypatch.setattr("src.test_generator.llm_call", request)
    monkeypatch.setattr("src.test_generator.escalation_is_configured", lambda: True)
    monkeypatch.setattr("src.test_generator.apply_escalation", escalation)

    result = await run_test_generation_attempts(generation_state, "no_candidate")

    assert result.reason == "token_budget_exceeded"
    assert calls == {"request": 1, "escalation": 0}
    assert generation_state.active_repair_plan == original_plan
    assert generation_state.patch_edits == original_edits
    assert generation_state.tool_patch_approval == original_approval
    assert generation_state.patch_content == original_patch
    assert generation_state.generated_test_approvals == []
    assert not (root / "tests" / "test_answer.py").exists()
    _assert_coverage_snapshot(generation_state, original_coverage_proof)


@pytest.mark.asyncio
async def test_budget_after_unknown_request_error_stops_without_escalation_or_retry(
    generation_state,
    monkeypatch,
):
    original_coverage_proof = _seed_coverage_snapshot(generation_state)
    calls = {"request": 0, "escalation": 0}
    generation_state.token_budget = 1

    async def request(*_args, **_kwargs):
        calls["request"] += 1
        raise _UnknownModelError()

    def escalation(*args, **kwargs):
        calls["escalation"] += 1
        return original_escalation(*args, **kwargs)

    original_escalation = test_generator.apply_escalation
    monkeypatch.setattr("src.test_generator.llm_call", request)
    monkeypatch.setattr("src.test_generator.escalation_is_configured", lambda: True)
    monkeypatch.setattr("src.test_generator.apply_escalation", escalation)

    result = await run_test_generation_attempts(generation_state, "no_candidate")

    assert result.reason == "token_budget_exceeded"
    assert calls == {"request": 1, "escalation": 0}
    _assert_coverage_snapshot(generation_state, original_coverage_proof)


@pytest.mark.asyncio
async def test_unknown_request_error_without_budget_is_reraised(
    generation_state,
    monkeypatch,
):
    error = _UnknownModelError()

    async def request(*_args, **_kwargs):
        raise error

    monkeypatch.setattr("src.test_generator.llm_call", request)

    with pytest.raises(_UnknownModelError) as raised:
        await run_test_generation_attempts(generation_state, "no_candidate")

    assert raised.value is error


@pytest.mark.asyncio
async def test_direct_cancellation_is_recorded_and_reraised_by_identity(
    generation_state,
    monkeypatch,
):
    error = asyncio.CancelledError("external cancellation details")
    captured: dict[str, str] = {}
    token_estimates: list[tuple[str, ...]] = []
    original_estimate = test_generator._estimate_tokens

    def estimate(*parts: str) -> int:
        token_estimates.append(parts)
        return original_estimate(*parts)

    async def cancelled(_system, prompt, **_kwargs):
        captured["prompt"] = prompt
        raise error

    monkeypatch.setattr("src.test_generator._estimate_tokens", estimate)
    monkeypatch.setattr("src.test_generator.llm_call", cancelled)

    with pytest.raises(asyncio.CancelledError) as raised:
        await request_test_batch(generation_state, "no_candidate")

    expected_input = original_estimate(_SYSTEM_PROMPT, captured["prompt"])
    assert raised.value is error
    assert generation_state.test_generation_attempts == 1
    assert generation_state.token_usage == expected_input
    assert token_estimates == [(_SYSTEM_PROMPT, captured["prompt"])]
    assert len(generation_state.model_history) == 1
    invocation = generation_state.model_history[0]
    assert invocation.node == "test_generation"
    assert invocation.status == "cancelled"
    assert invocation.input_tokens == expected_input
    assert invocation.output_tokens == 0
    assert invocation.elapsed_seconds >= 0
    assert invocation.error_class == "CancelledError"
    assert "external cancellation details" not in json.dumps(
        generation_state.model_dump(mode="json")
    )


@pytest.mark.asyncio
async def test_cancellation_drain_is_recorded_and_reraised_without_losing_evidence(
    generation_state,
    monkeypatch,
):
    cancellation = asyncio.CancelledError("cancel-secret")
    cleanup_error = RuntimeError("cleanup-secret")
    error = CancellationDrainError(
        "test generation",
        cancellation,
        cleanup_error,
    )
    captured: dict[str, str] = {}

    async def cancelled(_system, prompt, **_kwargs):
        captured["prompt"] = prompt
        raise error

    monkeypatch.setattr("src.test_generator.llm_call", cancelled)

    with pytest.raises(CancellationDrainError) as raised:
        await request_test_batch(generation_state, "no_candidate")

    assert raised.value is error
    assert raised.value.cancellation is cancellation
    assert raised.value.cleanup_error is cleanup_error
    expected_input = _estimate_tokens(_SYSTEM_PROMPT, captured["prompt"])
    assert generation_state.test_generation_attempts == 1
    assert generation_state.token_usage == expected_input
    assert len(generation_state.model_history) == 1
    invocation = generation_state.model_history[0]
    assert invocation.node == "test_generation"
    assert invocation.status == "cancelled"
    assert invocation.input_tokens == expected_input
    assert invocation.output_tokens == 0
    assert invocation.elapsed_seconds >= 0
    assert invocation.error_class == "CancellationDrainError"
    serialized = json.dumps(generation_state.model_dump(mode="json"))
    assert "cancel-secret" not in serialized
    assert "cleanup-secret" not in serialized


def test_allowed_test_path_requires_established_safe_layout(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    assert not is_allowed_test_path(root, "tests/test_new.py")
    (root / "tests").mkdir()
    assert is_allowed_test_path(root, "tests/test_new.py")
    assert not is_allowed_test_path(root, "tests/test_gold_patch.py")
    assert not is_allowed_test_path(root, "tests/helper.py")
    assert not is_allowed_test_path(root, "tests/data.bin")
    assert not is_allowed_test_path(root, "pyproject.toml")
    assert not is_allowed_test_path(root, ".github/workflows/test_ci.py")
    assert not is_allowed_test_path(root, "src/test_generated.py")


@pytest.mark.parametrize(
    "relative",
    [
        ".github/tests/test_ci.py",
        ".gitlab/tests/test_ci.py",
        "config/tests/test_config.py",
        "configs/tests/test_config.py",
        "vendor/tests/test_vendor.py",
        "third_party/tests/test_dependency.py",
        "dependencies/tests/test_dependency.py",
        "generated/tests/test_generated.py",
        "build/tests/test_build.py",
        ".venv/tests/test_env.py",
        "node_modules/tests/test_node.py",
        ".cache/tests/test_cache.py",
    ],
)
def test_allowed_test_path_rejects_sensitive_and_generated_trees(
    tmp_path,
    relative,
):
    root = tmp_path / "repo"
    path = root / relative
    path.parent.mkdir(parents=True)
    path.write_text("def test_hidden(): assert True\n")
    assert not is_allowed_test_path(root, relative)


@pytest.mark.asyncio
async def test_generator_prompt_excludes_credentials_and_evaluator_fields(
    generation_state,
    monkeypatch,
):
    raw_diff = "RAW-DIFF-SENTINEL-9381"
    hidden_id = "secret-instance-4820"
    hidden_test = "tests/test_hidden_oracle.py::test_secret"
    generation_state.issue_body = (
        "api_key=top-secret-value\n"
        f"gold_patch={raw_diff}\n"
        "FAIL_TO_PASS:\n"
        f"- {hidden_test}\n\n"
        f'{{"test_patch":"{raw_diff}","instance_id":"{hidden_id}"}}'
    )
    captured: dict[str, str] = {}

    async def fake_llm(system, user, **_kwargs):
        captured["system"] = system
        captured["user"] = user
        return _good_batch().model_dump()

    monkeypatch.setattr(
        "src.test_generator.collect_changed_targets",
        lambda *_: [ChangedTarget(file_path="src/calc.py", symbols=["gold_patch"])],
    )
    monkeypatch.setattr("src.test_generator.llm_call", fake_llm)
    await request_test_batch(generation_state, "base_also_passes")
    rendered = captured["system"] + captured["user"]
    assert "top-secret-value" not in rendered
    assert "gold_patch" not in rendered.casefold()
    assert "fail_to_pass" not in rendered.casefold()
    assert "instance_id" not in rendered.casefold()
    assert raw_diff not in rendered
    assert hidden_id not in rendered
    assert hidden_test not in rendered
