from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from src.coverage_gate import ChangedTarget, CoverageDecision
from src.patch_gate import validate_patch_batch
from src.state import AgentState, RepairPlan, VerifiedEdit, VerifiedEditBatch
from src.state import (
    PatchEdit,
    SnapshotManifestEntry,
    ToolPatchApproval,
    tool_manifest_fingerprint,
)
from src.test_generator import (
    is_allowed_test_path,
    request_test_batch,
    run_test_generation_attempts,
)


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

    async def request(_state, _reason):
        return next(responses)

    async def verify(_state, candidate):
        return CoverageDecision(
            verified=True,
            status="generated_verified",
            reason="verified",
            candidate=candidate,
        )

    monkeypatch.setattr("src.test_generator.request_test_batch", request)
    monkeypatch.setattr("src.test_generator.validate_differential_coverage", verify)
    monkeypatch.setenv("REPOPILOT_ESCALATION_ENABLED", "1")
    monkeypatch.setenv("LLM_ESCALATION_API_KEY", "sentinel-not-for-prompts")

    result = await run_test_generation_attempts(generation_state, "base_also_passes")

    assert result.verified is True
    assert generation_state.test_generation_attempts == 2
    assert generation_state.active_provider == "escalation"
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

    async def invalid(state, _reason):
        calls.append(state.active_provider)
        return _bad_batch()

    monkeypatch.setattr("src.test_generator.request_test_batch", invalid)
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

    async def invalid(*_):
        return _bad_batch()

    monkeypatch.setattr("src.test_generator.request_test_batch", invalid)
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

    async def good(*_):
        return _good_batch()

    async def verified(_state, candidate):
        return CoverageDecision(
            verified=True,
            status="generated_verified",
            reason="verified",
            candidate=candidate,
        )

    monkeypatch.setattr("src.test_generator.request_test_batch", good)
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

    async def good(*_):
        nonlocal requests
        requests += 1
        return _good_batch()

    async def rejected(_state, candidate):
        return CoverageDecision(
            verified=False,
            status="failed",
            reason="base_also_passes",
            candidate=candidate,
        )

    monkeypatch.setattr("src.test_generator.request_test_batch", good)
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
async def test_already_escalated_provider_never_downgrades(
    generation_state,
    monkeypatch,
):
    generation_state.active_provider = "escalation"
    generation_state.active_model = "claude-opus-4-8:stable"
    generation_state.escalated = True
    calls: list[str] = []

    async def invalid(state, _reason):
        calls.append(state.active_provider)
        return _bad_batch()

    monkeypatch.setattr("src.test_generator.request_test_batch", invalid)
    await run_test_generation_attempts(generation_state, "no_candidate")
    assert calls == ["escalation", "escalation"]
    assert generation_state.active_model == "claude-opus-4-8:stable"


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
