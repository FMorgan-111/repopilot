from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from src.coverage_gate import ChangedTarget, CoverageDecision
from src.patch_gate import validate_patch_batch
from src.state import AgentState, RepairPlan, VerifiedEdit, VerifiedEditBatch
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
    return AgentState(
        issue_url="https://github.com/a/b/issues/1",
        issue_title="Wrong answer",
        issue_body="answer should return two",
        repo_path=str(root),
        repo_ref=ref,
        active_provider="primary",
        active_model="gemini-3.5-flash:stable",
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


@pytest.mark.asyncio
async def test_generator_prompt_excludes_credentials_and_evaluator_fields(
    generation_state,
    monkeypatch,
):
    generation_state.issue_body = (
        "api_key=top-secret-value gold_patch FAIL_TO_PASS instance_id"
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
