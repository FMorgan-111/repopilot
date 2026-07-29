"""Tests for convergence fixes: patch diversification (solution 1) and
final-attempt force-execute (solution 2) in plan_fix / reflect_on_failure."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from src import model_policy, new_agent
from src.evidence import EvidenceStore
from src.nodes import execute as execute_node
from src.nodes import plan as plan_node
from src.nodes import reflect as reflect_node
from src.patch_authorization import authorize_plan_patch
from src.repair_rounds import begin_repair_round, bind_repair_round_author
from src.repair_rounds import validate_repair_round_state
from src.state import PatchEdit


@pytest.fixture(autouse=True)
def _enable_explicit_legacy_host_execution(monkeypatch):
    monkeypatch.setenv("REPOPILOT_UNSAFE_ALLOW_HOST_EXECUTION", "1")


def _failed_attempt(
    file_path="app/router.py", search="def handle():", replace="def handle(x):"
):
    return new_agent.FixAttempt(
        patch_edits=[PatchEdit(file_path=file_path, search=search, replace=replace)],
        test_result="failed",
        success=False,
    )


def _base_state(**kw):
    values = {
        "issue_url": "https://github.com/acme/widget/issues/7",
        "issue_title": "Login crash",
        "issue_body": "Crashes after submit.",
        "current_phase": new_agent.Phase.PLAN,
    }
    values.update(kw)
    return new_agent.AgentState(**values)


def _execute_response(file_path, search, replace, summary="Apply the fix."):
    return json.dumps(
        {
            "plan": summary,
            "patch": "",
            "patch_edits": [{"file": file_path, "search": search, "replace": replace}],
            "files": [file_path],
            "test_command": "pytest",
            "decision_frame": {
                "stage": "plan",
                "summary": summary,
                "recommended_action": "execute",
                "risk": "low",
                "confidence": 0.7,
            },
        }
    )


def _collect_more_context_response(summary="Need more context."):
    return json.dumps(
        {
            "plan": summary,
            "patch": "",
            "files": [],
            "test_command": "",
            "decision_frame": {
                "stage": "plan",
                "summary": summary,
                "recommended_action": "collect_more_context",
                "risk": "unknown",
                "confidence": 0.5,
            },
        }
    )


# ---- pure helpers -----------------------------------------------------------


def test_prior_failed_edits_context_empty_without_failures():
    assert plan_node._prior_failed_edits_context(_base_state()) == ""


def test_prior_failed_edits_context_lists_failed_edits():
    state = _base_state(fix_attempts=[_failed_attempt()])
    ctx = plan_node._prior_failed_edits_context(state)
    assert "ALREADY-TRIED EDITS" in ctx
    assert "app/router.py" in ctx


def test_prior_failed_edits_context_ignores_successful_attempts():
    ok = _failed_attempt()
    ok.success = True
    assert plan_node._prior_failed_edits_context(_base_state(fix_attempts=[ok])) == ""


def test_prior_failed_edits_context_sanitizes_the_complete_rendered_block():
    state = _base_state(
        fix_attempts=[
            new_agent.FixAttempt(
                patch_edits=[
                    PatchEdit(
                        file_path="src/safe-diversity-sentinel.py",
                        search="safe-search-sentinel",
                        replace="safe replacement",
                    ),
                    PatchEdit(
                        file_path="src/sk-AAAAAAAAAAAAAAAA.py",
                        search="secret-shaped-search-sentinel",
                        replace="safe replacement",
                    ),
                    PatchEdit(
                        file_path="src/RePaIrPlAn.py",
                        search="VeRiFiEdEdItBaTcH forbidden-diversity-sentinel",
                        replace="safe replacement",
                    ),
                ],
                success=False,
            )
        ]
    )

    context = plan_node._prior_failed_edits_context(state)
    folded = context.casefold()

    assert "safe-diversity-sentinel" in context
    assert "safe-search-sentinel" in context
    for forbidden in (
        "sk-aaaaaaaa",
        "repairplan",
        "verifiededitbatch",
        "forbidden-diversity-sentinel",
    ):
        assert forbidden not in folded


def test_prior_failed_edits_context_has_an_independent_total_bound():
    edits = [
        PatchEdit(
            file_path=f"src/safe-diversity-{index}.py",
            search=f"safe-search-{index}-" + "x" * 400,
            replace="safe replacement",
        )
        for index in range(40)
    ]
    state = _base_state(
        fix_attempts=[new_agent.FixAttempt(patch_edits=edits, success=False)]
    )

    context = plan_node._prior_failed_edits_context(state)

    assert "safe-diversity-0.py" in context
    assert len(context) <= 4_000


def test_is_final_attempt():
    assert (
        plan_node._is_final_attempt(_base_state(retry_count=3, max_retries=3)) is True
    )
    assert (
        plan_node._is_final_attempt(_base_state(retry_count=2, max_retries=3)) is False
    )


# ---- solution 1: diversification -------------------------------------------


async def test_identical_failed_patch_replay_uses_shared_retry_and_retires_approval(
    exact_repair_state, monkeypatch
):
    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        return _execute_response(
            "src/widget.py",
            "return 'old-sentinel'",
            "return 'new-sentinel'",
        )

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = exact_repair_state
    state.fix_attempts = [
        new_agent.FixAttempt(
            patch_edits=[
                PatchEdit(
                    file_path="src/widget.py",
                    search="return 'old-sentinel'",
                    replace="return 'new-sentinel'",
                )
            ],
            test_result="failed",
            success=False,
        )
    ]

    next_state = await plan_node.plan_fix(state)

    assert next_state.current_phase == new_agent.Phase.PLAN
    assert next_state.retry_count == 1
    assert next_state.primary_failed_repair_rounds == 1
    assert next_state.decision_frame.recommended_action == "plan"
    assert next_state.patch_edits == []
    assert next_state.tool_patch_approval is None
    assert any(
        w.get("warning") == "blocked_dead_patch" for w in next_state.decision_warnings
    )


async def test_materially_changed_proposal_receives_fresh_exact_approval(
    exact_repair_state, monkeypatch
):
    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        return _execute_response(
            "src/widget.py",
            "return 'old-sentinel'",
            "return 'new-sentinel'",
        )

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = exact_repair_state
    state.fix_attempts = [
        new_agent.FixAttempt(
            patch_edits=[
                PatchEdit(
                    file_path="src/widget.py",
                    search="return 'old-sentinel'",
                    replace="return 'wrong-sentinel'",
                )
            ],
            test_result="failed",
            success=False,
        )
    ]

    next_state = await plan_node.plan_fix(state)

    assert next_state.current_phase == new_agent.Phase.EXECUTE
    assert next_state.tool_patch_approval is not None
    assert next_state.patch_edits[0].exact_only is True
    assert next_state.authorized_repair_provider == "primary"


async def test_assertion_diversity_rejects_same_target_symbol(
    exact_repair_state, monkeypatch
):
    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        return json.dumps(
            {
                "kind": "plan",
                "plan": "Change replacement text on the same symbol.",
                "patch": "",
                "patch_edits": [
                    {
                        "file": "src/widget.py",
                        "node_target": "widget",
                        "replace": "def widget():\n    return 'new-sentinel'\n",
                    }
                ],
                "files": ["app/router.py"],
                "test_command": "pytest",
                "decision_frame": {
                    "stage": "plan",
                    "summary": "Reuse handle.",
                    "recommended_action": "execute",
                    "risk": "low",
                    "confidence": 0.7,
                },
            }
        )

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = exact_repair_state
    state.assertion_diversity_required = True
    state.fix_attempts = [
        new_agent.FixAttempt(
            patch_edits=[
                PatchEdit(
                    file_path="src/widget.py",
                    node_target="widget",
                    replace="def widget():\n    return 'wrong'\n",
                )
            ],
            failure_kind="assertion_failure",
            error_log="AssertionError: expected new",
        )
    ]

    result = await plan_node.plan_fix(state)

    assert result.current_phase == new_agent.Phase.PLAN
    assert not result.patch_edits
    assert result.retry_count == 1
    assert result.primary_failed_repair_rounds == 1
    assert result.tool_patch_approval is None
    assert result.decision_frame.recommended_action == "plan"
    assert any(
        warning.get("warning") == "assertion_target_not_diversified"
        for warning in result.decision_warnings
    )


async def test_assertion_diversity_rejects_different_search_in_same_resolved_symbol(
    exact_repair_state,
    monkeypatch,
):
    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        return _execute_response(
            "src/widget.py",
            "return 'old-sentinel'",
            "return 'new-sentinel'",
        )

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = exact_repair_state
    state.assertion_diversity_required = True
    state.fix_attempts = [
        new_agent.FixAttempt(
            patch_edits=[
                PatchEdit(
                    file_path="src/widget.py",
                    search="def widget():",
                    replace="def widget():\n    return 'wrong'",
                )
            ],
            failure_kind="assertion_failure",
            error_log="AssertionError: expected new",
        )
    ]

    result = await plan_node.plan_fix(state)

    assert result.current_phase == new_agent.Phase.PLAN
    assert not result.patch_edits
    assert result.retry_count == 1
    assert result.primary_failed_repair_rounds == 1
    assert result.decision_frame.recommended_action == "plan"


async def test_assertion_diversity_accepts_distinct_explicit_node_target(
    tmp_path, monkeypatch
):
    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        return json.dumps(
            {
                "kind": "plan",
                "plan": "Target another symbol.",
                "patch": "",
                "patch_edits": [
                    {
                        "file": "app/router.py",
                        "node_target": "other_helper",
                        "replace": "def other_helper():\n    return 'new'\n",
                    }
                ],
                "files": ["app/router.py"],
                "test_command": "pytest",
                "decision_frame": {
                    "stage": "plan",
                    "summary": "Target other_helper.",
                    "recommended_action": "execute",
                    "risk": "low",
                    "confidence": 0.8,
                },
            }
        )

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app").mkdir()
    (repo / "app" / "router.py").write_text(
        "def handle():\n    return 'wrong'\n\ndef other_helper():\n    return 'old'\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "--all"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "base",
        ],
        check=True,
    )
    ref = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    state = _base_state(
        repo_path=str(repo),
        repo_ref=ref,
        assertion_diversity_required=True,
        fix_attempts=[
            new_agent.FixAttempt(
                patch_edits=[
                    PatchEdit(
                        file_path="app/router.py",
                        node_target="handle",
                        replace="def handle():\n    return 'wrong'\n",
                    )
                ],
                failure_kind="assertion_failure",
                error_log="AssertionError: expected new",
            )
        ],
    )

    result = await plan_node.plan_fix(state)

    assert result.current_phase == new_agent.Phase.EXECUTE
    assert result.patch_edits[0].node_target == "other_helper"
    assert result.tool_patch_approval is not None
    assert result.assertion_diversity_required is False


@pytest.mark.parametrize(
    ("prior_search", "candidate_symbol", "expected_phase"),
    [
        ("    value = 1\n", "f", new_agent.Phase.PLAN),
        ("    value = 1\n", "g", new_agent.Phase.EXECUTE),
        ("    return value\n", "g", new_agent.Phase.PLAN),
    ],
)
async def test_assertion_diversity_resolves_prior_search_to_exact_ast_symbol(
    tmp_path,
    monkeypatch,
    prior_search,
    candidate_symbol,
    expected_phase,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = (
        "def f():\n"
        "    value = 1\n"
        "    return value\n\n"
        "def g():\n"
        "    value = 2\n"
        "    return value\n"
    )
    (repo / "module.py").write_text(source, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "module.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "base",
        ],
        check=True,
    )
    ref = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        return json.dumps(
            {
                "kind": "plan",
                "plan": f"Target {candidate_symbol}.",
                "patch": "",
                "patch_edits": [
                    {
                        "file": "module.py",
                        "node_target": candidate_symbol,
                        "replace": (f"def {candidate_symbol}():\n    return 'new'\n"),
                    }
                ],
                "files": ["module.py"],
                "test_command": "pytest",
                "decision_frame": {
                    "stage": "plan",
                    "summary": f"Target {candidate_symbol}.",
                    "recommended_action": "execute",
                    "risk": "low",
                    "confidence": 0.8,
                },
            }
        )

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = _base_state(
        assertion_diversity_required=True,
        repo_path=str(repo),
        repo_ref=ref,
        fix_attempts=[
            new_agent.FixAttempt(
                patch_edits=[
                    PatchEdit(
                        file_path="module.py",
                        search=prior_search,
                        replace="    return 'wrong'\n",
                    )
                ],
                error_log="AssertionError: inferred assertion failure",
            )
        ],
    )

    result = await plan_node.plan_fix(state)

    assert result.current_phase == expected_phase
    assert result.assertion_diversity_required is (
        expected_phase == new_agent.Phase.PLAN
    )


@pytest.mark.parametrize(
    ("candidate_symbol", "expected_phase"),
    [("f", new_agent.Phase.PLAN), ("g", new_agent.Phase.EXECUTE)],
)
async def test_applied_search_edit_persists_preimage_symbol_for_diversity(
    tmp_path,
    monkeypatch,
    candidate_symbol,
    expected_phase,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = (
        "def f():\n    old_value = 1\n    return old_value\n\ndef g():\n    return 2\n"
    )
    (repo / "module.py").write_text(source, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "module.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "base",
        ],
        check=True,
    )
    ref = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    async def failed_test(*args, **kwargs):
        return {
            "command": "pytest",
            "returncode": 1,
            "stdout": "AssertionError: expected new value",
            "stderr": "",
            "success": False,
        }

    monkeypatch.setattr(execute_node, "run_pytest", failed_test)
    monkeypatch.setattr(
        execute_node,
        "_create_venv",
        lambda _repo: {"python": "python3", "reason": "exists", "success": True},
    )
    state = _base_state(
        repo_path=str(repo),
        repo_ref=ref,
    )
    begin_repair_round(state)
    bind_repair_round_author(state)
    raw = json.loads(
        _execute_response(
            "module.py",
            "    old_value = 1\n",
            "    new_value = 3\n",
        )
    )
    raw["patch_edits"][0]["resolved_target_symbol"] = "g"
    assert authorize_plan_patch(state, raw).status == "accepted"
    state.current_phase = new_agent.Phase.EXECUTE

    state = await execute_node.execute_fix(state)

    assert "old_value = 1" not in (repo / "module.py").read_text(encoding="utf-8")
    assert state.fix_attempts[-1].patch_edits[0].resolved_target_symbol == "f"
    (repo / "module.py").write_text(source, encoding="utf-8")
    state.assertion_diversity_required = True
    state.current_phase = new_agent.Phase.PLAN

    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        return json.dumps(
            {
                "kind": "plan",
                "plan": f"Target {candidate_symbol}.",
                "patch": "",
                "patch_edits": [
                    {
                        "file": "module.py",
                        "node_target": candidate_symbol,
                        "replace": f"def {candidate_symbol}():\n    return 9\n",
                    }
                ],
                "files": ["module.py"],
                "test_command": "pytest",
                "decision_frame": {
                    "stage": "plan",
                    "summary": f"Target {candidate_symbol}.",
                    "recommended_action": "execute",
                    "risk": "low",
                    "confidence": 0.8,
                },
            }
        )

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    result = await plan_node.plan_fix(state)

    assert result.current_phase == expected_phase, result.failure_reason


@pytest.mark.parametrize(
    "forged_symbol",
    ["g", "../../forged-identity-trace-sentinel"],
)
async def test_plan_discards_model_authored_resolved_symbol_and_recomputes_exactly(
    tmp_path,
    monkeypatch,
    forged_symbol,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = "def f():\n    old_value = 1\n    return old_value\n"
    (repo / "module.py").write_text(source, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "module.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "base",
        ],
        check=True,
    )
    ref = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        return json.dumps(
            {
                "kind": "plan",
                "plan": "Change f.",
                "patch": "",
                "patch_edits": [
                    {
                        "file": "module.py",
                        "search": "    old_value = 1\n",
                        "replace": "    new_value = 3\n",
                        "resolved_target_symbol": forged_symbol,
                    }
                ],
                "files": ["module.py"],
                "test_command": "pytest",
                "decision_frame": {
                    "stage": "plan",
                    "summary": "Change f.",
                    "recommended_action": "execute",
                    "risk": "low",
                    "confidence": 0.8,
                },
            }
        )

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = _base_state(repo_path=str(repo), repo_ref=ref)

    result = await plan_node.plan_fix(state)

    assert result.current_phase == new_agent.Phase.EXECUTE
    assert result.patch_edits[0].resolved_target_symbol == ""
    assert "forged-identity-trace-sentinel" not in result.model_dump_json()

    async def passed_test(*_args, **_kwargs):
        return {
            "command": "pytest",
            "returncode": 0,
            "stdout": "passed",
            "stderr": "",
            "success": True,
        }

    monkeypatch.setattr(
        execute_node,
        "run_pytest",
        passed_test,
    )
    monkeypatch.setattr(
        execute_node,
        "_create_venv",
        lambda _repo: {"python": "python3", "reason": "exists", "success": True},
    )
    executed = await execute_node.execute_fix(result)
    assert executed.fix_attempts[-1].patch_edits[0].resolved_target_symbol == "f"


async def test_applied_search_edit_persists_dotted_method_symbol(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = (
        "class Widget:\n"
        "    def f(self):\n"
        "        old_value = 1\n"
        "        return old_value\n"
    )
    (repo / "module.py").write_text(source, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "module.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "base",
        ],
        check=True,
    )
    ref = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    async def failed_test(*args, **kwargs):
        return {
            "command": "pytest",
            "returncode": 1,
            "stdout": "AssertionError: expected new",
            "stderr": "",
            "success": False,
        }

    monkeypatch.setattr(execute_node, "run_pytest", failed_test)
    monkeypatch.setattr(
        execute_node,
        "_create_venv",
        lambda _repo: {"python": "python3", "reason": "exists", "success": True},
    )
    state = _base_state(
        repo_path=str(repo),
        repo_ref=ref,
    )
    begin_repair_round(state)
    bind_repair_round_author(state)
    raw = json.loads(
        _execute_response(
            "module.py",
            "        old_value = 1\n",
            "        new_value = 3\n",
        )
    )
    assert authorize_plan_patch(state, raw).status == "accepted"
    state.current_phase = new_agent.Phase.EXECUTE

    result = await execute_node.execute_fix(state)

    assert result.fix_attempts[-1].patch_edits[0].resolved_target_symbol == "Widget.f"


def test_legacy_patch_edit_defaults_resolved_symbol_for_saved_state_compatibility():
    edit = PatchEdit.model_validate(
        {
            "file_path": "module.py",
            "search": "old",
            "replace": "new",
        }
    )

    assert edit.resolved_target_symbol == ""


async def test_plan_prompt_includes_failed_edits(monkeypatch):
    captured = {}

    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        captured.update(system=system, user=user, model=model, provider=provider)
        return _collect_more_context_response()

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = _base_state(retry_count=1, max_retries=3, fix_attempts=[_failed_attempt()])

    await plan_node.plan_fix(state)

    assert "ALREADY-TRIED EDITS" in captured["user"]
    assert "app/router.py" in captured["user"]
    assert captured["system"] == plan_node.PLAN_SYSTEM
    assert captured["model"] == "gemini-3.5-flash:stable"
    assert captured["provider"] == "primary"


# ---- solution 2: final-attempt force execute -------------------------------


async def test_plan_fix_final_attempt_blocks_collect_more_context(monkeypatch):
    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        return _collect_more_context_response()

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = _base_state(retry_count=3, max_retries=3)

    next_state = await plan_node.plan_fix(state)

    assert next_state.current_phase == new_agent.Phase.FAILURE
    assert next_state.failure_reason == (
        "Context collection made no progress within its bounded cap."
    )
    assert next_state.retry_count == 3
    assert next_state.context_collection_count == 0


async def test_plan_fix_non_final_collect_more_context_still_loops(monkeypatch):
    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        return _collect_more_context_response()

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = _base_state(retry_count=0, max_retries=3)

    next_state = await plan_node.plan_fix(state)

    assert next_state.current_phase == new_agent.Phase.LOCATE
    assert next_state.context_collection_count == 1
    assert next_state.retry_count == 0
    assert next_state.current_repair_round_id == 1


async def test_plan_prompt_includes_final_attempt_instruction(monkeypatch):
    captured = {}

    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        captured.update(system=system, user=user, model=model, provider=provider)
        return _collect_more_context_response()

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = _base_state(retry_count=3, max_retries=3)

    await plan_node.plan_fix(state)

    assert captured["system"] == plan_node.PLAN_SYSTEM
    assert "FINAL planning attempt" in captured["user"]
    assert captured["model"] == "gemini-3.5-flash:stable"
    assert captured["provider"] == "primary"


async def test_complete_first_plan_prompt_uses_patch_edits_contract(monkeypatch):
    captured = {}

    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        captured.update(system=system, user=user, model=model, provider=provider)
        return _collect_more_context_response()

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    # First plan: no prior attempts, no context collected.
    await plan_node.plan_fix(_base_state())

    assert captured["system"] == plan_node.PLAN_SYSTEM
    prompt = captured["user"]
    assert "at least one item in patch_edits" in prompt
    assert "FIRST plan" in prompt
    assert "do NOT request a tool" in prompt
    assert "produce patch_edits from the supplied files" in prompt
    for stale in (
        "at least one patch_edit",
        "recommend collect_more_context",
        "recommended_action",
        "next_checks",
        "risk",
        "confidence",
        "DecisionFrame",
    ):
        assert stale not in prompt
    assert captured["model"] == "gemini-3.5-flash:stable"
    assert captured["provider"] == "primary"


async def test_plan_non_first_plan_still_requires_patch_but_allows_context(monkeypatch):
    captured = {}

    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        captured.update(system=system, user=user, model=model, provider=provider)
        return _collect_more_context_response()

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    # A later plan (context already collected once) still demands a patch but
    # does not carry the first-plan collect_more_context ban.
    await plan_node.plan_fix(_base_state(context_collection_count=1))

    assert captured["system"] == plan_node.PLAN_SYSTEM
    assert "at least one item in patch_edits" in captured["user"]
    assert "FIRST plan" not in captured["user"]
    assert captured["model"] == "gemini-3.5-flash:stable"
    assert captured["provider"] == "primary"


# ---- reflect wiring ---------------------------------------------------------


async def test_reflect_prompt_includes_failed_edits(monkeypatch):
    calls = []

    async def fake_summary(*_args, **_kwargs):
        return "safe reflection summary"

    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        calls.append(user)
        return json.dumps(
            {
                "root_cause": "wrong file",
                "what_went_wrong": "edit missed",
                "suggested_fix_approach": "try elsewhere",
                "files_that_also_need_changes": [],
                "decision_frame": {
                    "stage": "reflect",
                    "summary": "wrong file",
                    "recommended_action": "plan",
                    "risk": "medium",
                    "confidence": 0.4,
                },
            }
        )

    monkeypatch.setattr(reflect_node, "llm_call", fake_llm_call)
    monkeypatch.setattr(
        reflect_node,
        "summarize_attempt_outcome",
        fake_summary,
    )
    state = _base_state(retry_count=1, max_retries=3, fix_attempts=[_failed_attempt()])

    await reflect_node.reflect_on_failure(state)

    assert "ALREADY-TRIED EDITS" in calls[0]
    assert "DIFFERENT edit" in calls[0]


# ---- solution 1b: hard-block dead patches (do not execute known re-fails) ----


def _unappliable_attempt(
    file_path="app/router.py", search="def handle():", replace="def handle(x):"
):
    return new_agent.FixAttempt(
        patch_edits=[PatchEdit(file_path=file_path, search=search, replace=replace)],
        test_result="patch_apply_failed",
        failure_kind="patch_apply_failed",
        success=False,
    )


def test_dead_plan_reason_identical_patch():
    state = _base_state(fix_attempts=[_failed_attempt()])
    state.patch_edits = [
        PatchEdit(
            file_path="app/router.py", search="def handle():", replace="def handle(x):"
        )
    ]
    assert plan_node._dead_plan_reason(state) == "identical_to_failed_patch"


def test_dead_plan_reason_reuses_unappliable_anchor_even_with_new_replace():
    # Prior attempt could not be applied at all; a fresh replace on the SAME
    # anchor will still fail to apply, so it is dead.
    state = _base_state(fix_attempts=[_unappliable_attempt()])
    state.patch_edits = [
        PatchEdit(
            file_path="app/router.py", search="def handle():", replace="def handle(z):"
        )
    ]
    assert plan_node._dead_plan_reason(state) == "reuses_unappliable_anchor"


def test_dead_plan_reason_none_when_diversified():
    state = _base_state(fix_attempts=[_failed_attempt()])
    state.patch_edits = [
        PatchEdit(file_path="app/other.py", search="def other():", replace="Y")
    ]
    assert plan_node._dead_plan_reason(state) is None


async def test_plan_fix_blocks_identical_patch_with_shared_global_retry(
    exact_repair_state, monkeypatch
):
    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        return _execute_response(
            "src/widget.py",
            "return 'old-sentinel'",
            "return 'new-sentinel'",
        )

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = exact_repair_state
    state.retry_count = 1
    state.fix_attempts = [
        new_agent.FixAttempt(
            patch_edits=[
                PatchEdit(
                    file_path="src/widget.py",
                    search="return 'old-sentinel'",
                    replace="return 'new-sentinel'",
                )
            ],
            test_result="failed",
            success=False,
        )
    ]

    next_state = await plan_node.plan_fix(state)

    assert next_state.current_phase == new_agent.Phase.PLAN
    assert next_state.retry_count == 2
    assert next_state.decision_frame.recommended_action == "plan"
    assert new_agent.route_from_state(next_state) == "plan_fix"
    assert next_state.patch_edits == []
    assert next_state.tool_patch_approval is None
    assert any(
        w.get("warning") == "blocked_dead_patch" for w in next_state.decision_warnings
    )


async def test_legacy_block_counter_cannot_terminate_before_global_budget(
    exact_repair_state, monkeypatch
):
    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        return _execute_response(
            "src/widget.py",
            "return 'old-sentinel'",
            "return 'new-sentinel'",
        )

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = exact_repair_state
    state.retry_count = 1
    state.repeated_patch_block_count = 99
    state.fix_attempts = [
        new_agent.FixAttempt(
            patch_edits=[
                PatchEdit(
                    file_path="src/widget.py",
                    search="return 'old-sentinel'",
                    replace="return 'new-sentinel'",
                )
            ],
            test_result="failed",
            success=False,
        )
    ]

    next_state = await plan_node.plan_fix(state)

    assert next_state.current_phase == new_agent.Phase.PLAN
    assert next_state.retry_count == 2
    assert next_state.failure_reason == "repeated_failed_patch"


async def test_final_global_attempt_dead_patch_exhausts_shared_budget(
    exact_repair_state, monkeypatch
):
    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        return _execute_response(
            "src/widget.py",
            "return 'old-sentinel'",
            "return 'new-sentinel'",
        )

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = exact_repair_state
    state.retry_count = state.max_retries
    state.fix_attempts = [
        new_agent.FixAttempt(
            patch_edits=[
                PatchEdit(
                    file_path="src/widget.py",
                    search="return 'old-sentinel'",
                    replace="return 'new-sentinel'",
                )
            ],
            test_result="failed",
            success=False,
        )
    ]

    next_state = await plan_node.plan_fix(state)

    assert next_state.current_phase == new_agent.Phase.FAILURE
    assert next_state.failure_reason == (
        f"Maximum retries reached: {state.max_retries}."
    )


async def test_two_invalid_primary_anchors_switch_next_full_plan_to_opus(
    exact_repair_state, monkeypatch
):
    monkeypatch.setattr(model_policy, "escalation_is_configured", lambda: True)
    monkeypatch.setattr(
        model_policy,
        "get_model_config",
        lambda provider: SimpleNamespace(
            model=(
                "claude-opus-4-8:stable"
                if provider == "escalation"
                else "gemini-3.5-flash:stable"
            )
        ),
    )
    responses = [
        _execute_response(
            "src/widget.py",
            "return 'missing-sentinel'",
            "return 'new-sentinel'",
            summary="Use the missing anchor.",
        ),
        _execute_response(
            "src/widget.py",
            "return 'missing-sentinel'",
            "return 'new-sentinel'",
            summary="Use the missing anchor again.",
        ),
        _execute_response(
            "src/widget.py",
            "return 'old-sentinel'",
            "return 'new-sentinel'",
            summary="Use the exact anchor.",
        ),
    ]
    calls = []

    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        calls.append((system, user, model, provider))
        return responses.pop(0)

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = exact_repair_state

    state = await plan_node.plan_fix(state)
    assert state.current_phase == new_agent.Phase.PLAN
    assert state.retry_count == 1
    assert state.primary_failed_repair_rounds == 1
    assert state.escalated is False

    state = await plan_node.plan_fix(state)
    assert state.current_phase == new_agent.Phase.PLAN
    assert state.retry_count == 2
    assert state.primary_failed_repair_rounds == 2
    assert state.escalated is True
    assert state.active_provider == "escalation"
    assert state.escalation_reason == "primary_repair_round_limit"
    assert state.patch_edits == []
    assert state.tool_patch_approval is None

    state = await plan_node.plan_fix(state)

    assert state.current_phase == new_agent.Phase.EXECUTE
    assert [call[3] for call in calls] == ["primary", "primary", "escalation"]
    assert all(call[0] == plan_node.PLAN_SYSTEM for call in calls)
    assert state.authorized_repair_provider == "escalation"


async def test_changed_invalid_anchor_still_uses_one_shared_retry_per_round(
    exact_repair_state, monkeypatch
):
    monkeypatch.setattr(model_policy, "escalation_is_configured", lambda: False)
    responses = [
        _execute_response(
            "src/widget.py",
            "return 'missing-one'",
            "return 'new-sentinel'",
            summary="First missing anchor.",
        ),
        _execute_response(
            "src/widget.py",
            "return 'missing-two'",
            "return 'new-sentinel'",
            summary="Second missing anchor.",
        ),
    ]

    async def fake_llm_call(system, user, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = exact_repair_state

    state = await plan_node.plan_fix(state)
    state = await plan_node.plan_fix(state)

    assert state.current_phase == new_agent.Phase.PLAN
    assert state.retry_count == 2
    assert state.primary_failed_repair_rounds == 2
    assert state.repair_round_sequence == 2
    assert state.last_counted_repair_round_id == 2
    assert state.patch_edits == []
    assert state.tool_patch_approval is None
    assert state.escalated is False


async def test_repeated_assertion_reflection_requires_different_target_symbol(
    monkeypatch,
):
    captured = {}

    async def fake_llm_call(system, user, **kwargs):
        captured["user"] = user
        return {
            "root_cause": "The same assertion still fails.",
            "what_went_wrong": "The previous symbol was not causal.",
            "suggested_fix_approach": "Target another symbol.",
            "files_that_also_need_changes": [],
            "decision_frame": {
                "stage": "reflect",
                "summary": "Choose another target.",
                "recommended_action": "plan",
                "risk": "medium",
                "confidence": 0.6,
            },
        }

    async def fake_summary(state, **kwargs):
        return "same assertion; choose another symbol"

    monkeypatch.setattr(reflect_node, "llm_call", fake_llm_call)
    monkeypatch.setattr(reflect_node, "summarize_attempt_outcome", fake_summary)
    state = _base_state(
        current_phase=new_agent.Phase.REFLECT,
        no_progress_rounds=1,
        fix_attempts=[
            new_agent.FixAttempt(
                patch_edits=[
                    PatchEdit(
                        file_path="src/widget.py",
                        node_target="widget",
                        replace="def widget():\n    return 'still-wrong'\n",
                    )
                ],
                failure_kind="assertion_failure",
                error_log="AssertionError: expected new-sentinel",
            )
        ],
    )

    await reflect_node.reflect_on_failure(state)

    assert "different target symbol" in captured["user"].lower()


async def test_legacy_reflection_persists_only_normalized_decision_json(monkeypatch):
    async def fake_llm_call(system, user, **kwargs):
        return {
            "root_cause": "The causal branch is elsewhere.",
            "what_went_wrong": "The prior edit targeted the wrong symbol.",
            "suggested_fix_approach": "Change helper instead.",
            "files_that_also_need_changes": [],
            "hypotheses": [{"id": "H1", "claim": "helper", "score": 0.8}],
            "confidence": 0.8,
            "risk": "medium",
        }

    async def fake_summary(state, **kwargs):
        return "safe summary"

    monkeypatch.setattr(reflect_node, "llm_call", fake_llm_call)
    monkeypatch.setattr(reflect_node, "summarize_attempt_outcome", fake_summary)
    state = _base_state(
        current_phase=new_agent.Phase.REFLECT,
        fix_attempts=[
            new_agent.FixAttempt(
                failure_kind="assertion_failure",
                error_log="AssertionError: sentinel",
            )
        ],
    )

    result = await reflect_node.reflect_on_failure(state)
    persisted = json.loads(result.reflection_notes)

    assert set(persisted) == {
        "root_cause",
        "what_went_wrong",
        "suggested_fix_approach",
        "files_that_also_need_changes",
        "decision_frame",
    }


@pytest.mark.parametrize("terminal", ["model_stop", "tool_stop", "opus_limit"])
async def test_reflect_terminal_branches_finalize_summary_exactly_once(
    monkeypatch,
    terminal,
):
    summary_calls = 0

    async def fake_summary(state, **kwargs):
        nonlocal summary_calls
        summary_calls += 1
        return f"summary for {terminal}"

    if terminal == "model_stop":

        async def fake_llm_call(*args, **kwargs):
            return {"kind": "stop", "stop_reason": "done"}
    elif terminal == "tool_stop":

        async def fake_llm_call(*args, **kwargs):
            return {
                "kind": "tool",
                "tool_intent": {
                    "action": "finish_investigation",
                    "args": {},
                    "reason": "no repair",
                    "expected_evidence": "none",
                },
            }

        async def stop_router(state, intent, *, calls_this_round):
            return SimpleNamespace(
                status="ok",
                evidence_id=None,
                made_progress=False,
                control_action="finish_investigation",
            )

        monkeypatch.setattr(reflect_node, "route_tool_intent", stop_router)
    else:

        async def fake_llm_call(*args, **kwargs):
            raise ValueError("invalid escalated reflection")

    monkeypatch.setattr(reflect_node, "llm_call", fake_llm_call)
    monkeypatch.setattr(reflect_node, "summarize_attempt_outcome", fake_summary)
    state = _base_state(
        current_phase=new_agent.Phase.REFLECT,
        active_provider=("escalation" if terminal == "opus_limit" else "primary"),
        active_model=(
            "claude-opus-4-8:stable"
            if terminal == "opus_limit"
            else "gemini-3.5-flash:stable"
        ),
        escalated=terminal == "opus_limit",
        escalation_reason=("repeated_plan" if terminal == "opus_limit" else ""),
        fix_attempts=[
            new_agent.FixAttempt(
                failure_kind="assertion_failure",
                error_log="AssertionError: sentinel",
            )
        ],
    )

    result = await reflect_node.reflect_on_failure(state)

    assert result.current_phase == new_agent.Phase.FAILURE
    assert result.attempt_outcome_summary == f"summary for {terminal}"
    assert summary_calls == 1


async def test_reflect_duplicate_tools_are_diagnostic_until_primary_failure_threshold(
    monkeypatch,
):
    calls = []

    def reflection_response():
        return {
            "kind": "reflect",
            "root_cause": "The original target was wrong.",
            "what_went_wrong": "The edit missed the causal symbol.",
            "suggested_fix_approach": "Choose another symbol.",
            "files_that_also_need_changes": [],
            "decision_frame": {
                "stage": "reflect",
                "summary": "Choose another symbol.",
                "recommended_action": "plan",
                "risk": "medium",
                "confidence": 0.7,
            },
        }

    responses = [
        {
            "kind": "tool",
            "tool_intent": {
                "action": "search_text",
                "args": {"text": "same-sentinel"},
                "reason": "find source",
                "expected_evidence": "source location",
            },
        },
        {
            "kind": "tool",
            "tool_intent": {
                "action": "search_text",
                "args": {"text": "same-sentinel"},
                "reason": "find source",
                "expected_evidence": "source location",
            },
        },
        reflection_response(),
    ]

    async def fake_llm_call(system, user, **kwargs):
        calls.append((system, user, kwargs))
        return responses.pop(0)

    async def duplicate_route(state, intent, *, calls_this_round):
        return SimpleNamespace(
            status="duplicate",
            evidence_id=None,
            made_progress=False,
            control_action="",
        )

    async def fake_summary(state, **kwargs):
        return "reflection summary"

    monkeypatch.setattr(model_policy, "escalation_is_configured", lambda: True)
    monkeypatch.setattr(
        model_policy,
        "get_model_config",
        lambda provider: SimpleNamespace(model="claude-opus-4-8:stable"),
    )
    monkeypatch.setattr(reflect_node, "llm_call", fake_llm_call)
    monkeypatch.setattr(reflect_node, "route_tool_intent", duplicate_route)
    monkeypatch.setattr(reflect_node, "summarize_attempt_outcome", fake_summary)
    state = _base_state(
        current_phase=new_agent.Phase.REFLECT,
        repo_ref="a" * 40,
        fix_attempts=[
            new_agent.FixAttempt(
                failure_kind="assertion_failure",
                error_log="AssertionError: sentinel",
            )
        ],
    )

    result = await reflect_node.reflect_on_failure(state)

    assert result.active_provider == "primary"
    assert result.escalation_reason == ""
    assert result.no_progress_rounds == 2
    assert [event.kind for event in result.no_progress_history[-2:]] == [
        "unchanged_context",
        "unchanged_context",
    ]
    assert calls[2][0] != reflect_node.ESCALATED_REFLECT_SYSTEM
    assert calls[2][2] == {}

    threshold_calls = []

    async def threshold_llm_call(system, user, **kwargs):
        threshold_calls.append((system, user, kwargs))
        return reflection_response()

    monkeypatch.setattr(reflect_node, "llm_call", threshold_llm_call)
    threshold_state = _base_state(
        current_phase=new_agent.Phase.REFLECT,
        repo_ref="a" * 40,
        retry_count=2,
        repair_round_sequence=2,
        last_counted_repair_round_id=2,
        primary_failed_repair_rounds=2,
        fix_attempts=[
            new_agent.FixAttempt(
                failure_kind="assertion_failure",
                error_log="AssertionError: sentinel",
            )
        ],
    )
    validate_repair_round_state(threshold_state)

    threshold_result = await reflect_node.reflect_on_failure(threshold_state)

    assert threshold_result.active_provider == "escalation"
    assert threshold_result.escalation_reason == "primary_repair_round_limit"
    assert threshold_calls[0][0] == reflect_node.ESCALATED_REFLECT_SYSTEM
    assert threshold_calls[0][2]["provider"] == "escalation"


async def test_escalated_reflect_tool_reprompt_contains_only_delta_evidence(
    monkeypatch,
):
    calls = []
    responses = [
        {
            "kind": "tool",
            "tool_intent": {
                "action": "search_text",
                "args": {"text": "causal-helper"},
                "reason": "locate helper",
                "expected_evidence": "helper source",
            },
        },
        {
            "kind": "reflect",
            "root_cause": "The helper is causal.",
            "what_went_wrong": "The prior target was downstream.",
            "suggested_fix_approach": "Change causal_helper.",
            "files_that_also_need_changes": [],
            "decision_frame": {
                "stage": "reflect",
                "summary": "Change causal_helper.",
                "recommended_action": "plan",
                "risk": "medium",
                "confidence": 0.8,
            },
        },
    ]

    async def fake_llm_call(system, user, **kwargs):
        calls.append((system, user))
        return responses.pop(0)

    async def delta_router(state, intent, *, calls_this_round):
        added = EvidenceStore(state).add(
            tool="search_text",
            summary="new helper evidence",
            content="new-reflect-evidence-sentinel",
        )
        return SimpleNamespace(
            status="ok",
            evidence_id=added.evidence.evidence_id,
            made_progress=True,
            control_action="",
        )

    async def fake_summary(state, **kwargs):
        return "safe summary"

    monkeypatch.setattr(reflect_node, "llm_call", fake_llm_call)
    monkeypatch.setattr(reflect_node, "route_tool_intent", delta_router)
    monkeypatch.setattr(reflect_node, "summarize_attempt_outcome", fake_summary)
    state = _base_state(
        current_phase=new_agent.Phase.REFLECT,
        active_provider="escalation",
        active_model="claude-opus-4-8:stable",
        escalated=True,
        escalation_reason="repeated_plan",
        repo_ref="a" * 40,
        fix_attempts=[
            new_agent.FixAttempt(
                failure_kind="assertion_failure",
                error_log="AssertionError: sentinel",
            )
        ],
    )
    EvidenceStore(state).add(
        tool="read_range",
        summary="old evidence",
        content="old-reflect-evidence-sentinel",
    )

    result = await reflect_node.reflect_on_failure(state)

    assert result.current_phase == new_agent.Phase.PLAN
    assert all(system == reflect_node.ESCALATED_REFLECT_SYSTEM for system, _ in calls)
    assert "old-reflect-evidence-sentinel" in calls[0][1]
    assert "new-reflect-evidence-sentinel" in calls[1][1]
    assert "old-reflect-evidence-sentinel" not in calls[1][1]
    assert not any(
        item.get("event") == "opus_no_progress" for item in result.node_diagnostics
    )
