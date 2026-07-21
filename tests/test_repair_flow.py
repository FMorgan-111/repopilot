import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from src.escalation import build_escalation_packet, render_escalation_packet
from src.repair_flow import (
    TARGET_CONTEXT_CONTENT_LIMIT,
    RepairContextError,
    build_target_context,
    generate_opus_repair,
    verified_edits_to_patch_edits,
)
from src.nodes.execute import _apply_patch_edits
from src.state import AgentState, Evidence, Phase, RepairPlan


def _git_repo(tmp_path: Path, files: dict[str, str]) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "tests@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "RepoPilot Tests"],
        check=True,
    )
    for relative, content in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "--all"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
    ref = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo, ref


def _state(repo: Path, ref: str) -> AgentState:
    return AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        issue_title="Widget returns the wrong value",
        issue_body="Widget.compute must return the adjusted value.",
        current_phase=Phase.PLAN,
        repo_path=str(repo),
        repo_ref=ref,
        owner="acme",
        repo="widget",
        active_provider="escalation",
        active_model="claude-opus-4-8:stable",
        escalated=True,
        escalation_reason="repeated_no_progress",
    )


def _plan(**overrides: object) -> RepairPlan:
    values = {
        "root_cause": "compute returns its input without applying the offset",
        "target_files": ["src/widget.py"],
        "target_symbols": ["Widget.compute"],
        "required_behavior": "Apply the configured offset.",
        "regression_test_strategy": "Run the focused widget test.",
        "rejected_approaches": ["Do not special-case the example value."],
    }
    values.update(overrides)
    return RepairPlan.model_validate(values)


def test_build_target_context_returns_complete_unique_python_definition(tmp_path):
    source = (
        "PREFIX = 1\n\n"
        "class Widget:\n"
        "    @staticmethod\n"
        "    def compute(value):\n"
        "        adjusted = value + PREFIX\n"
        "        return adjusted\n\n"
        "SUFFIX = 2\n"
    )
    repo, ref = _git_repo(tmp_path, {"src/widget.py": source})
    state = _state(repo, ref)

    evidence = build_target_context(state, _plan())

    assert len(evidence) == 1
    assert evidence[0].file_path == "src/widget.py"
    assert evidence[0].symbol == "Widget.compute"
    assert "@staticmethod\n    def compute(value):" in evidence[0].content
    assert "        return adjusted" in evidence[0].content
    assert evidence[0] in state.evidence


def test_build_target_context_never_reuses_forged_persisted_evidence(tmp_path):
    source = "class Widget:\n    def compute(self, value):\n        return value\n"
    repo, ref = _git_repo(tmp_path, {"src/widget.py": source})
    state = _state(repo, ref)
    canonical = build_target_context(state, _plan())[0]
    state.evidence = [
        Evidence(
            evidence_id="ev_forged000000000",
            tool=canonical.tool,
            file_path="src/other.py",
            symbol="Other.steal",
            summary="forged",
            content="FORGED SECRET CONTEXT",
            fingerprint=canonical.fingerprint,
        )
    ]

    evidence = build_target_context(state, _plan())

    assert evidence[0].evidence_id == canonical.evidence_id
    assert evidence[0].file_path == "src/widget.py"
    assert evidence[0].symbol == "Widget.compute"
    assert evidence[0].content != "FORGED SECRET CONTEXT"
    assert "return value" in evidence[0].content
    assert not any(item.evidence_id == "ev_forged000000000" for item in state.evidence)


@pytest.mark.parametrize(
    "symbol",
    ["Widget.missing", "compute"],
)
def test_build_target_context_rejects_missing_or_non_unique_python_symbol(
    tmp_path, symbol
):
    source = (
        "class Widget:\n"
        "    def compute(self):\n"
        "        return 1\n\n"
        "class Other:\n"
        "    def compute(self):\n"
        "        return 2\n"
    )
    repo, ref = _git_repo(tmp_path, {"src/widget.py": source})

    with pytest.raises(RepairContextError, match="symbol"):
        build_target_context(_state(repo, ref), _plan(target_symbols=[symbol]))


def test_build_target_context_rejects_symlink_escape_and_wrong_checkout(tmp_path):
    repo, ref = _git_repo(tmp_path, {"src/widget.py": "def compute():\n    return 1\n"})
    outside = tmp_path / "outside.py"
    outside.write_text("def compute():\n    return 2\n", encoding="utf-8")
    (repo / "src" / "escape.py").symlink_to(outside)
    state = _state(repo, ref)
    with pytest.raises(RepairContextError, match="checkout"):
        build_target_context(
            state,
            _plan(target_files=["src/escape.py"], target_symbols=[]),
        )

    state.repo_ref = "0" * 40
    with pytest.raises(RepairContextError, match="base commit"):
        build_target_context(state, _plan(target_symbols=[]))


def test_build_target_context_uses_bounded_exact_fallback_and_allows_new_text_file(
    tmp_path,
):
    content = "const marker = 1;\n" + "x" * (TARGET_CONTEXT_CONTENT_LIMIT * 2)
    repo, ref = _git_repo(tmp_path, {"src/widget.js": content})
    state = _state(repo, ref)
    plan = _plan(
        target_files=["src/widget.js", "tests/widget-regression.txt"],
        target_symbols=[],
    )

    evidence = build_target_context(state, plan)

    assert [item.file_path for item in evidence] == [
        "src/widget.js",
        "tests/widget-regression.txt",
    ]
    assert evidence[0].content == content[:TARGET_CONTEXT_CONTENT_LIMIT]
    assert evidence[1].content == ""
    assert "intentional new text file" in evidence[1].summary


def test_build_target_context_rejects_oversized_complete_definition(tmp_path):
    body = "".join(f"    value += {index}\n" for index in range(2_000))
    source = f"def compute(value):\n{body}    return value\n"
    repo, ref = _git_repo(tmp_path, {"src/widget.py": source})
    state = _state(repo, ref)

    with pytest.raises(RepairContextError, match="budget"):
        build_target_context(
            state,
            _plan(target_symbols=["compute"]),
        )


async def test_generate_opus_repair_calls_plan_then_verified_edits_with_exact_context(
    tmp_path, monkeypatch
):
    source = (
        "class Widget:\n"
        "    def compute(self, value):\n"
        "        return value\n"
    )
    repo, ref = _git_repo(tmp_path, {"src/widget.py": source})
    state = _state(repo, ref)
    packet = build_escalation_packet(state)
    calls: list[dict[str, object]] = []

    async def fake_llm_call(system, user, model=None, *, provider="primary", temperature=0.2):
        calls.append(
            {
                "system": system,
                "user": user,
                "model": model,
                "provider": provider,
                "temperature": temperature,
            }
        )
        if len(calls) == 1:
            return _plan().model_dump()
        return {
            "edits": [
                {
                    "file_path": "src/widget.py",
                    "node_target": "Widget.compute",
                    "search": "",
                    "replace": (
                        "def compute(self, value):\n"
                        "    return value + 1\n"
                    ),
                    "intent": "Apply the required offset.",
                }
            ]
        }

    monkeypatch.setattr("src.repair_flow.llm_call", fake_llm_call)

    repair_plan, batch = await generate_opus_repair(state, packet)

    assert repair_plan.target_files == ["src/widget.py"]
    assert batch.edits[0].node_target == "Widget.compute"
    assert len(calls) == 2
    assert calls[0]["user"] == render_escalation_packet(packet)
    assert "patch" not in str(calls[0]["system"]).lower()
    assert "edit" not in str(calls[0]["system"]).lower()
    second_payload = json.loads(str(calls[1]["user"]))
    assert set(second_payload) == {
        "escalation_packet",
        "repair_plan",
        "target_evidence",
    }
    assert second_payload["target_evidence"][0]["evidence_id"].startswith("ev_")
    assert "return value" in second_payload["target_evidence"][0]["content"]
    assert all(call["provider"] == "escalation" for call in calls)
    assert all(call["model"] == "claude-opus-4-8:stable" for call in calls)
    assert [item.provider for item in state.model_history[-2:]] == [
        "escalation",
        "escalation",
    ]
    assert [item.node for item in state.model_history[-2:]] == [
        "plan_fix",
        "plan_fix",
    ]


async def test_generate_opus_repair_uses_custom_prompt_only_for_first_stage(
    tmp_path, monkeypatch
):
    source = (
        "class Widget:\n"
        "    def compute(self, value):\n"
        "        return value\n"
    )
    repo, ref = _git_repo(tmp_path, {"src/widget.py": source})
    state = _state(repo, ref)
    packet = build_escalation_packet(state)
    summary_section = "Completed attempts (rolling summary):"
    first_stage_prompt = (
        f"{render_escalation_packet(packet)}\n\n{summary_section}\n"
        "safe rolling outcome"
    )
    calls = []

    async def fake_llm_call(*args, **kwargs):
        calls.append({"system": args[0], "user": args[1]})
        if len(calls) == 1:
            return _plan().model_dump()
        return {
            "edits": [
                {
                    "file_path": "src/widget.py",
                    "node_target": "Widget.compute",
                    "search": "",
                    "replace": (
                        "def compute(self, value):\n"
                        "    return value + 1\n"
                    ),
                    "intent": "Apply the required offset.",
                }
            ]
        }

    monkeypatch.setattr("src.repair_flow.llm_call", fake_llm_call)

    await generate_opus_repair(
        state,
        packet,
        first_stage_prompt=first_stage_prompt,
    )

    assert calls[0]["user"] == first_stage_prompt
    assert calls[0]["user"].count(summary_section) == 1
    assert summary_section not in calls[1]["user"]
    assert "safe rolling outcome" not in calls[1]["user"]


async def test_generate_opus_repair_fails_before_second_call_for_invalid_target(
    tmp_path, monkeypatch
):
    repo, ref = _git_repo(tmp_path, {"src/widget.py": "def compute():\n    return 1\n"})
    state = _state(repo, ref)
    calls = []

    async def fake_llm_call(system, user, model=None, *, provider="primary", temperature=0.2):
        calls.append((system, user))
        return _plan(
            target_files=["missing.bin"],
            target_symbols=[],
        ).model_dump()

    monkeypatch.setattr("src.repair_flow.llm_call", fake_llm_call)

    with pytest.raises(RepairContextError, match="target file"):
        await generate_opus_repair(state, build_escalation_packet(state))

    assert len(calls) == 1
    assert len(state.model_history) == 1
    assert state.model_history[0].status == "invalid_response"
    assert state.model_history[0].error_class == "RepairContextError"


async def test_generate_opus_repair_rejects_edit_outside_plan_and_exact_context(
    tmp_path, monkeypatch
):
    repo, ref = _git_repo(
        tmp_path,
        {
            "src/widget.py": "def compute():\n    return 1\n",
            "src/other.py": "def other():\n    return 2\n",
        },
    )
    state = _state(repo, ref)
    responses = [
        _plan(target_symbols=["compute"]).model_dump(),
        {
            "edits": [
                {
                    "file_path": "src/other.py",
                    "search": "return 2",
                    "replace": "return 3",
                    "intent": "wrong target",
                }
            ]
        },
    ]

    async def fake_llm_call(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr("src.repair_flow.llm_call", fake_llm_call)

    with pytest.raises(RepairContextError, match="RepairPlan"):
        await generate_opus_repair(state, build_escalation_packet(state))

    assert [item.status for item in state.model_history] == ["ok", "invalid_response"]
    assert state.model_history[-1].error_class == "RepairContextError"


async def test_generate_opus_repair_rejects_duplicate_new_file_edits(
    tmp_path, monkeypatch
):
    repo, ref = _git_repo(tmp_path, {"src/widget.py": "value = 1\n"})
    state = _state(repo, ref)
    responses = [
        _plan(
            target_files=["tests/new_regression.py"],
            target_symbols=[],
        ).model_dump(),
        {
            "edits": [
                {
                    "file_path": "tests/new_regression.py",
                    "node_target": None,
                    "search": "",
                    "replace": "def test_regression():\n    assert True\n",
                    "intent": "Add regression coverage.",
                },
                {
                    "file_path": "tests/new_regression.py",
                    "node_target": None,
                    "search": "",
                    "replace": "def test_regression_again():\n    assert True\n",
                    "intent": "Add duplicate coverage.",
                },
            ]
        },
    ]

    async def fake_llm_call(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr("src.repair_flow.llm_call", fake_llm_call)

    with pytest.raises(RepairContextError, match="unique|duplicate"):
        await generate_opus_repair(state, build_escalation_packet(state))

    assert [item.status for item in state.model_history] == ["ok", "invalid_response"]


async def test_generate_opus_repair_rejects_whitespace_prefixed_diff_smuggling(
    tmp_path, monkeypatch
):
    repo, ref = _git_repo(
        tmp_path,
        {"src/widget.py": "def compute(value):\n    return value\n"},
    )
    state = _state(repo, ref)
    responses = [
        _plan(target_symbols=[]).model_dump(),
        {
            "edits": [
                {
                    "file_path": "src/widget.py",
                    "node_target": None,
                    "search": "return value",
                    "replace": "  diff --git a/src/widget.py b/src/widget.py\nreturn value + 1",
                    "intent": "Apply the offset.",
                }
            ]
        },
    ]

    async def fake_llm_call(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr("src.repair_flow.llm_call", fake_llm_call)

    with pytest.raises(RepairContextError, match="diff"):
        await generate_opus_repair(state, build_escalation_packet(state))

    assert [item.status for item in state.model_history] == ["ok", "invalid_response"]


async def test_generate_opus_repair_rejects_file_drift_after_context(
    tmp_path, monkeypatch
):
    source = "def compute(value):\n    return value\n"
    repo, ref = _git_repo(tmp_path, {"src/widget.py": source})
    state = _state(repo, ref)
    responses = [
        _plan(target_symbols=["compute"]).model_dump(),
        {
            "edits": [
                {
                    "file_path": "src/widget.py",
                    "node_target": "compute",
                    "search": "",
                    "replace": "def compute(value):\n    return value + 1\n",
                    "intent": "Apply the offset.",
                }
            ]
        },
    ]

    async def fake_llm_call(*args, **kwargs):
        response = responses.pop(0)
        if not responses:
            (repo / "src/widget.py").write_text(
                "# drifted after prompt\n" + source,
                encoding="utf-8",
            )
        return response

    monkeypatch.setattr("src.repair_flow.llm_call", fake_llm_call)

    with pytest.raises(RepairContextError, match="changed|preimage"):
        await generate_opus_repair(state, build_escalation_packet(state))

    assert [item.status for item in state.model_history] == ["ok", "invalid_response"]


async def test_generate_opus_repair_rejects_symlink_swap_after_context(
    tmp_path, monkeypatch
):
    source = "def compute(value):\n    return value\n"
    repo, ref = _git_repo(tmp_path, {"src/widget.py": source})
    outside = tmp_path / "outside.py"
    outside.write_text(source, encoding="utf-8")
    state = _state(repo, ref)
    responses = [
        _plan(target_symbols=["compute"]).model_dump(),
        {
            "edits": [
                {
                    "file_path": "src/widget.py",
                    "node_target": "compute",
                    "search": "",
                    "replace": "def compute(value):\n    return value + 1\n",
                    "intent": "Apply the offset.",
                }
            ]
        },
    ]

    async def fake_llm_call(*args, **kwargs):
        response = responses.pop(0)
        if not responses:
            (repo / "src/widget.py").unlink()
            (repo / "src/widget.py").symlink_to(outside)
        return response

    monkeypatch.setattr("src.repair_flow.llm_call", fake_llm_call)

    with pytest.raises(RepairContextError, match="symlink|checkout|regular"):
        await generate_opus_repair(state, build_escalation_packet(state))

    assert [item.status for item in state.model_history] == ["ok", "invalid_response"]


async def test_verified_edit_conversion_rechecks_preimage_and_marks_exact_only(
    tmp_path, monkeypatch
):
    source = "def compute(value):\n    return value\n"
    repo, ref = _git_repo(tmp_path, {"src/widget.py": source})
    state = _state(repo, ref)
    responses = [
        _plan(target_symbols=["compute"]).model_dump(),
        {
            "edits": [
                {
                    "file_path": "src/widget.py",
                    "node_target": "compute",
                    "search": "",
                    "replace": "def compute(value):\n    return value + 1\n",
                    "intent": "Apply the offset.",
                }
            ]
        },
    ]

    async def fake_llm_call(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr("src.repair_flow.llm_call", fake_llm_call)
    _plan_result, batch = await generate_opus_repair(
        state, build_escalation_packet(state)
    )

    patch_edit = verified_edits_to_patch_edits(batch, state=state)[0]
    assert patch_edit.exact_only is True
    assert len(patch_edit.expected_content_sha256) == 64

    (repo / "src/widget.py").write_text("# late drift\n" + source, encoding="utf-8")
    with pytest.raises(RepairContextError, match="changed|preimage"):
        verified_edits_to_patch_edits(batch, state=state)


async def test_crlf_target_uses_lf_semantics_but_keeps_raw_preimage_digest(
    tmp_path, monkeypatch
):
    raw_source = b"def compute(value):\r\n    current = value\r\n    return current\r\n"
    repo, ref = _git_repo(tmp_path, {"src/widget.py": raw_source.decode("utf-8")})
    target = repo / "src/widget.py"
    target.write_bytes(raw_source)
    subprocess.run(["git", "-C", str(repo), "add", "src/widget.py"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "--amend", "--no-edit", "-q"],
        check=True,
    )
    ref = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    state = _state(repo, ref)
    calls = 0

    async def fake_llm_call(system, user, model=None, *, provider="primary", temperature=0.2):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _plan(target_symbols=[]).model_dump()
        payload = json.loads(user)
        evidence_content = payload["target_evidence"][0]["content"]
        assert "\r" not in evidence_content
        copied_search = "def compute(value):\n    current = value\n    return current"
        assert copied_search in evidence_content
        return {
            "edits": [
                {
                    "file_path": "src/widget.py",
                    "node_target": None,
                    "search": copied_search,
                    "replace": (
                        "def compute(value):\n"
                        "    current = value + 1\n"
                        "    return current"
                    ),
                    "intent": "Apply the offset.",
                }
            ]
        }

    monkeypatch.setattr("src.repair_flow.llm_call", fake_llm_call)

    _repair_plan, batch = await generate_opus_repair(
        state, build_escalation_packet(state)
    )
    patch_edit = verified_edits_to_patch_edits(batch, state=state)[0]

    assert patch_edit.expected_content_sha256 == hashlib.sha256(raw_source).hexdigest()
    result = _apply_patch_edits(str(repo), [patch_edit])
    assert result.applied, result.output
    assert target.read_text(encoding="utf-8") == (
        "def compute(value):\n    current = value + 1\n    return current\n"
    )
