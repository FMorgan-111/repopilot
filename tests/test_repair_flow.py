import asyncio
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import repair_flow as repair_flow_module
from src.async_safety import CancellationDrainError
from src.escalation import (
    ESCALATION_PACKET_RENDER_LIMIT,
    EVIDENCE_CONTENT_LIMIT,
    EVIDENCE_LIMIT,
    EVIDENCE_TOTAL_LIMIT,
    build_escalation_packet,
    render_escalation_packet,
)
from src.evidence import EvidenceStore
from src.nodes.execute import _apply_patch_edits
from src.reasoning_loop import ReasoningStop
from src.repair_flow import (
    TARGET_CONTEXT_CONTENT_LIMIT,
    RepairContextError,
    build_target_context,
    generate_opus_repair,
    verified_edits_to_patch_edits,
)
from src.state import AgentState, Evidence, FileInfo, Phase, RepairPlan


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


def test_strict_symbol_resolution_returns_symbol_or_legitimate_none(tmp_path):
    source = "MODULE_VALUE = 1\n\ndef widget():\n    return 'old-sentinel'\n"
    repo, ref = _git_repo(tmp_path, {"src/widget.py": source})
    state = _state(repo, ref)

    assert (
        repair_flow_module.resolve_search_target_symbol_strict(
            state,
            "src/widget.py",
            "return 'old-sentinel'",
        )
        == "widget"
    )
    assert (
        repair_flow_module.resolve_search_target_symbol_strict(
            state,
            "src/widget.py",
            "MODULE_VALUE = 1",
        )
        is None
    )
    assert (
        repair_flow_module.resolve_search_target_symbol_strict(
            state, "src/widget.txt", "anything"
        )
        is None
    )


def test_strict_symbol_resolution_treats_syntax_error_as_no_symbol(tmp_path):
    repo, ref = _git_repo(
        tmp_path,
        {"src/widget.py": "def broken(:\n    target = 1\n"},
    )

    assert (
        repair_flow_module.resolve_search_target_symbol_strict(
            _state(repo, ref),
            "src/widget.py",
            "target = 1",
        )
        is None
    )


def test_strict_symbol_resolution_propagates_checkout_drift(tmp_path):
    repo, ref = _git_repo(
        tmp_path,
        {"src/widget.py": "def widget():\n    return 1\n"},
    )
    state = _state(repo, ref)
    state.repo_ref = "f" * 40

    with pytest.raises(RepairContextError, match="base commit"):
        repair_flow_module.resolve_search_target_symbol_strict(
            state,
            "src/widget.py",
            "return 1",
        )


def test_strict_symbol_resolution_normalizes_read_io_to_context_error(
    tmp_path, monkeypatch
):
    repo, ref = _git_repo(
        tmp_path,
        {"src/widget.py": "def widget():\n    return 1\n"},
    )

    def fail_read(*_args, **_kwargs):
        raise OSError("read failed")

    monkeypatch.setattr("src.repair_flow.read_exact_checkout_text", fail_read)
    with pytest.raises(RepairContextError, match="read"):
        repair_flow_module.resolve_search_target_symbol_strict(
            _state(repo, ref),
            "src/widget.py",
            "return 1",
        )


async def test_generate_opus_repair_calls_plan_then_verified_edits_with_exact_context(
    tmp_path, monkeypatch
):
    source = "class Widget:\n    def compute(self, value):\n        return value\n"
    repo, ref = _git_repo(tmp_path, {"src/widget.py": source})
    state = _state(repo, ref)
    packet = build_escalation_packet(state)
    calls: list[dict[str, object]] = []

    async def fake_llm_call(
        system, user, model=None, *, provider="primary", temperature=0.2
    ):
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
                    "replace": ("def compute(self, value):\n    return value + 1\n"),
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
    assert state.token_usage == sum(
        item.input_tokens + item.output_tokens for item in state.model_history
    )


async def test_escalated_plan_tool_drain_propagates_without_model_telemetry_or_debit(
    tmp_path,
    monkeypatch,
):
    repo, ref = _git_repo(
        tmp_path,
        {"src/widget.py": "def compute(value):\n    return value\n"},
    )
    state = _state(repo, ref)
    cancellation = asyncio.CancelledError("plan cancelled")
    cleanup_error = RuntimeError("tool cleanup failed")
    sentinel = CancellationDrainError(
        "escalated plan tool",
        cancellation,
        cleanup_error,
    )

    async def select_tool(*_args, **_kwargs):
        return {
            "kind": "tool",
            "tool_intent": {
                "action": "search_text",
                "args": {"text": "compute"},
                "reason": "confirm the target",
                "expected_evidence": "matching source",
            },
        }

    async def fail_during_tool(*_args, **_kwargs):
        raise sentinel

    monkeypatch.setattr("src.repair_flow.llm_call", select_tool)

    with pytest.raises(CancellationDrainError) as raised:
        await generate_opus_repair(
            state,
            build_escalation_packet(state),
            router=fail_during_tool,
        )

    assert raised.value is sentinel
    assert raised.value.cancellation is cancellation
    assert raised.value.cleanup_error is cleanup_error
    assert state.model_history == []
    assert state.token_usage == 0


async def test_opus_inner_repair_tool_uses_delta_evidence_and_pre_call_policy(
    tmp_path,
    monkeypatch,
):
    source = "class Widget:\n    def compute(self, value):\n        return value\n"
    repo, ref = _git_repo(tmp_path, {"src/widget.py": source})
    state = _state(repo, ref)
    EvidenceStore(state).add(
        tool="read_range",
        summary="old evidence",
        content="old-evidence-sentinel",
    )
    calls = []
    policy_calls = []
    responses = [
        {
            "kind": "tool",
            "tool_intent": {
                "action": "search_text",
                "args": {"text": "offset"},
                "reason": "locate cause",
                "expected_evidence": "offset definition",
            },
        },
        {"kind": "repair_plan", **_plan().model_dump(mode="json")},
        {
            "kind": "tool",
            "tool_intent": {
                "action": "read_symbol",
                "args": {"path": "src/widget.py", "symbol": "Widget.compute"},
                "reason": "confirm exact target",
                "expected_evidence": "target definition",
            },
        },
        {
            "kind": "verified_edits",
            "edits": [
                {
                    "file_path": "src/widget.py",
                    "node_target": "Widget.compute",
                    "search": "",
                    "replace": "def compute(self, value):\n    return value + 1\n",
                    "intent": "Apply offset.",
                }
            ],
        },
    ]

    async def fake_llm_call(system, user, **kwargs):
        calls.append(user)
        return responses.pop(0)

    router_calls = 0

    async def fake_router(current, intent, *, calls_this_round):
        nonlocal router_calls
        router_calls += 1
        added = EvidenceStore(current).add(
            tool=intent.action,
            summary=f"new evidence {router_calls}",
            content=f"new-evidence-sentinel-{router_calls}",
        )
        return SimpleNamespace(
            status="ok",
            evidence_id=added.evidence.evidence_id,
            made_progress=True,
            control_action="",
        )

    def policy_hook(current):
        policy_calls.append(current.active_provider)

    monkeypatch.setattr("src.repair_flow.llm_call", fake_llm_call)

    plan, batch = await generate_opus_repair(
        state,
        build_escalation_packet(state),
        router=fake_router,
        policy_hook=policy_hook,
    )

    assert plan.target_symbols == ["Widget.compute"]
    assert batch.edits[0].node_target == "Widget.compute"
    assert len(policy_calls) == 4
    assert "new-evidence-sentinel-1" in calls[1]
    assert "old-evidence-sentinel" not in calls[1]
    assert "new-evidence-sentinel-2" in calls[3]
    assert "old-evidence-sentinel" not in calls[3]
    assert state.token_usage == sum(
        item.input_tokens + item.output_tokens for item in state.model_history
    )


async def test_default_repairplan_tool_reprompt_retains_initial_source_evidence(
    tmp_path,
    monkeypatch,
):
    source = "class Widget:\n    def compute(self, value):\n        return value\n"
    repo, ref = _git_repo(tmp_path, {"src/widget.py": source})
    state = _state(repo, ref)
    store = EvidenceStore(state)
    seeded = [
        store.add(
            tool="planner_relevant_file",
            summary=f"supplied source {index}",
            content=f"{source}# supplied-source-{index}\n",
            file_path=f"src/supplied_{index}.py",
        ).evidence
        for index in range(4)
    ]
    stale = store.add(
        tool="read_range",
        summary="old context",
        content="old-evidence-sentinel",
    ).evidence
    calls = []
    responses = [
        {
            "kind": "tool",
            "tool_intent": {
                "action": "search_text",
                "args": {"text": "compute"},
                "reason": "confirm target",
                "expected_evidence": "source match",
            },
        },
        {"kind": "repair_plan", **_plan().model_dump(mode="json")},
        {
            "kind": "verified_edits",
            "edits": [
                {
                    "file_path": "src/widget.py",
                    "node_target": "Widget.compute",
                    "search": "",
                    "replace": "def compute(self, value):\n    return value + 1\n",
                    "intent": "Apply offset.",
                }
            ],
        },
    ]

    async def fake_llm_call(system, user, **kwargs):
        calls.append(user)
        return responses.pop(0)

    async def fake_router(current, intent, *, calls_this_round):
        added = EvidenceStore(current).add(
            tool=intent.action,
            summary="fresh tool evidence",
            content="fresh-tool-evidence-sentinel",
        )
        return SimpleNamespace(
            status="ok",
            evidence_id=added.evidence.evidence_id,
            made_progress=True,
            control_action="",
        )

    monkeypatch.setattr("src.repair_flow.llm_call", fake_llm_call)

    await generate_opus_repair(
        state,
        build_escalation_packet(state),
        router=fake_router,
    )

    reprompt = calls[1]
    packet = json.loads(reprompt)
    assert [item["tool"] for item in packet["evidence"]] == [
        "search_text",
        "planner_relevant_file",
        "planner_relevant_file",
        "planner_relevant_file",
    ]
    assert [item["evidence_id"] for item in packet["evidence"][1:]] == [
        item.evidence_id for item in seeded[:3]
    ]
    assert seeded[3].evidence_id not in {
        item["evidence_id"] for item in packet["evidence"]
    }
    assert stale.evidence_id not in {item["evidence_id"] for item in packet["evidence"]}
    assert all(
        len(item["content"]) <= EVIDENCE_CONTENT_LIMIT for item in packet["evidence"]
    )
    rendered_evidence = "\n".join(
        json.dumps(item, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        for item in packet["evidence"]
    )
    assert len(rendered_evidence) <= EVIDENCE_TOTAL_LIMIT
    assert len(reprompt) <= ESCALATION_PACKET_RENDER_LIMIT


async def test_direct_repair_hydrates_source_for_first_and_inner_plan_prompts(
    tmp_path,
    monkeypatch,
):
    long_header = "\n".join(f"import dependency_{index}" for index in range(700))
    source = (
        f"{long_header}\n"
        "class Widget:\n"
        "    def compute(self, value):\n"
        "        return value\n"
    )
    repo, ref = _git_repo(tmp_path, {"src/widget.py": source})
    state = _state(repo, ref)
    state.relevant_files = [FileInfo(path="src/widget.py", content=source)]
    packet = build_escalation_packet(state)
    assert packet.evidence == ()
    calls = []
    responses = [
        {
            "kind": "tool",
            "tool_intent": {
                "action": "search_text",
                "args": {"text": "compute"},
                "reason": "confirm the target",
                "expected_evidence": "matching source line",
            },
        },
        {"kind": "repair_plan", **_plan().model_dump(mode="json")},
        {
            "kind": "verified_edits",
            "edits": [
                {
                    "file_path": "src/widget.py",
                    "node_target": "Widget.compute",
                    "search": "",
                    "replace": ("def compute(self, value):\n    return value + 1\n"),
                    "intent": "Apply the required offset.",
                }
            ],
        },
    ]

    async def fake_llm_call(system, user, **kwargs):
        calls.append(user)
        return responses.pop(0)

    async def fake_router(current, intent, *, calls_this_round):
        added = EvidenceStore(current).add(
            tool=intent.action,
            summary="fresh tool evidence",
            content="fresh-tool-evidence-sentinel",
        )
        return SimpleNamespace(
            status="ok",
            evidence_id=added.evidence.evidence_id,
            made_progress=True,
            control_action="",
        )

    monkeypatch.setattr("src.repair_flow.llm_call", fake_llm_call)

    custom_reprompt_suffix = (
        "\n\nCompleted attempts (rolling summary):\nkeep bounded context"
    )
    await generate_opus_repair(
        state,
        packet,
        router=fake_router,
        first_stage_reprompt=lambda evidence_ids: (
            f"{render_escalation_packet(packet)}{custom_reprompt_suffix}"
        ),
    )

    first_packet = json.loads(calls[0])
    inner_packet, inner_offset = json.JSONDecoder().raw_decode(calls[1])
    assert [item["tool"] for item in first_packet["evidence"]] == [
        "planner_relevant_file"
    ]
    assert first_packet["evidence"][0]["file_path"] == "src/widget.py"
    assert "class Widget:" in first_packet["evidence"][0]["content"]
    assert [item["tool"] for item in inner_packet["evidence"]] == [
        "search_text",
        "planner_relevant_file",
    ]
    assert "fresh-tool-evidence-sentinel" in inner_packet["evidence"][0]["content"]
    assert "class Widget:" in inner_packet["evidence"][1]["content"]
    assert calls[1][inner_offset:] == custom_reprompt_suffix
    assert calls[1].count('"issue_title"') == 1


async def test_direct_repair_truncates_hydrated_source_candidates_to_three(
    tmp_path,
    monkeypatch,
):
    source = "class Widget:\n    def compute(self, value):\n        return value\n"
    repo, ref = _git_repo(tmp_path, {"src/widget.py": source})
    state = _state(repo, ref)
    state.relevant_files = [
        FileInfo(path="src/widget.py", content=source),
        FileInfo(path="src/second.py", content="second-source-sentinel"),
        FileInfo(path="src/third.py", content="third-source-sentinel"),
        FileInfo(path="src/fourth.py", content="fourth-source-sentinel"),
    ]
    packet = build_escalation_packet(state)
    calls = []
    responses = [
        _plan().model_dump(mode="json"),
        {
            "edits": [
                {
                    "file_path": "src/widget.py",
                    "node_target": "Widget.compute",
                    "search": "",
                    "replace": ("def compute(self, value):\n    return value + 1\n"),
                    "intent": "Apply the required offset.",
                }
            ]
        },
    ]

    async def fake_llm_call(system, user, **kwargs):
        calls.append(user)
        return responses.pop(0)

    monkeypatch.setattr("src.repair_flow.llm_call", fake_llm_call)

    await generate_opus_repair(state, packet)

    first_packet = json.loads(calls[0])
    source_evidence = [
        item
        for item in first_packet["evidence"]
        if item["tool"] == "planner_relevant_file"
    ]
    assert [item["file_path"] for item in source_evidence] == [
        "src/widget.py",
        "src/second.py",
        "src/third.py",
    ]
    assert all(
        len(item["content"]) <= EVIDENCE_CONTENT_LIMIT for item in source_evidence
    )
    assert "fourth-source-sentinel" not in calls[0]


async def test_direct_repair_reserves_state_capacity_for_real_tool_evidence(
    tmp_path,
    monkeypatch,
):
    source = "class Widget:\n    def compute(self, value):\n        return value\n"
    repo, ref = _git_repo(tmp_path, {"src/widget.py": source})
    state = _state(repo, ref)
    state.relevant_files = [FileInfo(path="src/widget.py", content=source)]
    store = EvidenceStore(state)
    for index in range(30):
        assert store.add(
            tool="read_range",
            summary=f"stale evidence {index}",
            content=f"stale-evidence-{index}",
        ).added
    packet = build_escalation_packet(state)
    calls = []
    responses = [
        {
            "kind": "tool",
            "tool_intent": {
                "action": "search_text",
                "args": {"text": "return value"},
                "reason": "locate the stale return",
                "expected_evidence": "matching source line",
            },
        },
        {"kind": "repair_plan", **_plan().model_dump(mode="json")},
        {
            "kind": "verified_edits",
            "edits": [
                {
                    "file_path": "src/widget.py",
                    "node_target": "Widget.compute",
                    "search": "",
                    "replace": ("def compute(self, value):\n    return value + 1\n"),
                    "intent": "Apply the required offset.",
                }
            ],
        },
    ]

    async def fake_llm_call(system, user, **kwargs):
        calls.append(user)
        return responses.pop(0)

    monkeypatch.setattr("src.repair_flow.llm_call", fake_llm_call)

    await generate_opus_repair(state, packet)

    inner_packet = json.loads(calls[1])
    assert [item["tool"] for item in inner_packet["evidence"][:2]] == [
        "search_text",
        "planner_relevant_file",
    ]
    assert len(state.evidence) <= 30
    assert any(item.tool == "search_text" for item in state.evidence)


async def test_opus_inner_repair_stop_terminates_without_schema_retry(
    tmp_path,
    monkeypatch,
):
    repo, ref = _git_repo(
        tmp_path,
        {"src/widget.py": "def compute(value):\n    return value\n"},
    )
    state = _state(repo, ref)
    calls = 0

    async def fake_llm_call(*args, **kwargs):
        nonlocal calls
        calls += 1
        return {"kind": "stop", "stop_reason": "no safe repair"}

    monkeypatch.setattr("src.repair_flow.llm_call", fake_llm_call)

    with pytest.raises(ReasoningStop, match="no safe repair"):
        await generate_opus_repair(state, build_escalation_packet(state))

    assert calls == 1
    [invocation] = state.model_history
    assert invocation.status == "invalid_response"
    assert state.token_usage == (invocation.input_tokens + invocation.output_tokens)


async def test_generate_opus_repair_uses_custom_prompt_only_for_first_stage(
    tmp_path, monkeypatch
):
    source = "class Widget:\n    def compute(self, value):\n        return value\n"
    repo, ref = _git_repo(tmp_path, {"src/widget.py": source})
    state = _state(repo, ref)
    packet = build_escalation_packet(state)
    summary_section = "Completed attempts (rolling summary):"
    first_stage_prompt = (
        f"{render_escalation_packet(packet)}\n\n{summary_section}\nsafe rolling outcome"
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
                    "replace": ("def compute(self, value):\n    return value + 1\n"),
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


async def test_custom_first_stage_prompt_retains_hydrated_source(tmp_path, monkeypatch):
    source = "def widget():\n    return 'old-sentinel'\n"
    repo, ref = _git_repo(tmp_path, {"src/widget.py": source})
    state = _state(repo, ref)
    state.relevant_files = [FileInfo(path="src/widget.py", content=source)]
    packet = build_escalation_packet(state)
    suffix = "\n\nCompleted attempts (rolling summary):\nsafe rolling outcome"
    first_stage_prompt = f"{render_escalation_packet(packet)}{suffix}"
    calls = []

    async def fake_llm_call(*args, **kwargs):
        calls.append(args[1])
        return {"kind": "stop", "stop_reason": "captured hydrated prompt"}

    monkeypatch.setattr("src.repair_flow.llm_call", fake_llm_call)

    with pytest.raises(ReasoningStop, match="captured hydrated prompt"):
        await generate_opus_repair(
            state,
            packet,
            first_stage_prompt=first_stage_prompt,
        )

    assert len(calls) == 1
    payload, offset = json.JSONDecoder().raw_decode(calls[0])
    assert calls[0][offset:] == suffix
    assert calls[0].count('"issue_title"') == 1
    assert [
        item["file_path"]
        for item in payload["evidence"]
        if item["tool"] == "planner_relevant_file"
    ] == ["src/widget.py"]


@pytest.mark.parametrize("tail", ["second_packet", "arbitrary_text"])
async def test_custom_first_stage_prompt_rejects_unclassified_tail(
    tmp_path, monkeypatch, tail
):
    repo, ref = _git_repo(
        tmp_path,
        {"src/widget.py": "def widget():\n    return 1\n"},
    )
    state = _state(repo, ref)
    packet = build_escalation_packet(state)
    rendered = render_escalation_packet(packet)
    suffix = rendered if tail == "second_packet" else "\n\narbitrary text"

    async def unexpected_llm_call(*args, **kwargs):
        raise AssertionError("invalid override must fail before the model call")

    monkeypatch.setattr("src.repair_flow.llm_call", unexpected_llm_call)

    with pytest.raises(RepairContextError, match="suffix"):
        await generate_opus_repair(
            state,
            packet,
            first_stage_prompt=f"{rendered}{suffix}",
        )


async def test_custom_inner_reprompt_rejects_second_packet_tail(tmp_path, monkeypatch):
    repo, ref = _git_repo(
        tmp_path,
        {"src/widget.py": "def widget():\n    return 1\n"},
    )
    state = _state(repo, ref)
    packet = build_escalation_packet(state)
    rendered = render_escalation_packet(packet)
    calls = []

    async def fake_llm_call(*args, **kwargs):
        calls.append(args[1])
        return {
            "kind": "tool",
            "tool_intent": {
                "action": "search_text",
                "args": {"text": "widget"},
                "reason": "confirm target",
                "expected_evidence": "matching source",
            },
        }

    async def fake_router(current, intent, *, calls_this_round):
        added = EvidenceStore(current).add(
            tool=intent.action,
            summary="fresh tool evidence",
            content="fresh-tool-evidence-sentinel",
        )
        return SimpleNamespace(
            status="ok",
            evidence_id=added.evidence.evidence_id,
            made_progress=True,
            control_action="",
        )

    monkeypatch.setattr("src.repair_flow.llm_call", fake_llm_call)

    with pytest.raises(RepairContextError, match="suffix"):
        await generate_opus_repair(
            state,
            packet,
            router=fake_router,
            first_stage_reprompt=lambda evidence_ids: f"{rendered}{rendered}",
        )

    assert len(calls) == 1


async def test_generate_opus_repair_renders_one_bounded_packet_with_suffix(
    tmp_path, monkeypatch
):
    files = {
        f"src/module_{index}.py": (
            f"def function_{index}():\n    return {'x' * 5900!r}\n"
        )
        for index in range(3)
    }
    repo, ref = _git_repo(tmp_path, files)
    state = _state(repo, ref)
    state.relevant_files = [
        FileInfo(path=path, content=content) for path, content in files.items()
    ]
    state.evidence = [
        Evidence(
            evidence_id=f"ev_stale{index:010d}",
            tool="search_text",
            summary=f"stale evidence {index}",
            content=f"stale-{index}-" + ("y" * 1900),
            fingerprint=hashlib.sha256(f"stale-{index}".encode("utf-8")).hexdigest(),
        )
        for index in range(15)
    ]
    packet = build_escalation_packet(state)
    suffix = "\n\nCompleted attempts (rolling summary):\nsafe rolling outcome"
    calls = []

    async def fake_llm_call(*args, **kwargs):
        calls.append(args[1])
        return {"kind": "stop", "stop_reason": "captured bounded prompt"}

    monkeypatch.setattr("src.repair_flow.llm_call", fake_llm_call)

    with pytest.raises(ReasoningStop, match="captured bounded prompt"):
        await generate_opus_repair(
            state,
            packet,
            first_stage_suffix=suffix,
        )

    assert len(calls) == 1
    rendered = calls[0]
    payload, offset = json.JSONDecoder().raw_decode(rendered)
    assert rendered[offset:] == suffix
    assert rendered.count('"issue_title"') == 1
    assert len(payload["evidence"]) <= EVIDENCE_LIMIT
    assert (
        sum(item["tool"] == "planner_relevant_file" for item in payload["evidence"])
        <= 3
    )
    assert (
        len(
            EvidenceStore.render_for_prompt(
                [Evidence.model_validate(item) for item in payload["evidence"]]
            )
        )
        <= EVIDENCE_TOTAL_LIMIT
    )


async def test_generate_opus_repair_fails_before_second_call_for_invalid_target(
    tmp_path, monkeypatch
):
    repo, ref = _git_repo(tmp_path, {"src/widget.py": "def compute():\n    return 1\n"})
    state = _state(repo, ref)
    calls = []

    async def fake_llm_call(
        system, user, model=None, *, provider="primary", temperature=0.2
    ):
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


async def test_generate_opus_repair_defers_missing_anchor_validation_when_disabled(
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
                    "search": "missing anchor",
                    "replace": "return value + 1",
                    "intent": "Apply the offset through PatchGate.",
                }
            ]
        },
    ]

    async def fake_llm_call(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr("src.repair_flow.llm_call", fake_llm_call)

    plan, batch = await generate_opus_repair(
        state,
        build_escalation_packet(state),
        validate_edits=False,
    )

    assert plan.target_files == ["src/widget.py"]
    assert batch.edits[0].search == "missing anchor"
    assert [item.status for item in state.model_history] == ["ok", "ok"]


async def test_generate_opus_repair_rejects_missing_anchor_when_validation_enabled(
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
                    "search": "missing anchor",
                    "replace": "return value + 1",
                    "intent": "Apply the offset.",
                }
            ]
        },
    ]

    async def fake_llm_call(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr("src.repair_flow.llm_call", fake_llm_call)

    with pytest.raises(RepairContextError, match="missing or not unique"):
        await generate_opus_repair(
            state,
            build_escalation_packet(state),
            validate_edits=True,
        )

    assert [item.status for item in state.model_history] == ["ok", "invalid_response"]


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

    async def fake_llm_call(
        system, user, model=None, *, provider="primary", temperature=0.2
    ):
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
    assert patch_edit.resolved_target_symbol == ""
    result = _apply_patch_edits(str(repo), [patch_edit])
    assert result.applied, result.output
    assert target.read_text(encoding="utf-8") == (
        "def compute(value):\n    current = value + 1\n    return current\n"
    )
