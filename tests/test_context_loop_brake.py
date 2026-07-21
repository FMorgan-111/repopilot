"""Regression tests for the collect_more_context loop brake.

Covers the three behaviors added to stop the PLAN <-> LOCATE loop from burning
the whole token budget:
  1. plan_fix forces `stop` once collect_more_context exceeds the hard cap.
  2. locate_code reads the files the planner named in next_checks/trace_notes.
  3. locate_code stops early when a round locates the exact same files again.
"""

import json
from types import SimpleNamespace

from src import new_agent
from src.nodes import locate as locate_node
from src.nodes import plan as plan_node
from src.state import Evidence, ToolInvocation


class EmptyMemoryStore:
    async def get_file_index(self, owner, repo, limit=8):
        return []


def _collect_more_context_response(summary="Need more context before patching."):
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
                "next_checks": ["Search for the request router middleware."],
                "risk": "unknown",
                "confidence": 0.5,
            },
        }
    )


def _base_state(**updates):
    values = {
        "issue_url": "https://github.com/acme/widget/issues/7",
        "issue_title": "sentinel",
        "issue_body": "update sentinel",
        "current_phase": new_agent.Phase.PLAN,
    }
    values.update(updates)
    return new_agent.AgentState(**values)


def _execute_response(file_path, search, replace):
    return json.dumps(
        {
            "kind": "plan",
            "plan": "Update the sentinel.",
            "patch_edits": [
                {"file": file_path, "search": search, "replace": replace}
            ],
            "files": [file_path],
            "test_command": "pytest tests/test_widget.py -q",
            "decision_frame": {
                "stage": "plan",
                "summary": "Update the sentinel.",
                "recommended_action": "execute",
                "risk": "low",
                "confidence": 0.9,
            },
        }
    )


async def test_plan_fix_collect_more_context_under_cap_still_routes_to_locate(
    monkeypatch,
):
    async def fake_llm_call(system, user):
        return _collect_more_context_response()

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        issue_title="Login crash",
        issue_body="Crashes after submit.",
        current_phase=new_agent.Phase.PLAN,
    )

    next_state = await plan_node.plan_fix(state)

    assert next_state.current_phase == new_agent.Phase.PLAN
    assert next_state.context_collection_count == 1
    assert next_state.failure_reason == ""
    assert next_state.decision_frame.recommended_action == "collect_more_context"


async def test_plan_fix_forces_stop_after_context_collection_cap(monkeypatch):
    async def fake_llm_call(system, user):
        return _collect_more_context_response()

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        issue_title="Login crash",
        issue_body="Crashes after submit.",
        current_phase=new_agent.Phase.PLAN,
    )

    # First MAX rounds keep routing to PLAN (so the router sends LOCATE).
    for round_num in range(1, plan_node.MAX_CONTEXT_COLLECTION_ROUNDS + 1):
        state = await plan_node.plan_fix(state)
        assert state.current_phase == new_agent.Phase.PLAN
        assert state.context_collection_count == round_num

    # The next collect_more_context exceeds the cap and is forced to stop.
    state = await plan_node.plan_fix(state)

    assert state.current_phase == new_agent.Phase.FAILURE
    assert state.decision_frame.recommended_action == "stop"
    assert "after" in state.failure_reason
    assert str(plan_node.MAX_CONTEXT_COLLECTION_ROUNDS) in state.failure_reason
    # Router must send a forced-stop frame to handle_failure, not back to LOCATE.
    assert new_agent.route_from_state(state) == "handle_failure"


async def test_locate_code_reads_files_named_by_planner(monkeypatch):
    read_paths = []

    async def fake_search_code(query, owner, repo):
        return []  # issue-text search finds nothing; frame paths must drive it

    async def fake_read_file(owner, repo, path):
        read_paths.append(path)
        return {"content": f"# contents of {path}\n", "sha": f"sha-{path}"}

    monkeypatch.setenv("REPOPILOT_DISABLE_PARALLEL", "1")
    monkeypatch.setattr(locate_node, "get_store", lambda: EmptyMemoryStore())
    monkeypatch.setattr(locate_node, "search_code", fake_search_code)
    monkeypatch.setattr(locate_node, "read_file", fake_read_file)

    state = new_agent.AgentState(
        issue_url="https://github.com/tox-dev/tox/issues/3075",
        owner="tox-dev",
        repo="tox",
        issue_title="env reuse bug",
        issue_body="tip-black reuses the black environment.",
        current_phase=new_agent.Phase.LOCATE,
    )
    plan_frame = new_agent.DecisionFrame(
        stage="plan",
        summary="Need to inspect env identity code.",
        recommended_action="collect_more_context",
        next_checks=[
            "Read src/tox/tox_env/runner.py to find where paths are assigned.",
        ],
        confidence=0.6,
        risk="medium",
        trace_notes=json.dumps({"files": ["src/tox/config/sets.py"]}),
    )
    new_agent._record_decision_frame(state, plan_frame)

    next_state = await locate_node.locate_code(state)

    # Both the trace_notes file and the next_checks path were read.
    assert "src/tox/config/sets.py" in read_paths
    assert "src/tox/tox_env/runner.py" in read_paths
    located = {f.path for f in next_state.relevant_files}
    assert "src/tox/config/sets.py" in located
    assert "src/tox/tox_env/runner.py" in located
    assert next_state.current_phase == new_agent.Phase.PLAN


async def test_locate_code_stops_when_no_new_files_located(monkeypatch):
    async def fake_search_code(query, owner, repo):
        return [{"path": "src/app/core.py", "sha": "sha-core"}]

    async def fake_read_file(owner, repo, path):
        return {"content": "def core():\n    return 1\n", "sha": "read-core"}

    monkeypatch.setenv("REPOPILOT_DISABLE_PARALLEL", "1")
    monkeypatch.setattr(locate_node, "get_store", lambda: EmptyMemoryStore())
    monkeypatch.setattr(locate_node, "search_code", fake_search_code)
    monkeypatch.setattr(locate_node, "read_file", fake_read_file)

    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        owner="acme",
        repo="widget",
        issue_title="core bug",
        issue_body="core misbehaves.",
        current_phase=new_agent.Phase.LOCATE,
    )

    first = await locate_node.locate_code(state)
    assert first.current_phase == new_agent.Phase.PLAN
    assert first.last_locate_signature == "src/app/core.py"

    # Second round locates the identical file set -> no progress -> stop.
    second = await locate_node.locate_code(first)
    assert second.current_phase == new_agent.Phase.FAILURE
    assert "no progress" in second.failure_reason


async def test_locate_code_carries_forward_files_when_new_paths_fail(monkeypatch):
    """A good file found in round 1 survives round 2 even if round 2's paths fail.

    This is the regression for the stateless-locate starvation: the planner
    found the real env file early, then a later round requested unreadable
    paths; without carry-forward the good file was discarded at the decisive
    round and the planner was fed garbage.
    """
    round_calls = {"n": 0}

    async def fake_search_code(query, owner, repo):
        # Round 1 surfaces the real target file; later rounds surface nothing.
        if round_calls["n"] == 0:
            return [{"path": "src/tox/tox_env/api.py", "sha": "sha-api"}]
        return []

    async def fake_read_file(owner, repo, path):
        if path == "src/tox/tox_env/api.py":
            return {"content": "class ToxEnv:\n    def env_dir(self): ...\n", "sha": "r"}
        # Any other path (e.g. a planner-hallucinated one) fails to read.
        raise RuntimeError(f"404 not found: {path}")

    monkeypatch.setenv("REPOPILOT_DISABLE_PARALLEL", "1")
    monkeypatch.setattr(locate_node, "get_store", lambda: EmptyMemoryStore())
    monkeypatch.setattr(locate_node, "search_code", fake_search_code)
    monkeypatch.setattr(locate_node, "read_file", fake_read_file)

    state = new_agent.AgentState(
        issue_url="https://github.com/tox-dev/tox/issues/3075",
        owner="tox-dev",
        repo="tox",
        issue_title="env reuse",
        issue_body="tip-black reuses black env",
        current_phase=new_agent.Phase.LOCATE,
    )

    # Round 1: finds and reads the real file.
    first = await locate_node.locate_code(state)
    round_calls["n"] = 1
    assert "src/tox/tox_env/api.py" in {f.path for f in first.relevant_files}

    # Round 2: planner now asks for an unreadable, hallucinated path; search
    # surfaces nothing new. The good file from round 1 must still be present.
    plan_frame = new_agent.DecisionFrame(
        stage="plan",
        summary="chase the wrong file",
        recommended_action="collect_more_context",
        next_checks=["Read src/tox/config/loader/ini/does_not_exist.py for clues."],
        confidence=0.5,
        risk="low",
        trace_notes=json.dumps({"files": []}),
    )
    new_agent._record_decision_frame(first, plan_frame)

    second = await locate_node.locate_code(first)

    assert "src/tox/tox_env/api.py" in {f.path for f in second.relevant_files}
    # And its content is intact (not a contentless husk).
    api = next(f for f in second.relevant_files if f.path == "src/tox/tox_env/api.py")
    assert "env_dir" in api.content


async def test_locate_code_excludes_docs_so_source_reaches_planner(monkeypatch):
    """Docs (.rst, docs/) must not crowd source code out of the planner's view.

    Regression: BM25 ranks huge keyword-dense docs above source, and once
    accumulated they filled every PLAN_MAX_FILES slot, starving the planner of
    code. A patch_edits + pytest agent fixes source, so docs are excluded.
    """
    async def fake_search_code(query, owner, repo):
        return [
            {"path": "docs/reference/config.rst", "sha": "sha-doc"},
            {"path": "docs/changelog.rst", "sha": "sha-cl"},
            {"path": "src/tox/tox_env/api.py", "sha": "sha-api"},
        ]

    async def fake_read_file(owner, repo, path):
        return {"content": f"# {path}\nclass X: ...\n", "sha": f"r-{path}"}

    monkeypatch.setenv("REPOPILOT_DISABLE_PARALLEL", "1")
    monkeypatch.setattr(locate_node, "get_store", lambda: EmptyMemoryStore())
    monkeypatch.setattr(locate_node, "search_code", fake_search_code)
    monkeypatch.setattr(locate_node, "read_file", fake_read_file)

    state = new_agent.AgentState(
        issue_url="https://github.com/tox-dev/tox/issues/3075",
        owner="tox-dev",
        repo="tox",
        issue_title="env reuse",
        issue_body="tip-black reuses black env",
        current_phase=new_agent.Phase.LOCATE,
    )

    next_state = await locate_node.locate_code(state)

    paths = {f.path for f in next_state.relevant_files}
    assert "src/tox/tox_env/api.py" in paths
    assert "docs/reference/config.rst" not in paths
    assert "docs/changelog.rst" not in paths
    assert not any(locate_node._is_doc_file(p) for p in paths)


async def test_plan_prompt_has_no_context_pressure_on_first_round(monkeypatch):
    captured = {}

    async def fake_llm_call(system, user):
        captured["user"] = user
        return _collect_more_context_response()

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        issue_title="Login crash",
        issue_body="Crashes after submit.",
        current_phase=new_agent.Phase.PLAN,
    )  # context_collection_count defaults to 0

    await plan_node.plan_fix(state)

    assert "Context Budget Instructions" not in captured["user"]


async def test_plan_prompt_adds_soft_pressure_mid_collection(monkeypatch):
    captured = {}

    async def fake_llm_call(system, user):
        captured["user"] = user
        return _collect_more_context_response()

    # Soft/hard pressure escalation only has a "middle" round when the cap >= 2;
    # the default cap is 1, so pin it to 3 to exercise the soft-pressure branch.
    monkeypatch.setattr(plan_node, "MAX_CONTEXT_COLLECTION_ROUNDS", 3)
    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        issue_title="Login crash",
        issue_body="Crashes after submit.",
        current_phase=new_agent.Phase.PLAN,
        context_collection_count=1,
    )

    await plan_node.plan_fix(state)

    assert "Context Budget Instructions" in captured["user"]
    assert "Strongly prefer producing" in captured["user"]
    # Not the final-round hard mandate yet.
    assert "FINAL context round" not in captured["user"]


async def test_plan_prompt_forces_commit_on_final_round(monkeypatch):
    captured = {}

    async def fake_llm_call(system, user):
        captured["user"] = user
        return _collect_more_context_response()

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        issue_title="Login crash",
        issue_body="Crashes after submit.",
        current_phase=new_agent.Phase.PLAN,
        context_collection_count=plan_node.MAX_CONTEXT_COLLECTION_ROUNDS,
    )

    await plan_node.plan_fix(state)

    assert "FINAL context round" in captured["user"]
    assert "MUST" in captured["user"]
    assert "patch_edits now" in captured["user"]


async def test_plan_tool_request_reprompts_with_only_new_evidence(monkeypatch):
    calls = []
    responses = [
        {
            "kind": "tool",
            "tool_intent": {
                "action": "read_symbol",
                "args": {"path": "src/widget.py", "symbol": "widget"},
                "reason": "inspect the sentinel",
                "expected_evidence": "widget source",
            },
        },
        {
            "kind": "plan",
            "plan": "Update the sentinel.",
            "patch_edits": [
                {
                    "file": "src/widget.py",
                    "search": "return 'old-sentinel'",
                    "replace": "return 'new-sentinel'",
                }
            ],
            "files": ["src/widget.py"],
            "test_command": "pytest tests/test_widget.py -q",
            "decision_frame": {
                "stage": "plan",
                "summary": "Update the sentinel.",
                "recommended_action": "execute",
                "risk": "low",
                "confidence": 0.9,
            },
        },
    ]

    async def fake_llm_call(system, user, **kwargs):
        calls.append(user)
        return responses.pop(0)

    async def fake_route(state, intent, *, calls_this_round):
        state.evidence.append(
            Evidence(
                evidence_id="ev_new_sentinel",
                tool="read_symbol",
                file_path="src/widget.py",
                symbol="widget",
                summary="new evidence",
                content="return 'old-sentinel'",
                fingerprint="new-fingerprint",
            )
        )
        state.tool_history.append(
            ToolInvocation(
                action="read_symbol",
                args_fingerprint="f" * 64,
                status="ok",
                evidence_id="ev_new_sentinel",
            )
        )
        return SimpleNamespace(
            status="ok",
            evidence_id="ev_new_sentinel",
            made_progress=True,
            control_action="",
        )

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    monkeypatch.setattr(plan_node, "route_tool_intent", fake_route, raising=False)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        issue_title="sentinel",
        issue_body="update sentinel",
        current_phase=new_agent.Phase.PLAN,
        repo_ref="a" * 40,
        relevant_files=[
            new_agent.FileInfo(
                path="src/widget.py",
                content="def widget():\n    return 'old-sentinel'\n",
            )
        ],
        evidence=[
            Evidence(
                evidence_id="ev_old_sentinel",
                tool="read_range",
                summary="old evidence",
                content="must not be reinjected",
                fingerprint="old-fingerprint",
            )
        ],
    )

    result = await plan_node.plan_fix(state)

    assert result.current_phase == new_agent.Phase.EXECUTE
    assert len(calls) == 2
    assert "ev_new_sentinel" in calls[1]
    assert "return 'old-sentinel'" in calls[1]
    assert "ev_old_sentinel" not in calls[1]


async def test_duplicate_tool_request_counts_no_progress_before_plan(monkeypatch):
    responses = [
        {
            "kind": "tool",
            "tool_intent": {
                "action": "search_text",
                "args": {"text": "sentinel"},
                "reason": "find sentinel",
                "expected_evidence": "source location",
            },
        },
        _execute_response(
            "src/widget.py",
            "return 'old-sentinel'",
            "return 'new-sentinel'",
        ),
    ]

    async def fake_llm_call(system, user, **kwargs):
        return responses.pop(0)

    async def duplicate_route(state, intent, *, calls_this_round):
        return SimpleNamespace(
            status="duplicate",
            evidence_id=None,
            made_progress=False,
            control_action="",
        )

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    monkeypatch.setattr(plan_node, "route_tool_intent", duplicate_route, raising=False)
    state = _base_state(
        relevant_files=[
            new_agent.FileInfo(
                path="src/widget.py",
                content="def widget():\n    return 'old-sentinel'\n",
            )
        ]
    )

    result = await plan_node.plan_fix(state)

    assert result.current_phase == new_agent.Phase.EXECUTE
    assert result.no_progress_rounds == 0
    assert result.no_progress_history[-1].kind == "unchanged_context"


async def test_plan_routes_no_more_than_eight_tool_requests_per_round(monkeypatch):
    routed = []

    async def fake_llm_call(system, user, **kwargs):
        return {
            "kind": "tool",
            "tool_intent": {
                "action": "search_text",
                "args": {"text": f"sentinel-{len(routed)}"},
                "reason": "bounded investigation",
                "expected_evidence": "one result",
            },
        }

    async def fake_route(state, intent, *, calls_this_round):
        routed.append(intent.args["text"])
        return SimpleNamespace(
            status="duplicate",
            evidence_id=None,
            made_progress=False,
            control_action="",
        )

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    monkeypatch.setattr(plan_node, "route_tool_intent", fake_route, raising=False)

    result = await plan_node.plan_fix(_base_state())

    assert len(routed) == 8
    assert result.current_phase == new_agent.Phase.FAILURE
    assert result.failure_reason == "tool_round_limit"
