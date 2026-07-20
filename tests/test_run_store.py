import json

from src import new_agent, run_store
from src.state import Evidence, ModelInvocation, NoProgressEvent


def paused_state(trace_id: str = "abc123def456"):
    frame = new_agent.DecisionFrame(
        frame_id="df_0001",
        stage="plan",
        summary="Need user approval before patching.",
        hypotheses=[
            new_agent.Hypothesis(
                id="H1",
                claim="The change is too risky to ship without confirmation.",
                evidence=["The issue requests a breaking API change."],
                score=0.88,
            )
        ],
        selected_hypothesis_id="H1",
        next_checks=["Confirm whether breaking changes are allowed."],
        recommended_action="ask_user",
        confidence=0.88,
        risk="high",
    )
    return new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        trace_id=trace_id,
        current_phase=new_agent.Phase.WAITING_FOR_USER,
        pending_human_input=True,
        human_input_request={
            "frame_id": "df_0001",
            "stage": "plan",
            "question": "Confirm whether breaking changes are allowed.",
            "summary": "Need user approval before patching.",
            "risk": "high",
            "confidence": 0.88,
        },
        decision_frame=frame,
        frame_history=[frame],
        route_decisions=[
            {
                "source": "decision_frame",
                "current_phase": "PLAN",
                "selected_phase": "WAITING_FOR_USER",
                "route": "WAITING_FOR_USER",
                "frame_id": "df_0001",
                "recommended_action": "ask_user",
            }
        ],
    )


def replay_state(trace_id: str = "abc123def456"):
    plan_frame = new_agent.DecisionFrame(
        frame_id="df_0001",
        stage="plan",
        summary="Need user approval before patching.",
        hypotheses=[
            new_agent.Hypothesis(
                id="H1",
                claim="The API change may be breaking.",
                evidence=["The issue asks to remove an existing field."],
                score=0.82,
                why_selected="It explains the compatibility risk.",
            )
        ],
        selected_hypothesis_id="H1",
        evidence=["Existing clients depend on the field."],
        next_checks=["Confirm whether breaking changes are allowed."],
        recommended_action="ask_user",
        confidence=0.82,
        risk="high",
        trace_notes="Planner stopped before patching.",
    )
    reflect_frame = new_agent.DecisionFrame(
        frame_id="df_0002",
        stage="reflect",
        summary="Previous patch failed because tests expect compatibility.",
        selected_hypothesis_id="H2",
        evidence=["Regression test failed on missing field."],
        recommended_action="plan",
        confidence=0.74,
        risk="medium",
        parent_frame_id="df_0001",
    )
    return new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        trace_id=trace_id,
        current_phase=new_agent.Phase.WAITING_FOR_USER,
        pending_human_input=True,
        human_input_request={
            "frame_id": "df_0001",
            "stage": "plan",
            "question": "Confirm whether breaking changes are allowed.",
            "summary": "Need user approval before patching.",
            "risk": "high",
            "confidence": 0.82,
        },
        decision_frame=plan_frame,
        frame_history=[plan_frame, reflect_frame],
        decision_warnings=[
            {
                "frame_id": "df_0001",
                "recommended_action": "ask_user",
                "expected_phase": "WAITING_FOR_USER",
                "actual_phase": "PLAN",
            }
        ],
        route_decisions=[
            {
                "source": "decision_frame",
                "current_phase": "PLAN",
                "selected_phase": "WAITING_FOR_USER",
                "route": "__end__",
                "frame_id": "df_0001",
                "recommended_action": "ask_user",
            },
            {
                "source": "current_phase",
                "current_phase": "PLAN",
                "selected_phase": "PLAN",
                "route": "plan_fix",
                "fallback_reason": "already_consumed",
            },
        ],
    )


def diagnostic_state(trace_id: str = "diag123"):
    state = replay_state(trace_id)
    state.node_diagnostics.append(
        {
            "node": "plan_fix",
            "event": "phase",
            "status": "timeout",
            "elapsed_seconds": 90.0,
            "error_type": "TimeoutError",
            "error": "TimeoutError",
            "phase_timeout_seconds": 90.0,
        }
    )
    return state


def test_save_and_load_paused_run_preserves_pause_state(tmp_path):
    root_dir = tmp_path / ".repopilot"
    state = paused_state()

    saved_path = run_store.save_run(state, root_dir=root_dir)
    loaded = run_store.load_run(state.trace_id, root_dir=root_dir)

    assert saved_path == root_dir / "runs" / f"{state.trace_id}.json"
    assert loaded.trace_id == state.trace_id
    assert loaded.current_phase == new_agent.Phase.WAITING_FOR_USER
    assert loaded.pending_human_input is True
    assert loaded.human_input_request == state.human_input_request
    assert loaded.frame_history == state.frame_history
    assert loaded.route_decisions == state.route_decisions


def test_save_and_load_preserves_model_routing_state(tmp_path):
    root_dir = tmp_path / ".repopilot"
    state = paused_state("routing-state")
    state.active_model = "claude-opus-4-8:stable"
    state.active_provider = "escalation"
    state.escalated = True
    state.escalation_reason = "repeated_edit"
    state.no_progress_rounds = 2
    state.last_plan_signature = "plan-sha"
    state.last_context_fingerprint = "context-sha"
    state.last_test_failure_signature = "test-sha"
    state.model_history = [
        ModelInvocation(
            model="gemini-3.5-flash:stable",
            provider="primary",
            node="plan_fix",
            elapsed_seconds=1.25,
            input_tokens=120,
            output_tokens=35,
            status="invalid_response",
            error_class="ValidationError: secret response body",
        )
    ]
    state.no_progress_history = [
        NoProgressEvent(
            kind="repeated_edit", fingerprint="edit-sha", node="plan_fix"
        )
    ]

    run_store.save_run(state, root_dir=root_dir)
    loaded = run_store.load_run(state.trace_id, root_dir=root_dir)

    assert loaded.active_model == state.active_model
    assert loaded.active_provider == state.active_provider
    assert loaded.escalated is True
    assert loaded.escalation_reason == "repeated_edit"
    assert loaded.no_progress_rounds == 2
    assert loaded.last_plan_signature == "plan-sha"
    assert loaded.last_context_fingerprint == "context-sha"
    assert loaded.last_test_failure_signature == "test-sha"
    assert loaded.model_history == state.model_history
    assert loaded.model_history[0].error_class == "ValidationError"
    assert loaded.no_progress_history == state.no_progress_history


def test_legacy_saved_state_loads_model_routing_defaults(tmp_path):
    root_dir = tmp_path / ".repopilot"
    path = run_store.run_path("legacy", root_dir=root_dir)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "issue_url": "https://github.com/acme/widget/issues/7",
                "trace_id": "legacy",
                "current_phase": "PLAN",
            }
        ),
        encoding="utf-8",
    )

    loaded = run_store.load_run("legacy", root_dir=root_dir)

    assert loaded.active_model == "gemini-3.5-flash:stable"
    assert loaded.active_provider == "primary"
    assert loaded.escalated is False
    assert loaded.escalation_reason == ""
    assert loaded.no_progress_rounds == 0
    assert loaded.model_history == []
    assert loaded.no_progress_history == []


def test_save_load_and_replay_preserve_outcome_summary_state(tmp_path):
    root_dir = tmp_path / ".repopilot"
    state = paused_state("outcome-summary")
    state.attempt_outcome_summary = "Guard moved before unsafe submit."
    state.summary_token_usage = 47

    run_store.save_run(state, root_dir=root_dir)
    loaded = run_store.load_run(state.trace_id, root_dir=root_dir)
    replay = run_store.replay_run(state.trace_id, root_dir=root_dir)

    assert loaded.attempt_outcome_summary == state.attempt_outcome_summary
    assert loaded.summary_token_usage == 47
    assert replay["attempt_outcome_summary"] == state.attempt_outcome_summary
    assert replay["summary_token_usage"] == 47


def test_legacy_saved_state_loads_outcome_summary_defaults(tmp_path):
    root_dir = tmp_path / ".repopilot"
    path = run_store.run_path("legacy-summary", root_dir=root_dir)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "issue_url": "https://github.com/acme/widget/issues/7",
                "trace_id": "legacy-summary",
                "current_phase": "PLAN",
            }
        ),
        encoding="utf-8",
    )

    loaded = run_store.load_run("legacy-summary", root_dir=root_dir)

    assert loaded.attempt_outcome_summary == ""
    assert loaded.summary_token_usage == 0


def test_save_and_load_preserves_evidence_and_legacy_state_defaults_to_empty(tmp_path):
    root_dir = tmp_path / ".repopilot"
    state = paused_state("evidence-state")
    state.evidence = [
        Evidence(
            evidence_id="ev_abc123",
            tool="read_file",
            file_path="src/widget.py",
            symbol="Widget.render",
            summary="Rendered widget source.",
            content="def render(): pass",
            fingerprint="abc123",
        )
    ]

    run_store.save_run(state, root_dir=root_dir)
    loaded = run_store.load_run(state.trace_id, root_dir=root_dir)

    assert loaded.evidence == state.evidence

    legacy_path = run_store.run_path("legacy-evidence", root_dir=root_dir)
    legacy_path.write_text(
        json.dumps(
            {
                "issue_url": "https://github.com/acme/widget/issues/7",
                "trace_id": "legacy-evidence",
                "current_phase": "PLAN",
            }
        ),
        encoding="utf-8",
    )

    assert run_store.load_run("legacy-evidence", root_dir=root_dir).evidence == []


def test_save_run_uses_repopilot_home_by_default(tmp_path, monkeypatch):
    repopilot_home = tmp_path / "custom-repopilot-home"
    state = paused_state()
    monkeypatch.setenv("REPOPILOT_HOME", str(repopilot_home))

    saved_path = run_store.save_run(state)

    assert run_store.default_runs_dir() == repopilot_home
    assert saved_path == repopilot_home / "runs" / f"{state.trace_id}.json"
    assert saved_path.exists()


def test_inspect_run_returns_stable_summary(tmp_path):
    root_dir = tmp_path / ".repopilot"
    state = paused_state()
    run_store.save_run(state, root_dir=root_dir)

    summary = run_store.inspect_run(state.trace_id, root_dir=root_dir)

    assert summary["run_id"] == "abc123def456"
    assert summary["issue_url"] == "https://github.com/acme/widget/issues/7"
    assert summary["current_phase"] == "WAITING_FOR_USER"
    assert summary["pending_human_input"] is True
    assert summary["human_input_question"] == "Confirm whether breaking changes are allowed."
    assert summary["latest_decision_frame"]["frame_id"] == "df_0001"
    assert summary["latest_decision_frame"]["recommended_action"] == "ask_user"
    assert summary["updated_at"].endswith("+00:00")


def test_list_runs_returns_saved_run_summaries_sorted_by_run_id(tmp_path):
    root_dir = tmp_path / ".repopilot"
    run_store.save_run(paused_state("run-b"), root_dir=root_dir)
    run_store.save_run(paused_state("run-a"), root_dir=root_dir)

    summaries = run_store.list_runs(root_dir=root_dir)

    assert [summary["run_id"] for summary in summaries] == ["run-a", "run-b"]
    assert all(summary["current_phase"] == "WAITING_FOR_USER" for summary in summaries)


def test_replay_run_returns_white_box_timeline(tmp_path):
    root_dir = tmp_path / ".repopilot"
    state = replay_state()
    run_store.save_run(state, root_dir=root_dir)

    replay = run_store.replay_run(state.trace_id, root_dir=root_dir)

    assert replay["run_id"] == "abc123def456"
    assert replay["issue_url"] == "https://github.com/acme/widget/issues/7"
    assert replay["current_phase"] == "WAITING_FOR_USER"
    assert replay["pause"]["question"] == "Confirm whether breaking changes are allowed."
    assert replay["timeline"] == [
        {
            "index": 1,
            "type": "decision_frame",
            "frame_id": "df_0001",
            "stage": "plan",
            "summary": "Need user approval before patching.",
            "selected_hypothesis_id": "H1",
            "selected_hypothesis": {
                "id": "H1",
                "claim": "The API change may be breaking.",
                "evidence": ["The issue asks to remove an existing field."],
                "score": 0.82,
                "why_selected": "It explains the compatibility risk.",
                "why_not_selected": "",
            },
            "recommended_action": "ask_user",
            "risk": "high",
            "confidence": 0.82,
            "route": {
                "source": "decision_frame",
                "current_phase": "PLAN",
                "selected_phase": "WAITING_FOR_USER",
                "route": "__end__",
                "frame_id": "df_0001",
                "recommended_action": "ask_user",
            },
            "warnings": [
                {
                    "frame_id": "df_0001",
                    "recommended_action": "ask_user",
                    "expected_phase": "WAITING_FOR_USER",
                    "actual_phase": "PLAN",
                }
            ],
            "next_checks": ["Confirm whether breaking changes are allowed."],
            "trace_notes": "Planner stopped before patching.",
        },
        {
            "index": 2,
            "type": "decision_frame",
            "frame_id": "df_0002",
            "stage": "reflect",
            "summary": "Previous patch failed because tests expect compatibility.",
            "selected_hypothesis_id": "H2",
            "selected_hypothesis": None,
            "recommended_action": "plan",
            "risk": "medium",
            "confidence": 0.74,
            "route": None,
            "warnings": [],
            "next_checks": [],
            "trace_notes": "",
        },
        {
            "index": 3,
            "type": "route_decision",
            "route": {
                "source": "current_phase",
                "current_phase": "PLAN",
                "selected_phase": "PLAN",
                "route": "plan_fix",
                "fallback_reason": "already_consumed",
            },
        },
    ]


def test_replay_run_includes_node_diagnostics(tmp_path):
    root_dir = tmp_path / ".repopilot"
    state = diagnostic_state()
    run_store.save_run(state, root_dir=root_dir)

    replay = run_store.replay_run(state.trace_id, root_dir=root_dir)

    assert replay["timeline"][-1] == {
        "index": 4,
        "type": "node_diagnostic",
        "diagnostic": {
            "node": "plan_fix",
            "event": "phase",
            "status": "timeout",
            "elapsed_seconds": 90.0,
            "error_type": "TimeoutError",
            "error": "TimeoutError",
            "phase_timeout_seconds": 90.0,
        },
    }


def test_format_replay_markdown_summarizes_timeline():
    replay = run_store.summarize_replay(replay_state())

    markdown = run_store.format_replay_markdown(replay)

    assert markdown == "\n".join(
        [
            "# RepoPilot Replay: abc123def456",
            "",
            "- Issue: https://github.com/acme/widget/issues/7",
            "- Final phase: WAITING_FOR_USER",
            "- Pending human input: yes",
            "- Question: Confirm whether breaking changes are allowed.",
            "",
            "## Timeline",
            "",
            "### 1. PLAN df_0001",
            "",
            "Need user approval before patching.",
            "",
            "- Selected hypothesis: H1",
            "- Hypothesis claim: The API change may be breaking.",
            "- Recommended action: ask_user",
            "- Risk: high",
            "- Confidence: 0.82",
            "- Route: __end__",
            "- Warning: expected WAITING_FOR_USER but actual PLAN",
            "- Next check: Confirm whether breaking changes are allowed.",
            "- Trace notes: Planner stopped before patching.",
            "",
            "### 2. REFLECT df_0002",
            "",
            "Previous patch failed because tests expect compatibility.",
            "",
            "- Selected hypothesis: H2",
            "- Recommended action: plan",
            "- Risk: medium",
            "- Confidence: 0.74",
            "",
            "### 3. Route Decision",
            "",
            "- Route: plan_fix",
            "- Source: current_phase",
            "- Fallback reason: already_consumed",
        ]
    )


def test_format_replay_markdown_includes_node_diagnostics():
    replay = run_store.summarize_replay(diagnostic_state())

    markdown = run_store.format_replay_markdown(replay)

    assert "### 4. Node Diagnostic" in markdown
    assert "- Node: plan_fix" in markdown
    assert "- Event: phase" in markdown
    assert "- Status: timeout" in markdown
    assert "- Error: TimeoutError" in markdown
    assert "- Phase timeout seconds: 90.0" in markdown
