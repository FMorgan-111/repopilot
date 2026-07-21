import json

import pytest

from src.nodes.plan import build_plan_user_prompt
from src.escalation import build_escalation_packet, render_escalation_packet
from src.outcome_summary import (
    MAX_OUTCOME_SUMMARY_CHARS,
    OutcomeSummaryInput,
    build_outcome_summary_input,
    deterministic_outcome_summary,
    summarize_attempt_outcome,
)
from src.state import (
    AgentState,
    DecisionFrame,
    FixAttempt,
    GeneratedTestApproval,
    Hypothesis,
    PatchEdit,
    Phase,
)
from src.summary_safety import sanitize_summary_text


def make_reflect_state(**updates):
    plan_frame = DecisionFrame(
        frame_id="df_0001",
        stage="plan",
        summary="Guard the missing user before submit.",
        recommended_action="execute",
    )
    reflect_frame = DecisionFrame(
        frame_id="df_0002",
        stage="reflect",
        summary="The guard was added after the unsafe call.",
        recommended_action="plan",
        parent_frame_id=plan_frame.frame_id,
    )
    values = {
        "issue_url": "https://github.com/acme/widget/issues/7",
        "current_phase": Phase.REFLECT,
        "fix_attempts": [
            FixAttempt(
                test_result="failed",
                failure_kind="assertion_failure",
                error_log="one focused assertion failed",
                success=False,
            )
        ],
        "decision_frame": reflect_frame,
        "frame_history": [plan_frame, reflect_frame],
    }
    values.update(updates)
    return AgentState(**values)


@pytest.mark.asyncio
async def test_summary_replaces_previous_value_and_is_injected_once(monkeypatch):
    state = make_reflect_state(attempt_outcome_summary="old failed approach")
    main_usage = state.token_usage

    async def fake_llm_call(*args, **kwargs):
        assert kwargs["provider"] == "primary"
        assert kwargs["temperature"] == 0.0
        assert kwargs["model"] == "gemini-3.5-flash:stable"
        payload = json.loads(args[1])
        assert payload["previous_summary"] == "old failed approach"
        assert set(payload) == {
            "previous_summary",
            "plan_signature",
            "patch_outcome",
            "test_failure_class",
            "reflection_action",
        }
        return {"summary": "new factual outcome"}

    monkeypatch.setattr("src.outcome_summary.llm_call", fake_llm_call)

    result = await summarize_attempt_outcome(state)
    state.attempt_outcome_summary = result
    prompt = build_plan_user_prompt(state)

    assert result == "new factual outcome"
    assert prompt.count("Completed attempts (rolling summary):") == 1
    assert "old failed approach" not in prompt
    assert state.token_usage == main_usage
    assert state.summary_token_usage > 0
    assert state.model_history[-1].provider == "primary"
    assert state.model_history[-1].node == "outcome_summary"
    assert state.model_history[-1].status == "ok"


@pytest.mark.asyncio
async def test_invalid_summary_uses_bounded_deterministic_fallback(monkeypatch):
    state = make_reflect_state()

    async def fake_llm_call(*args, **kwargs):
        return {"summary": "x" * 500}

    monkeypatch.setattr("src.outcome_summary.llm_call", fake_llm_call)

    summary = await summarize_attempt_outcome(state)

    assert summary == deterministic_outcome_summary(build_outcome_summary_input(state))
    assert 0 < len(summary) <= MAX_OUTCOME_SUMMARY_CHARS
    assert "Bearer" not in summary
    assert state.model_history[-1].status == "invalid_response"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [{}, {"summary": ""}, {"summary": 123}, "not-an-object"],
)
async def test_empty_or_malformed_summary_uses_deterministic_fallback(
    monkeypatch, response
):
    state = make_reflect_state()

    async def fake_llm_call(*args, **kwargs):
        return response

    monkeypatch.setattr("src.outcome_summary.llm_call", fake_llm_call)

    summary = await summarize_attempt_outcome(state)

    assert summary == deterministic_outcome_summary(build_outcome_summary_input(state))
    assert state.model_history[-1].status == "invalid_response"


@pytest.mark.asyncio
async def test_summary_call_failure_is_nonfatal_and_recorded_separately(monkeypatch):
    state = make_reflect_state(token_usage=987)

    async def fake_llm_call(*args, **kwargs):
        raise TimeoutError("provider timeout Bearer sk-summarytestsentinel")

    monkeypatch.setattr("src.outcome_summary.llm_call", fake_llm_call)

    summary = await summarize_attempt_outcome(state)

    assert summary == deterministic_outcome_summary(build_outcome_summary_input(state))
    assert state.token_usage == 987
    assert state.summary_token_usage > 0
    invocation = state.model_history[-1]
    assert invocation.status == "error"
    assert invocation.error_class == "TimeoutError"


@pytest.mark.asyncio
async def test_summary_input_and_output_exclude_denied_content(monkeypatch):
    generated_path = "tests/generated_secret_regression.py"
    old_marker = "old-attempt-must-not-appear"
    state = make_reflect_state(
        attempt_outcome_summary=(
            "previous safe result; Authorization: Bearer sk-summarytestsentinel"
        ),
        fix_attempts=[
            FixAttempt(error_log=old_marker, failure_kind="old_failure"),
            FixAttempt(
                test_result="assertion failed HTTP/1.1 raw response",
                failure_kind="FAIL_TO_PASS evaluator field",
                error_log=f"failure in {generated_path}",
            ),
        ],
        generated_test_approvals=[
            GeneratedTestApproval(
                path=generated_path,
                content_sha256="a" * 64,
                patch_gate_fingerprint="b" * 64,
            )
        ],
    )

    async def fake_llm_call(system, user, **kwargs):
        assert old_marker not in user
        assert "sk-summarytestsentinel" not in user
        assert "FAIL_TO_PASS" not in user
        assert "HTTP/1.1" not in user
        assert generated_path not in user
        return {
            "summary": (
                "safe prefix Authorization: Bearer sk-summarytestsentinel "
                "FAIL_TO_PASS HTTP/1.1 " + generated_path
            )
        }

    monkeypatch.setattr("src.outcome_summary.llm_call", fake_llm_call)

    summary = await summarize_attempt_outcome(state)

    assert summary == "safe prefix Authorization: [REDACTED]"
    assert "FAIL_TO_PASS" not in summary
    assert "HTTP/1.1" not in summary
    assert generated_path not in summary
    assert "sk-summarytestsentinel" not in summary


def test_plan_prompt_has_no_summary_section_when_summary_is_empty():
    prompt = build_plan_user_prompt(make_reflect_state(attempt_outcome_summary=""))

    assert "Completed attempts (rolling summary):" not in prompt


def test_nonempty_summary_is_the_only_completed_attempt_context_in_plan():
    sentinels = {
        "attempt-result-sentinel",
        "attempt-error-sentinel",
        "failed-search-sentinel",
        "failed-replace-sentinel",
        "reflection-notes-sentinel",
        "plan-frame-summary-sentinel",
        "plan-hypothesis-sentinel",
        "reflect-frame-summary-sentinel",
        "search-correction-sentinel",
        "semantic-recall-sentinel",
    }
    plan_frame = DecisionFrame(
        frame_id="df_plan",
        stage="plan",
        summary="plan-frame-summary-sentinel",
        hypotheses=[
            Hypothesis(
                id="H1",
                claim="plan-hypothesis-sentinel",
                evidence=["plan-hypothesis-evidence-sentinel"],
            )
        ],
        selected_hypothesis_id="H1",
        recommended_action="execute",
    )
    reflect_frame = DecisionFrame(
        frame_id="df_reflect",
        parent_frame_id="df_plan",
        stage="reflect",
        summary="reflect-frame-summary-sentinel",
        recommended_action="plan",
    )
    state = make_reflect_state(
        attempt_outcome_summary="one safe completed-attempt summary",
        reflection_notes="reflection-notes-sentinel",
        search_correction_context="search-correction-sentinel",
        fix_attempts=[
            FixAttempt(
                test_result="attempt-result-sentinel",
                failure_kind="patch_apply_failed",
                error_log="attempt-error-sentinel",
                patch_edits=[
                    PatchEdit(
                        file_path="src/widget.py",
                        search="failed-search-sentinel",
                        replace="failed-replace-sentinel",
                    )
                ],
            )
        ],
        decision_frame=reflect_frame,
        frame_history=[plan_frame, reflect_frame],
    )

    prompt = build_plan_user_prompt(
        state,
        recall_context="\n\nsemantic-recall-sentinel",
    )

    assert prompt.count("Completed attempts (rolling summary):") == 1
    assert prompt.count("one safe completed-attempt summary") == 1
    assert all(sentinel not in prompt for sentinel in sentinels)
    assert "plan-hypothesis-evidence-sentinel" not in prompt


def test_agent_state_sanitizes_summary_on_construction_assignment_and_dump():
    generated_path = "tests/regression_issue_123.py"
    state = make_reflect_state(
        generated_test_approvals=[
            GeneratedTestApproval(
                path=generated_path,
                content_sha256="a" * 64,
                patch_gate_fingerprint="b" * 64,
            )
        ],
        attempt_outcome_summary="x" * 500,
    )
    assert len(state.attempt_outcome_summary) == MAX_OUTCOME_SUMMARY_CHARS

    state.attempt_outcome_summary = (
        "safe assigned prefix Authorization: Bearer sk-assignment-sentinel "
        "FAIL_TO_PASS HTTP/1.1 " + generated_path
    )
    assert "sk-assignment-sentinel" not in state.attempt_outcome_summary
    assert "FAIL_TO_PASS" not in state.attempt_outcome_summary
    assert "HTTP/1.1" not in state.attempt_outcome_summary
    assert generated_path not in state.attempt_outcome_summary

    state.attempt_outcome_summary = f"safe dynamic path prefix {generated_path} tail"
    assert state.attempt_outcome_summary == "safe dynamic path prefix"

    object.__setattr__(
        state,
        "attempt_outcome_summary",
        "safe dumped prefix HTTP response payload: dump-http-sentinel",
    )
    dumped = state.model_dump(mode="json")["attempt_outcome_summary"]
    assert dumped == "safe dumped prefix"
    assert "dump-http-sentinel" not in dumped


@pytest.mark.parametrize(
    "value",
    [
        "safe prefix patch=private evaluator edit",
        "safe prefix patch: private evaluator edit",
        'safe prefix {"patch": "private evaluator edit"}',
    ],
)
def test_summary_sanitizer_stops_at_patch_field_boundaries(value):
    assert sanitize_summary_text(value) == "safe prefix"

    state = make_reflect_state(attempt_outcome_summary=value)
    prompt = build_plan_user_prompt(state)

    assert state.attempt_outcome_summary == "safe prefix"
    assert "private evaluator edit" not in prompt


def test_summary_sanitizer_preserves_patch_language_but_stops_at_raw_http():
    natural_language = "patch applied cleanly; tests still fail"

    assert sanitize_summary_text(natural_language) == natural_language
    assert (
        sanitize_summary_text(
            "safe prefix PATCH https://api.invalid/private HTTP body"
        )
        == "safe prefix"
    )


def test_deterministic_summary_uses_safe_edit_result_field():
    summary = deterministic_outcome_summary(
        OutcomeSummaryInput(
            plan_signature="a" * 64,
            patch_outcome="patch applied",
            test_failure_class="assertion_failure",
            reflection_action="plan",
        )
    )

    assert "edit_result=patch applied" in summary
    assert "patch=" not in summary


def test_plan_signature_binds_full_frame_and_ordered_edit_transaction():
    first = make_reflect_state()
    first.frame_history[0].risk = "low"
    first.fix_attempts[-1].patch_content = "raw-patch-sentinel"
    first.fix_attempts[-1].patch_edits = [
        PatchEdit(
            file_path="src/a.py",
            search="old-a",
            replace="new-a",
            exact_only=True,
            expected_content_sha256="a" * 64,
        ),
        PatchEdit(file_path="src/b.py", search="old-b", replace="new-b"),
    ]
    same = first.model_copy(deep=True)
    changed_frame = first.model_copy(deep=True)
    changed_frame.frame_history[0].risk = "high"
    changed_edit = first.model_copy(deep=True)
    changed_edit.fix_attempts[-1].patch_edits[0].replace = "different-new-a"
    changed_order = first.model_copy(deep=True)
    changed_order.fix_attempts[-1].patch_edits.reverse()
    changed_preimage = first.model_copy(deep=True)
    changed_preimage.fix_attempts[-1].patch_edits[0].expected_content_sha256 = "b" * 64

    first_input = build_outcome_summary_input(first)
    encoded = first_input.model_dump_json()

    assert first_input.plan_signature == build_outcome_summary_input(same).plan_signature
    assert first_input.plan_signature != build_outcome_summary_input(changed_frame).plan_signature
    assert first_input.plan_signature != build_outcome_summary_input(changed_edit).plan_signature
    assert first_input.plan_signature != build_outcome_summary_input(changed_order).plan_signature
    assert first_input.plan_signature != build_outcome_summary_input(changed_preimage).plan_signature
    assert "raw-patch-sentinel" not in encoded
    assert "old-a" not in encoded
    assert "new-a" not in encoded


def test_build_input_uses_only_latest_attempt_and_current_history_frames():
    state = make_reflect_state(
        issue_title="issue-title-must-not-be-in-summary-input",
        issue_body="issue-body-must-not-be-in-summary-input",
        conversation_history=[
            {"role": "user", "content": "conversation-must-not-be-in-input"}
        ],
        fix_attempts=[
            FixAttempt(error_log="old-attempt-must-not-be-in-input"),
            FixAttempt(test_result="failed", failure_kind="assertion_failure"),
        ],
    )

    item = build_outcome_summary_input(state)
    encoded = item.model_dump_json()

    assert "old-attempt-must-not-be-in-input" not in encoded
    assert "issue-title-must-not-be-in-summary-input" not in encoded
    assert "issue-body-must-not-be-in-summary-input" not in encoded
    assert "conversation-must-not-be-in-input" not in encoded
    assert item.reflection_action.startswith("plan")
    assert item.plan_signature


def test_escalation_packet_does_not_include_primary_plan_summary_section():
    state = make_reflect_state(
        active_model="claude-opus-4-8:stable",
        active_provider="escalation",
        escalated=True,
        escalation_reason="repeated_no_progress",
        attempt_outcome_summary="private rolling attempt outcome",
    )

    packet = render_escalation_packet(build_escalation_packet(state))

    assert "Completed attempts (rolling summary):" not in packet
    assert "private rolling attempt outcome" not in packet
