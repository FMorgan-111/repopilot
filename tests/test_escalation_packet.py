import json

import pytest
from pydantic import ValidationError

from src.escalation import (
    ESCALATION_PACKET_RENDER_LIMIT,
    EscalationPacket,
    build_escalation_packet,
    immediate_model_policy_reason,
    render_escalation_packet,
)
from src.http_client import LLMResponseError
from src.nodes import plan as plan_node
from src.nodes import reflect as reflect_node
from src.state import (
    AgentState,
    ConversationTurn,
    DecisionFrame,
    Evidence,
    FixAttempt,
    GeneratedTestApproval,
    Hypothesis,
    Phase,
    ToolCall,
    ToolInvocation,
)


def _valid_plan_response() -> dict:
    return {
        "plan": "Guard the missing user before submit.",
        "patch": "",
        "patch_edits": [
            {
                "file": "src/auth.py",
                "search": "if user is None:\n    submit(user)\n",
                "replace": "if user is None:\n    return\n",
            }
        ],
        "files": ["src/auth.py"],
        "test_command": "pytest tests/test_auth.py -q",
        "decision_frame": {
            "stage": "plan",
            "summary": "Guard missing users.",
            "recommended_action": "execute",
            "risk": "low",
            "confidence": 0.9,
        },
    }


def _valid_reflect_response() -> dict:
    return {
        "root_cause": "The previous edit changed the wrong branch.",
        "what_went_wrong": "The None case still reaches submit.",
        "suggested_fix_approach": "Guard None before submit.",
        "files_that_also_need_changes": ["src/auth.py"],
        "decision_frame": {
            "stage": "reflect",
            "summary": "Patch the None branch.",
            "recommended_action": "plan",
            "risk": "low",
            "confidence": 0.9,
        },
    }


def _escalated_state() -> AgentState:
    return AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        issue_title="Safe login crash",
        issue_body="Preserve a safe explanation.\nFAIL_TO_PASS: evaluator-case-sentinel",
        owner="acme",
        repo="widget",
        repo_ref="Ab" * 20,
        current_phase=Phase.PLAN,
        active_provider="escalation",
        active_model="claude-opus-4-8:stable",
        escalated=True,
        token_budget=100_000,
        token_usage=61_000,
        max_retries=4,
        retry_count=2,
    )


def test_empty_completion_after_transport_retries_is_an_immediate_policy_event():
    reason = immediate_model_policy_reason(
        LLMResponseError("empty chat completion response")
    )

    assert reason == "empty_completion_after_retries"


def test_escalation_packet_is_an_allowlist_boundary_with_independent_bounds():
    secret = "sk-secret-packet-sentinel"
    state = _escalated_state()
    state.issue_body = (
        "Keep this required behavior.\n"
        f"Authorization: Bearer {secret}\n"
        "gold_patch: evaluator-gold-sentinel\n"
        + "body-overflow-sentinel" * 1_000
    )
    state.evidence = [
        Evidence(
            evidence_id="ev_safe",
            tool="read_symbol",
            file_path="src/auth.py",
            symbol="submit",
            summary=f"Safe source window Authorization: Bearer {secret}",
            content="def submit(user):\n    return user\n" + "x" * 20_000,
            fingerprint="a" * 64,
        ),
        Evidence(
            evidence_id="ev_generated",
            tool="read_range",
            file_path="tests/test_generated_secret_name.py",
            summary="generated-filename-summary-sentinel",
            content="generated-filename-content-sentinel",
            fingerprint="b" * 64,
        ),
    ]
    state.generated_test_approvals = [
        GeneratedTestApproval(
            path="tests/test_generated_secret_name.py",
            content_sha256="c" * 64,
            patch_gate_fingerprint="d" * 64,
        )
    ]
    state.fix_attempts = [
        FixAttempt(
            patch_content="gold-patch-payload-sentinel",
            test_result="failed",
            failure_kind="test_failed",
            error_log=(
                "AssertionError: safe failure\n"
                f"api_key={secret}\n"
                "test_patch: evaluator-test-patch-sentinel\n"
                + "test-overflow-sentinel" * 1_000
            ),
        ),
        FixAttempt(
            patch_content="unrelated-patch-sentinel",
            test_result="patch_apply_failed",
            failure_kind="patch_apply_failed",
            error_log=(
                "Search block missing safely\n"
                "HTTP response payload: raw-http-error-sentinel\n"
                "tests/test_generated_secret_name.py\n"
                + "p" * 10_000
            ),
        ),
    ]
    state.frame_history = [
        DecisionFrame(
            stage="plan",
            summary="A failed safe approach",
            hypotheses=[
                Hypothesis(
                    id="H1",
                    claim="Patch the session cache instead",
                    why_not_selected="Evidence points to submit",
                )
            ],
            recommended_action="reflect",
        )
    ]
    state.conversation_history = [
        ConversationTurn(role="user", content="unrelated-conversation-sentinel")
    ]
    state.tool_calls = [
        ToolCall(
            tool_name="raw_http",
            args={"PASS_TO_PASS": "evaluator-pass-sentinel"},
            result="raw-http-payload-sentinel",
        )
    ]
    state.tool_history = [
        ToolInvocation(
            action="search_text",
            args_fingerprint="tool-history-sentinel",
            status="ok",
        )
    ]

    packet = build_escalation_packet(state)
    dumped = packet.model_dump_json()
    rendered = render_escalation_packet(packet)
    combined = dumped + rendered

    assert set(packet.model_dump()) == set(EscalationPacket.model_fields)
    assert packet.issue_title == "Safe login crash"
    assert "Keep this required behavior" in packet.issue_body
    assert packet.repository == "acme/widget"
    assert packet.base_commit == "Ab" * 20
    assert packet.evidence[0].evidence_id == "ev_safe"
    assert packet.evidence[0].content.startswith("def submit")
    assert packet.patch_errors[0].startswith("Search block missing safely")
    assert packet.test_error_summaries[0].startswith("AssertionError: safe failure")
    assert "A failed safe approach" in packet.rejected_approaches
    assert packet.remaining_token_budget == 39_000
    assert packet.remaining_execution_attempts == 2
    assert len(packet.issue_body) <= 4_000
    assert len(packet.evidence[0].content) <= 6_000
    assert all(len(item) <= 1_000 for item in packet.patch_errors)
    assert all(len(item) <= 1_000 for item in packet.test_error_summaries)
    assert json.loads(rendered)["base_commit"] == "Ab" * 20

    for forbidden in (
        secret,
        "evaluator-gold-sentinel",
        "evaluator-test-patch-sentinel",
        "gold-patch-payload-sentinel",
        "unrelated-patch-sentinel",
        "tests/test_generated_secret_name.py",
        "generated-filename-summary-sentinel",
        "generated-filename-content-sentinel",
        "unrelated-conversation-sentinel",
        "evaluator-pass-sentinel",
        "raw-http-payload-sentinel",
        "raw-http-error-sentinel",
        "tool-history-sentinel",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "gold_patch",
        "test_patch",
    ):
        assert forbidden not in combined


def test_raw_http_marker_is_a_fail_closed_boundary_in_every_context_field():
    state = _escalated_state()
    state.issue_title = (
        "Safe title prefix HTTP response payload: title-http-sentinel\n"
        "title-continuation-sentinel"
    )
    state.issue_body = (
        "Safe body prefix\ninline marker: raw HTTP request body: body-http-sentinel\n"
        "body-continuation-sentinel"
    )
    state.evidence = [
        Evidence(
            evidence_id="ev_safe",
            tool="read_symbol",
            file_path="src/auth.py",
            symbol="submit",
            summary=(
                "Safe evidence summary HTTP/1.1 503 evidence-summary-sentinel\n"
                "evidence-summary-continuation"
            ),
            content=(
                "def submit(user):\n"
                "    return user\n"
                "trace prefix GET https://provider.invalid/private\n"
                "evidence-content-continuation"
            ),
            fingerprint="a" * 64,
        )
    ]
    state.fix_attempts = [
        FixAttempt(
            test_result="patch_apply_failed",
            failure_kind="patch_apply_failed",
            error_log=(
                "Safe patch error POST https://provider.invalid/private\n"
                "patch-error-continuation"
            ),
        ),
        FixAttempt(
            test_result="failed",
            failure_kind="test_failed",
            error_log=(
                "Safe test error response headers: test-http-sentinel\n"
                "test-error-continuation"
            ),
        ),
    ]
    state.frame_history = [
        DecisionFrame(
            stage="plan",
            summary=(
                "Safe rejected approach HTTP request payload: rejected-http-sentinel\n"
                "rejected-continuation"
            ),
            recommended_action="reflect",
        )
    ]

    packet = build_escalation_packet(state)
    combined = packet.model_dump_json() + render_escalation_packet(packet)

    assert packet.issue_title == "Safe title prefix"
    assert packet.issue_body == "Safe body prefix\ninline marker:"
    assert packet.evidence[0].summary == "Safe evidence summary"
    assert packet.evidence[0].content.endswith("trace prefix")
    assert packet.patch_errors == ("Safe patch error",)
    assert packet.test_error_summaries == ("Safe test error",)
    assert packet.rejected_approaches == ("Safe rejected approach",)
    for forbidden in (
        "title-http-sentinel",
        "title-continuation-sentinel",
        "body-http-sentinel",
        "body-continuation-sentinel",
        "evidence-summary-sentinel",
        "evidence-summary-continuation",
        "provider.invalid",
        "evidence-content-continuation",
        "patch-error-continuation",
        "test-http-sentinel",
        "test-error-continuation",
        "rejected-http-sentinel",
        "rejected-continuation",
    ):
        assert forbidden not in combined


def test_standalone_raw_http_marker_truncates_evidence_and_test_error():
    state = _escalated_state()
    state.evidence = [
        Evidence(
            evidence_id="ev_raw_http_boundary",
            tool="read_symbol",
            file_path="src/auth.py",
            symbol="submit",
            summary="Safe summary inline Raw-HTTP: evidence-http-sentinel",
            content=(
                "def submit(user):\n"
                "    return user\n"
                "debug prefix RAW_HTTP: evidence-content-sentinel\n"
                "evidence-continuation-sentinel"
            ),
            fingerprint="a" * 64,
        )
    ]
    state.fix_attempts = [
        FixAttempt(
            test_result="failed",
            failure_kind="test_failed",
            error_log=(
                "AssertionError: safe failure inline raw HTTP: test-http-sentinel\n"
                "test-continuation-sentinel"
            ),
        )
    ]

    packet = build_escalation_packet(state)
    rendered = render_escalation_packet(packet)
    combined = packet.model_dump_json() + rendered

    assert packet.evidence[0].summary == "Safe summary inline"
    assert packet.evidence[0].content.endswith("debug prefix")
    assert packet.test_error_summaries == (
        "AssertionError: safe failure inline",
    )
    for forbidden in (
        "evidence-http-sentinel",
        "evidence-content-sentinel",
        "evidence-continuation-sentinel",
        "test-http-sentinel",
        "test-continuation-sentinel",
    ):
        assert forbidden not in combined


def test_direct_packet_construction_bounds_fields_and_is_deeply_immutable():
    packet = EscalationPacket(
        issue_title="t" * 5_000,
        issue_body="b" * 50_000,
        repository="r" * 5_000,
        base_commit="c" * 5_000,
        evidence=[
            Evidence(
                evidence_id=f"evidence-{index}-" + "i" * 500,
                tool="tool" * 100,
                file_path="src/" + "p" * 2_000,
                symbol="symbol" * 500,
                summary="s" * 10_000,
                content="x" * 50_000,
                fingerprint="f" * 500,
            )
            for index in range(30)
        ],
        failed_edit_signatures=["f" * 5_000] * 30,
        patch_errors=["p" * 5_000] * 30,
        test_error_summaries=["e" * 5_000] * 30,
        rejected_approaches=["r" * 5_000] * 30,
        required_behavior="required" * 5_000,
        remaining_token_budget=100_000,
        remaining_execution_attempts=3,
    )

    assert len(packet.issue_title) <= 500
    assert len(packet.issue_body) <= 4_000
    assert len(packet.repository) <= 300
    assert len(packet.base_commit) <= 500
    assert len(packet.required_behavior) <= 2_500
    assert len(packet.evidence) <= 12
    assert sum(
        len(json.dumps(item.model_dump(), ensure_ascii=False))
        for item in packet.evidence
    ) <= 24_000
    assert all(len(item.content) <= 6_000 for item in packet.evidence)
    assert all(len(item.summary) <= 500 for item in packet.evidence)
    for values in (
        packet.failed_edit_signatures,
        packet.patch_errors,
        packet.test_error_summaries,
        packet.rejected_approaches,
    ):
        assert len(values) <= 8
        assert all(len(item) <= 1_000 for item in values)

    with pytest.raises(ValidationError):
        packet.issue_body = "mutated"
    with pytest.raises(ValidationError):
        packet.evidence[0].content = "mutated"
    with pytest.raises(AttributeError):
        packet.patch_errors.append("mutated")


def test_render_reconstructs_and_rebounds_a_bypassed_mutation():
    packet = EscalationPacket(
        issue_title="safe",
        issue_body="safe",
        repository="acme/widget",
        base_commit="a" * 40,
        evidence=[],
        failed_edit_signatures=[],
        patch_errors=[],
        test_error_summaries=[],
        rejected_approaches=[],
        required_behavior="safe",
        remaining_token_budget=1,
        remaining_execution_attempts=1,
    )
    object.__setattr__(
        packet,
        "issue_body",
        "safe prefix HTTP response payload: mutated-http-sentinel\n"
        "mutated-continuation-sentinel" + "x" * 100_000,
    )
    object.__setattr__(packet, "patch_errors", ["p" * 100_000] * 100)
    object.__setattr__(
        packet,
        "evidence",
        [
            Evidence(
                evidence_id="e" * 10_000,
                tool="t" * 10_000,
                summary="s" * 100_000,
                content="c" * 100_000,
                fingerprint="f" * 10_000,
            )
        ]
        * 100,
    )

    rendered = render_escalation_packet(packet)
    payload = json.loads(rendered)

    assert payload["issue_body"] == "safe prefix"
    assert len(payload["patch_errors"]) <= 8
    assert all(len(item) <= 1_000 for item in payload["patch_errors"])
    assert len(payload["evidence"]) <= 12
    assert all(len(item["content"]) <= 6_000 for item in payload["evidence"])
    assert len(rendered) <= ESCALATION_PACKET_RENDER_LIMIT
    assert "mutated-http-sentinel" not in rendered
    assert "mutated-continuation-sentinel" not in rendered


async def test_primary_plan_keeps_current_bounded_prompt_and_default_call(monkeypatch):
    captured = {}

    async def fake_llm_call(system, user):
        captured.update(system=system, user=user)
        return _valid_plan_response()

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        issue_title="Login crash",
        issue_body="x" * 8_000,
        current_phase=Phase.PLAN,
    )

    result = await plan_node.plan_fix(state)

    assert result.current_phase == Phase.EXECUTE
    assert "Issue URL:" in captured["user"]
    assert "Relevant files:" in captured["user"]
    assert "x" * (plan_node.PLAN_ISSUE_BODY_LIMIT + 1) not in captured["user"]
    assert result.model_history[-1].provider == "primary"
    assert result.model_history[-1].model == "gemini-3.5-flash:stable"
    assert result.model_history[-1].status == "ok"


async def test_escalated_plan_uses_only_packet_and_active_escalation_model(monkeypatch):
    calls = []

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
        return _valid_plan_response()

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    state = _escalated_state()
    state.relevant_files = []
    state.conversation_history = [
        ConversationTurn(role="user", content="unrelated-conversation-sentinel")
    ]
    state.tool_calls = [
        ToolCall(tool_name="raw", result="raw-http-payload-sentinel")
    ]
    expected_packet = render_escalation_packet(build_escalation_packet(state))

    result = await plan_node.plan_fix(state)

    assert result.current_phase == Phase.EXECUTE
    assert calls[0]["provider"] == "escalation"
    assert calls[0]["model"] == "claude-opus-4-8:stable"
    assert calls[0]["user"] == expected_packet
    assert "decision_frame" in calls[0]["system"]
    assert "patch_edits" in calls[0]["system"]
    assert "tool_intent" in calls[0]["system"]
    assert "Issue URL:" not in calls[0]["user"]
    assert "Relevant files:" not in calls[0]["user"]
    assert "unrelated-conversation-sentinel" not in calls[0]["user"]
    assert "raw-http-payload-sentinel" not in calls[0]["user"]
    assert result.model_history[-1].provider == "escalation"
    assert result.model_history[-1].model == "claude-opus-4-8:stable"
    assert result.model_history[-1].node == "plan_fix"


async def test_escalated_reflect_uses_packet_and_active_escalation_model(monkeypatch):
    calls = []

    async def fake_llm_call(system, user, model=None, *, provider="primary", temperature=0.2):
        calls.append((system, user, model, provider))
        return _valid_reflect_response()

    monkeypatch.setattr(reflect_node, "llm_call", fake_llm_call)
    state = _escalated_state()
    state.current_phase = Phase.REFLECT
    state.fix_attempts = [
        FixAttempt(
            patch_content="secret-full-patch-sentinel",
            test_result="failed",
            error_log="AssertionError: public failure",
        )
    ]
    expected_packet = render_escalation_packet(build_escalation_packet(state))

    result = await reflect_node.reflect_on_failure(state)

    system, user, model, provider = calls[0]
    assert result.current_phase == Phase.PLAN
    assert provider == "escalation"
    assert model == "claude-opus-4-8:stable"
    assert user == expected_packet
    assert "decision_frame" in system
    assert "tool_intent" in system
    assert "Patch Applied:" not in user
    assert "secret-full-patch-sentinel" not in user
    assert result.model_history[-1].provider == "escalation"
    assert result.model_history[-1].node == "reflect_on_failure"
