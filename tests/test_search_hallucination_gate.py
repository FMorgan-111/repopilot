"""PLAN integration coverage for exact PatchGate search-anchor rejection."""

from src.nodes import plan as plan_node
from src.state import Phase


def _plan_response(search: str) -> dict:
    return {
        "kind": "plan",
        "plan": "Update the widget sentinel.",
        "patch": "",
        "patch_edits": [
            {
                "file": "src/widget.py",
                "search": search,
                "replace": "return 'new-sentinel'",
            }
        ],
        "files": ["src/widget.py"],
        "test_command": "pytest -q",
        "decision_frame": {
            "stage": "plan",
            "summary": "Update the sentinel.",
            "recommended_action": "execute",
            "risk": "low",
            "confidence": 0.8,
        },
    }


async def test_patchgate_search_missing_routes_to_new_full_plan_with_real_window(
    exact_repair_state, monkeypatch
):
    async def fake_llm(_system, _user, model=None, *, provider="primary", **_kwargs):
        return _plan_response("def widget(value):\n    return old")

    monkeypatch.setattr(plan_node, "llm_call", fake_llm)
    result = await plan_node.plan_fix(exact_repair_state)

    assert result.current_phase == Phase.PLAN
    assert result.decision_frame.recommended_action == "plan"
    assert result.retry_count == 1
    assert result.primary_failed_repair_rounds == 1
    assert result.last_counted_repair_round_id == 1
    assert result.tool_patch_approval is None
    assert result.patch_edits == []
    assert "search_missing" in result.repair_correction_context
    assert "old-sentinel" in result.repair_correction_context
    assert len(result.repair_correction_context) <= 8_000


async def test_patchgate_search_missing_obeys_terminal_global_budget(
    exact_repair_state, monkeypatch
):
    exact_repair_state.max_retries = 0

    async def fake_llm(_system, _user, model=None, *, provider="primary", **_kwargs):
        return _plan_response("missing final anchor")

    monkeypatch.setattr(plan_node, "llm_call", fake_llm)
    result = await plan_node.plan_fix(exact_repair_state)

    assert result.current_phase == Phase.FAILURE
    assert result.decision_frame.recommended_action == "stop"
    assert result.retry_count == 0
    assert result.primary_failed_repair_rounds == 1
    assert result.last_counted_repair_round_id == 1
    assert result.repair_round_sequence == 1
    assert result.tool_patch_approval is None


async def test_exact_search_acceptance_clears_obsolete_correction_context(
    exact_repair_state, monkeypatch
):
    exact_repair_state.repair_correction_context = "stale-anchor-correction"

    async def fake_llm(_system, _user, model=None, *, provider="primary", **_kwargs):
        return _plan_response("return 'old-sentinel'")

    monkeypatch.setattr(plan_node, "llm_call", fake_llm)
    result = await plan_node.plan_fix(exact_repair_state)

    assert result.current_phase == Phase.EXECUTE
    assert result.repair_correction_context == ""
    assert result.tool_patch_approval is not None
    assert result.patch_edits[0].exact_only is True
