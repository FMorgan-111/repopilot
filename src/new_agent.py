"""RepoPilot v2 agent: graph-based issue fixing loop.

This module is a thin re-export wrapper. The implementation lives in:
  src/state.py        — models, enums, and helper functions
  src/nodes/          — individual phase implementations
  src/graph.py        — graph runner, router, and fallback classes
"""

from __future__ import annotations

import asyncio
from typing import Any

from .state import (
    AgentState,
    ConversationTurn,
    FileInfo,
    FinalReport,
    FixAttempt,
    NodeFn,
    Phase,
    ToolCall,
    _as_state,
    _estimate_tokens,
    _extract_json_object,
    _is_budget_exceeded,
    _issue_search_terms,
    _primary_patch_file,
    _rank_reason,
    _record_tool,
    _remember,
)
from .nodes.understand import understand_issue
from .nodes.locate import locate_code
from .nodes.plan import plan_fix
from .nodes.execute import execute_fix, git_clone, apply_patch, run_pytest
from .nodes.verify import verify_fix
from .nodes.reflect import reflect_on_failure
from .nodes.commit import commit_fix, push_files, create_pr
from .nodes.failure import handle_failure
from .graph import (
    END,
    FallbackCompiledGraph,
    FallbackStateGraph,
    route_from_state,
    run_graph,
    StateGraph,
)
from .tracer import Tracer


def build_agent_graph() -> Any:
    """Build the RepoPilot v2 state graph.

    Defined here (not in graph.py) so that monkeypatching the node-function
    attributes on this module (as tests do) flows through to the graph.
    """
    if StateGraph is None:
        graph = FallbackStateGraph()
        for name, fn in {
            "understand_issue": understand_issue,
            "locate_code": locate_code,
            "plan_fix": plan_fix,
            "reflect_on_failure": reflect_on_failure,
            "execute_fix": execute_fix,
            "verify_fix": verify_fix,
            "commit_fix": commit_fix,
            "handle_failure": handle_failure,
        }.items():
            graph.add_node(name, fn)
        graph.set_entry_point("understand_issue")
        return graph.compile()

    graph = StateGraph(AgentState)
    graph.add_node("understand_issue", understand_issue)
    graph.add_node("locate_code", locate_code)
    graph.add_node("plan_fix", plan_fix)
    graph.add_node("execute_fix", execute_fix)
    graph.add_node("verify_fix", verify_fix)
    graph.add_node("reflect_on_failure", reflect_on_failure)
    graph.add_node("commit_fix", commit_fix)
    graph.add_node("handle_failure", handle_failure)
    for node in [
        "understand_issue",
        "locate_code",
        "plan_fix",
        "reflect_on_failure",
        "execute_fix",
        "verify_fix",
        "commit_fix",
        "handle_failure",
    ]:
        graph.add_conditional_edges(
            node,
            route_from_state,
            {
                "understand_issue": "understand_issue",
                "locate_code": "locate_code",
                "plan_fix": "plan_fix",
                "reflect_on_failure": "reflect_on_failure",
                "execute_fix": "execute_fix",
                "verify_fix": "verify_fix",
                "commit_fix": "commit_fix",
                "handle_failure": "handle_failure",
                END: END,
            },
        )
    graph.set_entry_point("understand_issue")
    return graph.compile()


def final_report_from_state(state: AgentState, turns_taken: int) -> FinalReport:
    return FinalReport(
        issue_url=state.issue_url,
        fix_applied=state.current_phase == Phase.DONE,
        pr_url=state.pr_url,
        test_results=state.fix_attempts[-1].test_result if state.fix_attempts else "",
        turns_taken=turns_taken,
        token_used=state.token_usage,
    )


async def agent_v2(issue_url: str, max_retries: int = 3, token_budget: int = 50000) -> dict:
    tracer = Tracer()
    state = AgentState(
        issue_url=issue_url,
        max_retries=max_retries,
        token_budget=token_budget,
        trace_id=tracer.trace_id,
    )
    graph = build_agent_graph()
    final_state = await run_graph(graph, state)
    report = final_report_from_state(final_state, len(final_state.tool_calls))
    tracer.log(
        "agent_v2_done",
        {"issue_url": issue_url},
        {"phase": final_state.current_phase.value, "pr_url": final_state.pr_url},
        error=final_state.failure_reason or None,
    )
    payload = report.model_dump()
    payload.update(
        {
            "done": final_state.current_phase in {Phase.DONE, Phase.FAILED},
            "success": final_state.current_phase == Phase.DONE,
            "final_phase": final_state.current_phase.value,
            "trace_id": tracer.trace_id,
            "relevant_files": [file.model_dump() for file in final_state.relevant_files],
            "fix_attempts": [attempt.model_dump() for attempt in final_state.fix_attempts],
            "error": final_state.failure_reason or None,
        }
    )
    return payload


async def intelligent_analyze_issue(
    issue_url: str, max_retries: int = 3, token_budget: int = 50000
) -> dict:
    """Backward-compatible alias for the previous experimental endpoint."""
    return await agent_v2(issue_url, max_retries=max_retries, token_budget=token_budget)


if __name__ == "__main__":  # pragma: no cover
    print(
        asyncio.run(
            agent_v2(
                "https://github.com/example/repo/issues/1",
                max_retries=1,
                token_budget=10000,
            )
        )
    )
