"""Balance-aware context tightening: the PLAN file window shrinks as the token
budget depletes so late retries can still afford to emit a patch."""

from src.nodes import plan as plan_node
from src.nodes.plan import PLAN_FILE_CONTENT_LIMIT, PLAN_MAX_FILES
from src.state import AgentState


def _state(usage, budget=100000):
    return AgentState(
        issue_url="https://github.com/a/b/issues/1",
        token_usage=usage,
        token_budget=budget,
    )


def test_full_context_when_budget_fresh():
    limit, files = plan_node._budget_scaled_file_limits(_state(0))
    assert limit == PLAN_FILE_CONTENT_LIMIT
    assert files == PLAN_MAX_FILES


def test_full_context_above_half():
    limit, files = plan_node._budget_scaled_file_limits(_state(40000))
    assert limit == PLAN_FILE_CONTENT_LIMIT
    assert files == PLAN_MAX_FILES


def test_shrinks_window_below_half():
    limit, files = plan_node._budget_scaled_file_limits(_state(65000))
    assert limit == PLAN_FILE_CONTENT_LIMIT * 2 // 3
    assert files == PLAN_MAX_FILES


def test_shrinks_more_below_quarter():
    limit, files = plan_node._budget_scaled_file_limits(_state(90000))
    assert limit == PLAN_FILE_CONTENT_LIMIT // 2
    assert files == max(1, PLAN_MAX_FILES - 1)


def test_never_below_half_and_one_file():
    limit, files = plan_node._budget_scaled_file_limits(_state(100000))
    assert limit >= PLAN_FILE_CONTENT_LIMIT // 2
    assert files >= 1


def test_zero_budget_is_safe():
    limit, files = plan_node._budget_scaled_file_limits(_state(0, budget=0))
    assert limit == PLAN_FILE_CONTENT_LIMIT
    assert files == PLAN_MAX_FILES
