"""Relevance-centered file windowing for the PLAN prompt.

Head-truncation hid fix sites below a file's imports and drove the planner to
hallucinate search blocks. `_relevance_window` centers the same char budget on
the issue-relevant lines instead.
"""

from src.nodes.plan import _relevance_window


def test_short_content_returned_whole():
    content = "line1\nline2\n"
    assert _relevance_window(content, ["anything"], 6000) == content


def test_centers_on_matching_line_below_head():
    # Fix site ("def target_fn") is far below the head; a small budget centered
    # on the term must include it, unlike head-truncation.
    head = "\n".join(f"import mod{i}" for i in range(200))
    body = "def target_fn():\n    return compute_bug()"
    content = head + "\n" + body
    window = _relevance_window(content, ["target_fn"], 400)
    assert "def target_fn():" in window
    assert "return compute_bug()" in window
    assert "truncated" in window  # window markers present


def test_no_term_match_falls_back_to_head():
    content = "\n".join(f"line{i}" for i in range(500))
    window = _relevance_window(content, ["nonexistentxyz"], 200)
    assert window.startswith("line0")
    assert window.endswith("...")


def test_window_respects_char_budget_roughly():
    content = "\n".join(f"line{i} filler filler filler" for i in range(500))
    limit = 300
    window = _relevance_window(content, ["line250"], limit)
    assert "line250 filler filler filler" in window
    # Body (excluding marker annotations) stays near the budget, not the whole file.
    assert len(window) < limit + 120


def test_picks_densest_term_line():
    content = (
        "\n".join(f"pad{i}" for i in range(100))
        + "\nfoo bar\n"
        + "\n".join(f"pad{i}" for i in range(100, 200))
        + "\nfoo bar baz qux\n"
        + "\n".join(f"pad{i}" for i in range(200, 300))
    )
    window = _relevance_window(content, ["foo", "bar", "baz", "qux"], 300)
    assert "foo bar baz qux" in window  # the line matching more terms wins
