"""Pure locators shared by EXECUTE (fuzzy apply) and PLAN (hallucination gate)."""

from src.patch_match import (
    closest_region,
    find_normalized_span,
    locate_search_block,
    reindent,
)


def test_locate_exact():
    assert locate_search_block("def f():\n    return 1\n", "    return 1") is True


def test_locate_normalized_indent_drift():
    content = "class C:\n    def f(self):\n        return 1\n"
    assert locate_search_block(content, "def f(self):\n    return 1") is True


def test_locate_absent_block():
    assert locate_search_block("def f():\n    return 1\n", "    return 999") is False


def test_locate_empty_search():
    assert locate_search_block("anything", "") is False


def test_locate_ambiguous_normalized_is_not_located():
    content = (
        "class A:\n    def f():\n        return 1\n"
        "class B:\n    def f():\n        return 1\n"
    )
    assert locate_search_block(content, "def f():\n    return 1") is False


def test_reindent_positive_and_negative():
    assert reindent("a\n    b", 2) == "  a\n      b"
    assert reindent("    a\n        b", -4) == "a\n    b"
    # Negative delta refused when a line lacks the indentation to remove.
    assert reindent("  a\nb", -4) == "  a\nb"


def test_find_normalized_span_unique():
    content = "class C:\n    def f(self):\n        return 1\n"
    span = find_normalized_span(content, "def f(self):\n    return 1")
    assert span is not None
    start, end, delta = span
    assert delta == 4  # method body sits 4 spaces deeper than the search block


def test_closest_region_returns_real_nearby_lines():
    content = (
        "\n".join(f"pad{i}" for i in range(50))
        + "\ndef compute(value):\n    return value * 2\n"
        + "\n".join(f"tail{i}" for i in range(50))
    )
    # Model hallucinated a paraphrase; closest_region surfaces the real lines.
    region = closest_region(content, "def compute(val):\n    return val*2")
    assert "def compute(value):" in region
    assert "return value * 2" in region


def test_closest_region_empty_inputs():
    assert closest_region("", "x") == ""
    assert closest_region("a\nb", "   \n  ") == ""
