"""AST-anchored (node_target) patch edits: locate a def/class by dotted name and
replace its whole span, with no verbatim text anchoring."""

import pytest

from src.nodes.execute import _apply_patch_edits
from src.patch_match import locate_node_span
from src.state import PatchEdit

SRC = '''\
import os


def top():
    return 1


class C:
    def m(self, x):
        return x + 1

    @property
    def p(self):
        return self._p
'''


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


# ---- locate_node_span -------------------------------------------------------

def test_locate_top_level_function():
    span = locate_node_span(SRC, "top")
    assert span is not None
    start, end, indent = span
    assert SRC[start:end] == "def top():\n    return 1\n"
    assert indent == 0


def test_locate_method_qualname():
    span = locate_node_span(SRC, "C.m")
    start, end, indent = span
    assert SRC[start:end] == "    def m(self, x):\n        return x + 1\n"
    assert indent == 4


def test_locate_includes_decorators():
    span = locate_node_span(SRC, "C.p")
    start, end, _ = span
    assert SRC[start:end].startswith("    @property\n")
    assert "return self._p" in SRC[start:end]


def test_locate_class():
    span = locate_node_span(SRC, "C")
    start, end, _ = span
    assert SRC[start:end].startswith("class C:")


def test_locate_unknown_returns_none():
    assert locate_node_span(SRC, "nope") is None


def test_locate_syntax_error_returns_none():
    assert locate_node_span("def (:\n", "x") is None


def test_locate_ambiguous_returns_none():
    dup = "def f():\n    return 1\n\n\ndef f():\n    return 2\n"
    assert locate_node_span(dup, "f") is None


# ---- execute node-target apply ---------------------------------------------

def test_node_target_replaces_whole_function(tmp_path):
    _write(tmp_path, "a.py", SRC)
    result = _apply_patch_edits(
        str(tmp_path),
        [PatchEdit(file_path="a.py", node_target="top",
                   replace="def top():\n    return 42")],
    )
    assert result.applied, result.output
    text = (tmp_path / "a.py").read_text()
    assert "return 42" in text
    assert "return 1" not in text
    # Rest of the file preserved (comments/format untouched).
    assert "class C:" in text and "@property" in text


def test_node_target_reindents_method_replacement(tmp_path):
    _write(tmp_path, "a.py", SRC)
    # Model supplies the method at column 0; executor reindents to 4.
    result = _apply_patch_edits(
        str(tmp_path),
        [PatchEdit(file_path="a.py", node_target="C.m",
                   replace="def m(self, x):\n    return x + 100")],
    )
    assert result.applied, result.output
    text = (tmp_path / "a.py").read_text()
    assert "    def m(self, x):\n        return x + 100" in text


def test_node_target_missing_node_fails(tmp_path):
    _write(tmp_path, "a.py", SRC)
    result = _apply_patch_edits(
        str(tmp_path),
        [PatchEdit(file_path="a.py", node_target="C.nonexistent", replace="x=1")],
    )
    assert not result.applied
    assert "could not locate" in result.output


def test_patch_edit_requires_an_anchor():
    with pytest.raises(ValueError):
        PatchEdit(file_path="a.py")  # neither search nor node_target


def test_patch_edit_both_anchors_prefers_search():
    # The model sometimes supplies both; tolerate it (prefer search) rather than
    # crashing the whole plan phase with a validation error.
    edit = PatchEdit(file_path="a.py", search="x", node_target="f")
    assert edit.search == "x"
    assert edit.node_target == ""


def test_normalize_plan_decision_drops_malformed_edit():
    from src.nodes.plan import _normalize_plan_decision

    resp = {
        "plan": "p",
        "patch": "",
        "patch_edits": [
            {"file": "a.py", "search": "x", "replace": "y"},  # valid
            {"file": "b.py"},  # malformed: no anchor → dropped, not a crash
        ],
        "files": ["a.py"],
        "test_command": "pytest",
        "decision_frame": {
            "stage": "plan", "summary": "p",
            "recommended_action": "execute", "risk": "low", "confidence": 0.7,
        },
    }
    decision = _normalize_plan_decision(resp)
    assert len(decision.patch_edits) == 1
    assert decision.patch_edits[0].file_path == "a.py"

