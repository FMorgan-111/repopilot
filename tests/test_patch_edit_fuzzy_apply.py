"""Normalized (whitespace-tolerant) fallback for search/replace patch edits.

The exact-match path is unchanged; these tests pin the fuzzy fallback that
recovers from the dominant Gemini failure mode — indentation drift between the
model's multi-line search block and the real file. (Single-line leading/trailing
drift still contains the search as a substring, so it stays on the exact path.)
"""

from src.nodes.execute import _apply_patch_edits
from src.state import PatchEdit


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_exact_match_still_applies(tmp_path):
    _write(tmp_path, "a.py", "def f():\n    return 1\n")
    result = _apply_patch_edits(
        str(tmp_path),
        [PatchEdit(file_path="a.py", search="    return 1", replace="    return 2")],
    )
    assert result.applied
    assert (tmp_path / "a.py").read_text() == "def f():\n    return 2\n"


def test_indent_drift_recovers_and_reindents(tmp_path):
    # Real method body is indented 4/8; model wrote the block at 0/4.
    _write(tmp_path, "a.py", "class C:\n    def f(self):\n        return 1\n")
    result = _apply_patch_edits(
        str(tmp_path),
        [
            PatchEdit(
                file_path="a.py",
                search="def f(self):\n    return 1",
                replace="def f(self):\n    return 2",
            )
        ],
    )
    assert result.applied, result.output
    # Replacement is reindented to the file's actual 4/8 level.
    assert (tmp_path / "a.py").read_text() == (
        "class C:\n    def f(self):\n        return 2\n"
    )


def test_ambiguous_normalized_match_is_rejected(tmp_path):
    # Two indent-drifted sites both normalize to the search block: must NOT
    # fuzzy-replace either one (still reported as not found).
    _write(
        tmp_path,
        "a.py",
        "class A:\n    def f():\n        return 1\n"
        "class B:\n    def f():\n        return 1\n",
    )
    result = _apply_patch_edits(
        str(tmp_path),
        [PatchEdit(file_path="a.py", search="def f():\n    return 1", replace="def f():\n    return 2")],
    )
    assert not result.applied
    assert "was not found" in result.output


def test_genuinely_absent_block_still_fails(tmp_path):
    _write(tmp_path, "a.py", "def f():\n    return 1\n")
    result = _apply_patch_edits(
        str(tmp_path),
        [PatchEdit(file_path="a.py", search="    return 999", replace="    return 2")],
    )
    assert not result.applied
    assert "was not found" in result.output


def test_replace_all_does_not_use_fuzzy_fallback(tmp_path):
    # A block that WOULD fuzzy-match, but replace_all must stay exact-only.
    _write(tmp_path, "a.py", "class C:\n    def f(self):\n        return 1\n")
    result = _apply_patch_edits(
        str(tmp_path),
        [
            PatchEdit(
                file_path="a.py",
                search="def f(self):\n    return 1",
                replace="def f(self):\n    return 2",
                replace_all=True,
            )
        ],
    )
    assert not result.applied
    assert "was not found" in result.output
