"""Deterministic search->node_target converter: rescues a whole-function rewrite
whose search text drifted too far to match, WITHOUT ever truncating a partial
edit (the size gate is the safety core)."""

from src.nodes.execute import _apply_patch_edits
from src.patch_match import try_upgrade_to_node_target
from src.state import PatchEdit

# A function big enough that a stub replace is clearly a partial edit.
BIG = (
    "class Svc:\n"
    "    def modify_user(self, name):\n"
    "        user = lookup(name)\n"
    "        if user is None:\n"
    "            raise KeyError(name)\n"
    "        user.touch()\n"
    "        audit(user)\n"
    "        return user\n"
)


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


# ---- converter gate (pure) --------------------------------------------------

def test_upgrade_when_whole_function_rewrite():
    # search is the whole function (but with drifted text so it won't match),
    # replace is a comparably-sized whole function → upgrade.
    search = (
        "def modify_user(self, name):\n"
        "    user = lookup(name)\n"
        "    if user is None:\n"
        "        raise KeyError(name)\n"
        "    user.touch()\n"
        "    audit(user)\n"
        "    return user"
    )
    replace = (
        "def modify_user(self, name):\n"
        "    user = lookup(name)\n"
        "    if user is None:\n"
        "        raise KeyError(name)\n"
        "    user.touch()\n"
        "    audit(user)\n"
        "    log(user)\n"
        "    return user"
    )
    assert try_upgrade_to_node_target(BIG, search, replace) == "Svc.modify_user"


def test_no_upgrade_for_partial_edit_stub():
    # THE SAFETY CASE: search/replace are the def line + one line only. Upgrading
    # would replace the whole 8-line function with a 2-line stub → must refuse.
    search = "def modify_user(self, name):\n    user = lookup(name)"
    replace = "def modify_user(self, name):\n    user = lookup(name.lower())"
    assert try_upgrade_to_node_target(BIG, search, replace) is None


def test_no_upgrade_when_replace_not_a_def():
    search = "def modify_user(self, name):\n    return 1"
    replace = "    user = lookup(name)\n    return user"  # not a def
    assert try_upgrade_to_node_target(BIG, search, replace) is None


def test_no_upgrade_when_name_mismatch():
    search = "def other(self):\n    return 1"
    replace = "def other(self):\n    return 2"
    assert try_upgrade_to_node_target(BIG, search, replace) is None


def test_no_upgrade_when_ambiguous():
    dup = (
        "def f(x):\n    return x\n\n\n"
        "class A:\n    def f(x):\n        return x\n"
    )
    search = "def f(x):\n    return x"
    replace = "def f(x):\n    return x + 1"
    # 'f' appears twice → ambiguous → refuse.
    assert try_upgrade_to_node_target(dup, search, replace) is None


# ---- end-to-end apply -------------------------------------------------------

def test_apply_rescues_drifted_whole_function(tmp_path):
    _write(tmp_path, "s.py", BIG)
    # Search has wrong indentation AND a drifted line, so exact and normalized
    # both miss; replace is a full function → converter rescues via node_target.
    search = (
        "def modify_user(self, name):\n"
        "  user = find(name)\n"              # 'find' not 'lookup' — content drift
        "  if user is None:\n"
        "    raise KeyError(name)\n"
        "  user.touch()\n"
        "  audit(user)\n"
        "  return user"
    )
    replace = (
        "def modify_user(self, name):\n"
        "    user = lookup(name)\n"
        "    if user is None:\n"
        "        raise KeyError(name)\n"
        "    user.touch()\n"
        "    audit(user)\n"
        "    log(user)\n"
        "    return user"
    )
    result = _apply_patch_edits(
        str(tmp_path),
        [PatchEdit(file_path="s.py", search=search, replace=replace)],
    )
    assert result.applied, result.output
    text = (tmp_path / "s.py").read_text()
    assert "log(user)" in text                     # new line applied
    assert text.count("def modify_user") == 1      # not duplicated
    assert "class Svc:" in text                    # class header preserved
    # Method reindented under the class (8-space body).
    assert "\n        user = lookup(name)" in text


def test_apply_still_fails_for_partial_drift(tmp_path):
    # A partial edit whose search drifted and can't fuzzy-match must still fail
    # (NOT be silently upgraded and truncate the function).
    _write(tmp_path, "s.py", BIG)
    search = "def modify_user(self, name):\n  user = find(name)"  # stub + drift
    replace = "def modify_user(self, name):\n    user = lookup(name.lower())"
    result = _apply_patch_edits(
        str(tmp_path),
        [PatchEdit(file_path="s.py", search=search, replace=replace)],
    )
    assert not result.applied
    assert "was not found" in result.output
    # File untouched — no truncation.
    assert (tmp_path / "s.py").read_text() == BIG
