import hashlib
import subprocess
from pathlib import Path

import pytest

from src.patch_gate import revalidate_approved_patch, validate_patch_batch
from src.state import AgentState, RepairPlan, VerifiedEdit, VerifiedEditBatch


def _repo(tmp_path: Path, files: dict[str, str]) -> tuple[Path, str]:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "--all"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "base"], check=True)
    ref = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return root, ref


def _plan(*paths: str, symbols: list[str] | None = None) -> RepairPlan:
    return RepairPlan(
        root_cause="A bounded defect exists.",
        target_files=list(paths),
        target_symbols=symbols or [],
        required_behavior="Correct the behavior.",
        regression_test_strategy="Run the focused test.",
    )


def _state(root: Path, ref: str, plan: RepairPlan) -> AgentState:
    return AgentState(
        issue_url="https://github.com/a/b/issues/1",
        repo_path=str(root),
        repo_ref=ref,
        active_provider="escalation",
        active_model="claude-opus-4-8:stable",
        escalated=True,
        active_repair_plan=plan,
    )


def _edit(path: str, *, search: str = "", replace: str, node: str | None = None, before: bytes = b"") -> VerifiedEdit:
    edit = VerifiedEdit(
        file_path=path,
        node_target=node,
        search=search,
        replace=replace,
        intent="Make the bounded change.",
    )
    edit._expected_content_sha256 = hashlib.sha256(before).hexdigest()
    edit._exact_only = True
    return edit


def test_gate_accepts_exact_search_without_mutating_and_builds_approval(tmp_path):
    source = b"def value():\n    return 1\n"
    root, ref = _repo(tmp_path, {"src/a.py": source.decode()})
    plan = _plan("src/a.py")
    state = _state(root, ref, plan)
    batch = VerifiedEditBatch(
        edits=[_edit("src/a.py", search="return 1", replace="return 2", before=source)]
    )

    result = validate_patch_batch(state, plan, batch)

    assert result.accepted
    assert (root / "src/a.py").read_bytes() == source
    assert result.edits[0].exact_only
    assert state.tool_patch_approval is not None
    assert state.tool_patch_approval.base_ref == ref
    assert state.tool_patch_approval.changed_manifest[0].path == "src/a.py"
    revalidate_approved_patch(state)


def test_gate_accepts_unique_node_and_intentional_new_text_file_atomically(tmp_path):
    source = b"def value():\n    return 1\n"
    root, ref = _repo(tmp_path, {"src/a.py": source.decode()})
    plan = _plan("src/a.py", "tests/test_new.py", symbols=["value"])
    state = _state(root, ref, plan)
    batch = VerifiedEditBatch(
        edits=[
            _edit("src/a.py", node="value", replace="def value():\n    return 2\n", before=source),
            _edit("tests/test_new.py", replace="def test_value():\n    assert True\n"),
        ]
    )

    result = validate_patch_batch(state, plan, batch)

    assert result.accepted
    assert [item.file_path for item in result.edits] == ["src/a.py", "tests/test_new.py"]
    assert not (root / "tests/test_new.py").exists()
    assert [item.path for item in state.tool_patch_approval.changed_manifest] == ["src/a.py", "tests/test_new.py"]


def test_gate_simulates_multiple_distinct_edits_to_one_file_atomically(tmp_path):
    source = b"first = 1\nsecond = 1\n"
    root, ref = _repo(tmp_path, {"a.py": source.decode()})
    plan = _plan("a.py")
    state = _state(root, ref, plan)
    batch = VerifiedEditBatch(
        edits=[
            _edit("a.py", search="first = 1", replace="first = 2", before=source),
            _edit("a.py", search="second = 1", replace="second = 2", before=source),
        ]
    )

    result = validate_patch_batch(state, plan, batch)

    assert result.accepted
    assert len(result.edits) == 2
    assert state.tool_patch_approval.changed_manifest[0].size == len("first = 2\nsecond = 2\n")
    assert (root / "a.py").read_bytes() == source


def test_gate_distinguishes_ambiguous_node_target(tmp_path):
    source = b"def value():\n    return 1\n\ndef value():\n    return 2\n"
    root, ref = _repo(tmp_path, {"a.py": source.decode()})
    plan = _plan("a.py", symbols=["value"])
    state = _state(root, ref, plan)
    batch = VerifiedEditBatch(
        edits=[_edit("a.py", node="value", replace="def value():\n    return 3\n", before=source)]
    )

    result = validate_patch_batch(state, plan, batch)

    assert result.issues[0].code == "target_ambiguous"


@pytest.mark.parametrize(
    ("path", "code"),
    [
        ("../escape.py", "scope_violation"),
        ("dist/package.whl", "generated_artifact"),
        ("src/payload.bin", "binary_artifact"),
        (".env", "scope_violation"),
    ],
)
def test_gate_rejects_unsafe_new_targets(tmp_path, path, code):
    root, ref = _repo(tmp_path, {"src/a.py": "x = 1\n"})
    # RepairPlan canonical validation catches traversal, so use a valid plan and
    # construct the adversarial edit without trusting normal schema parsing.
    plan = _plan("src/new.py")
    state = _state(root, ref, plan)
    edit = _edit("src/new.py", replace="x = 2\n")
    object.__setattr__(edit, "file_path", path)

    result = validate_patch_batch(state, plan, VerifiedEditBatch(edits=[edit]))

    assert not result.accepted
    assert result.issues[0].code == code


def test_gate_rejects_missing_ambiguous_noop_and_keeps_batch_atomic(tmp_path):
    source = b"x = 1\nx = 1\n"
    root, ref = _repo(tmp_path, {"src/a.py": source.decode()})
    plan = _plan("src/a.py")
    state = _state(root, ref, plan)

    ambiguous = validate_patch_batch(
        state,
        plan,
        VerifiedEditBatch(edits=[_edit("src/a.py", search="x = 1", replace="x = 2", before=source)]),
    )
    missing = validate_patch_batch(
        state,
        plan,
        VerifiedEditBatch(edits=[_edit("src/a.py", search="missing", replace="x = 2", before=source)]),
    )
    noop = validate_patch_batch(
        state,
        plan,
        VerifiedEditBatch(edits=[_edit("src/a.py", search=source.decode(), replace=source.decode(), before=source)]),
    )

    assert ambiguous.issues[0].code == "target_ambiguous"
    assert missing.issues[0].code == "search_missing"
    assert noop.issues[0].code == "empty_patch"
    assert (root / "src/a.py").read_bytes() == source


def test_gate_rejects_symlink_and_late_preimage_drift(tmp_path):
    source = b"x = 1\n"
    root, ref = _repo(tmp_path, {"src/a.py": source.decode()})
    plan = _plan("src/a.py")
    state = _state(root, ref, plan)
    batch = VerifiedEditBatch(edits=[_edit("src/a.py", search="x = 1", replace="x = 2", before=source)])
    assert validate_patch_batch(state, plan, batch).accepted

    (root / "src/a.py").write_text("x = 9\n", encoding="utf-8")
    with pytest.raises(ValueError, match="preimage|approval"):
        revalidate_approved_patch(state)


def test_gate_revalidation_rejects_post_approval_edit_tampering(tmp_path):
    source = b"value = 1\n"
    root, ref = _repo(tmp_path, {"a.py": source.decode()})
    plan = _plan("a.py")
    state = _state(root, ref, plan)
    batch = VerifiedEditBatch(
        edits=[_edit("a.py", search="value = 1", replace="value = 2", before=source)]
    )
    assert validate_patch_batch(state, plan, batch).accepted
    state.patch_edits[0].replace = "value = 999"

    with pytest.raises(ValueError, match="approval|manifest|patch"):
        revalidate_approved_patch(state)


def test_gate_generated_patch_applies_to_exact_base(tmp_path):
    source = b"value = 1\n"
    root, ref = _repo(tmp_path, {"a.py": source.decode()})
    plan = _plan("a.py", "tests/test_value.py")
    state = _state(root, ref, plan)
    batch = VerifiedEditBatch(
        edits=[
            _edit("a.py", search="value = 1", replace="value = 2", before=source),
            _edit("tests/test_value.py", replace="def test_value():\n    assert True\n"),
        ]
    )
    assert validate_patch_batch(state, plan, batch).accepted

    checked = subprocess.run(
        ["git", "-C", str(root), "apply", "--check", "-"],
        input=state.patch_content,
        capture_output=True,
        text=True,
    )

    assert checked.returncode == 0, checked.stderr
