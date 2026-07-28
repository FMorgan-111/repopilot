import hashlib
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.patch_gate import (
    apply_approved_patch,
    revalidate_approved_patch,
    validate_patch_batch,
)
from src.repair_rounds import begin_repair_round
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
    state = AgentState(
        issue_url="https://github.com/a/b/issues/1",
        repo_path=str(root),
        repo_ref=ref,
        active_provider="escalation",
        active_model="claude-opus-4-8:stable",
        escalated=True,
        active_repair_plan=plan,
    )
    begin_repair_round(state)
    return state


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
    assert state.authorized_repair_round_id == state.current_repair_round_id
    assert state.authorized_repair_provider == "escalation"
    assert state.authorized_repair_model == "claude-opus-4-8:stable"
    revalidate_approved_patch(state)


def test_gate_approval_survives_resolved_search_target_metadata(tmp_path):
    source = b"class Nominal:\n    def _setup(self):\n        return 1\n"
    root, ref = _repo(tmp_path, {"src/a.py": source.decode()})
    plan = _plan("src/a.py")
    state = _state(root, ref, plan)
    batch = VerifiedEditBatch(
        edits=[_edit("src/a.py", search="return 1", replace="return 2", before=source)]
    )
    assert validate_patch_batch(state, plan, batch).accepted

    state.patch_edits[0].resolved_target_symbol = "Nominal._setup"

    revalidate_approved_patch(state)
    apply_approved_patch(state)
    assert (root / "src/a.py").read_text(encoding="utf-8") == source.decode().replace(
        "return 1", "return 2"
    )


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
    assert result.issues[0].failure_class == "model_correctable"


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
    state.active_repair_plan = plan
    missing = validate_patch_batch(
        state,
        plan,
        VerifiedEditBatch(edits=[_edit("src/a.py", search="missing", replace="x = 2", before=source)]),
    )
    state.active_repair_plan = plan
    noop = validate_patch_batch(
        state,
        plan,
        VerifiedEditBatch(edits=[_edit("src/a.py", search=source.decode(), replace=source.decode(), before=source)]),
    )

    assert ambiguous.issues[0].code == "target_ambiguous"
    assert ambiguous.issues[0].failure_class == "model_correctable"
    assert missing.issues[0].code == "search_missing"
    assert missing.issues[0].failure_class == "model_correctable"
    assert noop.issues[0].code == "empty_patch"
    assert noop.issues[0].failure_class == "model_correctable"
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


@pytest.mark.parametrize("newline", [b"\r\n", b"\r"])
def test_gate_preserves_existing_raw_newline_style_and_applies(tmp_path, newline):
    root, _ = _repo(tmp_path, {"a.py": "value = 1\nother = 2\n"})
    source = newline.join([b"value = 1", b"other = 2", b""])
    (root / "a.py").write_bytes(source)
    subprocess.run(["git", "-C", str(root), "add", "a.py"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "--amend", "-qm", "raw-newlines"], check=True)
    ref = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    plan = _plan("a.py")
    state = _state(root, ref, plan)
    batch = VerifiedEditBatch(
        edits=[_edit("a.py", search="value = 1", replace="value = 3", before=source)]
    )

    assert validate_patch_batch(state, plan, batch).accepted
    apply_approved_patch(state)

    assert (root / "a.py").read_bytes() == newline.join(
        [b"value = 3", b"other = 2", b""]
    )


def test_gate_rejects_when_generated_patch_fails_real_git_apply_check(tmp_path, monkeypatch):
    source = b"value = 1\n"
    root, ref = _repo(tmp_path, {"a.py": source.decode()})
    plan = _plan("a.py")
    state = _state(root, ref, plan)
    actual_run = subprocess.run

    def fail_only_apply_check(command, **kwargs):
        if command[-3:] == ["apply", "--check", "-"]:
            return subprocess.CompletedProcess(command, 1, stdout=b"", stderr=b"rejected")
        return actual_run(command, **kwargs)

    monkeypatch.setattr("src.patch_gate.subprocess.run", fail_only_apply_check)
    result = validate_patch_batch(
        state,
        plan,
        VerifiedEditBatch(
            edits=[_edit("a.py", search="value = 1", replace="value = 2", before=source)]
        ),
    )

    assert not result.accepted
    assert result.issues[0].code == "apply_failed"
    assert result.issues[0].failure_class == "model_correctable"
    assert state.tool_patch_approval is None


def test_exact_checkout_failure_is_environment(tmp_path):
    source = b"value = 1\n"
    root, _ref = _repo(tmp_path, {"a.py": source.decode()})
    plan = _plan("a.py")
    state = _state(root, "0" * 40, plan)

    result = validate_patch_batch(
        state,
        plan,
        VerifiedEditBatch(
            edits=[_edit("a.py", search="value = 1", replace="value = 2", before=source)]
        ),
    )

    assert not result.accepted
    assert result.issues[0].code == "apply_failed"
    assert result.issues[0].failure_class == "environment"


def test_active_plan_invariant_failure_is_environment(tmp_path):
    source = b"value = 1\n"
    root, ref = _repo(tmp_path, {"a.py": source.decode()})
    active = _plan("a.py")
    submitted = active.model_copy(update={"root_cause": "A different bounded defect."})
    state = _state(root, ref, active)

    result = validate_patch_batch(
        state,
        submitted,
        VerifiedEditBatch(
            edits=[_edit("a.py", search="value = 1", replace="value = 2", before=source)]
        ),
    )

    assert not result.accepted
    assert result.issues[0].failure_class == "environment"


def test_live_preimage_drift_is_environment(tmp_path):
    source = b"value = 1\n"
    root, ref = _repo(tmp_path, {"a.py": source.decode()})
    plan = _plan("a.py")
    state = _state(root, ref, plan)
    (root / "a.py").write_text("value = 9\n", encoding="utf-8")

    result = validate_patch_batch(
        state,
        plan,
        VerifiedEditBatch(
            edits=[_edit("a.py", search="value = 1", replace="value = 2", before=source)]
        ),
    )

    assert not result.accepted
    assert result.issues[0].code == "apply_failed"
    assert result.issues[0].failure_class == "environment"


def test_filesystem_failure_is_environment(tmp_path, monkeypatch):
    source = b"value = 1\n"
    root, ref = _repo(tmp_path, {"a.py": source.decode()})
    plan = _plan("a.py")
    state = _state(root, ref, plan)

    def fail_read(*_args, **_kwargs):
        raise OSError("injected read failure")

    monkeypatch.setattr("src.patch_gate._read_regular_no_follow", fail_read)
    result = validate_patch_batch(
        state,
        plan,
        VerifiedEditBatch(
            edits=[_edit("a.py", search="value = 1", replace="value = 2", before=source)]
        ),
    )

    assert not result.accepted
    assert result.issues[0].failure_class == "environment"


@pytest.mark.parametrize("content", [b"\xff", b"value = 1\0\n"])
def test_tracked_non_text_content_is_model_correctable(tmp_path, content):
    root, _ref = _repo(tmp_path, {"a.py": "value = 1\n"})
    (root / "a.py").write_bytes(content)
    subprocess.run(["git", "-C", str(root), "add", "a.py"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "commit", "--amend", "-qm", "binary-base"],
        check=True,
    )
    ref = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    plan = _plan("a.py")
    state = _state(root, ref, plan)

    result = validate_patch_batch(
        state,
        plan,
        VerifiedEditBatch(
            edits=[_edit("a.py", search="value = 1", replace="value = 2", before=content)]
        ),
    )

    assert not result.accepted
    assert result.issues[0].code == "binary_artifact"
    assert result.issues[0].failure_class == "model_correctable"


def test_absent_tracked_path_is_model_correctable(tmp_path):
    root, ref = _repo(tmp_path, {"a.py": "value = 1\n"})
    plan = _plan("src/missing.py")
    state = _state(root, ref, plan)

    result = validate_patch_batch(
        state,
        plan,
        VerifiedEditBatch(
            edits=[
                _edit(
                    "src/missing.py",
                    search="missing anchor",
                    replace="replacement",
                )
            ]
        ),
    )

    assert not result.accepted
    assert result.issues[0].code == "target_missing"
    assert result.issues[0].failure_class == "model_correctable"


def test_untracked_existing_regular_file_is_environment_drift(tmp_path):
    root, ref = _repo(tmp_path, {"a.py": "value = 1\n"})
    (root / "src").mkdir()
    (root / "src" / "new.py").write_text("untracked = True\n", encoding="utf-8")
    plan = _plan("src/new.py")
    state = _state(root, ref, plan)

    result = validate_patch_batch(
        state,
        plan,
        VerifiedEditBatch(
            edits=[_edit("src/new.py", replace="created = True\n")]
        ),
    )

    assert not result.accepted
    assert result.issues[0].code == "apply_failed"
    assert result.issues[0].failure_class == "environment"
    assert state.patch_content == ""
    assert state.patch_edits == []
    assert state.active_repair_plan is None
    assert state.tool_patch_approval is None
    assert state.authorized_repair_round_id == 0


@pytest.mark.parametrize("failing_git_command", ["ls-tree", "show"])
def test_git_tree_or_blob_failure_is_environment(
    tmp_path, monkeypatch, failing_git_command
):
    source = b"value = 1\n"
    root, ref = _repo(tmp_path, {"a.py": source.decode()})
    plan = _plan("a.py")
    state = _state(root, ref, plan)
    actual_git = __import__("src.patch_gate", fromlist=["_git"])._git

    def fail_selected_git(repo, *args):
        if args and args[0] == failing_git_command:
            return subprocess.CompletedProcess(args, 128, stdout=b"", stderr=b"injected")
        return actual_git(repo, *args)

    monkeypatch.setattr("src.patch_gate._git", fail_selected_git)
    result = validate_patch_batch(
        state,
        plan,
        VerifiedEditBatch(
            edits=[_edit("a.py", search="value = 1", replace="value = 2", before=source)]
        ),
    )

    assert not result.accepted
    assert result.issues[0].code == "apply_failed"
    assert result.issues[0].failure_class == "environment"


def test_git_tree_decode_failure_is_environment(tmp_path, monkeypatch):
    source = b"value = 1\n"
    root, ref = _repo(tmp_path, {"a.py": source.decode()})
    plan = _plan("a.py")
    state = _state(root, ref, plan)
    actual_git = __import__("src.patch_gate", fromlist=["_git"])._git

    def undecodable_mode(repo, *args):
        if args and args[0] == "ls-tree":
            record = b"\xff blob " + b"0" * 40 + b"\ta.py\0"
            return subprocess.CompletedProcess(args, 0, stdout=record, stderr=b"")
        return actual_git(repo, *args)

    monkeypatch.setattr("src.patch_gate._git", undecodable_mode)
    result = validate_patch_batch(
        state,
        plan,
        VerifiedEditBatch(
            edits=[_edit("a.py", search="value = 1", replace="value = 2", before=source)]
        ),
    )

    assert not result.accepted
    assert result.issues[0].code == "apply_failed"
    assert result.issues[0].failure_class == "environment"


def test_untracked_target_filesystem_failure_is_environment(tmp_path, monkeypatch):
    root, ref = _repo(
        tmp_path,
        {
            "a.py": "value = 1\n",
            "tests/test_existing.py": "def test_existing():\n    assert True\n",
        },
    )
    plan = _plan("tests/test_new.py")
    state = _state(root, ref, plan)
    actual_lstat = Path.lstat

    def fail_target_lstat(path, *args, **kwargs):
        if path.name == "test_new.py":
            raise OSError("injected lstat failure")
        return actual_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", fail_target_lstat)
    result = validate_patch_batch(
        state,
        plan,
        VerifiedEditBatch(
            edits=[_edit("tests/test_new.py", replace="def test_new():\n    assert True\n")]
        ),
    )

    assert not result.accepted
    assert result.issues[0].code == "apply_failed"
    assert result.issues[0].failure_class == "environment"


def test_production_acceptance_without_runtime_binding_fails_closed(tmp_path):
    source = b"value = 1\n"
    root, ref = _repo(tmp_path, {"a.py": source.decode()})
    plan = _plan("a.py")
    state = AgentState(
        issue_url="https://github.com/a/b/issues/1",
        repo_path=str(root),
        repo_ref=ref,
        active_provider="primary",
        active_model="gemini-3.5-flash:stable",
        active_repair_plan=plan,
    )

    result = validate_patch_batch(
        state,
        plan,
        VerifiedEditBatch(
            edits=[_edit("a.py", search="value = 1", replace="value = 2", before=source)]
        ),
    )

    assert not result.accepted
    assert result.issues[0].failure_class == "environment"
    assert state.patch_content == ""
    assert state.patch_edits == []
    assert state.active_repair_plan is None
    assert state.tool_patch_approval is None
    assert state.authorized_repair_round_id == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("authorized_repair_round_id", 2),
        ("authorized_repair_provider", "primary"),
        ("authorized_repair_model", "tampered-model"),
    ],
)
def test_revalidation_rejects_authorized_attribution_tampering(
    tmp_path, field, value
):
    source = b"value = 1\n"
    root, ref = _repo(tmp_path, {"a.py": source.decode()})
    plan = _plan("a.py")
    state = _state(root, ref, plan)
    assert validate_patch_batch(
        state,
        plan,
        VerifiedEditBatch(
            edits=[_edit("a.py", search="value = 1", replace="value = 2", before=source)]
        ),
    ).accepted

    object.__setattr__(state, field, value)

    with pytest.raises(ValueError, match="PatchGate|authorization|fingerprint|binding"):
        revalidate_approved_patch(state)


def test_test_only_success_never_replaces_production_attribution(tmp_path):
    source = b"value = 1\n"
    root, ref = _repo(
        tmp_path,
        {
            "a.py": source.decode(),
            "tests/test_existing.py": "def test_existing():\n    assert True\n",
        },
    )
    production_plan = _plan("a.py")
    state = _state(root, ref, production_plan)
    assert validate_patch_batch(
        state,
        production_plan,
        VerifiedEditBatch(
            edits=[_edit("a.py", search="value = 1", replace="value = 2", before=source)]
        ),
    ).accepted
    frozen = (
        state.authorized_repair_round_id,
        state.authorized_repair_provider,
        state.authorized_repair_model,
    )
    object.__setattr__(state, "current_repair_provider", "primary")
    object.__setattr__(state, "current_repair_model", "different-runtime-model")
    test_plan = _plan("tests/test_generated.py")
    state.active_repair_plan = test_plan

    result = validate_patch_batch(
        state,
        test_plan,
        VerifiedEditBatch(
            edits=[_edit("tests/test_generated.py", replace="def test_it():\n    assert True\n")]
        ),
        test_only=True,
    )

    assert result.accepted
    assert (
        state.authorized_repair_round_id,
        state.authorized_repair_provider,
        state.authorized_repair_model,
    ) == frozen


def test_test_only_rejection_never_clears_production_attribution(tmp_path):
    source = b"value = 1\n"
    root, ref = _repo(tmp_path, {"a.py": source.decode()})
    production_plan = _plan("a.py")
    state = _state(root, ref, production_plan)
    assert validate_patch_batch(
        state,
        production_plan,
        VerifiedEditBatch(
            edits=[_edit("a.py", search="value = 1", replace="value = 2", before=source)]
        ),
    ).accepted
    frozen = (
        state.authorized_repair_round_id,
        state.authorized_repair_provider,
        state.authorized_repair_model,
    )
    test_plan = _plan("a.py")
    state.active_repair_plan = test_plan

    result = validate_patch_batch(
        state,
        test_plan,
        VerifiedEditBatch(
            edits=[_edit("a.py", search="value = 1", replace="value = 3", before=source)]
        ),
        test_only=True,
    )

    assert not result.accepted
    assert (
        state.authorized_repair_round_id,
        state.authorized_repair_provider,
        state.authorized_repair_model,
    ) == frozen


def test_test_only_malformed_attribution_fails_without_mutating_it(tmp_path):
    root, ref = _repo(
        tmp_path,
        {
            "a.py": "value = 1\n",
            "tests/test_existing.py": "def test_existing():\n    assert True\n",
        },
    )
    plan = _plan("tests/test_new.py")
    state = _state(root, ref, plan)
    object.__setattr__(state, "authorized_repair_round_id", 9)
    malformed = (
        state.authorized_repair_round_id,
        state.authorized_repair_provider,
        state.authorized_repair_model,
    )

    result = validate_patch_batch(
        state,
        plan,
        VerifiedEditBatch(
            edits=[_edit("tests/test_new.py", replace="def test_new():\n    assert True\n")]
        ),
        test_only=True,
    )

    assert not result.accepted
    assert result.issues[0].failure_class == "environment"
    assert (
        state.authorized_repair_round_id,
        state.authorized_repair_provider,
        state.authorized_repair_model,
    ) == malformed


def test_test_only_path_admission_filesystem_error_is_environment(
    tmp_path, monkeypatch
):
    root, ref = _repo(
        tmp_path,
        {
            "a.py": "value = 1\n",
            "tests/test_existing.py": "def test_existing():\n    assert True\n",
        },
    )
    plan = _plan("tests/test_new.py")
    state = _state(root, ref, plan)

    def fail_admission(*_args, **_kwargs):
        raise OSError("injected test path inspection failure")

    monkeypatch.setattr("src.patch_gate.inspect_allowed_test_path", fail_admission)
    result = validate_patch_batch(
        state,
        plan,
        VerifiedEditBatch(
            edits=[_edit("tests/test_new.py", replace="def test_new():\n    assert True\n")]
        ),
        test_only=True,
    )

    assert not result.accepted
    assert result.issues[0].code == "apply_failed"
    assert result.issues[0].failure_class == "environment"


def _approved_two_edit_state(tmp_path):
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
    assert validate_patch_batch(state, plan, batch).accepted
    return root, state


def test_approval_is_frozen_and_revalidation_requires_exact_active_plan(tmp_path):
    _root, state = _approved_two_edit_state(tmp_path)
    approval = state.tool_patch_approval
    assert approval is not None
    with pytest.raises(ValidationError):
        approval.patch_sha256 = "0" * 64

    state.active_repair_plan = None
    with pytest.raises(ValueError, match="active RepairPlan"):
        revalidate_approved_patch(state)


@pytest.mark.parametrize("tamper", ["plan", "fingerprint", "edit", "order", "output"])
def test_revalidation_rejects_every_canonical_approval_binding_tamper(tmp_path, tamper):
    _root, state = _approved_two_edit_state(tmp_path)
    approval = state.tool_patch_approval
    assert approval is not None
    if tamper == "plan":
        state.active_repair_plan = state.active_repair_plan.model_copy(
            update={"root_cause": "different validated plan"}
        )
    elif tamper == "fingerprint":
        state.tool_patch_approval = approval.model_copy(
            update={"patch_gate_fingerprint": "0" * 64}
        )
    elif tamper == "edit":
        state.patch_edits[0].replace = "first = 9"
    elif tamper == "order":
        state.patch_edits.reverse()
    else:
        state.patch_content = state.patch_content.replace("second = 2", "second = 9")
        state.tool_patch_approval = approval.model_copy(
            update={
                "patch_sha256": hashlib.sha256(state.patch_content.encode()).hexdigest()
            }
        )

    with pytest.raises(ValueError, match="PatchGate|approval|fingerprint|plan"):
        revalidate_approved_patch(state)


def test_approved_multifile_apply_rolls_back_every_target_on_second_replace_error(
    tmp_path, monkeypatch
):
    sources = {"a.py": "a = 1\n", "b.py": "b = 1\n"}
    root, ref = _repo(tmp_path, sources)
    plan = _plan("a.py", "b.py")
    state = _state(root, ref, plan)
    batch = VerifiedEditBatch(
        edits=[
            _edit("a.py", search="a = 1", replace="a = 2", before=b"a = 1\n"),
            _edit("b.py", search="b = 1", replace="b = 2", before=b"b = 1\n"),
        ]
    )
    assert validate_patch_batch(state, plan, batch).accepted
    actual_replace = __import__("os").replace
    failed = False

    def fail_second_target(src, dst, *args, **kwargs):
        nonlocal failed
        if dst == "b.py" and not failed:
            failed = True
            raise OSError("injected second target failure")
        return actual_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr("src.patch_gate.os.replace", fail_second_target)
    with pytest.raises(OSError, match="injected"):
        apply_approved_patch(state)

    assert (root / "a.py").read_text() == sources["a.py"]
    assert (root / "b.py").read_text() == sources["b.py"]
    assert not list(root.glob(".repopilot-*"))


def test_approved_apply_detects_target_symlink_swap_before_backup(tmp_path, monkeypatch):
    source = b"value = 1\n"
    root, ref = _repo(tmp_path, {"a.py": source.decode(), "outside.py": "safe\n"})
    plan = _plan("a.py")
    state = _state(root, ref, plan)
    assert validate_patch_batch(
        state,
        plan,
        VerifiedEditBatch(
            edits=[_edit("a.py", search="value = 1", replace="value = 2", before=source)]
        ),
    ).accepted
    actual_link = __import__("os").link
    swapped = False

    def swap_before_backup(src, dst, *args, **kwargs):
        nonlocal swapped
        if src == "a.py" and not swapped:
            swapped = True
            (root / "a.py").unlink()
            (root / "a.py").symlink_to("outside.py")
        return actual_link(src, dst, *args, **kwargs)

    monkeypatch.setattr("src.patch_gate.os.link", swap_before_backup)
    with pytest.raises(ValueError, match="identity|symlink|preimage"):
        apply_approved_patch(state)

    assert (root / "outside.py").read_text() == "safe\n"


def test_approved_new_file_failure_rolls_back_prior_existing_write(tmp_path, monkeypatch):
    source = b"value = 1\n"
    root, ref = _repo(tmp_path, {"a.py": source.decode()})
    plan = _plan("a.py", "tests/test_new.py")
    state = _state(root, ref, plan)
    batch = VerifiedEditBatch(
        edits=[
            _edit("a.py", search="value = 1", replace="value = 2", before=source),
            _edit("tests/test_new.py", replace="def test_new():\n    assert True\n"),
        ]
    )
    assert validate_patch_batch(state, plan, batch).accepted
    actual_link = __import__("os").link

    def fail_new_target(src, dst, *args, **kwargs):
        if dst == "test_new.py":
            raise OSError("injected new-file link failure")
        return actual_link(src, dst, *args, **kwargs)

    monkeypatch.setattr("src.patch_gate.os.link", fail_new_target)
    with pytest.raises(OSError, match="new-file"):
        apply_approved_patch(state)

    assert (root / "a.py").read_bytes() == source
    assert not (root / "tests/test_new.py").exists()


def test_partial_stage_write_failure_removes_stage_and_created_directories(
    tmp_path, monkeypatch
):
    root, ref = _repo(tmp_path, {"a.py": "value = 1\n"})
    plan = _plan("generated/deep/test_new.py")
    state = _state(root, ref, plan)
    assert validate_patch_batch(
        state,
        plan,
        VerifiedEditBatch(
            edits=[
                _edit(
                    "generated/deep/test_new.py",
                    replace="def test_new():\n    assert True\n",
                )
            ]
        ),
    ).accepted
    actual_write = __import__("os").write
    calls = 0

    def partial_then_fail(fd, data):
        nonlocal calls
        calls += 1
        if calls == 1:
            return actual_write(fd, data[:4])
        raise OSError("injected partial stage write")

    monkeypatch.setattr("src.patch_gate.os.write", partial_then_fail)
    with pytest.raises(OSError, match="partial stage"):
        apply_approved_patch(state)

    assert not (root / "generated").exists()
    assert not list(root.rglob(".repopilot-stage-*"))
    assert (root / "a.py").read_text() == "value = 1\n"


def test_ancestor_namespace_drift_after_precommit_check_rolls_back_detached_write(
    tmp_path, monkeypatch
):
    source = b"value = 1\n"
    root, ref = _repo(tmp_path, {"src/a.py": source.decode()})
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a.py").write_text("outside = 1\n")
    plan = _plan("src/a.py")
    state = _state(root, ref, plan)
    assert validate_patch_batch(
        state,
        plan,
        VerifiedEditBatch(
            edits=[
                _edit(
                    "src/a.py",
                    search="value = 1",
                    replace="value = 2",
                    before=source,
                )
            ]
        ),
    ).accepted
    actual_replace = __import__("os").replace
    attacked = False

    def drift_namespace_then_replace(src, dst, *args, **kwargs):
        nonlocal attacked
        if dst == "a.py" and not attacked:
            attacked = True
            (root / "src").rename(root / "moved-src")
            (root / "src").symlink_to(outside, target_is_directory=True)
        return actual_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr("src.patch_gate.os.replace", drift_namespace_then_replace)
    with pytest.raises(ValueError, match="namespace|ancestor|identity"):
        apply_approved_patch(state)

    assert (outside / "a.py").read_text() == "outside = 1\n"
    assert (root / "moved-src/a.py").read_bytes() == source
    assert not list((root / "moved-src").rglob(".repopilot-*"))
