from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.safe_subprocess import minimal_subprocess_env
from src.state import (
    AgentState,
    GeneratedTestApproval,
    SnapshotManifestEntry,
    ToolInvocation,
    ToolPatchApproval,
    ToolSandboxConfig,
)
from src.tool_policy import PYTEST_BOOTSTRAP, ToolIntent, ToolPolicy


def _repo(tmp_path: Path) -> tuple[Path, str]:
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "tests" / "test_widget.py").write_text(
        "def test_widget():\n    assert True\n", encoding="utf-8"
    )
    (tmp_path / "src" / "widget.py").write_text(
        "def widget():\n    return 1\n", encoding="utf-8"
    )
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "base"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return tmp_path, commit


def _state(root: Path, commit: str, **updates: object) -> AgentState:
    values: dict[str, object] = {
        "issue_url": "https://github.com/acme/widget/issues/1",
        "repo_path": str(root),
        "repo_ref": commit,
        "tool_sandbox_config": ToolSandboxConfig(
            backend="docker",
            image="registry.example/repopilot-tests@sha256:" + "1" * 64,
            python_executable="/usr/bin/python3",
            project_executables={"npm": "/usr/bin/npm"},
        ),
    }
    values.update(updates)
    return AgentState(**values)


def _manifest_fingerprint(entries: list[SnapshotManifestEntry]) -> str:
    payload = json.dumps(
        [entry.model_dump(mode="json") for entry in entries],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _intent(action: str, args: dict[str, object]) -> ToolIntent:
    return ToolIntent(
        action=action, args=args, reason="diagnose", expected_evidence="source"
    )


@pytest.mark.parametrize(
    ("action", "args"),
    [
        ("search_symbol", {"symbol": "widget"}),
        ("search_text", {"text": "return 1", "path": "src"}),
        ("read_symbol", {"path": "src/widget.py", "symbol": "widget"}),
        ("read_range", {"path": "src/widget.py", "start_line": 1, "end_line": 2}),
        ("find_references", {"symbol": "widget"}),
        ("list_related_tests", {"path": "src/widget.py"}),
        ("run_targeted_test", {"command": "pytest tests/test_widget.py::test_widget -q"}),
        ("inspect_git_diff", {}),
        ("validate_patch", {}),
        ("request_repair", {}),
        ("finish_investigation", {}),
    ],
)
def test_authorizes_each_allowlisted_action(tmp_path, action, args):
    root, commit = _repo(tmp_path)

    decision = ToolPolicy().authorize(
        _state(root, commit), _intent(action, args), calls_this_round=0
    )

    assert decision.approved is True
    assert decision.status == "approved"
    assert len(decision.args_fingerprint) == 64


@pytest.mark.parametrize(
    ("action", "args"),
    [
        ("read_range", {"path": "../outside.py", "start_line": 1, "end_line": 2}),
        ("read_range", {"path": "/tmp/outside.py", "start_line": 1, "end_line": 2}),
        ("read_symbol", {"path": "src/widget.py", "symbol": "widget", "extra": 1}),
        ("search_text", {"text": "x", "command": "cat /etc/passwd"}),
        ("run_targeted_test", {"command": "bash -c pytest"}),
        ("run_targeted_test", {"command": "pytest tests/test_widget.py | tee /tmp/x"}),
        ("run_targeted_test", {"command": "TOKEN=x pytest tests/test_widget.py"}),
        ("run_targeted_test", {"command": "curl https://example.com"}),
        ("run_targeted_test", {"command": "pytest ../tests/test_widget.py"}),
        (
            "run_targeted_test",
            {"command": "pytest tests/test_widget.py --rootdir=/tmp/outside"},
        ),
    ],
)
def test_rejects_unsafe_or_unknown_arguments(tmp_path, action, args):
    root, commit = _repo(tmp_path)

    decision = ToolPolicy().authorize(
        _state(root, commit), _intent(action, args), calls_this_round=0
    )

    assert decision.approved is False
    assert decision.status == "rejected"


def test_rejects_symlink_escape(tmp_path):
    root, commit = _repo(tmp_path)
    outside = tmp_path.parent / "outside-tool-policy.py"
    outside.write_text("secret", encoding="utf-8")
    (root / "escape.py").symlink_to(outside)

    decision = ToolPolicy().authorize(
        _state(root, commit),
        _intent("read_range", {"path": "escape.py", "start_line": 1, "end_line": 1}),
        calls_this_round=0,
    )

    assert decision.approved is False


def test_normalized_key_order_cannot_bypass_duplicate_detection(tmp_path):
    root, commit = _repo(tmp_path)
    policy = ToolPolicy()
    state = _state(root, commit)
    first = policy.authorize(
        state,
        _intent("read_range", {"path": "src/widget.py", "start_line": 1, "end_line": 2}),
        calls_this_round=0,
    )
    state.tool_history.append(
        ToolInvocation(
            action="read_range",
            args_fingerprint=first.args_fingerprint,
            status="ok",
        )
    )

    duplicate = policy.authorize(
        state,
        _intent("read_range", {"end_line": 2, "start_line": 1, "path": "src/widget.py"}),
        calls_this_round=1,
    )

    assert duplicate.approved is False
    assert duplicate.status == "duplicate"
    assert duplicate.args_fingerprint == first.args_fingerprint


@pytest.mark.parametrize(("round_calls", "history", "reason"), [(8, 0, "round_limit"), (0, 30, "sample_limit")])
def test_enforces_call_budgets(tmp_path, round_calls, history, reason):
    root, commit = _repo(tmp_path)
    state = _state(root, commit)
    state.tool_history = [
        ToolInvocation(
            action="finish_investigation",
            args_fingerprint=f"{index:064x}",
            status="ok",
        )
        for index in range(history)
    ]

    decision = ToolPolicy().authorize(
        state,
        _intent("search_symbol", {"symbol": "widget"}),
        calls_this_round=round_calls,
    )

    assert decision.approved is False
    assert decision.reason == reason


@pytest.mark.parametrize(
    "updates",
    [
        {"repo_path": ""},
        {"repo_ref": "main"},
        {"repo_ref": "f" * 40},
    ],
)
def test_rejects_missing_or_mismatched_exact_checkout(tmp_path, updates):
    root, commit = _repo(tmp_path)
    state = _state(root, commit, **updates)

    decision = ToolPolicy().authorize(
        state, _intent("search_symbol", {"symbol": "widget"}), calls_this_round=0
    )

    assert decision.approved is False


def test_accepts_python_module_pytest_and_existing_project_command(tmp_path):
    root, commit = _repo(tmp_path)
    (root / "package.json").write_text('{"scripts":{"test":"pytest"}}', encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "package.json"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "--amend", "--no-edit"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    state = _state(root, commit, test_command="npm test")
    policy = ToolPolicy()

    python = policy.authorize(
        state,
        _intent("run_targeted_test", {"command": "python -m pytest tests/test_widget.py"}),
        calls_this_round=0,
    )
    project = policy.authorize(
        state,
        _intent("run_targeted_test", {"command": "npm test -- tests/test_widget.py"}),
        calls_this_round=0,
    )

    assert python.approved is True
    assert project.approved is True
    assert python.argv[:4] == ["/usr/bin/python3", "-P", "-c", PYTEST_BOOTSTRAP]
    assert project.argv[0] == "/usr/bin/npm"


def test_targeted_test_fails_closed_without_immutable_oci_config(tmp_path):
    root, commit = _repo(tmp_path)

    decision = ToolPolicy().authorize(
        _state(root, commit, tool_sandbox_config=None),
        _intent("run_targeted_test", {"command": "pytest tests/test_widget.py"}),
        calls_this_round=0,
    )

    assert decision.approved is False


@pytest.mark.parametrize(
    "image",
    [
        "registry.example/repopilot-tests:latest",
        "registry.example/repopilot-tests@sha256:short",
        "--privileged@sha256:" + "1" * 64,
    ],
)
def test_tool_sandbox_image_must_be_digest_pinned(image):
    with pytest.raises(ValidationError):
        ToolSandboxConfig(backend="docker", image=image)


def test_tool_sandbox_project_executables_are_deeply_immutable():
    config = ToolSandboxConfig(
        backend="docker",
        image="sha256:" + "1" * 64,
        project_executables={"npm": "/usr/bin/npm"},
    )

    assert isinstance(config.project_executables, tuple)
    with pytest.raises(TypeError):
        config.project_executables[0] = ("npm", "/tmp/npm")


@pytest.mark.parametrize("path", ["cafe\u0301/widget.py", "src/control\x01.py"])
def test_rejects_noncanonical_unicode_or_control_paths(tmp_path, path):
    root, commit = _repo(tmp_path)

    decision = ToolPolicy().authorize(
        _state(root, commit),
        _intent("search_text", {"text": "widget", "path": path}),
        calls_this_round=0,
    )

    assert decision.approved is False


def test_rejects_network_program_hidden_in_configured_project_command(tmp_path):
    root, commit = _repo(tmp_path)
    (root / "package.json").write_text("{}", encoding="utf-8")
    state = _state(root, commit, test_command="npm exec curl")

    decision = ToolPolicy().authorize(
        state,
        _intent(
            "run_targeted_test",
            {"command": "npm exec curl tests/test_widget.py"},
        ),
        calls_this_round=0,
    )

    assert decision.approved is False


def test_rejects_non_test_program_hidden_in_project_command(tmp_path):
    root, commit = _repo(tmp_path)
    (root / "package.json").write_text("{}", encoding="utf-8")
    state = _state(root, commit, test_command="npm exec echo")

    decision = ToolPolicy().authorize(
        state,
        _intent(
            "run_targeted_test",
            {"command": "npm exec echo tests/test_widget.py"},
        ),
        calls_this_round=0,
    )

    assert decision.approved is False


@pytest.mark.parametrize(
    ("state_command", "requested"),
    [
        ("npm run contest", "npm run contest -- tests/test_widget.py"),
        ("npm test -- --runInBand", "npm test -- --runInBand tests/test_widget.py"),
        ("npm exec pytest", "npm exec pytest tests/test_widget.py"),
    ],
)
def test_rejects_inexact_or_preconfigured_project_commands(
    tmp_path, state_command, requested
):
    root, commit = _repo(tmp_path)
    (root / "package.json").write_text(
        '{"scripts":{"test":"pytest","contest":"pytest"}}', encoding="utf-8"
    )
    subprocess.run(["git", "-C", str(root), "add", "package.json"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "--amend", "--no-edit"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    decision = ToolPolicy().authorize(
        _state(root, commit, test_command=state_command),
        _intent("run_targeted_test", {"command": requested}),
        calls_this_round=0,
    )

    assert decision.approved is False


def test_accepts_exact_colon_scoped_test_script_from_committed_manifest(tmp_path):
    root, commit = _repo(tmp_path)
    (root / "package.json").write_text(
        '{"scripts":{"test:unit":"pytest"}}', encoding="utf-8"
    )
    subprocess.run(["git", "-C", str(root), "add", "package.json"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "--amend", "--no-edit"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    decision = ToolPolicy().authorize(
        _state(root, commit, test_command="npm run test:unit"),
        _intent(
            "run_targeted_test",
            {"command": "npm run test:unit -- tests/test_widget.py"},
        ),
        calls_this_round=0,
    )

    assert decision.approved is True


def test_rejects_modified_or_symlinked_project_manifest(tmp_path):
    root, commit = _repo(tmp_path)
    manifest = root / "package.json"
    manifest.write_text('{"scripts":{"test":"pytest"}}', encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "package.json"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "--amend", "--no-edit"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest.write_text('{"scripts":{"test":"curl example.invalid"}}', encoding="utf-8")

    modified = ToolPolicy().authorize(
        _state(root, commit, test_command="npm test"),
        _intent("run_targeted_test", {"command": "npm test -- tests/test_widget.py"}),
        calls_this_round=0,
    )

    manifest.unlink()
    outside = tmp_path.parent / "external-package.json"
    outside.write_text('{"scripts":{"test":"pytest"}}', encoding="utf-8")
    manifest.symlink_to(outside)
    symlinked = ToolPolicy().authorize(
        _state(root, commit, test_command="npm test"),
        _intent("run_targeted_test", {"command": "npm test -- tests/test_widget.py"}),
        calls_this_round=0,
    )

    assert modified.approved is False
    assert symlinked.approved is False


@pytest.mark.parametrize(
    ("action", "args"),
    [
        ("read_range", {"path": ".git/config", "start_line": 1, "end_line": 1}),
        ("read_range", {"path": ".env", "start_line": 1, "end_line": 1}),
        ("search_text", {"text": "token", "path": ".git"}),
        ("search_text", {"text": "token", "path": ".env"}),
        ("read_range", {"path": "private.pem", "start_line": 1, "end_line": 1}),
        ("read_range", {"path": "credentials.yaml", "start_line": 1, "end_line": 1}),
        (
            "read_range",
            {"path": ".env.production/nested.py", "start_line": 1, "end_line": 1},
        ),
    ],
)
def test_rejects_vcs_metadata_and_common_secret_paths(tmp_path, action, args):
    root, commit = _repo(tmp_path)
    (root / ".env").write_text("TOKEN=sentinel\n", encoding="utf-8")
    (root / "private.pem").write_text("sentinel\n", encoding="utf-8")
    (root / "credentials.yaml").write_text("sentinel\n", encoding="utf-8")
    (root / ".env.production").mkdir(exist_ok=True)
    (root / ".env.production" / "nested.py").write_text(
        "sentinel\n", encoding="utf-8"
    )

    decision = ToolPolicy().authorize(
        _state(root, commit), _intent(action, args), calls_this_round=0
    )

    assert decision.approved is False


def test_untracked_test_selector_requires_persisted_patchgate_approval(tmp_path):
    root, commit = _repo(tmp_path)
    generated = root / "tests" / "test_generated_regression.py"
    generated.write_text("def test_generated():\n    assert True\n", encoding="utf-8")
    command = "pytest tests/test_generated_regression.py"

    unmarked = ToolPolicy().authorize(
        _state(root, commit),
        _intent("run_targeted_test", {"command": command}),
        calls_this_round=0,
    )
    merely_marked = ToolPolicy().authorize(
        _state(
            root,
            commit,
            patch_content=(
                "diff --git a/tests/test_generated_regression.py "
                "b/tests/test_generated_regression.py\n"
                "new file mode 100644\n"
                "--- /dev/null\n"
                "+++ b/tests/test_generated_regression.py\n"
            ),
        ),
        _intent("run_targeted_test", {"command": command}),
        calls_this_round=0,
    )

    assert unmarked.approved is False
    assert merely_marked.approved is False


@pytest.mark.parametrize(
    "path",
    [
        ".github/tests/test_ci.py",
        "vendor/tests/test_vendor.py",
        "config/tests/test_config.py",
    ],
)
def test_tracked_test_selector_rejects_forbidden_tree(tmp_path, path):
    root, _commit = _repo(tmp_path)
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("def test_hidden(): assert True\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", path], check=True)
    subprocess.run(
        ["git", "-C", str(root), "commit", "--amend", "--no-edit"], check=True
    )
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    decision = ToolPolicy().authorize(
        _state(root, commit),
        _intent("run_targeted_test", {"command": f"pytest {path}"}),
        calls_this_round=0,
    )

    assert decision.approved is False


def test_approved_generated_test_requires_exact_current_content(tmp_path):
    root, commit = _repo(tmp_path)
    path = "tests/test_generated_regression.py"
    generated = root / path
    content = "def test_generated():\n    assert True\n"
    generated.write_text(content, encoding="utf-8")
    patch = (
        f"diff --git a/{path} b/{path}\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        f"+++ b/{path}\n"
        "@@ -0,0 +1,2 @@\n"
        "+def test_generated():\n"
        "+    assert True\n"
    )
    gate_fingerprint = "a" * 64
    manifest = [
        SnapshotManifestEntry(
            path=path,
            change="added",
            mode="100644",
            content_sha256=hashlib.sha256(content.encode()).hexdigest(),
            size=len(content.encode()),
        )
    ]
    state = _state(
        root,
        commit,
        patch_content=patch,
        tool_patch_approval=ToolPatchApproval(
            base_ref=commit,
            patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),
            patch_gate_fingerprint=gate_fingerprint,
            changed_manifest=manifest,
            manifest_fingerprint=_manifest_fingerprint(manifest),
        ),
        generated_test_approvals=[
            GeneratedTestApproval(
                path=path,
                content_sha256=hashlib.sha256(content.encode()).hexdigest(),
                patch_gate_fingerprint=gate_fingerprint,
            )
        ],
    )
    command = f"pytest {path}"

    approved = ToolPolicy().authorize(
        state,
        _intent("run_targeted_test", {"command": command}),
        calls_this_round=0,
    )
    generated.write_text(content + "# tampered\n", encoding="utf-8")
    tampered = ToolPolicy().authorize(
        state,
        _intent("run_targeted_test", {"command": command}),
        calls_this_round=0,
    )

    assert approved.approved is True
    assert tampered.approved is False


def test_generated_test_approval_round_trips_and_legacy_defaults_empty(tmp_path):
    root, commit = _repo(tmp_path)
    state = _state(
        root,
        commit,
        tool_patch_approval=ToolPatchApproval(
            base_ref=commit,
            patch_sha256="b" * 64,
            patch_gate_fingerprint="c" * 64,
            changed_manifest=[],
            manifest_fingerprint=_manifest_fingerprint([]),
        ),
        generated_test_approvals=[
            GeneratedTestApproval(
                path="tests/test_generated.py",
                content_sha256="d" * 64,
                patch_gate_fingerprint="c" * 64,
            )
        ],
    )

    loaded = AgentState.model_validate_json(state.model_dump_json())
    legacy = AgentState(issue_url="https://github.com/acme/widget/issues/1")

    assert loaded.tool_patch_approval == state.tool_patch_approval
    assert loaded.generated_test_approvals == state.generated_test_approvals
    assert legacy.tool_patch_approval is None
    assert legacy.generated_test_approvals == []
    assert legacy.tool_sandbox_config is None


def test_rejects_source_and_index_only_test_selectors(tmp_path):
    root, commit = _repo(tmp_path)
    staged = root / "tests" / "test_staged.py"
    staged.write_text("def test_staged():\n    assert True\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", str(staged)], check=True)
    policy = ToolPolicy()

    source = policy.authorize(
        _state(root, commit),
        _intent("run_targeted_test", {"command": "pytest src/widget.py"}),
        calls_this_round=0,
    )
    staged_only = policy.authorize(
        _state(root, commit),
        _intent("run_targeted_test", {"command": "pytest tests/test_staged.py"}),
        calls_this_round=0,
    )

    assert source.approved is False
    assert staged_only.approved is False


def test_generated_test_marker_must_belong_to_the_same_new_file_diff(tmp_path):
    root, commit = _repo(tmp_path)
    generated = root / "tests" / "test_generated_regression.py"
    generated.write_text("def test_generated():\n    assert True\n", encoding="utf-8")
    misleading_patch = (
        "diff --git a/tests/test_other.py b/tests/test_other.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/tests/test_other.py\n"
        "diff --git a/tests/test_generated_regression.py "
        "b/tests/test_generated_regression.py\n"
        "--- a/tests/test_generated_regression.py\n"
        "+++ b/tests/test_generated_regression.py\n"
    )

    decision = ToolPolicy().authorize(
        _state(root, commit, patch_content=misleading_patch),
        _intent(
            "run_targeted_test",
            {"command": "pytest tests/test_generated_regression.py"},
        ),
        calls_this_round=0,
    )

    assert decision.approved is False


def test_public_fixed_test_argv_uses_sandbox_python_and_preserves_selector(tmp_path):
    import src.tool_policy as policy

    root, commit = _repo(tmp_path)
    state = _state(
        root,
        commit,
        test_command="python -m pytest tests/test_widget.py -q",
    )
    argv, normalized = policy.fixed_test_argv(
        root,
        state,
        ["python", "-m", "pytest", "tests/test_widget.py", "-q"],
    )
    assert argv == [
        "/usr/bin/python3",
        "-P",
        "-c",
        PYTEST_BOOTSTRAP,
        "tests/test_widget.py",
        "-q",
    ]
    assert normalized == "python -m pytest tests/test_widget.py -q"


def test_fixed_pytest_argv_imports_trusted_pytest_before_workspace(tmp_path):
    import src.tool_policy as policy

    root, _commit = _repo(tmp_path)
    (root / "pytest.py").write_text(
        'raise RuntimeError("hostile repository pytest.py imported")\n',
        encoding="utf-8",
    )
    (root / "tests" / "test_widget.py").write_text(
        "import pytest\n"
        "from src.widget import widget\n\n"
        "def test_widget():\n"
        "    assert widget() == 1\n"
        "    assert pytest.__file__ != 'pytest.py'\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(root), "commit", "--amend", "--no-edit"],
        check=True,
        capture_output=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    state = _state(
        root,
        commit,
        tool_sandbox_config=ToolSandboxConfig(
            backend="docker",
            image="registry.example/repopilot-tests@sha256:" + "1" * 64,
            python_executable=sys.executable,
        ),
    )

    argv, _normalized = policy.fixed_test_argv(
        root,
        state,
        ["pytest", "tests/test_widget.py", "-q"],
    )
    result = subprocess.run(
        argv,
        cwd=root,
        env=minimal_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout
