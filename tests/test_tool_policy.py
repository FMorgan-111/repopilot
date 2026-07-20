from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.state import AgentState, ToolInvocation
from src.tool_policy import ToolIntent, ToolPolicy


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
    }
    values.update(updates)
    return AgentState(**values)


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
        _intent("read_range", {"end_line": 2, "start_line": 1, "path": "./src/widget.py"}),
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
    assert python.argv[1:3] == ["-m", "pytest"]


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
    ],
)
def test_rejects_vcs_metadata_and_common_secret_paths(tmp_path, action, args):
    root, commit = _repo(tmp_path)
    (root / ".env").write_text("TOKEN=sentinel\n", encoding="utf-8")
    (root / "private.pem").write_text("sentinel\n", encoding="utf-8")

    decision = ToolPolicy().authorize(
        _state(root, commit), _intent(action, args), calls_this_round=0
    )

    assert decision.approved is False


def test_untracked_test_selector_requires_explicit_generated_patch_marker(tmp_path):
    root, commit = _repo(tmp_path)
    generated = root / "tests" / "test_generated_regression.py"
    generated.write_text("def test_generated():\n    assert True\n", encoding="utf-8")
    command = "pytest tests/test_generated_regression.py"

    unmarked = ToolPolicy().authorize(
        _state(root, commit),
        _intent("run_targeted_test", {"command": command}),
        calls_this_round=0,
    )
    marked = ToolPolicy().authorize(
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
    assert marked.approved is True


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
