from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.evidence import EvidenceStore
from src.safe_subprocess import BoundedProcessResult
from src.state import AgentState
from src.tool_policy import ToolIntent
from src.tool_router import route_tool_intent


def _repo(tmp_path: Path) -> tuple[Path, str]:
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "widget.py").write_text(
        "class Widget:\n    def render(self):\n        return 'old'\n\ndef caller():\n    return Widget().render()\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_widget.py").write_text(
        "from src.widget import Widget\n\ndef test_render():\n    assert Widget().render() == 'old'\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "base"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
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


def _intent(action: str, **args: object) -> ToolIntent:
    return ToolIntent(action=action, args=args, reason="diagnose", expected_evidence="result")


@pytest.mark.parametrize(
    ("intent", "needle"),
    [
        (_intent("search_symbol", symbol="Widget"), "src/widget.py"),
        (_intent("search_text", text="return 'old'"), "return 'old'"),
        (_intent("read_symbol", path="src/widget.py", symbol="Widget.render"), "def render"),
        (_intent("read_range", path="src/widget.py", start_line=1, end_line=3), "class Widget"),
        (_intent("find_references", symbol="Widget"), "tests/test_widget.py"),
        (_intent("list_related_tests", path="src/widget.py"), "tests/test_widget.py"),
    ],
)
async def test_data_tools_add_bounded_evidence(tmp_path, intent, needle):
    root, commit = _repo(tmp_path)
    state = _state(root, commit)

    result = await route_tool_intent(state, intent, calls_this_round=0)

    assert result.status == "ok"
    assert result.made_progress is True
    assert result.evidence_id == state.evidence[-1].evidence_id
    assert needle in state.evidence[-1].content
    assert len(state.evidence[-1].content) <= 8_000
    assert state.tool_history[-1].status == "ok"


async def test_targeted_test_uses_fixed_argv_without_shell(tmp_path, monkeypatch):
    root, commit = _repo(tmp_path)
    captured: dict[str, object] = {}

    def fake_run(argv, root, **kwargs):
        captured["argv"] = argv
        captured["root"] = root
        captured.update(kwargs)
        return BoundedProcessResult(argv=argv, returncode=0, stdout="one passed", stderr="")

    monkeypatch.setattr("src.tool_router._run", fake_run)
    state = _state(root, commit)

    result = await route_tool_intent(
        state,
        _intent("run_targeted_test", command="pytest tests/test_widget.py::test_render -q"),
        calls_this_round=0,
    )

    assert result.status == "ok"
    assert isinstance(captured["argv"], list)
    assert captured["argv"][1:3] == ["-m", "pytest"]
    assert captured["root"] == root
    assert captured["isolate_network"] is True
    assert "one passed" in state.evidence[-1].content


async def test_diff_and_patch_validation_use_state_baseline(tmp_path):
    root, commit = _repo(tmp_path)
    state = _state(root, commit)
    old = (root / "src" / "widget.py").read_text(encoding="utf-8")
    (root / "src" / "widget.py").write_text(old.replace("'old'", "'new'"), encoding="utf-8")
    state.patch_content = subprocess.run(
        ["git", "-C", str(root), "diff"], check=True, capture_output=True, text=True
    ).stdout

    diff = await route_tool_intent(state, _intent("inspect_git_diff"), calls_this_round=0)
    valid = await route_tool_intent(state, _intent("validate_patch"), calls_this_round=1)

    assert diff.status == "ok"
    assert "+        return 'new'" in state.evidence[0].content
    assert valid.status == "ok"
    assert "valid" in state.evidence[1].content.lower()


async def test_duplicate_evidence_is_recorded_as_no_progress(tmp_path):
    root, commit = _repo(tmp_path)
    state = _state(root, commit)

    first = await route_tool_intent(
        state, _intent("search_text", text="return 'old'"), calls_this_round=0
    )
    # A different allowed intent can produce the same normalized evidence payload only
    # when the store's fingerprint is already present; force replay through a fresh history.
    state.tool_history.clear()
    second = await route_tool_intent(
        state, _intent("search_text", text="return 'old'"), calls_this_round=1
    )

    assert first.status == "ok"
    assert second.status == "duplicate"
    assert second.made_progress is False
    assert len(state.evidence) == 1
    assert state.tool_history[-1].status == "duplicate"


@pytest.mark.parametrize("action", ["request_repair", "finish_investigation"])
async def test_control_actions_execute_nothing_and_add_no_evidence(tmp_path, monkeypatch, action):
    root, commit = _repo(tmp_path)
    monkeypatch.setattr(
        "src.tool_router._run",
        lambda *args, **kwargs: pytest.fail("control action executed a command"),
    )
    state = _state(root, commit)

    result = await route_tool_intent(state, _intent(action), calls_this_round=0)

    assert result.status == "ok"
    assert result.control_action == action
    assert result.evidence_id is None
    assert state.evidence == []


async def test_errors_store_only_sanitized_exception_class(tmp_path, monkeypatch):
    root, commit = _repo(tmp_path)
    secret = "sk-sensitive-tool-error"

    def explode(*args, **kwargs):
        raise RuntimeError(f"failed with {secret}")

    monkeypatch.setattr("src.tool_router._run", explode)
    state = _state(root, commit)

    result = await route_tool_intent(
        state,
        _intent("run_targeted_test", command="pytest tests/test_widget.py"),
        calls_this_round=0,
    )

    assert result.status == "error"
    assert result.error_class == "RuntimeError"
    assert secret not in state.model_dump_json()
    assert state.tool_history[-1].error_class == "RuntimeError"


async def test_evidence_capacity_is_an_error_without_nonexistent_id(tmp_path):
    root, commit = _repo(tmp_path)
    state = _state(root, commit)
    store = EvidenceStore(state, max_items=30)
    for index in range(30):
        added = store.add(tool="seed", summary=str(index), content=str(index))
        assert added.added is True

    result = await route_tool_intent(
        state,
        _intent("search_text", text="return 'old'"),
        calls_this_round=0,
    )

    assert result.status == "error"
    assert result.error_class == "EvidenceCapacityError"
    assert result.evidence_id is None
    assert state.tool_history[-1].evidence_id is None


async def test_git_diff_excludes_tracked_dotenv_content(tmp_path):
    root, commit = _repo(tmp_path)
    dotenv = root / ".env"
    dotenv.write_text("TOKEN=old-sentinel\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", ".env"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "--amend", "--no-edit"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dotenv.write_text("TOKEN=new-sentinel\n", encoding="utf-8")
    state = _state(root, commit)

    result = await route_tool_intent(
        state, _intent("inspect_git_diff"), calls_this_round=0
    )

    assert result.status == "ok"
    assert "old-sentinel" not in state.evidence[-1].content
    assert "new-sentinel" not in state.evidence[-1].content


async def test_policy_rejection_is_persisted_without_execution(tmp_path):
    root, commit = _repo(tmp_path)
    state = _state(root, commit)

    result = await route_tool_intent(
        state,
        _intent("run_targeted_test", command="curl https://example.com"),
        calls_this_round=0,
    )

    assert result.status == "rejected"
    assert state.tool_history[-1].status == "rejected"
    assert state.evidence == []
