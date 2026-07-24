from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from src.evidence import EvidenceStore
from src.safe_subprocess import BoundedProcessResult
from src.state import (
    AgentState,
    GeneratedTestApproval,
    SnapshotManifestEntry,
    ToolPatchApproval,
    ToolSandboxConfig,
)
from src.tool_policy import PYTEST_BOOTSTRAP, ToolIntent
from src.tool_router import (
    _disposable_test_snapshot,
    _preflight_exact_tree,
    _scan_snapshot,
    disposable_test_snapshot,
    route_tool_intent,
)

_IMAGE = "registry.example/repopilot-tests@sha256:" + "1" * 64


def _sandbox_config() -> ToolSandboxConfig:
    return ToolSandboxConfig(
        backend="docker",
        image=_IMAGE,
        python_executable="/usr/bin/python3",
        project_executables={"npm": "/usr/bin/npm"},
    )


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
        "tool_sandbox_config": _sandbox_config(),
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


def _approval(
    commit: str, patch: str, entries: list[SnapshotManifestEntry]
) -> ToolPatchApproval:
    return ToolPatchApproval(
        base_ref=commit,
        patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),
        patch_gate_fingerprint="e" * 64,
        changed_manifest=entries,
        manifest_fingerprint=_manifest_fingerprint(entries),
    )


def _approved_generated_overlay_state(
    root: Path,
    commit: str,
    path: str,
) -> AgentState:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("def test_generated(): assert True\n", encoding="utf-8")
    from src.nodes.execute import git_diff

    patch = git_diff(str(root))
    content = target.read_bytes()
    fingerprint = "9" * 64
    entry = SnapshotManifestEntry(
        path=path,
        change="added",
        mode="100644",
        content_sha256=hashlib.sha256(content).hexdigest(),
        size=len(content),
    )
    return _state(
        root,
        commit,
        patch_content=patch,
        coverage_test_files=[path],
        generated_test_approvals=[
            GeneratedTestApproval(
                path=path,
                content_sha256=entry.content_sha256,
                patch_gate_fingerprint=fingerprint,
            )
        ],
        tool_patch_approval=ToolPatchApproval(
            base_ref=commit,
            patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),
            patch_gate_fingerprint=fingerprint,
            changed_manifest=[entry],
            manifest_fingerprint=_manifest_fingerprint([entry]),
        ),
    )


def _approved_modified_overlay_state(
    root: Path,
    commit: str,
    path: str,
) -> AgentState:
    target = root / path
    base = target.read_bytes()
    target.write_text("def test_modified(): assert True\n", encoding="utf-8")
    from src.nodes.execute import git_diff

    patch = git_diff(str(root))
    content = target.read_bytes()
    fingerprint = "8" * 64
    entry = SnapshotManifestEntry(
        path=path,
        change="modified",
        mode="100644",
        content_sha256=hashlib.sha256(content).hexdigest(),
        size=len(content),
    )
    return _state(
        root,
        commit,
        patch_content=patch,
        coverage_test_files=[path],
        generated_test_approvals=[
            GeneratedTestApproval(
                path=path,
                change="modified",
                base_content_sha256=hashlib.sha256(base).hexdigest(),
                content_sha256=entry.content_sha256,
                patch_gate_fingerprint=fingerprint,
            )
        ],
        tool_patch_approval=ToolPatchApproval(
            base_ref=commit,
            patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),
            patch_gate_fingerprint=fingerprint,
            changed_manifest=[entry],
            manifest_fingerprint=_manifest_fingerprint([entry]),
        ),
    )


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

    async def fake_oci(argv, *, sandbox, config, **kwargs):
        captured["argv"] = argv
        captured["root"] = sandbox.workspace
        captured["sandbox"] = sandbox
        captured["config"] = config
        captured.update(kwargs)
        assert sandbox.workspace != root_repo
        assert not (sandbox.workspace / ".git").exists()
        assert (sandbox.workspace / "tests" / "test_widget.py").is_file()
        return BoundedProcessResult(argv=argv, returncode=0, stdout="one passed", stderr="")

    monkeypatch.setattr("src.tool_router.run_oci_process_async", fake_oci, raising=False)
    root_repo = root
    state = _state(root, commit)

    result = await route_tool_intent(
        state,
        _intent("run_targeted_test", command="pytest tests/test_widget.py::test_render -q"),
        calls_this_round=0,
    )

    assert result.status == "ok"
    assert isinstance(captured["argv"], list)
    assert captured["argv"][:4] == [
        "/usr/bin/python3",
        "-P",
        "-c",
        PYTEST_BOOTSTRAP,
    ]
    assert captured["root"] != root_repo
    assert captured["sandbox"].workspace == captured["root"]
    assert "one passed" in state.evidence[-1].content


async def test_targeted_test_route_fails_closed_when_oci_is_unconfigured(tmp_path):
    root, commit = _repo(tmp_path)
    state = _state(root, commit, tool_sandbox_config=None)

    result = await route_tool_intent(
        state,
        _intent("run_targeted_test", command="pytest tests/test_widget.py"),
        calls_this_round=0,
    )

    assert result.status == "rejected"
    assert state.evidence == []


async def test_targeted_test_snapshot_contains_only_base_plus_approved_patch(
    tmp_path, monkeypatch
):
    root, commit = _repo(tmp_path)
    source = root / "src" / "widget.py"
    source.write_text(
        source.read_text(encoding="utf-8").replace("'old'", "'approved'"),
        encoding="utf-8",
    )
    (root / "host-only-sentinel").write_text("must-not-copy", encoding="utf-8")
    patch = subprocess.run(
        ["git", "-C", str(root), "diff", "--binary", commit, "--"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    changed = source.read_bytes()
    manifest = [
        SnapshotManifestEntry(
            path="src/widget.py",
            change="modified",
            mode="100644",
            content_sha256=hashlib.sha256(changed).hexdigest(),
            size=len(changed),
        )
    ]
    state = _state(
        root,
        commit,
        patch_content=patch,
        tool_patch_approval=_approval(commit, patch, manifest),
    )

    async def fake_oci(argv, *, sandbox, config, **kwargs):
        snapshot = sandbox.workspace
        assert "'approved'" in (snapshot / "src" / "widget.py").read_text()
        assert not (snapshot / "host-only-sentinel").exists()
        assert not (snapshot / ".git").exists()
        assert config.image == _IMAGE
        return BoundedProcessResult(argv=argv, returncode=0, stdout="passed", stderr="")

    monkeypatch.setattr("src.tool_router.run_oci_process_async", fake_oci, raising=False)

    result = await route_tool_intent(
        state,
        _intent("run_targeted_test", command="pytest tests/test_widget.py::test_render"),
        calls_this_round=0,
    )

    assert result.status == "ok"


async def test_targeted_test_keeps_snapshot_alive_until_cancelled_worker_drains(
    tmp_path, monkeypatch
):
    root, commit = _repo(tmp_path)
    state = _state(root, commit)
    started = asyncio.Event()
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    captured: dict[str, Path] = {}

    async def draining_oci(argv, *, sandbox, **_kwargs):
        captured["workspace"] = sandbox.workspace
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            assert sandbox.workspace.is_dir()
            cleanup_started.set()
            await cleanup_release.wait()
            assert sandbox.workspace.is_dir()
            raise

    monkeypatch.setattr(
        "src.tool_router.run_oci_process_async", draining_oci, raising=False
    )
    task = asyncio.create_task(
        route_tool_intent(
            state,
            _intent("run_targeted_test", command="pytest tests/test_widget.py"),
            calls_this_round=0,
        )
    )
    await asyncio.wait_for(started.wait(), timeout=3)
    task.cancel("cancel targeted OCI")
    await asyncio.wait_for(cleanup_started.wait(), timeout=3)

    workspace = captured["workspace"]
    assert workspace.is_dir()
    cleanup_release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()
    assert not workspace.exists()


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

    async def explode(*args, **kwargs):
        raise RuntimeError(f"failed with {secret}")

    monkeypatch.setattr(
        "src.tool_router.run_oci_process_async", explode, raising=False
    )
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


async def test_git_diff_filters_case_insensitive_secret_and_sensitive_rename(tmp_path):
    root, commit = _repo(tmp_path)
    upper = root / ".ENV"
    renamed = root / ".env.production"
    upper.write_text("TOKEN=upper-old\n", encoding="utf-8")
    renamed.write_text("TOKEN=rename-old\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", ".ENV", ".env.production"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "--amend", "--no-edit"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    upper.write_text("TOKEN=upper-new\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(root), "mv", ".env.production", "public.py"], check=True
    )
    state = _state(root, commit)

    result = await route_tool_intent(
        state, _intent("inspect_git_diff"), calls_this_round=0
    )

    assert result.status == "ok"
    content = state.evidence[-1].content
    assert "upper-old" not in content
    assert "upper-new" not in content
    assert "rename-old" not in content
    assert "public.py" not in content


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


async def test_patch_manifest_fingerprint_tamper_fails_before_oci_run(
    tmp_path, monkeypatch
):
    root, commit = _repo(tmp_path)
    approval = ToolPatchApproval(
        base_ref=commit,
        patch_sha256=hashlib.sha256(b"").hexdigest(),
        patch_gate_fingerprint="e" * 64,
        changed_manifest=[],
        manifest_fingerprint="f" * 64,
    )
    state = _state(root, commit, tool_patch_approval=approval)
    monkeypatch.setattr(
        "src.tool_router.run_oci_process_async",
        lambda *args, **kwargs: pytest.fail("tampered manifest reached OCI"),
    )

    result = await route_tool_intent(
        state,
        _intent("run_targeted_test", command="pytest tests/test_widget.py"),
        calls_this_round=0,
    )

    assert result.status == "error"
    assert result.error_class == "ValueError"


@pytest.mark.parametrize("protected", [".ENV", "package.json"])
async def test_postapply_scan_rejects_traditional_sensitive_or_manifest_patch(
    tmp_path, monkeypatch, protected
):
    root, _ = _repo(tmp_path)
    target = root / protected
    old = "TOKEN=old\n" if protected == ".ENV" else '{"scripts":{"test":"old"}}\n'
    new = "TOKEN=new\n" if protected == ".ENV" else '{"scripts":{"test":"new"}}\n'
    target.write_text(old, encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", protected], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "--amend", "--no-edit"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    patch = f"--- {protected}\n+++ {protected}\n@@ -1 +1 @@\n-{old.rstrip()}\n+{new.rstrip()}\n"
    manifest = [
        SnapshotManifestEntry(
            path=protected,
            change="modified",
            mode="100644",
            content_sha256=hashlib.sha256(new.encode()).hexdigest(),
            size=len(new.encode()),
        )
    ]
    state = _state(
        root,
        commit,
        patch_content=patch,
        tool_patch_approval=_approval(commit, patch, manifest),
    )
    monkeypatch.setattr(
        "src.tool_router.run_oci_process_async",
        lambda *args, **kwargs: pytest.fail("protected patch reached OCI"),
    )

    result = await route_tool_intent(
        state,
        _intent("run_targeted_test", command="pytest tests/test_widget.py"),
        calls_this_round=0,
    )

    assert result.status == "error"
    assert result.error_class == "ValueError"


async def test_postapply_manifest_must_match_exact_snapshot_changes(
    tmp_path, monkeypatch
):
    root, commit = _repo(tmp_path)
    source = root / "src" / "widget.py"
    source.write_text(source.read_text().replace("'old'", "'new'"), encoding="utf-8")
    patch = subprocess.run(
        ["git", "-C", str(root), "diff", "--binary", commit, "--"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    state = _state(
        root,
        commit,
        patch_content=patch,
        tool_patch_approval=_approval(commit, patch, []),
    )
    monkeypatch.setattr(
        "src.tool_router.run_oci_process_async",
        lambda *args, **kwargs: pytest.fail("unexpected change reached OCI"),
    )

    result = await route_tool_intent(
        state,
        _intent("run_targeted_test", command="pytest tests/test_widget.py"),
        calls_this_round=0,
    )

    assert result.status == "error"


def test_exact_tree_preflight_caps_file_count_and_blob_bytes(tmp_path, monkeypatch):
    root, commit = _repo(tmp_path)
    monkeypatch.setattr("src.tool_router._MAX_SNAPSHOT_FILES", 1)
    with pytest.raises(ValueError, match="file count"):
        _preflight_exact_tree(root, commit)

    monkeypatch.setattr("src.tool_router._MAX_SNAPSHOT_FILES", 100)
    monkeypatch.setattr("src.tool_router._MAX_SNAPSHOT_BYTES", 1)
    with pytest.raises(ValueError, match="blob bytes"):
        _preflight_exact_tree(root, commit)


def test_approved_patch_bytes_are_capped_before_apply(tmp_path, monkeypatch):
    root, commit = _repo(tmp_path)
    patch = "x" * 32
    state = _state(
        root,
        commit,
        patch_content=patch,
        tool_patch_approval=_approval(commit, patch, []),
    )
    monkeypatch.setattr("src.tool_router._MAX_PATCH_BYTES", 16)

    with pytest.raises(ValueError, match="byte limit"):
        with _disposable_test_snapshot(state):
            pytest.fail("oversize patch snapshot was yielded")


def test_binary_patch_is_rejected_before_apply(tmp_path):
    root, commit = _repo(tmp_path)
    patch = "diff --git a/blob b/blob\nGIT binary patch\nliteral 1000000\n"
    state = _state(
        root,
        commit,
        patch_content=patch,
        tool_patch_approval=_approval(commit, patch, []),
    )

    with pytest.raises(ValueError, match="binary patches"):
        with _disposable_test_snapshot(state):
            pytest.fail("binary patch snapshot was yielded")


def test_manifest_size_preflight_rejects_impossible_text_patch_inflation(tmp_path):
    root, commit = _repo(tmp_path)
    source = root / "src" / "widget.py"
    source.write_text(source.read_text().replace("'old'", "'new'"), encoding="utf-8")
    patch = subprocess.run(
        ["git", "-C", str(root), "diff", commit, "--", "src/widget.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    content = source.read_bytes()
    manifest = [
        SnapshotManifestEntry(
            path="src/widget.py",
            change="modified",
            mode="100644",
            content_sha256=hashlib.sha256(content).hexdigest(),
            size=500_000_000,
        )
    ]
    state = _state(
        root,
        commit,
        patch_content=patch,
        tool_patch_approval=_approval(commit, patch, manifest),
    )

    with pytest.raises(ValueError, match="impossible size"):
        with _disposable_test_snapshot(state):
            pytest.fail("impossible manifest snapshot was yielded")


def test_snapshot_scan_rejects_special_files_and_postpatch_size(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fifo = workspace / "pipe"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="special"):
        _scan_snapshot(workspace)

    fifo.unlink()
    (workspace / "large.py").write_bytes(b"x" * 32)
    monkeypatch.setattr("src.tool_router._MAX_SNAPSHOT_BYTES", 16)
    with pytest.raises(ValueError, match="bytes"):
        _scan_snapshot(workspace)


def test_snapshot_scan_rejects_postpatch_symlink(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "escape").symlink_to(tmp_path)

    with pytest.raises(ValueError, match="symlink"):
        _scan_snapshot(workspace)


def test_public_disposable_snapshot_can_select_exact_base_or_approved_fixed(
    tmp_path,
):
    root, commit = _repo(tmp_path)
    source = root / "src" / "widget.py"
    source.write_text(source.read_text().replace("'old'", "'fixed'"))
    patch = subprocess.run(
        ["git", "-C", str(root), "diff", "--binary", commit, "--"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    content = source.read_bytes()
    manifest = [
        SnapshotManifestEntry(
            path="src/widget.py",
            change="modified",
            mode="100644",
            content_sha256=hashlib.sha256(content).hexdigest(),
            size=len(content),
        )
    ]
    state = _state(
        root,
        commit,
        patch_content=patch,
        tool_patch_approval=_approval(commit, patch, manifest),
    )
    import src.tool_router as router

    with router.disposable_test_snapshot(
        state, apply_approved_changes=False
    ) as base:
        assert "'old'" in (base.workspace / "src" / "widget.py").read_text()
    with router.disposable_test_snapshot(
        state, apply_approved_changes=True
    ) as fixed:
        assert "'fixed'" in (fixed.workspace / "src" / "widget.py").read_text()


def test_public_disposable_snapshot_rejects_caller_supplied_overlay_bytes(tmp_path):
    root, commit = _repo(tmp_path)
    state = _state(root, commit)

    with pytest.raises(TypeError):
        with disposable_test_snapshot(
            state,
            apply_approved_changes=False,
            test_overlays={"tests/test_widget.py": b"caller-controlled"},
        ):
            pytest.fail("caller-supplied overlay bytes were accepted")


def test_public_disposable_snapshot_derives_only_fully_approved_test_overlays(
    tmp_path,
):
    root, commit = _repo(tmp_path)
    state = _state(
        root,
        commit,
        coverage_test_files=["tests/test_widget.py"],
    )

    with pytest.raises(ValueError, match="generated test approval"):
        with disposable_test_snapshot(
            state,
            apply_approved_changes=False,
            test_overlay_paths=["tests/test_widget.py"],
        ):
            pytest.fail("unapproved existing test was overlaid")


def test_generated_test_approval_persists_modified_base_preimage_binding(tmp_path):
    root, commit = _repo(tmp_path)
    base = (root / "tests" / "test_widget.py").read_bytes()
    approval = GeneratedTestApproval(
        path="tests/test_widget.py",
        change="modified",
        base_content_sha256=hashlib.sha256(base).hexdigest(),
        content_sha256="d" * 64,
        patch_gate_fingerprint="e" * 64,
    )

    loaded = GeneratedTestApproval.model_validate_json(approval.model_dump_json())
    assert loaded.change == "modified"
    assert loaded.base_content_sha256 == hashlib.sha256(base).hexdigest()


@pytest.mark.parametrize(
    "path",
    [
        ".github/tests/test_ci.py",
        "vendor/tests/test_vendor.py",
        "config/tests/test_config.py",
    ],
)
def test_overlay_rejects_forbidden_test_trees_even_with_forged_approval(
    tmp_path,
    path,
):
    root, commit = _repo(tmp_path)
    state = _approved_generated_overlay_state(root, commit, path)

    with pytest.raises(ValueError, match="approved test path"):
        with disposable_test_snapshot(
            state,
            apply_approved_changes=False,
            test_overlay_paths=[path],
        ):
            pytest.fail("forbidden-tree overlay was accepted")


@pytest.mark.parametrize(
    ("existing", "generated"),
    [
        ("test_existing.py", "test_generated.py"),
        ("src/existing_test.py", "src/generated_test.py"),
    ],
)
def test_overlay_accepts_approved_established_root_or_colocated_test_convention(
    tmp_path,
    existing,
    generated,
):
    root, _commit = _repo(tmp_path)
    existing_path = root / existing
    existing_path.parent.mkdir(parents=True, exist_ok=True)
    existing_path.write_text("def test_existing(): assert True\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", existing], check=True)
    subprocess.run(
        ["git", "-C", str(root), "commit", "--amend", "--no-edit"], check=True
    )
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    state = _approved_generated_overlay_state(root, commit, generated)

    with disposable_test_snapshot(
        state,
        apply_approved_changes=False,
        test_overlay_paths=[generated],
    ) as sandbox:
        assert (sandbox.workspace / generated).is_file()


@pytest.mark.parametrize(
    ("path", "accepted"),
    [
        (".github/tests/test_ci.py", False),
        ("src/existing_test.py", True),
    ],
)
def test_modified_overlay_uses_the_same_shared_test_path_policy(
    tmp_path,
    path,
    accepted,
):
    root, _commit = _repo(tmp_path)
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("def test_base(): assert True\n", encoding="utf-8")
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
    state = _approved_modified_overlay_state(root, commit, path)

    context = disposable_test_snapshot(
        state,
        apply_approved_changes=False,
        test_overlay_paths=[path],
    )
    if not accepted:
        with pytest.raises(ValueError, match="approved test path"):
            with context:
                pytest.fail("forbidden modified overlay was accepted")
        return
    with context as sandbox:
        assert "test_modified" in (sandbox.workspace / path).read_text()


async def test_sensitive_delete_and_public_add_suppresses_all_diff_evidence(tmp_path):
    root, _ = _repo(tmp_path)
    secret = "propagated-secret-sentinel"
    (root / ".env").write_text(secret + "\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", ".env"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "--amend", "--no-edit"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (root / ".env").unlink()
    (root / "public.py").write_text(secret + "\n", encoding="utf-8")
    source = root / "src" / "widget.py"
    source.write_text(source.read_text().replace("'old'", "'safe-new'"), encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    state = _state(root, commit)

    result = await route_tool_intent(
        state, _intent("inspect_git_diff"), calls_this_round=0
    )

    assert result.status == "ok"
    content = state.evidence[-1].content
    assert secret not in content
    assert "safe-new" not in content
    assert "public.py" not in content
    assert "omitted" in content.lower()


async def test_unchanged_sensitive_base_taints_public_copy_diff(tmp_path):
    root, _ = _repo(tmp_path)
    secret = "unchanged-sensitive-base-sentinel"
    (root / ".env").write_text(secret + "\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", ".env"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "--amend", "--no-edit"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (root / "public.py").write_text(secret + "\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "public.py"], check=True)
    state = _state(root, commit)

    result = await route_tool_intent(
        state, _intent("inspect_git_diff"), calls_this_round=0
    )

    assert result.status == "ok"
    assert secret not in state.evidence[-1].content
    assert "public.py" not in state.evidence[-1].content
    assert "omitted" in state.evidence[-1].content.lower()


async def test_untracked_safe_files_are_conservatively_omitted_from_diff(tmp_path):
    root, commit = _repo(tmp_path)
    marker = "untracked-content-must-not-enter-evidence"
    (root / "notes.py").write_text(marker, encoding="utf-8")
    source = root / "src" / "widget.py"
    source.write_text(
        source.read_text().replace("'old'", "'tracked-new'"), encoding="utf-8"
    )
    state = _state(root, commit)

    result = await route_tool_intent(
        state, _intent("inspect_git_diff"), calls_this_round=0
    )

    assert result.status == "ok"
    content = state.evidence[-1].content
    assert "tracked-new" in content
    assert marker not in content
    assert "notes.py" not in content
    assert "untracked files" in content.lower()
