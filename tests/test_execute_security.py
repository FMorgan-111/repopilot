from __future__ import annotations

import asyncio
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import tool_policy
from src.nodes import execute as execute_node
from src.patch_gate import validate_patch_batch
from src.safe_subprocess import BoundedProcessResult, minimal_subprocess_env
from src.state import (
    AgentState,
    RepairPlan,
    ToolSandboxConfig,
    VerifiedEdit,
    VerifiedEditBatch,
)

_IMAGE = "registry.example/repopilot-tests@sha256:" + "7" * 64


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _bounded_git_metadata(root: Path, *, limit: int = 2_000_000) -> bytes:
    metadata = bytearray()
    for path in sorted((root / ".git").rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        content = path.read_bytes()
        assert len(metadata) + len(content) <= limit
        metadata.extend(content)
    return bytes(metadata)


def _approved_state(tmp_path: Path, *, command: str) -> AgentState:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Test")
    (root / "src").mkdir()
    (root / "tests").mkdir()
    source = root / "src" / "widget.py"
    source.write_text("def answer():\n    return 1\n", encoding="utf-8")
    (root / "tests" / "test_widget.py").write_text(
        "from src.widget import answer\n\n"
        "def test_answer():\n"
        "    assert answer() == 2\n",
        encoding="utf-8",
    )
    _git(root, "add", "--all")
    _git(root, "commit", "-qm", "base")
    ref = _git(root, "rev-parse", "HEAD")
    plan = RepairPlan(
        root_cause="answer returns the old value",
        target_files=["src/widget.py"],
        target_symbols=["answer"],
        required_behavior="answer returns two",
        regression_test_strategy="run the focused answer test",
    )
    state = AgentState(
        issue_url="https://github.com/acme/widget/issues/1",
        owner="acme",
        repo="widget",
        repo_path=str(root),
        repo_ref=ref,
        trace_id="abc123def456",
        test_command=command,
        active_repair_plan=plan,
        tool_sandbox_config=ToolSandboxConfig(
            backend="docker",
            image=_IMAGE,
            python_executable="/sandbox/bin/python",
        ),
    )
    batch = VerifiedEditBatch(
        edits=[
            VerifiedEdit(
                file_path="src/widget.py",
                search="return 1",
                replace="return 2",
                intent="return the corrected answer",
            )
        ]
    )
    assert validate_patch_batch(state, plan, batch).accepted
    assert state.tool_patch_approval is not None
    return state


@pytest.mark.parametrize("unsafe_value", [None, "", "0", "true", "01", "1 "])
def test_repository_execution_mode_fails_closed_without_exact_opt_in(
    monkeypatch, unsafe_value
):
    if unsafe_value is None:
        monkeypatch.delenv("REPOPILOT_UNSAFE_ALLOW_HOST_EXECUTION", raising=False)
    else:
        monkeypatch.setenv("REPOPILOT_UNSAFE_ALLOW_HOST_EXECUTION", unsafe_value)
    state = AgentState(issue_url="https://github.com/acme/widget/issues/1")

    with pytest.raises(RuntimeError, match="host execution is disabled"):
        tool_policy.repository_execution_mode(state)


def test_repository_execution_mode_prefers_oci_over_unsafe_host(monkeypatch):
    monkeypatch.setenv("REPOPILOT_UNSAFE_ALLOW_HOST_EXECUTION", "1")
    state = AgentState(
        issue_url="https://github.com/acme/widget/issues/1",
        tool_sandbox_config=ToolSandboxConfig(
            backend="docker",
            image=_IMAGE,
            python_executable="/sandbox/bin/python",
        ),
    )

    assert tool_policy.repository_execution_mode(state) == "oci"


async def test_execute_fails_closed_before_clone_or_patch_without_sandbox(monkeypatch):
    monkeypatch.delenv("REPOPILOT_UNSAFE_ALLOW_HOST_EXECUTION", raising=False)

    async def forbidden_clone(_state):
        raise AssertionError("clone must not run before execution mode is approved")

    async def forbidden_patch(*_args, **_kwargs):
        raise AssertionError("patch must not run before execution mode is approved")

    monkeypatch.setattr(execute_node, "git_clone", forbidden_clone)
    monkeypatch.setattr(execute_node, "apply_patch_with_repair", forbidden_patch)
    state = AgentState(
        issue_url="https://github.com/acme/widget/issues/1",
        owner="acme",
        repo="widget",
        patch_content="hostile patch sentinel",
        trace_id="abc123def456",
    )

    result = await execute_node.execute_fix(state)

    assert result.fix_attempts[-1].failure_kind == "infra_error"
    assert "host execution is disabled" in result.fix_attempts[-1].error_log


async def test_oci_requires_exact_approval_before_clone_or_patch(monkeypatch):
    monkeypatch.setenv("REPOPILOT_UNSAFE_ALLOW_HOST_EXECUTION", "1")
    clone_called = False

    async def forbidden_clone(_state):
        nonlocal clone_called
        clone_called = True
        raise AssertionError("clone reached")

    monkeypatch.setattr(execute_node, "git_clone", forbidden_clone)
    state = AgentState(
        issue_url="https://github.com/acme/widget/issues/1",
        owner="acme",
        repo="widget",
        repo_ref="a" * 40,
        trace_id="abc123def456",
        patch_content="unapproved patch",
        tool_sandbox_config=ToolSandboxConfig(
            backend="docker",
            image=_IMAGE,
            python_executable="/sandbox/bin/python",
        ),
    )

    result = await execute_node.execute_fix(state)

    assert clone_called is False
    assert result.fix_attempts[-1].failure_kind == "infra_error"
    assert "exact PatchGate approval" in result.fix_attempts[-1].error_log


async def test_oci_execute_uses_exact_snapshot_and_skips_all_host_execution(
    tmp_path, monkeypatch
):
    state = _approved_state(
        tmp_path, command="pytest tests/test_widget.py::test_answer -q"
    )
    captured: dict[str, object] = {}

    def forbidden(*_args, **_kwargs):
        raise AssertionError("OCI mode must not invoke host build or test helpers")

    def fake_oci(argv, *, sandbox, config, **kwargs):
        captured["argv"] = list(argv)
        captured["config"] = config
        captured["workspace"] = sandbox.workspace
        captured.update(kwargs)
        assert "return 2" in (sandbox.workspace / "src" / "widget.py").read_text()
        assert sandbox.workspace != Path(state.repo_path)
        assert not (sandbox.workspace / ".git").exists()
        return BoundedProcessResult(list(argv), 0, "1 passed", "")

    for name in (
        "_create_venv",
        "_pip_install_editable",
        "_ensure_pytest_available",
        "run_pytest",
    ):
        monkeypatch.setattr(execute_node, name, forbidden)
    monkeypatch.setattr(execute_node, "run_oci_process", fake_oci, raising=False)

    result = await execute_node.execute_fix(state)

    assert result.fix_attempts[-1].success is True
    assert captured["argv"] == [
        "/sandbox/bin/python",
        "-P",
        "-m",
        "pytest",
        "tests/test_widget.py::test_answer",
        "-q",
    ]


async def test_oci_execute_ignores_hostile_test_text_and_uses_fixed_full_suite(
    tmp_path, monkeypatch
):
    sentinel = "python setup.py build; touch /tmp/planner-command-ran"
    state = _approved_state(tmp_path, command=sentinel)
    captured: dict[str, object] = {}

    def fake_oci(argv, *, sandbox, **_kwargs):
        captured["argv"] = list(argv)
        captured["workspace"] = sandbox.workspace
        return BoundedProcessResult(list(argv), 1, "failed", "")

    monkeypatch.setattr(execute_node, "run_oci_process", fake_oci, raising=False)

    result = await execute_node.execute_fix(state)

    assert captured["argv"] == [
        "/sandbox/bin/python",
        "-P",
        "-m",
        "pytest",
        "-q",
    ]
    assert all("planner-command-ran" not in token for token in captured["argv"])
    assert result.fix_attempts[-1].success is False


@pytest.mark.parametrize(
    "tamper_kind",
    ["fingerprint", "plan", "edit", "base"],
)
async def test_oci_revalidates_canonical_approval_before_patch_mutation(
    tmp_path, monkeypatch, tamper_kind
):
    state = _approved_state(tmp_path, command="pytest tests/test_widget.py -q")
    source = Path(state.repo_path) / "src" / "widget.py"
    patch_mutation_called = False

    if tamper_kind == "fingerprint":
        assert state.tool_patch_approval is not None
        state.tool_patch_approval = state.tool_patch_approval.model_copy(
            update={"patch_gate_fingerprint": "f" * 64}
        )
    elif tamper_kind == "plan":
        assert state.active_repair_plan is not None
        state.active_repair_plan = state.active_repair_plan.model_copy(
            update={"required_behavior": "tampered plan"}
        )
    elif tamper_kind == "edit":
        state.patch_edits[0].replace = "return 3"
    else:
        source.write_text(
            source.read_text(encoding="utf-8") + "# tampered base\n",
            encoding="utf-8",
        )
    before = source.read_bytes()

    def forbidden_patch_mutation(_state):
        nonlocal patch_mutation_called
        patch_mutation_called = True
        raise AssertionError("patch mutation reached before canonical revalidation")

    monkeypatch.setattr(
        "src.patch_gate.apply_approved_patch", forbidden_patch_mutation
    )

    result = await execute_node.execute_fix(state)

    assert patch_mutation_called is False
    assert source.read_bytes() == before
    assert result.fix_attempts[-1].failure_kind == "infra_error"
    assert "PatchGate" in result.fix_attempts[-1].error_log


def test_legacy_host_helpers_receive_only_minimal_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "model-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "github-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "cloud-secret")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("CI", "true")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='sample'\n")
    seen: list[dict[str, str]] = []

    def fake_run(cmd, **kwargs):
        seen.append(kwargs["env"])
        if cmd[:3] == ["python3", "-m", "venv"]:
            python = execute_node._venv_python_path(str(tmp_path))
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(execute_node.subprocess, "run", fake_run)

    execute_node._create_venv(str(tmp_path))
    execute_node._pip_install_editable(str(tmp_path))
    execute_node._ensure_pytest_available("python3")
    pytest_root = tmp_path / "pytest-root"
    pytest_root.mkdir()
    asyncio.run(execute_node.run_pytest(str(pytest_root), "pytest -q"))

    assert seen
    assert all(env == minimal_subprocess_env() for env in seen)
    assert all(
        key not in env
        for env in seen
        for key in (
            "LLM_API_KEY",
            "GITHUB_TOKEN",
            "AWS_SECRET_ACCESS_KEY",
            "GITHUB_ACTIONS",
            "CI",
        )
    )


async def test_explicit_unsafe_host_child_receives_real_minimal_environment(
    tmp_path, monkeypatch
):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Test")
    source = root / "value.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    probe = root / "environment_probe.py"
    secret_keys = [
        "LLM_API_KEY",
        "GITHUB_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
        "GITHUB_ACTIONS",
        "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
        "CI",
    ]
    probe.write_text(
        "import json, os\n"
        "print(json.dumps("
        f"{{key: os.environ.get(key) for key in {secret_keys!r}}}, "
        "sort_keys=True))\n",
        encoding="utf-8",
    )
    _git(root, "add", "--all")
    _git(root, "commit", "-qm", "base")
    source.write_text("VALUE = 2\n", encoding="utf-8")
    patch = subprocess.run(
        ["git", "-C", str(root), "diff", "--binary", "HEAD", "--"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _git(root, "checkout", "--", "value.py")
    monkeypatch.setenv("REPOPILOT_UNSAFE_ALLOW_HOST_EXECUTION", "1")
    for key in secret_keys:
        monkeypatch.setenv(key, f"host-{key.lower()}-sentinel")
    state = AgentState(
        issue_url="https://github.com/acme/widget/issues/1",
        repo_path=str(root),
        trace_id="abc123def456",
        patch_content=patch,
        test_command=shlex.join([sys.executable, probe.name]),
    )

    result = await execute_node.execute_fix(state)

    assert result.fix_attempts[-1].success is True
    child_environment = json.loads(result.fix_attempts[-1].error_log.strip())
    assert child_environment == {key: None for key in secret_keys}


def test_mutable_checkout_and_venv_paths_use_only_trace_sha256(monkeypatch, tmp_path):
    monkeypatch.setenv("REPOPILOT_HOME", str(tmp_path / "home"))
    hostile_trace = "../../trace-traversal-sentinel"
    digest = hashlib.sha256(hostile_trace.encode()).hexdigest()

    first = execute_node._repo_work_path("acme", "widget", "a" * 40, hostile_trace)
    second = execute_node._repo_work_path("acme", "widget", "a" * 40, "other")
    venv = execute_node._venv_dir_for(str(first))

    assert digest in first.name
    assert "trace-traversal-sentinel" not in first.as_posix()
    assert first.parent == tmp_path / "home" / "repos"
    assert first != second
    assert digest in venv.name
    assert venv.parent == first.parent


async def test_concurrent_clones_serialize_shared_cache_and_recheck_under_lock(
    monkeypatch, tmp_path
):
    home = tmp_path / "home"
    monkeypatch.setenv("REPOPILOT_HOME", str(home))
    ref = "a" * 40
    populate_calls = 0
    active_clones = 0
    max_active_clones = 0

    async def populate(_state, cache_path, _repo_ref, **_kwargs):
        nonlocal populate_calls
        populate_calls += 1
        (cache_path / ".git").mkdir(parents=True)

    async def healthy(path):
        return (Path(path) / ".git").exists()

    async def head(_path):
        return ref

    async def safe_origin(_state, _cache_path):
        return True

    async def clone(cache_path, target):
        nonlocal active_clones, max_active_clones
        assert (cache_path / ".git").exists()
        active_clones += 1
        max_active_clones = max(max_active_clones, active_clones)
        await asyncio.sleep(0.02)
        (Path(target) / ".git").mkdir(parents=True)
        active_clones -= 1

    async def clean(_path, _ref):
        return None

    monkeypatch.setattr(execute_node, "_populate_ref_cache", populate)
    monkeypatch.setattr(execute_node, "_worktree_is_healthy", healthy)
    monkeypatch.setattr(execute_node, "_worktree_head", head)
    monkeypatch.setattr(execute_node, "_cache_origin_matches", safe_origin)
    monkeypatch.setattr(execute_node, "_clone_local_repo_async", clone)
    monkeypatch.setattr(execute_node, "_verify_clean_worktree", clean, raising=False)
    states = [
        AgentState(
            issue_url=f"https://github.com/acme/widget/issues/{index}",
            owner="acme",
            repo="widget",
            repo_ref=ref,
            trace_id=f"trace-{index}",
        )
        for index in (1, 2)
    ]

    paths = await asyncio.gather(*(execute_node.git_clone(state) for state in states))

    assert populate_calls == 1
    assert max_active_clones == 1
    assert paths[0] != paths[1]


async def test_cache_lock_rejects_world_writable_lock_directory(tmp_path):
    cache = tmp_path / "repos" / "acme-widget"
    lock_root = cache.parent / ".locks"
    lock_root.mkdir(parents=True)
    lock_root.chmod(0o777)

    with pytest.raises(RuntimeError, match="lock directory is unsafe"):
        async with execute_node._cache_lock(cache):
            raise AssertionError("unsafe cache lock was acquired")


async def test_cancelled_cache_lock_waiter_never_strands_an_acquisition(tmp_path):
    cache = tmp_path / "repos" / "acme-widget"
    holder_ready = asyncio.Event()
    release_holder = asyncio.Event()

    async def hold_lock():
        async with execute_node._cache_lock(cache):
            holder_ready.set()
            await release_holder.wait()

    async def wait_for_lock():
        async with execute_node._cache_lock(cache):
            raise AssertionError("cancelled waiter acquired the cache lock")

    holder = asyncio.create_task(hold_lock())
    await holder_ready.wait()
    waiter = asyncio.create_task(wait_for_lock())
    await asyncio.sleep(0.05)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(waiter, timeout=0.5)
    release_holder.set()
    await holder

    async with asyncio.timeout(0.5):
        async with execute_node._cache_lock(cache):
            pass


async def test_cache_rebuild_stops_when_checked_removal_fails(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    work = tmp_path / "work"
    (cache / ".git").mkdir(parents=True)
    ref = "a" * 40
    removal_called = False

    async def unhealthy(_path):
        return False

    def failed_removal(path, *, ignore_errors=False):
        nonlocal removal_called
        removal_called = True
        if ignore_errors:
            return
        raise OSError(f"injected removal failure: {path}")

    failed_removal.avoids_symlink_attacks = True

    async def forbidden_repopulation(*_args, **_kwargs):
        raise AssertionError("cache repopulation reached after removal failure")

    monkeypatch.setattr(execute_node, "_worktree_is_healthy", unhealthy)
    monkeypatch.setattr(execute_node.shutil, "rmtree", failed_removal)
    monkeypatch.setattr(execute_node, "_populate_ref_cache", forbidden_repopulation)
    state = AgentState(
        issue_url="https://github.com/acme/widget/issues/1",
        owner="acme",
        repo="widget",
        repo_ref=ref,
        trace_id="abc123def456",
    )

    with pytest.raises(RuntimeError, match="checked cache removal failed"):
        await execute_node._git_clone_locked(state, ref, cache, str(work))

    assert removal_called is True


async def test_reset_failure_and_dirty_status_fail_closed(monkeypatch, tmp_path):
    work = tmp_path / "work"
    (work / ".git").mkdir(parents=True)
    ref = "a" * 40

    async def failed_reset(args, timeout, cwd=None):
        if "reset" in args:
            return execute_node._ProcResult(1, "", "reset failed")
        return execute_node._ProcResult(0, "", "")

    monkeypatch.setattr(execute_node, "_run_git_async", failed_reset)
    with pytest.raises(RuntimeError, match="reset failed"):
        await execute_node._reset_work_tree_async(str(work), ref)

    async def dirty_tree(args, timeout, cwd=None):
        if "rev-parse" in args:
            return execute_node._ProcResult(0, ref + "\n", "")
        if "status" in args:
            return execute_node._ProcResult(0, "?? hostile-build-output\n", "")
        return execute_node._ProcResult(0, "", "")

    monkeypatch.setattr(execute_node, "_run_git_async", dirty_tree)
    with pytest.raises(RuntimeError, match="not clean"):
        await execute_node._reset_work_tree_async(str(work), ref)


async def test_reset_removes_tracked_untracked_and_ignored_build_outputs(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    _git(work, "init", "-q")
    _git(work, "config", "user.email", "test@example.invalid")
    _git(work, "config", "user.name", "Test")
    (work / ".gitignore").write_text("build/\n", encoding="utf-8")
    (work / "tracked.py").write_text("value = 1\n", encoding="utf-8")
    _git(work, "add", "--all")
    _git(work, "commit", "-qm", "base")
    ref = _git(work, "rev-parse", "HEAD")
    (work / "tracked.py").write_text("value = 2\n", encoding="utf-8")
    (work / "untracked.txt").write_text("sentinel\n", encoding="utf-8")
    (work / "build").mkdir()
    (work / "build" / "hostile.py").write_text("raise SystemExit\n", encoding="utf-8")

    await execute_node._reset_work_tree_async(str(work), ref)

    assert (work / "tracked.py").read_text(encoding="utf-8") == "value = 1\n"
    assert not (work / "untracked.txt").exists()
    assert not (work / "build").exists()
    assert _git(work, "status", "--porcelain=v1", "--ignored=matching") == ""


async def test_exact_ref_transport_credentials_are_command_scoped(
    monkeypatch, tmp_path
):
    cache = tmp_path / "cache"
    ref = "a" * 40
    token = "synthetic-command-scope-token"
    safe_url = "https://github.com/acme/widget.git"
    authenticated_url = (
        f"https://x-access-token:{token}@github.com/acme/widget.git"
    )
    commands: list[list[str]] = []

    async def checked(args, timeout):
        commands.append(args)
        return execute_node._ProcResult(0, "", "")

    monkeypatch.setenv("GITHUB_TOKEN", token)
    monkeypatch.setattr(execute_node, "_checked_git", checked)
    state = SimpleNamespace(owner="acme", repo="widget")

    await execute_node._populate_ref_cache(state, cache, ref, retry_delays=())

    assert [
        "git",
        "-C",
        str(cache),
        "remote",
        "add",
        "origin",
        safe_url,
    ] in commands
    fetch = next(command for command in commands if "fetch" in command)
    assert fetch[:3] == [
        "git",
        "-c",
        f"url.{authenticated_url}.insteadOf={safe_url}",
    ]
    assert all("set-url" not in command for command in commands)


async def test_live_clone_transport_credentials_are_command_scoped(
    monkeypatch, tmp_path
):
    cache = tmp_path / "cache"
    token = "synthetic-live-clone-token"
    safe_url = "https://github.com/acme/widget.git"
    authenticated_url = (
        f"https://x-access-token:{token}@github.com/acme/widget.git"
    )
    commands: list[list[str]] = []

    async def run_git(args, timeout, cwd=None):
        commands.append(args)
        return execute_node._ProcResult(0, "", "")

    monkeypatch.setenv("GITHUB_TOKEN", token)
    monkeypatch.setattr(execute_node, "_run_git_async", run_git)
    state = SimpleNamespace(owner="acme", repo="widget")

    await execute_node._populate_live_cache(state, cache)

    assert commands == [
        [
            "git",
            "-c",
            f"url.{authenticated_url}.insteadOf={safe_url}",
            "clone",
            "--depth",
            "1",
            "--single-branch",
            safe_url,
            str(cache),
        ]
    ]


@pytest.mark.parametrize("operation", ["exact_ref", "live_clone", "live_refresh"])
async def test_completed_cache_git_metadata_contains_no_transport_credentials(
    monkeypatch, tmp_path, operation
):
    marker = "synthetic-transport-credential"
    source = tmp_path / marker / "source"
    source.mkdir(parents=True)
    _git(source, "init", "-q")
    _git(source, "config", "user.email", "test@example.invalid")
    _git(source, "config", "user.name", "Test")
    (source / "README.md").write_text("safe cache\n", encoding="utf-8")
    _git(source, "add", "--all")
    _git(source, "commit", "-qm", "base")
    ref = _git(source, "rev-parse", "HEAD")
    authenticated_url = source.resolve().as_uri()
    safe_url = "https://github.example.invalid/acme/widget.git"
    cache = tmp_path / "cache"
    state = SimpleNamespace(owner="acme", repo="widget")

    def repo_url(_state, *, include_token):
        return authenticated_url if include_token else safe_url

    monkeypatch.setattr(execute_node, "_repo_url", repo_url)
    if operation == "exact_ref":
        await execute_node._populate_ref_cache(
            state,
            cache,
            ref,
            retry_delays=(),
        )
    elif operation == "live_clone":
        await execute_node._populate_live_cache(state, cache)
    else:
        cache.mkdir()
        _git(cache, "init", "-q")
        _git(cache, "remote", "add", "origin", safe_url)
        await execute_node._refresh_live_cache(state, cache)

    assert _git(
        cache,
        "config",
        "--local",
        "--get-all",
        "remote.origin.url",
    ) == safe_url
    assert marker.encode() not in _bounded_git_metadata(cache)


async def test_tokenized_legacy_cache_is_removed_and_rebuilt(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    work = tmp_path / "work"
    (cache / ".git").mkdir(parents=True)
    (work / ".git").mkdir(parents=True)
    sentinel = cache / "legacy-tokenized-cache"
    sentinel.write_text("legacy", encoding="utf-8")
    ref = "a" * 40
    safe_url = "https://github.com/acme/widget.git"
    populate_calls = 0
    origin_queries: list[list[str]] = []

    async def run_git(args, timeout, cwd=None):
        if args[-4:] == [
            "config",
            "--local",
            "--get-all",
            "remote.origin.url",
        ]:
            origin_queries.append(args)
            return execute_node._ProcResult(
                0,
                "https://x-access-token:synthetic-legacy-token@github.com/"
                "acme/widget.git\n",
                "",
            )
        return execute_node._ProcResult(0, "", "")

    async def healthy(_path):
        return True

    async def head(_path):
        return ref

    async def populate(_state, cache_path, _repo_ref, **_kwargs):
        nonlocal populate_calls
        populate_calls += 1
        (cache_path / ".git").mkdir(parents=True)
        (cache_path / "rebuilt-origin").write_text(safe_url, encoding="utf-8")

    async def forbidden_clone(_cache_path, _target):
        raise AssertionError("healthy mutable worktree should be reset, not cloned")

    async def reset(_target, _ref):
        return None

    monkeypatch.setattr(execute_node, "_run_git_async", run_git)
    monkeypatch.setattr(execute_node, "_worktree_is_healthy", healthy)
    monkeypatch.setattr(execute_node, "_worktree_head", head)
    monkeypatch.setattr(execute_node, "_populate_ref_cache", populate)
    monkeypatch.setattr(execute_node, "_clone_local_repo_async", forbidden_clone)
    monkeypatch.setattr(execute_node, "_reset_work_tree_async", reset)
    monkeypatch.setattr(
        execute_node, "_verify_clean_worktree", lambda *_args: asyncio.sleep(0)
    )
    state = AgentState(
        issue_url="https://github.com/acme/widget/issues/1",
        owner="acme",
        repo="widget",
        repo_ref=ref,
        trace_id="legacy-cache-test",
    )

    await execute_node._git_clone_locked(state, ref, cache, str(work))

    assert populate_calls == 1
    assert origin_queries == [
        [
            "git",
            "-C",
            str(cache),
            "config",
            "--local",
            "--get-all",
            "remote.origin.url",
        ]
    ]
    assert not sentinel.exists()
    assert (cache / "rebuilt-origin").read_text(encoding="utf-8") == safe_url


async def test_live_refresh_cancellation_removes_mutated_cache(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    (cache / ".git").mkdir(parents=True)

    async def checked(args, timeout):
        if "fetch" in args:
            raise asyncio.CancelledError("cancel live refresh")
        return execute_node._ProcResult(0, "", "")

    monkeypatch.setattr(execute_node, "_checked_git", checked)
    state = SimpleNamespace(owner="acme", repo="widget")

    with pytest.raises(asyncio.CancelledError, match="cancel live refresh"):
        await execute_node._refresh_live_cache(state, cache)

    assert not cache.exists()


async def test_exact_ref_cancellation_removes_partial_cache(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    ref = "a" * 40

    async def checked(args, timeout):
        if args[:2] == ["git", "init"]:
            (cache / ".git").mkdir(parents=True)
        if "fetch" in args:
            raise asyncio.CancelledError("cancel exact ref")
        return execute_node._ProcResult(0, "", "")

    monkeypatch.setattr(execute_node, "_checked_git", checked)
    state = SimpleNamespace(owner="acme", repo="widget")

    with pytest.raises(asyncio.CancelledError, match="cancel exact ref"):
        await execute_node._populate_ref_cache(state, cache, ref, retry_delays=())

    assert not cache.exists()


async def test_live_clone_cancellation_removes_partial_cache(monkeypatch, tmp_path):
    cache = tmp_path / "cache"

    async def run_git(_args, timeout, cwd=None):
        (cache / ".git").mkdir(parents=True)
        raise asyncio.CancelledError("cancel live clone")

    monkeypatch.setattr(execute_node, "_run_git_async", run_git)
    state = SimpleNamespace(owner="acme", repo="widget")

    with pytest.raises(asyncio.CancelledError, match="cancel live clone"):
        await execute_node._populate_live_cache(state, cache)

    assert not cache.exists()


async def test_local_clone_cancellation_removes_worktree_but_preserves_cache(
    monkeypatch, tmp_path
):
    cache = tmp_path / "cache"
    work = tmp_path / "work"
    (cache / ".git").mkdir(parents=True)
    verified_cache = cache / "verified-cache"
    verified_cache.write_text("immutable", encoding="utf-8")
    ref = "a" * 40
    safe_url = "https://github.com/acme/widget.git"

    async def run_git(args, timeout, cwd=None):
        if args[-4:] == [
            "config",
            "--local",
            "--get-all",
            "remote.origin.url",
        ]:
            return execute_node._ProcResult(0, safe_url + "\n", "")
        return execute_node._ProcResult(0, "", "")

    async def healthy(_path):
        return True

    async def head(_path):
        return ref

    async def cancelled_clone(_cache_path, target):
        (Path(target) / ".git").mkdir(parents=True)
        raise asyncio.CancelledError("cancel local clone")

    monkeypatch.setattr(execute_node, "_run_git_async", run_git)
    monkeypatch.setattr(execute_node, "_worktree_is_healthy", healthy)
    monkeypatch.setattr(execute_node, "_worktree_head", head)
    monkeypatch.setattr(execute_node, "_clone_local_repo_async", cancelled_clone)
    state = AgentState(
        issue_url="https://github.com/acme/widget/issues/1",
        owner="acme",
        repo="widget",
        repo_ref=ref,
        trace_id="local-clone-cancel-test",
    )

    with pytest.raises(asyncio.CancelledError, match="cancel local clone"):
        await execute_node._git_clone_locked(state, ref, cache, str(work))

    assert verified_cache.read_text(encoding="utf-8") == "immutable"
    assert not work.exists()


async def test_local_reset_cancellation_removes_worktree_but_preserves_cache(
    monkeypatch, tmp_path
):
    cache = tmp_path / "cache"
    work = tmp_path / "work"
    (cache / ".git").mkdir(parents=True)
    (work / ".git").mkdir(parents=True)
    verified_cache = cache / "verified-cache"
    verified_cache.write_text("immutable", encoding="utf-8")
    ref = "a" * 40

    async def healthy(_path):
        return True

    async def head(_path):
        return ref

    async def safe_origin(_state, _cache_path):
        return True

    async def cancelled_reset(target, _ref):
        (Path(target) / "partial-reset").write_text("partial", encoding="utf-8")
        raise asyncio.CancelledError("cancel local reset")

    monkeypatch.setattr(execute_node, "_worktree_is_healthy", healthy)
    monkeypatch.setattr(execute_node, "_worktree_head", head)
    monkeypatch.setattr(execute_node, "_cache_origin_matches", safe_origin)
    monkeypatch.setattr(execute_node, "_reset_work_tree_async", cancelled_reset)
    state = AgentState(
        issue_url="https://github.com/acme/widget/issues/1",
        owner="acme",
        repo="widget",
        repo_ref=ref,
        trace_id="local-reset-cancel-test",
    )

    with pytest.raises(asyncio.CancelledError, match="cancel local reset"):
        await execute_node._git_clone_locked(state, ref, cache, str(work))

    assert verified_cache.read_text(encoding="utf-8") == "immutable"
    assert not work.exists()


async def test_cleanup_failure_is_chained_from_original_cancellation(
    monkeypatch, tmp_path
):
    cache = tmp_path / "cache"

    async def run_git(_args, timeout, cwd=None):
        (cache / ".git").mkdir(parents=True)
        raise asyncio.CancelledError("original cancellation")

    def failed_cleanup(_path):
        raise RuntimeError("injected cleanup failure")

    monkeypatch.setattr(execute_node, "_run_git_async", run_git)
    monkeypatch.setattr(execute_node, "_checked_remove_tree", failed_cleanup)
    state = SimpleNamespace(owner="acme", repo="widget")

    with pytest.raises(RuntimeError, match="injected cleanup failure") as caught:
        await execute_node._populate_live_cache(state, cache)

    assert isinstance(caught.value.__cause__, asyncio.CancelledError)
    assert str(caught.value.__cause__) == "original cancellation"


async def test_live_clone_timeout_cleanup_failure_preserves_timeout(
    monkeypatch, tmp_path
):
    cache = tmp_path / "cache"
    original_timeout = asyncio.TimeoutError("synthetic clone timeout")
    clone_attempts = 0
    cleanup_attempts = 0

    async def run_git(_args, timeout, cwd=None):
        nonlocal clone_attempts
        clone_attempts += 1
        (cache / ".git").mkdir(parents=True, exist_ok=True)
        raise original_timeout

    def failed_cleanup(_path):
        nonlocal cleanup_attempts
        cleanup_attempts += 1
        raise RuntimeError("injected timeout cleanup failure")

    monkeypatch.setattr(execute_node, "_run_git_async", run_git)
    monkeypatch.setattr(execute_node, "_checked_remove_tree", failed_cleanup)
    state = SimpleNamespace(owner="acme", repo="widget")

    with pytest.raises(
        RuntimeError,
        match="injected timeout cleanup failure",
    ) as caught:
        await execute_node._populate_live_cache(state, cache)

    assert caught.value.__cause__ is original_timeout
    assert clone_attempts == 1
    assert cleanup_attempts == 1


async def test_live_clone_result_cleanup_failure_preserves_clone_failure(
    monkeypatch, tmp_path
):
    cache = tmp_path / "cache"
    clone_attempts = 0
    cleanup_attempts = 0

    async def run_git(_args, timeout, cwd=None):
        nonlocal clone_attempts
        clone_attempts += 1
        (cache / ".git").mkdir(parents=True, exist_ok=True)
        return execute_node._ProcResult(
            128,
            "",
            "fatal: synthetic clone transport failure",
        )

    def failed_cleanup(_path):
        nonlocal cleanup_attempts
        cleanup_attempts += 1
        raise RuntimeError("injected result cleanup failure")

    monkeypatch.setattr(execute_node, "_run_git_async", run_git)
    monkeypatch.setattr(execute_node, "_checked_remove_tree", failed_cleanup)
    state = SimpleNamespace(owner="acme", repo="widget")

    with pytest.raises(
        RuntimeError,
        match="injected result cleanup failure",
    ) as caught:
        await execute_node._populate_live_cache(state, cache)

    assert isinstance(caught.value.__cause__, RuntimeError)
    assert str(caught.value.__cause__) == "fatal: synthetic clone transport failure"
    assert clone_attempts == 1
    assert cleanup_attempts == 1


def test_workflow_never_enables_unsafe_host_execution():
    workflow = Path(".github/workflows/swe-bench-oci-eval.yml").read_text(
        encoding="utf-8"
    )

    assert "REPOPILOT_UNSAFE_ALLOW_HOST_EXECUTION" not in workflow
