from __future__ import annotations

import asyncio
import io
import json
import sys
import threading
import time
from pathlib import Path

import pytest

import src.safe_subprocess as safe_subprocess
from src.safe_subprocess import (
    BoundedProcessResult,
    IsolationCleanupError,
    NetworkIsolationUnavailable,
    ProcessOutputLimitExceeded,
    ProcessTimeoutError,
    SandboxPaths,
    build_oci_command,
    minimal_subprocess_env,
    run_bounded_process,
    run_oci_process,
    tool_sandbox_config_from_env,
    trusted_executable,
)
from src.state import ToolSandboxConfig

_IMAGE = "registry.example/repopilot-tests@sha256:" + "1" * 64


class _InjectedBaseException(BaseException):
    pass


def _config() -> ToolSandboxConfig:
    return ToolSandboxConfig(
        backend="docker",
        image=_IMAGE,
        python_executable="/usr/bin/python3",
        project_executables={"npm": "/usr/bin/npm"},
    )


def _inspect_result(argv, sandbox: SandboxPaths, *, exists: bool):
    if not exists:
        return BoundedProcessResult(list(argv), 1, "", "No such container")
    payload = [
        {
            "Config": {"User": "65532:65532", "Image": _IMAGE},
            "HostConfig": {
                "NetworkMode": "none",
                "ReadonlyRootfs": True,
                "Privileged": False,
                "CapDrop": ["ALL"],
                "SecurityOpt": ["no-new-privileges"],
                "PidsLimit": 128,
                "Memory": 1_073_741_824,
                "NanoCpus": 1_000_000_000,
                "Tmpfs": {
                    "/tmp": "rw,noexec,nosuid,nodev,size=256m,mode=1777",
                    "/home/repopilot": (
                        "rw,noexec,nosuid,nodev,size=64m,mode=0700,uid=65532,gid=65532"
                    ),
                },
            },
            "Mounts": [
                {
                    "Type": "bind",
                    "Source": str(sandbox.workspace),
                    "Destination": "/workspace",
                    "RW": False,
                }
            ],
        }
    ]
    return BoundedProcessResult(list(argv), 0, json.dumps(payload), "")


def test_child_environment_is_allowlisted_without_api_or_cloud_credentials(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("LLM_API_KEY", "sentinel-primary-credential")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "sentinel-cloud-credential")
    monkeypatch.setenv("GITHUB_TOKEN", "sentinel-github-credential")
    names = ["LLM_API_KEY", "AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN"]
    script = (
        "import os; names=" + repr(names) + "; "
        "print('clean' if not any(name in os.environ for name in names) else 'leaked')"
    )

    result = run_bounded_process(
        [sys.executable, "-c", script],
        cwd=tmp_path,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "clean"


def test_environment_overrides_cannot_reintroduce_credentials():
    with pytest.raises(ValueError):
        minimal_subprocess_env({"LLM_API_KEY": "synthetic-value"})


def test_operator_oci_configuration_is_validated_and_persistable(monkeypatch):
    monkeypatch.setenv("REPOPILOT_TOOL_OCI_BACKEND", "docker")
    monkeypatch.setenv("REPOPILOT_TOOL_OCI_IMAGE", _IMAGE)
    monkeypatch.setenv("REPOPILOT_TOOL_PYTHON_EXECUTABLE", "/opt/python/bin/python3")
    monkeypatch.setenv(
        "REPOPILOT_TOOL_PROJECT_EXECUTABLES", '{"npm":"/usr/bin/npm"}'
    )

    config = tool_sandbox_config_from_env()

    assert config is not None
    assert config.backend == "docker"
    assert config.image == _IMAGE
    assert config.python_executable == "/opt/python/bin/python3"
    assert config.project_executables == (("npm", "/usr/bin/npm"),)


def test_operator_oci_configuration_reads_bounded_resource_limits(monkeypatch):
    monkeypatch.setenv("REPOPILOT_TOOL_OCI_BACKEND", "docker")
    monkeypatch.setenv("REPOPILOT_TOOL_OCI_IMAGE", _IMAGE)
    monkeypatch.setenv("REPOPILOT_TOOL_MEMORY", "4g")
    monkeypatch.setenv("REPOPILOT_TOOL_CPUS", "2.0")
    monkeypatch.setenv("REPOPILOT_TOOL_PIDS_LIMIT", "256")

    config = tool_sandbox_config_from_env()

    assert config is not None
    assert config.memory == "4g"
    assert config.cpus == 2.0
    assert config.pids_limit == 256


def test_partial_operator_oci_configuration_fails_closed(monkeypatch):
    monkeypatch.setenv("REPOPILOT_TOOL_OCI_BACKEND", "docker")
    monkeypatch.delenv("REPOPILOT_TOOL_OCI_IMAGE", raising=False)

    with pytest.raises(ValueError, match="both OCI"):
        tool_sandbox_config_from_env()


def test_strict_subprocess_decode_rejects_non_utf8_plumbing(tmp_path):
    with pytest.raises(UnicodeDecodeError):
        run_bounded_process(
            [sys.executable, "-c", "import os; os.write(1, b'bad-\\xff')"],
            cwd=tmp_path,
            decode_errors="strict",
        )


def test_oci_command_is_immutable_and_mounts_only_disposable_workspace(tmp_path):
    sandbox = SandboxPaths.create(tmp_path / "bundle")
    cidfile = sandbox.root / "test.cid"

    argv = build_oci_command(
        _config(),
        sandbox,
        ["/usr/bin/python3", "-P", "-m", "pytest", "tests/test_widget.py"],
        backend="/usr/bin/docker",
        name="repopilot-test-fixed",
        cidfile=cidfile,
    )

    joined = " ".join(argv)
    assert argv[:2] == ["/usr/bin/docker", "create"]
    for option in (
        "--pull=never",
        "--network=none",
        "--ipc=private",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--user=65532:65532",
        "--pids-limit=128",
        "--memory=1g",
        "--cpus=1.0",
    ):
        assert option in argv
    mounts = [token for token in argv if token.startswith("--mount=")]
    assert mounts == [
        f"--mount=type=bind,src={sandbox.workspace},dst=/workspace,ro"
    ]
    assert str(Path.home()) not in joined
    assert "/usr/local" not in joined
    assert "docker.sock" not in joined
    assert "--pid=host" not in argv
    assert "--ipc=host" not in argv
    assert "--privileged" not in argv
    assert "--entrypoint=/usr/bin/python3" in argv
    assert _IMAGE in argv
    assert "-P" in argv
    assert any(token.startswith("--tmpfs=/tmp:") for token in argv)
    assert any(token.startswith("--tmpfs=/home/repopilot:") for token in argv)


def test_oci_command_rejects_forged_workspace_bind(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    forged = SandboxPaths(root=root, workspace=tmp_path)

    with pytest.raises(ValueError, match="exact disposable"):
        build_oci_command(
            _config(),
            forged,
            ["/usr/bin/python3", "-P", "-m", "pytest", "tests/test_widget.py"],
            backend="/usr/bin/docker",
            name="repopilot-test-fixed",
            cidfile=root / "test.cid",
        )


@pytest.mark.parametrize("outcome", ["success", "timeout", "overflow"])
def test_oci_runner_probes_python_and_pytest_and_forces_cleanup(
    tmp_path, monkeypatch, outcome
):
    sandbox = SandboxPaths.create(tmp_path / "bundle")
    calls: list[list[str]] = []
    starts = 0
    active: set[str] = set()

    def fake_bounded(argv, **kwargs):
        nonlocal starts
        calls.append(list(argv))
        if argv[1] == "rm":
            active.discard(argv[-1])
            return BoundedProcessResult(list(argv), 0, "", "")
        if argv[1] == "inspect":
            return _inspect_result(argv, sandbox, exists=argv[-1] in active)
        if argv[1] == "create":
            name = next(token for token in argv if token.startswith("--name="))[7:]
            active.add(name)
            return BoundedProcessResult(list(argv), 0, "created", "")
        if argv[1] == "start":
            starts += 1
            if starts == 1:
                return BoundedProcessResult(list(argv), 0, "pytest 9.0.0\n", "")
            if outcome == "timeout":
                raise ProcessTimeoutError("", "")
            if outcome == "overflow":
                raise ProcessOutputLimitExceeded("", "")
        return BoundedProcessResult(list(argv), 0, "passed", "")

    monkeypatch.setattr(
        "src.safe_subprocess.trusted_executable",
        lambda name, required=True: "/usr/bin/docker" if name == "docker" else None,
    )
    monkeypatch.setattr("src.safe_subprocess.run_bounded_process", fake_bounded)

    if outcome == "success":
        result = run_oci_process(
            ["/usr/bin/python3", "-P", "-m", "pytest", "tests/test_widget.py"],
            sandbox=sandbox,
            config=_config(),
        )
        assert result.returncode == 0
    else:
        expected = (
            ProcessTimeoutError if outcome == "timeout" else ProcessOutputLimitExceeded
        )
        with pytest.raises(expected):
            run_oci_process(
                ["/usr/bin/python3", "-P", "-m", "pytest", "tests/test_widget.py"],
                sandbox=sandbox,
                config=_config(),
            )

    create_calls = [argv for argv in calls if argv[1] == "create"]
    cleanup_calls = [argv for argv in calls if argv[1:3] == ["rm", "-f"]]
    inspect_calls = [argv for argv in calls if argv[1] == "inspect"]
    assert len(create_calls) == 2
    assert len(cleanup_calls) == 2
    assert len(inspect_calls) == 4
    assert all("--pull=never" in argv for argv in create_calls)
    run_names = {
        next(token for token in argv if token.startswith("--name="))[7:]
        for argv in create_calls
    }
    assert {call[-1] for call in cleanup_calls} == run_names
    probe = " ".join(create_calls[0])
    assert "/usr/bin/python3" in probe
    assert " -P " in f" {probe} "
    assert " -m pytest --version" in probe
    assert all(call[-1].startswith("repopilot-") for call in cleanup_calls)


def test_oci_runner_fails_closed_without_backend_or_successful_probe(
    tmp_path, monkeypatch
):
    sandbox = SandboxPaths.create(tmp_path / "bundle")
    monkeypatch.setattr(
        "src.safe_subprocess.trusted_executable", lambda *args, **kwargs: None
    )
    with pytest.raises(NetworkIsolationUnavailable):
        run_oci_process(
            ["/usr/bin/python3", "-P", "-m", "pytest", "tests/test_widget.py"],
            sandbox=sandbox,
            config=_config(),
        )


def test_oci_runner_fails_closed_on_bad_probe_and_still_cleans(tmp_path, monkeypatch):
    sandbox = SandboxPaths.create(tmp_path / "bundle")
    calls: list[list[str]] = []
    active: set[str] = set()

    def fake_bounded(argv, **kwargs):
        calls.append(list(argv))
        if argv[1] == "rm":
            active.discard(argv[-1])
            return BoundedProcessResult(list(argv), 0, "", "")
        if argv[1] == "inspect":
            return _inspect_result(argv, sandbox, exists=argv[-1] in active)
        if argv[1] == "create":
            name = next(token for token in argv if token.startswith("--name="))[7:]
            active.add(name)
            return BoundedProcessResult(list(argv), 0, "created", "")
        return BoundedProcessResult(list(argv), 0, "wrong marker", "")

    monkeypatch.setattr(
        "src.safe_subprocess.trusted_executable",
        lambda *args, **kwargs: "/usr/bin/docker",
    )
    monkeypatch.setattr("src.safe_subprocess.run_bounded_process", fake_bounded)

    with pytest.raises(NetworkIsolationUnavailable):
        run_oci_process(
            ["/usr/bin/python3", "-P", "-m", "pytest", "tests/test_widget.py"],
            sandbox=sandbox,
            config=_config(),
        )

    assert [argv[1] for argv in calls] == [
        "create",
        "inspect",
        "start",
        "rm",
        "inspect",
    ]


def test_cleanup_failure_overrides_timeout_and_preserves_recovery_record(
    tmp_path, monkeypatch
):
    sandbox = SandboxPaths.create(tmp_path / "bundle")
    active: set[str] = set()

    def fake_bounded(argv, **kwargs):
        if argv[1] == "create":
            name = next(token for token in argv if token.startswith("--name="))[7:]
            active.add(name)
            return BoundedProcessResult(list(argv), 0, "created", "")
        if argv[1] == "start":
            raise ProcessTimeoutError("", "")
        if argv[1] == "rm":
            return BoundedProcessResult(list(argv), 1, "", "still running")
        return _inspect_result(argv, sandbox, exists=argv[-1] in active)

    monkeypatch.setattr(
        "src.safe_subprocess.trusted_executable", lambda *args, **kwargs: "/usr/bin/docker"
    )
    monkeypatch.setattr("src.safe_subprocess.run_bounded_process", fake_bounded)

    with pytest.raises(IsolationCleanupError) as caught:
        run_oci_process(
            ["/usr/bin/python3", "-P", "-m", "pytest", "tests/test_widget.py"],
            sandbox=sandbox,
            config=_config(),
        )

    assert isinstance(caught.value.__cause__, ProcessTimeoutError)
    assert caught.value.recovery_path.is_file()
    assert caught.value.container_name in caught.value.recovery_path.read_text()
    caught.value.recovery_path.unlink()


def test_configured_project_runner_gets_a_real_executable_probe(tmp_path, monkeypatch):
    sandbox = SandboxPaths.create(tmp_path / "bundle")
    calls: list[list[str]] = []
    starts = 0
    active: set[str] = set()

    def fake_bounded(argv, **kwargs):
        nonlocal starts
        calls.append(list(argv))
        if argv[1] == "create":
            name = next(token for token in argv if token.startswith("--name="))[7:]
            active.add(name)
            return BoundedProcessResult(list(argv), 0, "created", "")
        if argv[1] == "start":
            starts += 1
            output = "pytest 9\n" if starts == 1 else "10.0.0\n"
            return BoundedProcessResult(list(argv), 0, output, "")
        if argv[1] == "rm":
            active.discard(argv[-1])
            return BoundedProcessResult(list(argv), 0, "", "")
        return _inspect_result(argv, sandbox, exists=argv[-1] in active)

    monkeypatch.setattr(
        "src.safe_subprocess.trusted_executable", lambda *args, **kwargs: "/usr/bin/docker"
    )
    monkeypatch.setattr("src.safe_subprocess.run_bounded_process", fake_bounded)

    result = run_oci_process(
        ["/usr/bin/npm", "test", "--", "tests/test_widget.py"],
        sandbox=sandbox,
        config=_config(),
    )

    assert result.returncode == 0
    creates = [argv for argv in calls if argv[1] == "create"]
    assert len(creates) == 3
    assert any("--entrypoint=/usr/bin/npm" in argv and argv[-1] == "--version" for argv in creates)


def test_project_probe_cancellation_prevents_command_container(tmp_path, monkeypatch):
    sandbox = SandboxPaths.create(tmp_path / "bundle")
    cancellation = threading.Event()
    calls: list[list[str]] = []
    active: set[str] = set()
    cleanup_started = False
    starts = 0

    def fake_bounded(argv, **kwargs):
        nonlocal cleanup_started, starts
        calls.append(list(argv))
        if argv[1] == "create":
            cleanup_started = False
        if argv[1] != "rm" and not cleanup_started:
            assert kwargs["cancellation_event"] is cancellation
        if argv[1] == "create":
            name = next(token for token in argv if token.startswith("--name="))[7:]
            active.add(name)
            return BoundedProcessResult(list(argv), 0, "created", "")
        if argv[1] == "start":
            starts += 1
            if starts == 1:
                return BoundedProcessResult(list(argv), 0, "pytest 9.0.0", "")
            cancellation.set()
            raise safe_subprocess.ProcessCancellationRequested
        if argv[1] == "rm":
            cleanup_started = True
            assert "cancellation_event" not in kwargs
            active.discard(argv[-1])
            return BoundedProcessResult(list(argv), 0, "", "")
        if cleanup_started:
            assert "cancellation_event" not in kwargs
            return BoundedProcessResult(list(argv), 1, "", "No such container")
        return _inspect_result(argv, sandbox, exists=argv[-1] in active)

    monkeypatch.setattr(
        safe_subprocess,
        "trusted_executable",
        lambda *_args, **_kwargs: "/usr/bin/docker",
    )
    monkeypatch.setattr(safe_subprocess, "run_bounded_process", fake_bounded)

    with pytest.raises(safe_subprocess.ProcessCancellationRequested):
        run_oci_process(
            ["/usr/bin/npm", "test", "--", "tests/test_widget.py"],
            sandbox=sandbox,
            config=_config(),
            cancellation_event=cancellation,
        )

    create_calls = [argv for argv in calls if argv[1] == "create"]
    assert len(create_calls) == 2
    assert create_calls[-1][-1] == "--version"


def _wait_for_file(path: Path, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.02)
    return path.exists()


async def _wait_for_thread_event(
    event: threading.Event, timeout: float = 3.0
) -> None:
    deadline = time.monotonic() + timeout
    while not event.is_set():
        if time.monotonic() >= deadline:
            raise AssertionError("thread event was not set before timeout")
        await asyncio.sleep(0.01)


def test_pre_cancelled_process_never_spawns(tmp_path, monkeypatch):
    cancellation = threading.Event()
    cancellation.set()
    monkeypatch.setattr(
        safe_subprocess.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("pre-cancelled process was spawned"),
    )

    with pytest.raises(safe_subprocess.ProcessCancellationRequested):
        run_bounded_process(
            [sys.executable, "-c", "pass"],
            cwd=tmp_path,
            cancellation_event=cancellation,
        )


def test_cancellation_reaps_process_group_and_completes_io_threads(
    tmp_path, monkeypatch
):
    cancellation = threading.Event()
    parent_marker = tmp_path / "cancel-parent"
    child_marker = tmp_path / "cancel-child"
    ready = tmp_path / "cancel-ready"
    reader_completions: list[int] = []
    writer_completed = threading.Event()
    original_reader = safe_subprocess._reader
    original_writer = safe_subprocess._write_input

    def tracked_reader(*args):
        try:
            original_reader(*args)
        finally:
            reader_completions.append(threading.get_ident())

    def tracked_writer(*args):
        try:
            original_writer(*args)
        finally:
            writer_completed.set()

    monkeypatch.setattr(safe_subprocess, "_reader", tracked_reader)
    monkeypatch.setattr(safe_subprocess, "_write_input", tracked_writer)
    child = (
        "import pathlib,signal,sys,time; marker=pathlib.Path(sys.argv[1]); "
        "ready=pathlib.Path(sys.argv[2]); "
        "signal.signal(signal.SIGTERM, lambda *_: (marker.write_text('yes'), sys.exit(0))); "
        "ready.write_text('yes'); time.sleep(60)"
    )
    parent = (
        "import pathlib,signal,subprocess,sys,time; marker=pathlib.Path(sys.argv[1]); "
        "child=sys.argv[2]; child_marker=sys.argv[3]; ready=pathlib.Path(sys.argv[4]); "
        "signal.signal(signal.SIGTERM, lambda *_: (marker.write_text('yes'), sys.exit(0))); "
        "subprocess.Popen([sys.executable, '-c', child, child_marker, str(ready)]); "
        "\nwhile not ready.exists():\n time.sleep(0.01)"
        "\ntime.sleep(60)"
    )
    outcome: list[BaseException] = []

    def invoke() -> None:
        try:
            run_bounded_process(
                [
                    sys.executable,
                    "-c",
                    parent,
                    str(parent_marker),
                    child,
                    str(child_marker),
                    str(ready),
                ],
                cwd=tmp_path,
                input_text="x" * 8_000_000,
                timeout=10,
                cancellation_event=cancellation,
            )
        except BaseException as exc:
            outcome.append(exc)

    worker = threading.Thread(target=invoke)
    worker.start()
    assert _wait_for_file(ready)
    cancellation.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert len(outcome) == 1
    assert isinstance(outcome[0], safe_subprocess.ProcessCancellationRequested)
    assert _wait_for_file(parent_marker)
    assert _wait_for_file(child_marker)
    assert len(reader_completions) == 2
    assert writer_completed.is_set()


def test_arbitrary_base_exception_still_terminates_and_joins_io(
    tmp_path, monkeypatch
):
    release_io = threading.Event()
    terminated = threading.Event()
    reader_completions: list[int] = []
    writer_completed = threading.Event()

    class FakeProcess:
        pid = 4242
        returncode = None
        stdout = io.BytesIO()
        stderr = io.BytesIO()
        stdin = io.BytesIO()

        def poll(self):
            raise _InjectedBaseException("poll interrupted")

    process = FakeProcess()

    def blocking_reader(*_args):
        release_io.wait(timeout=2)
        reader_completions.append(threading.get_ident())

    def blocking_writer(*_args):
        release_io.wait(timeout=2)
        writer_completed.set()

    def terminate(received):
        assert received is process
        process.returncode = -15
        terminated.set()
        release_io.set()

    monkeypatch.setattr(safe_subprocess.subprocess, "Popen", lambda *_a, **_k: process)
    monkeypatch.setattr(safe_subprocess, "_reader", blocking_reader)
    monkeypatch.setattr(safe_subprocess, "_write_input", blocking_writer)
    monkeypatch.setattr(safe_subprocess, "_terminate_process_group", terminate)

    try:
        with pytest.raises(_InjectedBaseException, match="poll interrupted"):
            run_bounded_process(
                ["/usr/bin/fake"],
                cwd=tmp_path,
                input_text="payload",
            )
        cleanup_observed = (
            terminated.is_set()
            and len(reader_completions) == 2
            and writer_completed.is_set()
        )
    finally:
        release_io.set()

    assert cleanup_observed


def test_partial_reader_start_failure_closes_unowned_pipes_and_reaps(
    tmp_path, monkeypatch
):
    class FakeProcess:
        pid = 4243
        returncode = None
        stdout = io.BytesIO()
        stderr = io.BytesIO()
        stdin = io.BytesIO()

    process = FakeProcess()
    reaped = threading.Event()
    thread_index = 0

    class FakeThread:
        def __init__(self, *, target, args, daemon):
            nonlocal thread_index
            self.index = thread_index
            thread_index += 1
            self.target = target
            self.args = args
            self.daemon = daemon
            self.joined = False

        def start(self):
            if self.index == 1:
                raise _InjectedBaseException("second reader start interrupted")
            self.target(*self.args)

        def join(self):
            self.joined = True

    def terminate(received):
        assert received is process
        received.returncode = -15
        reaped.set()

    monkeypatch.setattr(safe_subprocess.subprocess, "Popen", lambda *_a, **_k: process)
    monkeypatch.setattr(safe_subprocess.threading, "Thread", FakeThread)
    monkeypatch.setattr(safe_subprocess, "_terminate_process_group", terminate)

    with pytest.raises(_InjectedBaseException, match="second reader start interrupted"):
        run_bounded_process(
            ["/usr/bin/fake"],
            cwd=tmp_path,
            input_text="payload",
        )

    assert reaped.is_set()
    assert process.stdout.closed
    assert process.stderr.closed
    assert process.stdin.closed


def test_reader_construction_failure_after_spawn_still_closes_and_reaps(
    tmp_path, monkeypatch
):
    class FakeProcess:
        pid = 4245
        returncode = None
        stdout = io.BytesIO()
        stderr = io.BytesIO()
        stdin = io.BytesIO()

    process = FakeProcess()
    reaped = threading.Event()
    constructions = 0

    class FailingThread:
        def __init__(self, **_kwargs):
            nonlocal constructions
            constructions += 1
            if constructions == 2:
                raise _InjectedBaseException("reader construction interrupted")

    def terminate(received):
        assert received is process
        received.returncode = -15
        reaped.set()

    monkeypatch.setattr(safe_subprocess.subprocess, "Popen", lambda *_a, **_k: process)
    monkeypatch.setattr(safe_subprocess.threading, "Thread", FailingThread)
    monkeypatch.setattr(safe_subprocess, "_terminate_process_group", terminate)

    with pytest.raises(_InjectedBaseException, match="reader construction interrupted"):
        run_bounded_process(
            ["/usr/bin/fake"],
            cwd=tmp_path,
            input_text="payload",
        )

    assert reaped.is_set()
    assert process.stdout.closed
    assert process.stderr.closed
    assert process.stdin.closed


def test_reader_started_before_start_raises_is_joined_before_return(
    tmp_path, monkeypatch
):
    class FakeProcess:
        pid = 4246
        returncode = None
        stdout = io.BytesIO()
        stderr = io.BytesIO()
        stdin = io.BytesIO()

    process = FakeProcess()
    reaped = threading.Event()
    cleanup_signal = threading.Event()
    reader_entered = threading.Event()
    reader_finished = threading.Event()
    original_start = threading.Thread.start
    interrupted_thread: threading.Thread | None = None

    def delayed_reader(pipe, *_args):
        reader_entered.set()
        cleanup_signal.wait(timeout=2)
        time.sleep(0.2)
        pipe.close()
        reader_finished.set()

    def interrupted_start(self):
        nonlocal interrupted_thread
        target = self._target
        original_start(self)
        if target is delayed_reader and interrupted_thread is None:
            interrupted_thread = self
            assert reader_entered.wait(timeout=2)
            raise _InjectedBaseException("reader start return interrupted")

    def terminate(received):
        assert received is process
        received.returncode = -15
        reaped.set()
        cleanup_signal.set()

    monkeypatch.setattr(safe_subprocess.subprocess, "Popen", lambda *_a, **_k: process)
    monkeypatch.setattr(safe_subprocess, "_reader", delayed_reader)
    monkeypatch.setattr(safe_subprocess.threading.Thread, "start", interrupted_start)
    monkeypatch.setattr(safe_subprocess, "_terminate_process_group", terminate)

    try:
        with pytest.raises(
            _InjectedBaseException, match="reader start return interrupted"
        ):
            run_bounded_process(["/usr/bin/fake"], cwd=tmp_path)
        assert interrupted_thread is not None
        cleanup_completed_before_return = (
            reaped.is_set()
            and process.stdout.closed
            and process.stderr.closed
            and process.stdin.closed
            and reader_finished.is_set()
            and not interrupted_thread.is_alive()
        )
    finally:
        cleanup_signal.set()
        if interrupted_thread is not None:
            interrupted_thread.join(timeout=2)

    assert cleanup_completed_before_return


def test_writer_started_before_start_raises_is_joined_before_return(
    tmp_path, monkeypatch
):
    class FakeProcess:
        pid = 4247
        returncode = None
        stdout = io.BytesIO()
        stderr = io.BytesIO()
        stdin = io.BytesIO()

    process = FakeProcess()
    reaped = threading.Event()
    cleanup_signal = threading.Event()
    writer_entered = threading.Event()
    writer_finished = threading.Event()
    original_start = threading.Thread.start
    interrupted_thread: threading.Thread | None = None

    def delayed_writer(pipe, _value):
        writer_entered.set()
        cleanup_signal.wait(timeout=2)
        time.sleep(0.2)
        pipe.close()
        writer_finished.set()

    def interrupted_start(self):
        nonlocal interrupted_thread
        target = self._target
        original_start(self)
        if target is delayed_writer:
            interrupted_thread = self
            assert writer_entered.wait(timeout=2)
            raise _InjectedBaseException("writer start return interrupted")

    def terminate(received):
        assert received is process
        received.returncode = -15
        reaped.set()
        cleanup_signal.set()

    monkeypatch.setattr(safe_subprocess.subprocess, "Popen", lambda *_a, **_k: process)
    monkeypatch.setattr(safe_subprocess, "_write_input", delayed_writer)
    monkeypatch.setattr(safe_subprocess.threading.Thread, "start", interrupted_start)
    monkeypatch.setattr(safe_subprocess, "_terminate_process_group", terminate)

    try:
        with pytest.raises(
            _InjectedBaseException, match="writer start return interrupted"
        ):
            run_bounded_process(
                ["/usr/bin/fake"], cwd=tmp_path, input_text="payload"
            )
        assert interrupted_thread is not None
        cleanup_completed_before_return = (
            reaped.is_set()
            and process.stdout.closed
            and process.stderr.closed
            and process.stdin.closed
            and writer_finished.is_set()
            and not interrupted_thread.is_alive()
        )
    finally:
        cleanup_signal.set()
        if interrupted_thread is not None:
            interrupted_thread.join(timeout=2)

    assert cleanup_completed_before_return


def test_process_lookup_during_group_signal_still_waits_for_leader(monkeypatch):
    class FakeProcess:
        pid = 4244

        def __init__(self):
            self.wait_calls = 0

        def wait(self, timeout=None):
            self.wait_calls += 1
            return 0

    process = FakeProcess()
    monkeypatch.setattr(
        safe_subprocess.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(ProcessLookupError()),
    )

    safe_subprocess._terminate_process_group(process)

    assert process.wait_calls == 1


def test_oci_removal_runs_for_arbitrary_base_exception(tmp_path, monkeypatch):
    sandbox = SandboxPaths.create(tmp_path / "bundle")
    calls: list[list[str]] = []
    active: set[str] = set()
    starts = 0

    def fake_bounded(argv, **_kwargs):
        nonlocal starts
        calls.append(list(argv))
        if argv[1] == "create":
            name = next(token for token in argv if token.startswith("--name="))[7:]
            active.add(name)
            return BoundedProcessResult(list(argv), 0, "created", "")
        if argv[1] == "start":
            starts += 1
            if starts == 1:
                return BoundedProcessResult(list(argv), 0, "pytest 9.0.0", "")
            raise _InjectedBaseException("start interrupted")
        if argv[1] == "rm":
            active.discard(argv[-1])
            return BoundedProcessResult(list(argv), 0, "", "")
        return _inspect_result(argv, sandbox, exists=argv[-1] in active)

    monkeypatch.setattr(
        safe_subprocess,
        "trusted_executable",
        lambda *_args, **_kwargs: "/usr/bin/docker",
    )
    monkeypatch.setattr(safe_subprocess, "run_bounded_process", fake_bounded)

    with pytest.raises(_InjectedBaseException, match="start interrupted"):
        run_oci_process(
            ["/usr/bin/python3", "-P", "-m", "pytest", "tests/test_widget.py"],
            sandbox=sandbox,
            config=_config(),
        )

    assert len([argv for argv in calls if argv[1:3] == ["rm", "-f"]]) == 2


def test_oci_force_removal_never_receives_cancellation_token(tmp_path, monkeypatch):
    sandbox = SandboxPaths.create(tmp_path / "bundle")
    cancellation = threading.Event()
    cleanup_started = False
    starts = 0

    def fake_bounded(argv, **kwargs):
        nonlocal cleanup_started, starts
        if argv[1] == "create":
            cleanup_started = False
        if argv[1] != "rm" and not cleanup_started:
            assert kwargs["cancellation_event"] is cancellation
        if argv[1] == "create":
            name = next(token for token in argv if token.startswith("--name="))[7:]
            return BoundedProcessResult(list(argv), 0, name, "")
        if argv[1] == "start":
            starts += 1
            if starts == 1:
                return BoundedProcessResult(list(argv), 0, "pytest 9.0.0", "")
            cancellation.set()
            raise safe_subprocess.ProcessCancellationRequested
        if argv[1] == "rm":
            cleanup_started = True
            assert "cancellation_event" not in kwargs
            return BoundedProcessResult(list(argv), 0, "", "")
        if cleanup_started:
            assert "cancellation_event" not in kwargs
            return BoundedProcessResult(list(argv), 1, "", "No such container")
        return _inspect_result(argv, sandbox, exists=True)

    monkeypatch.setattr(
        safe_subprocess,
        "trusted_executable",
        lambda *_args, **_kwargs: "/usr/bin/docker",
    )
    monkeypatch.setattr(safe_subprocess, "run_bounded_process", fake_bounded)

    with pytest.raises(safe_subprocess.ProcessCancellationRequested):
        run_oci_process(
            ["/usr/bin/python3", "-P", "-m", "pytest", "tests/test_widget.py"],
            sandbox=sandbox,
            config=_config(),
            cancellation_event=cancellation,
        )


async def test_async_oci_adapter_keeps_event_loop_responsive(tmp_path, monkeypatch):
    sandbox = SandboxPaths.create(tmp_path / "bundle")
    started = threading.Event()
    release = threading.Event()

    def blocking_oci(argv, *, cancellation_event, **_kwargs):
        started.set()
        assert release.wait(timeout=3)
        return BoundedProcessResult(list(argv), 0, "passed", "")

    monkeypatch.setattr(safe_subprocess, "run_oci_process", blocking_oci)
    task = asyncio.create_task(
        safe_subprocess.run_oci_process_async(
            ["/usr/bin/python3", "-m", "pytest"],
            sandbox=sandbox,
            config=_config(),
        )
    )
    await _wait_for_thread_event(started)
    heartbeat = 0
    for _ in range(5):
        await asyncio.sleep(0)
        heartbeat += 1
    release.set()

    result = await task
    assert heartbeat == 5
    assert result.returncode == 0


async def test_async_oci_cancellation_waits_for_worker_cleanup(tmp_path, monkeypatch):
    sandbox = SandboxPaths.create(tmp_path / "bundle")
    started = threading.Event()
    cleanup_started = threading.Event()
    cleanup_release = threading.Event()
    worker_finished = threading.Event()

    def cancellable_oci(argv, *, cancellation_event, **_kwargs):
        started.set()
        assert cancellation_event.wait(timeout=3)
        cleanup_started.set()
        assert cleanup_release.wait(timeout=3)
        worker_finished.set()
        raise safe_subprocess.ProcessCancellationRequested

    monkeypatch.setattr(safe_subprocess, "run_oci_process", cancellable_oci)
    task = asyncio.create_task(
        safe_subprocess.run_oci_process_async(
            ["/usr/bin/python3", "-m", "pytest"],
            sandbox=sandbox,
            config=_config(),
        )
    )
    await _wait_for_thread_event(started)
    task.cancel("original cancellation")
    await _wait_for_thread_event(cleanup_started)
    await asyncio.sleep(0)
    assert not task.done()
    cleanup_release.set()

    with pytest.raises(asyncio.CancelledError) as caught:
        await task
    assert caught.value.args == ("original cancellation",)
    assert worker_finished.is_set()


async def test_async_oci_drain_survives_repeated_cancellation(tmp_path, monkeypatch):
    sandbox = SandboxPaths.create(tmp_path / "bundle")
    started = threading.Event()
    cleanup_started = threading.Event()
    cleanup_release = threading.Event()

    def cancellable_oci(argv, *, cancellation_event, **_kwargs):
        started.set()
        assert cancellation_event.wait(timeout=3)
        cleanup_started.set()
        assert cleanup_release.wait(timeout=3)
        raise safe_subprocess.ProcessCancellationRequested

    monkeypatch.setattr(safe_subprocess, "run_oci_process", cancellable_oci)
    task = asyncio.create_task(
        safe_subprocess.run_oci_process_async(
            ["/usr/bin/python3", "-m", "pytest"],
            sandbox=sandbox,
            config=_config(),
        )
    )
    await _wait_for_thread_event(started)
    task.cancel("first cancellation")
    await _wait_for_thread_event(cleanup_started)
    task.cancel("second cancellation")
    await asyncio.sleep(0)
    assert not task.done()
    cleanup_release.set()

    with pytest.raises(asyncio.CancelledError) as caught:
        await task
    assert caught.value.args == ("first cancellation",)


async def test_async_oci_cleanup_error_is_chained_to_original_cancellation(
    tmp_path, monkeypatch
):
    sandbox = SandboxPaths.create(tmp_path / "bundle")
    started = threading.Event()
    recovery = tmp_path / "recovery.json"

    def failing_cleanup(argv, *, cancellation_event, **_kwargs):
        started.set()
        assert cancellation_event.wait(timeout=3)
        raise IsolationCleanupError("repopilot-test-cleanup", recovery)

    monkeypatch.setattr(safe_subprocess, "run_oci_process", failing_cleanup)
    task = asyncio.create_task(
        safe_subprocess.run_oci_process_async(
            ["/usr/bin/python3", "-m", "pytest"],
            sandbox=sandbox,
            config=_config(),
        )
    )
    await _wait_for_thread_event(started)
    task.cancel("original cancellation")

    with pytest.raises(asyncio.CancelledError) as caught:
        await task
    assert caught.value.args == ("original cancellation",)
    assert isinstance(caught.value.__cause__, IsolationCleanupError)


def test_streaming_output_cap_terminates_entire_process_group(tmp_path):
    parent_marker = tmp_path / "parent-terminated"
    child_marker = tmp_path / "child-terminated"
    child_ready = tmp_path / "child-ready"
    child_script = (
        "import pathlib,signal,time,sys; marker=pathlib.Path(sys.argv[1]); "
        "ready=pathlib.Path(sys.argv[2]); "
        "signal.signal(signal.SIGTERM, lambda *_: (marker.write_text('yes'), sys.exit(0))); "
        "ready.write_text('yes'); "
        "time.sleep(60)"
    )
    script = (
        "import pathlib,signal,subprocess,sys,time; "
        "parent=pathlib.Path(sys.argv[1]); child=sys.argv[2]; child_marker=sys.argv[3]; "
        "ready=pathlib.Path(sys.argv[4]); "
        "signal.signal(signal.SIGTERM, lambda *_: (parent.write_text('yes'), sys.exit(0))); "
        "subprocess.Popen([sys.executable, '-c', child, child_marker, str(ready)]); "
        "\nwhile not ready.exists():\n time.sleep(0.01)"
        "\nwhile True:\n print('x' * 4096, flush=True)"
    )

    with pytest.raises(ProcessOutputLimitExceeded) as caught:
        run_bounded_process(
            [
                sys.executable,
                "-c",
                script,
                str(parent_marker),
                child_script,
                str(child_marker),
                str(child_ready),
            ],
            cwd=tmp_path,
            max_output_bytes=8_192,
            timeout=10,
        )

    assert len(caught.value.stdout.encode("utf-8")) + len(
        caught.value.stderr.encode("utf-8")
    ) <= 8_192
    assert _wait_for_file(parent_marker)
    assert _wait_for_file(child_marker)


def test_timeout_terminates_entire_process_group(tmp_path):
    parent_marker = tmp_path / "timeout-parent"
    child_marker = tmp_path / "timeout-child"
    ready = tmp_path / "timeout-ready"
    child = (
        "import pathlib,signal,sys,time; marker=pathlib.Path(sys.argv[1]); "
        "ready=pathlib.Path(sys.argv[2]); "
        "signal.signal(signal.SIGTERM, lambda *_: (marker.write_text('yes'), sys.exit(0))); "
        "ready.write_text('yes'); time.sleep(60)"
    )
    parent = (
        "import pathlib,signal,subprocess,sys,time; marker=pathlib.Path(sys.argv[1]); "
        "child=sys.argv[2]; child_marker=sys.argv[3]; ready=pathlib.Path(sys.argv[4]); "
        "signal.signal(signal.SIGTERM, lambda *_: (marker.write_text('yes'), sys.exit(0))); "
        "subprocess.Popen([sys.executable, '-c', child, child_marker, str(ready)]); "
        "\nwhile not ready.exists():\n time.sleep(0.01)"
        "\ntime.sleep(60)"
    )

    with pytest.raises(ProcessTimeoutError):
        run_bounded_process(
            [
                sys.executable,
                "-c",
                parent,
                str(parent_marker),
                child,
                str(child_marker),
                str(ready),
            ],
            cwd=tmp_path,
            timeout=0.2,
        )

    assert _wait_for_file(parent_marker)
    assert _wait_for_file(child_marker)


def test_watched_archive_output_is_hard_capped_while_process_runs(tmp_path):
    output = tmp_path / "base.tar"
    script = (
        "import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); "
        "f=p.open('wb'); "
        "\nwhile True:\n f.write(b'x' * 4096); f.flush(); time.sleep(0.001)"
    )

    with pytest.raises(ProcessOutputLimitExceeded):
        run_bounded_process(
            [sys.executable, "-c", script, str(output)],
            cwd=tmp_path,
            timeout=10,
            max_output_bytes=8_000,
            watched_output_path=output,
            max_watched_output_bytes=16_384,
        )

    assert output.stat().st_size <= 24_576


def test_normal_leader_exit_still_terminates_detached_group_child(tmp_path):
    marker = tmp_path / "normal-child-terminated"
    ready = tmp_path / "normal-child-ready"
    child = (
        "import pathlib,signal,sys,time; marker=pathlib.Path(sys.argv[1]); "
        "ready=pathlib.Path(sys.argv[2]); "
        "signal.signal(signal.SIGTERM, lambda *_: (marker.write_text('yes'), sys.exit(0))); "
        "ready.write_text('yes'); time.sleep(60)"
    )
    parent = (
        "import pathlib,subprocess,sys,time; child=sys.argv[1]; marker=sys.argv[2]; "
        "ready=pathlib.Path(sys.argv[3]); "
        "subprocess.Popen([sys.executable, '-c', child, marker, str(ready)]); "
        "\nwhile not ready.exists():\n time.sleep(0.01)"
    )

    result = run_bounded_process(
        [sys.executable, "-c", parent, child, str(marker), str(ready)],
        cwd=tmp_path,
        timeout=5,
    )

    assert result.returncode == 0
    assert _wait_for_file(marker)


def test_trusted_launcher_ignores_path_shadow(tmp_path, monkeypatch):
    shadow = tmp_path / "docker"
    shadow.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    shadow.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))

    resolved = trusted_executable("docker", required=False)

    assert resolved is None or resolved != str(shadow)
