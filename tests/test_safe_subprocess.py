from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

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


def _wait_for_file(path: Path, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.02)
    return path.exists()


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
