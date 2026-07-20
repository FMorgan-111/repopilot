from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import pytest

from src.safe_subprocess import (
    NetworkIsolation,
    NetworkIsolationUnavailable,
    ProcessOutputLimitExceeded,
    ProcessTimeoutError,
    network_isolation,
    run_bounded_process,
    SandboxPaths,
    build_darwin_profile,
    build_linux_isolation,
    minimal_subprocess_env,
    trusted_executable,
    _probe_network_isolation,
)


def test_child_environment_is_allowlisted_without_api_or_cloud_credentials(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("LLM_API_KEY", "sentinel-primary-credential")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "sentinel-cloud-credential")
    monkeypatch.setenv("GITHUB_TOKEN", "sentinel-github-credential")
    monkeypatch.setattr(
        "src.safe_subprocess.network_isolation",
        lambda _sandbox: NetworkIsolation(capability="test", argv_prefix=()),
    )
    names = ["LLM_API_KEY", "AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN"]
    script = (
        "import os; names=" + repr(names) + "; "
        "print('clean' if not any(name in os.environ for name in names) else 'leaked')"
    )

    sandbox = SandboxPaths.create(tmp_path / "bundle")
    result = run_bounded_process(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        isolate_network=True,
        sandbox=sandbox,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "clean"


def test_environment_overrides_cannot_reintroduce_credentials():
    with pytest.raises(ValueError):
        minimal_subprocess_env({"LLM_API_KEY": "synthetic-value"})


def test_network_isolation_fails_closed_without_supported_launcher(tmp_path, monkeypatch):
    monkeypatch.setattr("src.safe_subprocess.platform.system", lambda: "Plan9")

    with pytest.raises(NetworkIsolationUnavailable):
        network_isolation(SandboxPaths.create(tmp_path / "bundle"))


def test_network_isolation_exposes_explicit_darwin_capability(monkeypatch):
    monkeypatch.setattr("src.safe_subprocess.platform.system", lambda: "Darwin")
    monkeypatch.setattr(
        "src.safe_subprocess.trusted_executable",
        lambda name, required=False: "/usr/bin/sandbox-exec" if name == "sandbox-exec" else None,
    )
    monkeypatch.setattr("src.safe_subprocess._probe_network_isolation", lambda *_: True)

    with tempfile.TemporaryDirectory() as directory:
        isolation = network_isolation(SandboxPaths.create(Path(directory) / "bundle"))

    assert isolation.capability == "darwin-sandbox-exec-deny-default"
    assert isolation.argv_prefix[0] == "/usr/bin/sandbox-exec"
    assert "deny network" in isolation.argv_prefix[2]


def test_network_isolation_rejects_launcher_that_cannot_apply_policy(monkeypatch):
    monkeypatch.setattr("src.safe_subprocess.platform.system", lambda: "Darwin")
    monkeypatch.setattr(
        "src.safe_subprocess.trusted_executable",
        lambda name, required=False: "/usr/bin/sandbox-exec" if name == "sandbox-exec" else None,
    )
    monkeypatch.setattr("src.safe_subprocess._probe_network_isolation", lambda *_: False)

    with tempfile.TemporaryDirectory() as directory, pytest.raises(NetworkIsolationUnavailable):
        network_isolation(SandboxPaths.create(Path(directory) / "bundle"))


def test_negative_probe_rejects_launcher_that_exposes_host_file_and_socket(tmp_path):
    sandbox = SandboxPaths.create(tmp_path / "bundle")
    unisolated = NetworkIsolation(capability="fake", argv_prefix=())

    assert _probe_network_isolation(unisolated, sandbox) is False


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


def test_linux_isolation_spec_has_no_host_root_or_unshare_only_fallback(tmp_path):
    sandbox = SandboxPaths.create(tmp_path / "bundle")

    isolation = build_linux_isolation(
        sandbox,
        launcher="/usr/bin/bwrap",
        runtime_mounts=(Path("/usr"),),
        python_path=(),
    )

    argv = isolation.argv_prefix
    assert "--unshare-user" in argv
    assert "--unshare-pid" in argv
    assert "--unshare-net" in argv
    assert "--proc" in argv
    assert not any(argv[index : index + 3] == ("--bind", "/", "/") for index in range(len(argv) - 2))
    assert "unshare" not in Path(argv[0]).name


def test_darwin_profile_is_deny_default_and_scoped_to_runtime_and_bundle(tmp_path):
    sandbox = SandboxPaths.create(tmp_path / "bundle")

    profile = build_darwin_profile(
        sandbox,
        runtime_mounts=(Path("/usr"), Path("/System/Library")),
        python_path=(),
    )

    assert "(deny default)" in profile
    assert "(allow default)" not in profile
    assert "(deny network*)" in profile
    assert str(sandbox.workspace) in profile
    assert str(Path.home()) not in profile
    assert "/proc" not in profile


def test_trusted_launcher_ignores_path_shadow(tmp_path, monkeypatch):
    shadow = tmp_path / "sandbox-exec"
    shadow.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    shadow.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))

    resolved = trusted_executable("sandbox-exec", required=False)

    assert resolved is None or resolved != str(shadow)
