"""Tests for the best-effort editable install and isolated venv added before
pytest runs."""

import subprocess

from src.nodes import execute as execute_node


def test_pip_install_skipped_without_packaging_metadata(tmp_path):
    # Empty dir: no pyproject/setup.* -> install is not attempted.
    record = execute_node._pip_install_editable(str(tmp_path))
    assert record == {"attempted": False, "reason": "no_packaging_metadata"}


def test_pip_install_succeeds_on_first_extras_candidate(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(execute_node.subprocess, "run", fake_run)

    record = execute_node._pip_install_editable(str(tmp_path))

    assert record["attempted"] is True
    assert record["success"] is True
    assert record["command"].endswith(".[test]")
    assert len(calls) == 1  # stopped at first success, didn't try fallbacks


def test_pip_install_falls_back_to_bare_editable(tmp_path, monkeypatch):
    (tmp_path / "setup.py").write_text("from setuptools import setup; setup()\n")
    rc_by_cmd = {}

    def fake_run(cmd, **kwargs):
        # Every extras variant fails; only the bare last candidate succeeds.
        bare = cmd[-1] == "."
        rc = 0 if bare else 1
        rc_by_cmd[tuple(cmd)] = rc
        return subprocess.CompletedProcess(cmd, returncode=rc, stdout="", stderr="boom")

    monkeypatch.setattr(execute_node.subprocess, "run", fake_run)

    record = execute_node._pip_install_editable(str(tmp_path))

    assert record["success"] is True
    assert record["command"].endswith("-e .")
    assert len(rc_by_cmd) == 4


def test_pip_install_reports_timeout(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, execute_node.INSTALL_TIMEOUT_SECONDS)

    monkeypatch.setattr(execute_node.subprocess, "run", fake_run)

    record = execute_node._pip_install_editable(str(tmp_path))

    assert record["success"] is False
    assert record["reason"] == "timeout"


def test_pip_install_uses_given_python_exe(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    seen = []

    def fake_run(cmd, **kwargs):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(execute_node.subprocess, "run", fake_run)

    # The install command must invoke the passed interpreter, not system python3.
    record = execute_node._pip_install_editable(
        str(tmp_path), python_exe="/venv/bin/python"
    )

    assert record["success"] is True
    assert seen[0][0] == "/venv/bin/python"


def test_ensure_pytest_available_installs_runner_deps_when_missing(monkeypatch):
    seen = []

    def fake_run(cmd, **kwargs):
        seen.append(cmd)
        if cmd[:2] == ["/venv/bin/python", "-c"]:
            return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(execute_node.subprocess, "run", fake_run)

    record = execute_node._ensure_pytest_available("/venv/bin/python")

    assert record["attempted"] is True
    assert record["success"] is True
    assert seen[-1] == [
        "/venv/bin/python",
        "-m",
        "pip",
        "install",
        "pytest",
        "pytest-twisted",
    ]


def test_ensure_pytest_available_skips_install_when_runner_deps_exist(monkeypatch):
    seen = []

    def fake_run(cmd, **kwargs):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(execute_node.subprocess, "run", fake_run)

    record = execute_node._ensure_pytest_available("/venv/bin/python")

    assert record == {"attempted": False, "reason": "pytest_available"}
    assert len(seen) == 1


def test_create_venv_uses_system_site_packages(tmp_path, monkeypatch):
    seen = []

    def fake_run(cmd, **kwargs):
        seen.append(cmd)
        marker = execute_node._venv_python_path(str(tmp_path))
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(execute_node.subprocess, "run", fake_run)

    record = execute_node._create_venv(str(tmp_path))

    assert record["created"] is True
    assert "--system-site-packages" in seen[0]
    assert record["python"] == str(execute_node._venv_python_path(str(tmp_path)))


def test_create_venv_reports_failure(tmp_path, monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(execute_node.subprocess, "run", fake_run)

    record = execute_node._create_venv(str(tmp_path))

    assert record["created"] is False
    assert record["python"] is None


def test_create_venv_falls_back_to_uv_when_stdlib_venv_lacks_ensurepip(
    tmp_path, monkeypatch
):
    seen = []

    def fake_run(cmd, **kwargs):
        seen.append(cmd)
        if cmd[:3] == ["python3", "-m", "venv"]:
            marker = execute_node._venv_python_path(str(tmp_path))
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("")
            return subprocess.CompletedProcess(
                cmd,
                returncode=1,
                stdout="",
                stderr="ensurepip is not available",
            )
        if cmd[:2] == ["uv", "venv"]:
            marker = execute_node._venv_python_path(str(tmp_path))
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("")
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(execute_node.subprocess, "run", fake_run)

    record = execute_node._create_venv(str(tmp_path))

    assert record["created"] is True
    assert record["python"] == str(execute_node._venv_python_path(str(tmp_path)))
    assert record["creator"] == "uv"
    assert seen[0][:3] == ["python3", "-m", "venv"]
    assert seen[1][:2] == ["uv", "venv"]


def test_run_pytest_prefers_venv_interpreter(tmp_path, monkeypatch):
    marker = execute_node._venv_python_path(str(tmp_path))
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("")
    execute_node._venv_ready_marker(str(tmp_path)).write_text("ready\n")
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env", {})
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(execute_node.subprocess, "run", fake_run)

    import asyncio
    asyncio.run(execute_node.run_pytest(str(tmp_path)))

    assert captured["cmd"][0] == str(marker)
    assert str(marker.parent) in captured["env"]["PATH"]
    assert captured["env"]["VIRTUAL_ENV"] == str(
        execute_node._venv_dir_for(str(tmp_path))
    )


def test_run_pytest_rewrites_bare_pytest_command_to_venv_python(
    tmp_path, monkeypatch
):
    marker = execute_node._venv_python_path(str(tmp_path))
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("")
    execute_node._venv_ready_marker(str(tmp_path)).write_text("ready\n")
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(execute_node.subprocess, "run", fake_run)

    import asyncio
    asyncio.run(execute_node.run_pytest(str(tmp_path), "pytest tests/test_one.py"))

    assert captured["cmd"] == [str(marker), "-m", "pytest", "tests/test_one.py"]


def test_run_pytest_ignores_partial_venv_without_ready_marker(tmp_path, monkeypatch):
    marker = execute_node._venv_python_path(str(tmp_path))
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("")
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env", {})
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(execute_node.subprocess, "run", fake_run)

    import asyncio
    asyncio.run(execute_node.run_pytest(str(tmp_path)))

    assert captured["cmd"][:3] == ["python3", "-m", "pytest"]
    assert "VIRTUAL_ENV" not in captured["env"]


def test_run_pytest_falls_back_to_system_interpreter(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(execute_node.subprocess, "run", fake_run)

    import asyncio
    asyncio.run(execute_node.run_pytest(str(tmp_path)))

    assert captured["cmd"][:3] == ["python3", "-m", "pytest"]
