from __future__ import annotations

import os
import re
import shutil
import subprocess

import pytest

from src.safe_subprocess import ProcessTimeoutError, SandboxPaths, run_oci_process
from src.state import ToolSandboxConfig


def test_real_docker_boundary_and_cleanup(tmp_path, monkeypatch):
    if os.getenv("REPOPILOT_RUN_OCI_INTEGRATION") != "1":
        pytest.skip("real OCI boundary is enforced by the dedicated CI job")
    docker = shutil.which("docker")
    if not docker:
        pytest.fail("Docker CLI is required by the OCI integration gate")

    build = tmp_path / "image"
    build.mkdir()
    (build / "Dockerfile").write_text(
        "FROM python:3.12-slim\n"
        "RUN python -m pip install --no-cache-dir 'pytest>=9,<10'\n",
        encoding="utf-8",
    )
    built = subprocess.run(
        [docker, "build", "-q", str(build)],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    ).stdout.strip()
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", built)

    sandbox = SandboxPaths.create(tmp_path / "sandbox")
    boundary = sandbox.workspace / "check_boundary.py"
    boundary.write_text(
        "import os, pathlib, socket\n"
        "assert os.getuid() != 0\n"
        "assert not pathlib.Path('/var/run/docker.sock').exists()\n"
        "for target in (pathlib.Path('/rootfs-write'), pathlib.Path('/workspace/write')):\n"
        "    try:\n"
        "        target.write_text('forbidden')\n"
        "    except OSError:\n"
        "        pass\n"
        "    else:\n"
        "        raise AssertionError(f'writable boundary: {target}')\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=0.2)\n"
        "except OSError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('network was reachable')\n"
        "mounts=[line for line in pathlib.Path('/proc/self/mountinfo').read_text().splitlines() "
        "if line.split()[4] == '/workspace']\n"
        "assert len(mounts) == 1 and 'ro' in mounts[0].split()[5].split(',')\n"
        "print('OCI_BOUNDARY_OK')\n",
        encoding="utf-8",
    )
    config = ToolSandboxConfig(
        backend="docker",
        image=built,
        python_executable="/usr/local/bin/python",
    )
    names: list[str] = []

    def deterministic_identity(prefix, current_sandbox):
        name = f"repopilot-integration-{prefix}-{len(names)}"
        names.append(name)
        return name, current_sandbox.root / f"{name}.cid"

    monkeypatch.setattr(
        "src.safe_subprocess._container_identity", deterministic_identity
    )

    result = run_oci_process(
        ["/usr/local/bin/python", "-P", "/workspace/check_boundary.py"],
        sandbox=sandbox,
        config=config,
        timeout=30,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "OCI_BOUNDARY_OK"

    with pytest.raises(ProcessTimeoutError):
        run_oci_process(
            ["/usr/local/bin/python", "-P", "-c", "import time; time.sleep(60)"],
            sandbox=sandbox,
            config=config,
            timeout=0.5,
        )

    for name in names:
        inspected = subprocess.run(
            [docker, "inspect", name], capture_output=True, timeout=30
        )
        assert inspected.returncode != 0
