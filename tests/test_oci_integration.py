from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from src.nodes import execute as execute_node
from src.patch_gate import validate_patch_batch
from src.repair_rounds import begin_repair_round, bind_repair_round_author
from src.safe_subprocess import ProcessTimeoutError, SandboxPaths, run_oci_process
from src.state import (
    AgentState,
    RepairPlan,
    ToolSandboxConfig,
    VerifiedEdit,
    VerifiedEditBatch,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


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
        "targets = (pathlib.Path('/rootfs-write'), "
        "pathlib.Path('/workspace/write'))\n"
        "for target in targets:\n"
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
        "mountinfo = pathlib.Path('/proc/self/mountinfo').read_text()\n"
        "mounts=[line for line in mountinfo.splitlines() "
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

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    (repo / "src").mkdir()
    (repo / "tests").mkdir()
    source = repo / "src" / "widget.py"
    source.write_text("def answer():\n    return 1\n", encoding="utf-8")
    sentinel_scope = hashlib.sha256(str(tmp_path).encode()).hexdigest()[:16]
    setup_sentinel = Path(f"/tmp/repopilot-setup-{sentinel_scope}")
    test_sentinel = Path(f"/tmp/repopilot-test-{sentinel_scope}")
    (repo / "setup.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(setup_sentinel)!r}).write_text('container-only')\n",
        encoding="utf-8",
    )
    secret_keys = (
        "LLM_API_KEY",
        "GITHUB_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
        "GITHUB_ACTIONS",
        "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
        "CI",
    )
    (repo / "tests" / "test_hostile_boundary.py").write_text(
        "import os, runpy\n"
        "from pathlib import Path\n"
        "from src.widget import answer\n\n"
        "def test_hostile_boundary():\n"
        "    assert answer() == 2\n"
        f"    keys = {secret_keys!r}\n"
        "    assert all(os.environ.get(key) is None for key in keys)\n"
        "    runpy.run_path('/workspace/setup.py')\n"
        f"    assert Path({str(setup_sentinel)!r}).read_text() == 'container-only'\n"
        f"    Path({str(test_sentinel)!r}).write_text('container-only')\n",
        encoding="utf-8",
    )
    _git(repo, "add", "--all")
    _git(repo, "commit", "-qm", "base")
    ref = _git(repo, "rev-parse", "HEAD")
    plan = RepairPlan(
        root_cause="answer returns the old value",
        target_files=["src/widget.py"],
        target_symbols=["answer"],
        required_behavior="answer returns two",
        regression_test_strategy="run the hostile boundary test",
    )
    state = AgentState(
        issue_url="https://github.com/acme/widget/issues/1",
        owner="acme",
        repo="widget",
        repo_path=str(repo),
        repo_ref=ref,
        trace_id="real-oci-boundary",
        test_command="pytest tests/test_hostile_boundary.py -q",
        active_repair_plan=plan,
        tool_sandbox_config=config,
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
    begin_repair_round(state)
    bind_repair_round_author(state)
    assert validate_patch_batch(state, plan, batch).accepted
    assert state.authorized_repair_round_id > 0
    for key in secret_keys:
        monkeypatch.setenv(key, f"host-{key.lower()}-sentinel")
    assert not setup_sentinel.exists()
    assert not test_sentinel.exists()

    execution = asyncio.run(execute_node.execute_fix(state))

    assert execution.fix_attempts[-1].success is True
    assert not setup_sentinel.exists()
    assert not test_sentinel.exists()

    for name in names:
        inspected = subprocess.run(
            [docker, "inspect", name], capture_output=True, timeout=30
        )
        assert inspected.returncode != 0
