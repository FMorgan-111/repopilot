"""Credential-safe subprocesses and immutable OCI test execution."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import stat
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Literal, Sequence

from .async_safety import CancellationDrainError, drain_task
from .state import ToolSandboxConfig

_TRUSTED_EXECUTABLE_DIRS = (
    Path("/usr/bin"),
    Path("/bin"),
    Path("/usr/sbin"),
    Path("/sbin"),
    Path("/usr/local/bin"),
)
_SAFE_HOST_ENV_KEYS = frozenset(
    {"COMSPEC", "LANG", "LC_ALL", "PATHEXT", "SYSTEMROOT", "TZ", "WINDIR"}
)
_SAFE_OVERRIDE_KEYS = frozenset(
    {"HOME", "PYTHONPATH", "TEMP", "TMP", "TMPDIR", "VIRTUAL_ENV"}
)


@dataclass(frozen=True)
class SandboxPaths:
    """Private host root containing only the disposable workspace and OCI IDs."""

    root: Path
    workspace: Path

    @classmethod
    def create(cls, root: str | Path) -> SandboxPaths:
        resolved = Path(root).resolve()
        workspace = resolved / "workspace"
        resolved.mkdir(parents=True, exist_ok=True)
        resolved.chmod(0o700)
        workspace.mkdir(parents=True, exist_ok=True)
        # The configured container user must be able to traverse and mutate the
        # disposable bind. Its parent and cid files remain private on the host.
        workspace.chmod(0o777)
        return cls(root=resolved, workspace=workspace)


@dataclass(frozen=True)
class BoundedProcessResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str


class NetworkIsolationUnavailableError(RuntimeError):
    """Raised instead of executing repository code outside a proven OCI boundary."""


NetworkIsolationUnavailable = NetworkIsolationUnavailableError


class IsolationCleanupError(NetworkIsolationUnavailableError):
    """A container may still exist; recovery metadata is preserved on disk."""

    def __init__(self, name: str, recovery_path: Path) -> None:
        super().__init__(f"OCI cleanup could not be verified for {name}")
        self.container_name = name
        self.recovery_path = recovery_path


class ProcessOutputLimitError(RuntimeError):
    def __init__(self, stdout: str, stderr: str) -> None:
        super().__init__("subprocess output exceeded the configured bound")
        self.stdout = stdout
        self.stderr = stderr


ProcessOutputLimitExceeded = ProcessOutputLimitError


class ProcessTimeoutError(RuntimeError):
    def __init__(self, stdout: str, stderr: str) -> None:
        super().__init__("subprocess exceeded the configured timeout")
        self.stdout = stdout
        self.stderr = stderr


class ProcessCancellationRequested(RuntimeError):
    """Raised in a worker thread after cooperative subprocess cleanup."""


def _is_trusted_executable(path: Path) -> bool:
    try:
        resolved = path.resolve(strict=True)
        info = resolved.stat()
    except OSError:
        return False
    executable_is_trusted = (
        resolved.is_file()
        and bool(info.st_mode & stat.S_IXUSR)
        and info.st_uid == 0
        and not bool(info.st_mode & (stat.S_IWGRP | stat.S_IWOTH))
    )
    if not executable_is_trusted:
        return False
    for ancestor in resolved.parents:
        try:
            ancestor_info = ancestor.stat()
        except OSError:
            return False
        if ancestor_info.st_uid != 0 or bool(
            ancestor_info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            return False
    return True


def trusted_executable(name: str, *, required: bool = True) -> str | None:
    """Resolve only canonical root-owned launchers, ignoring the host PATH."""
    if not name or Path(name).name != name:
        raise ValueError("trusted executable must be a basename")
    for directory in _TRUSTED_EXECUTABLE_DIRS:
        candidate = directory / name
        if _is_trusted_executable(candidate):
            return str(candidate.resolve())
    if required:
        raise NetworkIsolationUnavailableError(f"trusted launcher unavailable: {name}")
    return None


def _trusted_path() -> str:
    directories = []
    for path in _TRUSTED_EXECUTABLE_DIRS[:4]:
        try:
            info = path.stat()
        except OSError:
            continue
        if path.is_dir() and info.st_uid == 0 and not bool(
            info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            directories.append(str(path))
    return os.pathsep.join(directories)


def minimal_subprocess_env(
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a credential-free environment with a non-shadowable PATH."""
    env = {key: value for key, value in os.environ.items() if key in _SAFE_HOST_ENV_KEYS}
    env["PATH"] = _trusted_path()
    if overrides:
        if not set(overrides).issubset(_SAFE_OVERRIDE_KEYS):
            raise ValueError("unsafe subprocess environment override")
        env.update({str(key): str(value) for key, value in overrides.items()})
    return env


def tool_sandbox_config_from_env() -> ToolSandboxConfig | None:
    """Load operator-owned OCI settings; repository state cannot supply them."""
    backend = os.getenv("REPOPILOT_TOOL_OCI_BACKEND", "").strip()
    image = os.getenv("REPOPILOT_TOOL_OCI_IMAGE", "").strip()
    if not backend and not image:
        return None
    if not backend or not image:
        raise ValueError("both OCI backend and digest-pinned image are required")
    raw_executables = os.getenv("REPOPILOT_TOOL_PROJECT_EXECUTABLES", "{}").strip()
    try:
        project_executables = json.loads(raw_executables)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid OCI project executable configuration") from exc
    return ToolSandboxConfig(
        backend=backend,
        image=image,
        python_executable=os.getenv(
            "REPOPILOT_TOOL_PYTHON_EXECUTABLE", "/usr/bin/python3"
        ).strip(),
        project_executables=project_executables,
        memory=os.getenv("REPOPILOT_TOOL_MEMORY", "1g").strip(),
        cpus=os.getenv("REPOPILOT_TOOL_CPUS", "1.0").strip(),
        pids_limit=os.getenv("REPOPILOT_TOOL_PIDS_LIMIT", "128").strip(),
    )


def build_oci_command(
    config: ToolSandboxConfig,
    sandbox: SandboxPaths,
    command: Sequence[str],
    *,
    backend: str,
    name: str,
    cidfile: Path,
) -> list[str]:
    """Build one locked-down OCI invocation with exactly one host bind."""
    if not command or any(not isinstance(token, str) or not token for token in command):
        raise ValueError("OCI command must contain non-empty strings")
    try:
        root = sandbox.root.resolve(strict=True)
        workspace = sandbox.workspace.resolve(strict=True)
    except OSError as exc:
        raise ValueError("OCI sandbox paths are unavailable") from exc
    if (
        sandbox.root.is_symlink()
        or sandbox.workspace.is_symlink()
        or root != sandbox.root
        or workspace != sandbox.workspace
        or workspace != root / "workspace"
        or not root.is_dir()
        or not workspace.is_dir()
    ):
        raise ValueError("OCI workspace is not the exact disposable bind")
    if not name.startswith("repopilot-") or not name.replace("-", "").isalnum():
        raise ValueError("invalid OCI container name")
    if any(character in str(workspace) for character in ",\n\r"):
        raise ValueError("OCI workspace path cannot be encoded safely")
    cidfile = cidfile.resolve()
    if not cidfile.is_relative_to(sandbox.root) or cidfile.parent != sandbox.root:
        raise ValueError("OCI cidfile escaped private sandbox root")
    if (
        Path(command[0]).as_posix() != command[0]
        or not command[0].startswith("/")
        or ".." in Path(command[0]).parts
        or any(character.isspace() or ord(character) < 32 for character in command[0])
    ):
        raise ValueError("OCI entrypoint must be an absolute clean path")
    return [
        backend,
        "create",
        "--pull=never",
        f"--name={name}",
        f"--cidfile={cidfile}",
        "--network=none",
        "--ipc=private",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        f"--user={config.user}",
        f"--pids-limit={config.pids_limit}",
        f"--memory={config.memory}",
        f"--cpus={config.cpus}",
        "--workdir=/workspace",
        "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=256m,mode=1777",
        "--tmpfs=/home/repopilot:rw,noexec,nosuid,nodev,size=64m,mode=0700,uid="
        f"{config.user.partition(':')[0]},gid={config.user.partition(':')[2]}",
        "--env=HOME=/home/repopilot",
        "--env=TMPDIR=/tmp",
        "--env=PYTHONDONTWRITEBYTECODE=1",
        f"--mount=type=bind,src={workspace},dst=/workspace,ro",
        f"--entrypoint={command[0]}",
        config.image,
        *command[1:],
    ]


def _container_identity(prefix: str, sandbox: SandboxPaths) -> tuple[str, Path]:
    token = uuid.uuid4().hex
    return f"repopilot-{prefix}-{token}", sandbox.root / f"{prefix}-{token}.cid"


def _recovery_path(name: str, cidfile: Path) -> Path:
    recovery_root = Path(tempfile.gettempdir()).resolve() / "repopilot-oci-recovery"
    if recovery_root.is_symlink():
        raise NetworkIsolationUnavailableError("unsafe OCI recovery directory")
    recovery_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    recovery_root.chmod(0o700)
    info = recovery_root.stat()
    if recovery_root.is_symlink() or info.st_uid != os.getuid() or bool(
        info.st_mode & (stat.S_IRWXG | stat.S_IRWXO)
    ):
        raise NetworkIsolationUnavailableError("unsafe OCI recovery directory")
    return recovery_root / f"{name}.json"


def _write_recovery_record(backend: str, name: str, cidfile: Path) -> Path:
    recovery = _recovery_path(name, cidfile)
    payload = {
        "backend": backend,
        "container_name": name,
        "cidfile": str(cidfile),
        "container_id": "",
    }
    recovery.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    recovery.chmod(0o600)
    return recovery


def _known_missing_container(result: BoundedProcessResult) -> bool:
    message = f"{result.stdout}\n{result.stderr}".casefold()
    return any(
        marker in message
        for marker in (
            "no such container",
            "no such object",
            "no container with",
            "not found",
            "does not exist",
        )
    )


def _force_remove_container(backend: str, name: str, cidfile: Path) -> bool:
    target = name
    try:
        raw = cidfile.read_text(encoding="ascii").strip()
    except OSError:
        raw = ""
    if len(raw) in {12, 64} and all(character in "0123456789abcdef" for character in raw):
        target = raw
        recovery = _recovery_path(name, cidfile)
        try:
            payload = json.loads(recovery.read_text(encoding="utf-8"))
            payload["container_id"] = raw
            recovery.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            recovery.chmod(0o600)
        except (OSError, ValueError, TypeError):
            pass
    for _attempt in range(3):
        try:
            _removed = run_bounded_process(
                [backend, "rm", "-f", target],
                cwd=cidfile.parent,
                timeout=15,
                max_output_bytes=8_000,
            )
            inspected = run_bounded_process(
                [backend, "inspect", target],
                cwd=cidfile.parent,
                timeout=15,
                max_output_bytes=8_000,
            )
        except (OSError, RuntimeError, ValueError):
            time.sleep(0.1)
            continue
        if inspected.returncode != 0 and _known_missing_container(inspected):
            for path in (cidfile, _recovery_path(name, cidfile)):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            return True
        time.sleep(0.1)
    return False


def _verify_created_container(
    backend: str,
    name: str,
    sandbox: SandboxPaths,
    config: ToolSandboxConfig,
    cancellation_event: threading.Event | None,
) -> None:
    inspected = run_bounded_process(
        [backend, "inspect", name],
        cwd=sandbox.root,
        timeout=15,
        max_output_bytes=128_000,
        decode_errors="strict",
        cancellation_event=cancellation_event,
    )
    if inspected.returncode != 0:
        raise NetworkIsolationUnavailableError("OCI created-container inspect failed")
    try:
        decoded = json.loads(inspected.stdout)
        details = decoded[0]
        host = details["HostConfig"]
        container_config = details["Config"]
        mounts = details.get("Mounts", [])
    except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise NetworkIsolationUnavailableError("invalid OCI inspect response") from exc
    persistent_mounts = [
        mount for mount in mounts if str(mount.get("Type", "")).lower() != "tmpfs"
    ]
    tmpfs = host.get("Tmpfs", {})
    tmp_size = str(tmpfs.get("/tmp", "")).lower()
    home_size = str(tmpfs.get("/home/repopilot", "")).lower()
    expected_source = str(sandbox.workspace)
    if (
        len(persistent_mounts) != 1
        or persistent_mounts[0].get("Type") != "bind"
        or persistent_mounts[0].get("Source") != expected_source
        or persistent_mounts[0].get("Destination") != "/workspace"
        or persistent_mounts[0].get("RW") is not False
        or str(host.get("NetworkMode", "")).lower() != "none"
        or host.get("ReadonlyRootfs") is not True
        or host.get("Privileged") is not False
        or "all" not in {str(item).lower() for item in host.get("CapDrop", [])}
        or "no-new-privileges" not in " ".join(
            str(item).lower() for item in host.get("SecurityOpt", [])
        )
        or int(host.get("PidsLimit") or 0) != config.pids_limit
        or int(host.get("Memory") or 0) <= 0
        or int(host.get("NanoCpus") or host.get("CpuQuota") or 0) <= 0
        or set(tmpfs) != {"/tmp", "/home/repopilot"}
        or not any(value in tmp_size for value in ("size=256m", "size=268435456"))
        or not any(value in home_size for value in ("size=64m", "size=67108864"))
        or not str(container_config.get("User", ""))
        or str(container_config.get("User", "")).split(":", 1)[0] == "0"
        or str(container_config.get("Image", "")) != config.image
    ):
        raise NetworkIsolationUnavailableError("OCI runtime did not preserve isolation")


def _run_oci_container(
    backend: str,
    config: ToolSandboxConfig,
    sandbox: SandboxPaths,
    command: Sequence[str],
    *,
    prefix: str,
    timeout: float,
    max_output_bytes: int,
    cancellation_event: threading.Event | None,
) -> BoundedProcessResult:
    name, cidfile = _container_identity(prefix, sandbox)
    argv = build_oci_command(
        config,
        sandbox,
        command,
        backend=backend,
        name=name,
        cidfile=cidfile,
    )
    recovery = _write_recovery_record(backend, name, cidfile)
    result: BoundedProcessResult | None = None
    operation_error: BaseException | None = None
    try:
        created = run_bounded_process(
            argv,
            cwd=sandbox.root,
            timeout=min(timeout, 60),
            max_output_bytes=max_output_bytes,
            cancellation_event=cancellation_event,
        )
        if created.returncode != 0:
            result = created
        else:
            _verify_created_container(
                backend, name, sandbox, config, cancellation_event
            )
            result = run_bounded_process(
                [backend, "start", "-a", name],
                cwd=sandbox.root,
                timeout=timeout,
                max_output_bytes=max_output_bytes,
                cancellation_event=cancellation_event,
            )
    except BaseException as exc:
        operation_error = exc
    finally:
        try:
            cleaned = _force_remove_container(backend, name, cidfile)
        except BaseException:
            cleaned = False
        if not cleaned:
            raise IsolationCleanupError(name, recovery) from operation_error
    if operation_error is not None:
        raise operation_error.with_traceback(operation_error.__traceback__)
    assert result is not None
    return result


def run_oci_process(
    command: Sequence[str],
    *,
    sandbox: SandboxPaths,
    config: ToolSandboxConfig,
    timeout: float = 300,
    max_output_bytes: int = 8_000,
    cancellation_event: threading.Event | None = None,
) -> BoundedProcessResult:
    """Probe and execute repository code only in a digest-pinned OCI image."""
    backend = trusted_executable(config.backend, required=True)
    if backend is None:
        raise NetworkIsolationUnavailableError("trusted OCI backend unavailable")
    probe = _run_oci_container(
        backend,
        config,
        sandbox,
        [
            config.python_executable,
            "-I",
            "-m",
            "pytest",
            "--version",
        ],
        prefix="probe",
        timeout=min(timeout, 30),
        max_output_bytes=8_000,
        cancellation_event=cancellation_event,
    )
    if probe.returncode != 0 or "pytest" not in f"{probe.stdout}\n{probe.stderr}".casefold():
        raise NetworkIsolationUnavailableError("OCI Python/pytest capability probe failed")
    if command[0] in dict(config.project_executables).values():
        project_probe = _run_oci_container(
            backend,
            config,
            sandbox,
            [command[0], "--version"],
            prefix="project-probe",
            timeout=min(timeout, 30),
            max_output_bytes=8_000,
            cancellation_event=cancellation_event,
        )
        if project_probe.returncode != 0:
            raise NetworkIsolationUnavailableError(
                "OCI project executable capability probe failed"
            )
    return _run_oci_container(
        backend,
        config,
        sandbox,
        command,
        prefix="command",
        timeout=timeout,
        max_output_bytes=max_output_bytes,
        cancellation_event=cancellation_event,
    )


async def run_oci_process_async(
    command: Sequence[str],
    *,
    sandbox: SandboxPaths,
    config: ToolSandboxConfig,
    timeout: float = 300,
    max_output_bytes: int = 8_000,
) -> BoundedProcessResult:
    """Run OCI work off-loop and drain its cleanup before propagating cancellation."""
    cancellation_event = threading.Event()
    worker = asyncio.create_task(
        asyncio.to_thread(
            run_oci_process,
            command,
            sandbox=sandbox,
            config=config,
            timeout=timeout,
            max_output_bytes=max_output_bytes,
            cancellation_event=cancellation_event,
        )
    )
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError as original_cancel:
        cancellation_event.set()
        outcome = await drain_task(worker)
        if isinstance(outcome.error, ProcessCancellationRequested):
            raise original_cancel
        if outcome.error is not None:
            raise CancellationDrainError(
                "OCI process", original_cancel, outcome.error
            ) from outcome.error
        raise original_cancel


def _reader(
    pipe: IO[bytes],
    output: bytearray,
    budget: list[int],
    lock: threading.Lock,
    overflow: threading.Event,
) -> None:
    try:
        while True:
            chunk = pipe.read(4096)
            if not chunk:
                return
            if overflow.is_set():
                continue
            with lock:
                accepted = chunk[: budget[0]]
                output.extend(accepted)
                budget[0] -= len(accepted)
                if len(accepted) != len(chunk):
                    overflow.set()
    finally:
        pipe.close()


def _write_input(pipe: IO[bytes], value: bytes) -> None:
    try:
        pipe.write(value)
        pipe.flush()
    except (BrokenPipeError, OSError):
        pass
    finally:
        pipe.close()


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            process.kill()
        else:
            time.sleep(0.2)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (PermissionError, ProcessLookupError):
                pass
    else:  # pragma: no cover
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
        except OSError:
            process.kill()
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_bounded_process(
    argv: Sequence[str],
    *,
    cwd: str | Path,
    input_text: str | None = None,
    timeout: float = 300,
    max_output_bytes: int = 8_000,
    env_overrides: dict[str, str] | None = None,
    watched_output_path: str | Path | None = None,
    max_watched_output_bytes: int | None = None,
    decode_errors: Literal["strict", "replace"] = "replace",
    cancellation_event: threading.Event | None = None,
) -> BoundedProcessResult:
    """Run fixed argv with bounded capture and reap the group on every exit."""
    if not argv or any(not isinstance(token, str) for token in argv):
        raise ValueError("argv must contain strings")
    if timeout <= 0 or max_output_bytes < 0:
        raise ValueError("invalid subprocess bounds")
    if (watched_output_path is None) != (max_watched_output_bytes is None):
        raise ValueError("watched output path and bound must be configured together")
    if max_watched_output_bytes is not None and max_watched_output_bytes < 0:
        raise ValueError("invalid watched output bound")
    original_argv = list(argv)
    kwargs: dict[str, object] = {
        "cwd": Path(cwd),
        "env": minimal_subprocess_env(env_overrides),
        "stdin": subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
    }
    if os.name == "posix":
        kwargs["start_new_session"] = True
    else:  # pragma: no cover
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    stdout = bytearray()
    stderr = bytearray()
    budget = [max_output_bytes]
    lock = threading.Lock()
    overflow = threading.Event()
    readers: list[threading.Thread] = []
    writer: threading.Thread | None = None
    timed_out = False
    watched_overflow = False
    watched = Path(watched_output_path) if watched_output_path is not None else None
    input_bytes = input_text.encode("utf-8") if input_text is not None else None
    if cancellation_event is not None and cancellation_event.is_set():
        raise ProcessCancellationRequested
    process = subprocess.Popen(original_argv, **kwargs)  # type: ignore[arg-type]
    try:
        assert process.stdout is not None and process.stderr is not None
        readers.extend(
            [
                threading.Thread(
                    target=_reader,
                    args=(process.stdout, stdout, budget, lock, overflow),
                    daemon=True,
                ),
                threading.Thread(
                    target=_reader,
                    args=(process.stderr, stderr, budget, lock, overflow),
                    daemon=True,
                ),
            ]
        )
        for reader in readers:
            reader.start()
        if input_bytes is not None:
            assert process.stdin is not None
            writer = threading.Thread(
                target=_write_input,
                args=(process.stdin, input_bytes),
                daemon=True,
            )
            writer.start()

        deadline = time.monotonic() + timeout
        while True:
            if cancellation_event is not None and cancellation_event.is_set():
                raise ProcessCancellationRequested
            if process.poll() is not None:
                break
            if watched is not None:
                try:
                    watched_overflow = watched.stat().st_size > int(
                        max_watched_output_bytes or 0
                    )
                except FileNotFoundError:
                    pass
            if overflow.is_set() or watched_overflow or time.monotonic() >= deadline:
                timed_out = not overflow.is_set() and not watched_overflow
                break
            time.sleep(0.01)
        if watched is not None:
            try:
                watched_overflow = watched_overflow or watched.stat().st_size > int(
                    max_watched_output_bytes or 0
                )
            except FileNotFoundError:
                pass
    finally:
        try:
            _terminate_process_group(process)
        finally:
            for reader in readers:
                if getattr(reader, "ident", None) is not None:
                    reader.join()
            if writer is not None and getattr(writer, "ident", None) is not None:
                writer.join()
            for pipe in (process.stdout, process.stderr):
                if pipe is not None:
                    pipe.close()
            if process.stdin is not None:
                process.stdin.close()
    if watched_overflow and watched is not None and max_watched_output_bytes is not None:
        try:
            with watched.open("r+b") as output:
                output.truncate(max_watched_output_bytes)
        except OSError:
            pass
    stdout_text = stdout.decode("utf-8", errors=decode_errors)
    stderr_text = stderr.decode("utf-8", errors=decode_errors)
    if overflow.is_set() or watched_overflow:
        raise ProcessOutputLimitError(stdout_text, stderr_text)
    if timed_out:
        raise ProcessTimeoutError(stdout_text, stderr_text)
    return BoundedProcessResult(
        argv=original_argv,
        returncode=int(process.returncode or 0),
        stdout=stdout_text,
        stderr=stderr_text,
    )
