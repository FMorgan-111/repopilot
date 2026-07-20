"""Credential-safe subprocess execution and fail-closed test isolation."""

from __future__ import annotations

import json
import os
import platform
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Sequence

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
_SAFE_OVERRIDE_KEYS = frozenset({"HOME", "PYTHONPATH", "TEMP", "TMP", "TMPDIR"})
_PROBE_ENV_NAME = "REPOPILOT_ISOLATION_PARENT_SENTINEL"
_PROBE_LOCK = threading.Lock()


@dataclass(frozen=True)
class SandboxPaths:
    """Private host directories mapped into one disposable test sandbox."""

    root: Path
    workspace: Path
    home: Path
    temp: Path

    @classmethod
    def create(cls, root: str | Path) -> SandboxPaths:
        resolved = Path(root).resolve()
        workspace = resolved / "workspace"
        home = resolved / "home"
        temp = resolved / "tmp"
        for path in (resolved, workspace, home, temp):
            path.mkdir(parents=True, exist_ok=True)
            path.chmod(0o700)
        return cls(root=resolved, workspace=workspace, home=home, temp=temp)


@dataclass(frozen=True)
class NetworkIsolation:
    """A proved OS isolation launcher plus its private environment."""

    capability: str
    argv_prefix: tuple[str, ...]
    env_overrides: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BoundedProcessResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str


class NetworkIsolationUnavailableError(RuntimeError):
    """Raised instead of executing repository code without proven isolation."""


NetworkIsolationUnavailable = NetworkIsolationUnavailableError


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


def _is_trusted_executable(path: Path) -> bool:
    try:
        resolved = path.resolve(strict=True)
        info = resolved.stat()
    except OSError:
        return False
    return (
        resolved.is_file()
        and bool(info.st_mode & stat.S_IXUSR)
        and info.st_uid == 0
        and not bool(info.st_mode & (stat.S_IWGRP | stat.S_IWOTH))
    )


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


def trusted_python() -> str:
    """Return a canonical root-owned interpreter, never a writable venv shim."""
    candidate = Path(sys.executable).resolve()
    if _is_trusted_executable(candidate):
        return str(candidate)
    for name in ("python3", "python"):
        resolved = trusted_executable(name, required=False)
        if resolved:
            return resolved
    raise NetworkIsolationUnavailableError("trusted Python launcher unavailable")


def _trusted_path() -> str:
    directories = [
        str(path)
        for path in _TRUSTED_EXECUTABLE_DIRS[:4]
        if path.is_dir()
        and path.stat().st_uid == 0
        and not bool(path.stat().st_mode & (stat.S_IWGRP | stat.S_IWOTH))
    ]
    return os.pathsep.join(directories)


def minimal_subprocess_env(
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a credential-free environment with a non-shadowable PATH."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key in _SAFE_HOST_ENV_KEYS
    }
    env["PATH"] = _trusted_path()
    if overrides:
        if not set(overrides).issubset(_SAFE_OVERRIDE_KEYS):
            raise ValueError("unsafe subprocess environment override")
        env.update({str(key): str(value) for key, value in overrides.items()})
    return env


def _runtime_mounts() -> tuple[Path, ...]:
    candidates = (
        Path("/usr"),
        Path("/bin"),
        Path("/lib"),
        Path("/lib64"),
        Path("/usr/local"),
        Path("/System/Library"),
        Path("/Library/Frameworks"),
    )
    trusted: list[Path] = []
    for path in candidates:
        try:
            info = path.stat()
        except OSError:
            continue
        if info.st_uid == 0 and not bool(info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)):
            trusted.append(path)
    return tuple(trusted)


def _python_import_paths() -> tuple[Path, ...]:
    paths: list[Path] = []
    for entry in sys.path:
        if not entry:
            continue
        path = Path(entry).resolve()
        try:
            info = path.stat()
        except OSError:
            continue
        if (
            path.is_dir()
            and ("site-packages" in path.parts or "dist-packages" in path.parts)
            and info.st_uid == 0
            and not bool(info.st_mode & (stat.S_IWGRP | stat.S_IWOTH))
        ):
            paths.append(path)
    return tuple(dict.fromkeys(paths))


def build_linux_isolation(
    sandbox: SandboxPaths,
    *,
    launcher: str,
    runtime_mounts: Sequence[Path],
    python_path: Sequence[Path],
) -> NetworkIsolation:
    """Build a mount+PID+user+network namespace without binding host root."""
    argv: list[str] = [
        launcher,
        "--die-with-parent",
        "--new-session",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-net",
        "--unshare-ipc",
        "--unshare-uts",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/run",
        "--dir",
        "/runtime",
        "--dir",
        "/runtime/pythonpath",
    ]
    for path in runtime_mounts:
        argv.extend(("--ro-bind", str(path), str(path)))
    mapped_python: list[str] = []
    for index, path in enumerate(python_path):
        destination = f"/runtime/pythonpath/{index}"
        argv.extend(("--ro-bind", str(path), destination))
        mapped_python.append(destination)
    argv.extend(
        (
            "--bind",
            str(sandbox.workspace),
            "/workspace",
            "--bind",
            str(sandbox.home),
            "/home/repopilot",
            "--bind",
            str(sandbox.temp),
            "/tmp",
            "--chdir",
            "/workspace",
            "--",
        )
    )
    return NetworkIsolation(
        capability="linux-bwrap-namespaces",
        argv_prefix=tuple(argv),
        env_overrides={
            "HOME": "/home/repopilot",
            "TMPDIR": "/tmp",
            "TMP": "/tmp",
            "TEMP": "/tmp",
            "PYTHONPATH": os.pathsep.join(mapped_python),
        },
    )


def _sandbox_literal(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace('"', '\\"')


def build_darwin_profile(
    sandbox: SandboxPaths,
    *,
    runtime_mounts: Sequence[Path],
    python_path: Sequence[Path],
) -> str:
    """Build a deny-default profile scoped to runtime and disposable paths."""
    read_paths = [*runtime_mounts, *python_path, sandbox.workspace]
    write_paths = [sandbox.workspace, sandbox.home, sandbox.temp]
    exec_rules = " ".join(
        f'(subpath "{_sandbox_literal(path)}")' for path in runtime_mounts
    )
    read_rules = " ".join(
        f'(subpath "{_sandbox_literal(path)}")' for path in read_paths
    )
    write_rules = " ".join(
        f'(subpath "{_sandbox_literal(path)}")' for path in write_paths
    )
    return "".join(
        (
            "(version 1)",
            "(deny default)",
            "(deny network*)",
            "(allow process-fork)",
            "(allow signal (target self))",
            "(allow sysctl-read)",
            f"(allow process-exec {exec_rules})",
            f"(allow file-read* {read_rules} (literal \"/dev/null\"))",
            f"(allow file-write* {write_rules} (literal \"/dev/null\"))",
        )
    )


def _build_darwin_isolation(
    sandbox: SandboxPaths,
    launcher: str,
    runtime_mounts: Sequence[Path],
    python_path: Sequence[Path],
) -> NetworkIsolation:
    profile = build_darwin_profile(
        sandbox, runtime_mounts=runtime_mounts, python_path=python_path
    )
    return NetworkIsolation(
        capability="darwin-sandbox-exec-deny-default",
        argv_prefix=(launcher, "-p", profile),
        env_overrides={
            "HOME": str(sandbox.home),
            "TMPDIR": str(sandbox.temp),
            "TMP": str(sandbox.temp),
            "TEMP": str(sandbox.temp),
            "PYTHONPATH": os.pathsep.join(str(path) for path in python_path),
        },
    )


_PROBE_SCRIPT = r"""
import json, os, pathlib, socket, sys
sentinel, env_name, host_pid, port = sys.argv[1:]
try:
    pathlib.Path(sentinel).read_text()
    file_blocked = False
except Exception:
    file_blocked = True
env_blocked = env_name not in os.environ
proc_private = not pathlib.Path('/proc', host_pid).exists()
try:
    sock = socket.create_connection(('127.0.0.1', int(port)), timeout=0.2)
    sock.close()
    socket_blocked = False
except Exception:
    socket_blocked = True
print(json.dumps({'file': file_blocked, 'env': env_blocked, 'proc': proc_private, 'socket': socket_blocked}))
"""


def _probe_network_isolation(
    isolation: NetworkIsolation,
    sandbox: SandboxPaths,
) -> bool:
    """Prove host file/env/proc/socket boundaries, not merely launcher startup."""
    with _PROBE_LOCK, tempfile.TemporaryDirectory(
        prefix="repopilot-host-probe-"
    ) as probe_dir:
        sentinel = Path(probe_dir) / "host-sentinel"
        sentinel.write_text("must-not-be-readable", encoding="utf-8")
        listener: socket.socket | None = None
        try:
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
        except OSError:
            if listener is not None:
                listener.close()
            return False
        previous = os.environ.get(_PROBE_ENV_NAME)
        os.environ[_PROBE_ENV_NAME] = "must-not-cross"
        try:
            result = run_bounded_process(
                [
                    *isolation.argv_prefix,
                    trusted_python(),
                    "-c",
                    _PROBE_SCRIPT,
                    str(sentinel),
                    _PROBE_ENV_NAME,
                    str(os.getpid()),
                    str(listener.getsockname()[1]),
                ],
                cwd=sandbox.workspace,
                timeout=10,
                max_output_bytes=8_000,
                env_overrides=isolation.env_overrides,
            )
        except (OSError, RuntimeError, ValueError):
            return False
        finally:
            listener.close()
            if previous is None:
                os.environ.pop(_PROBE_ENV_NAME, None)
            else:
                os.environ[_PROBE_ENV_NAME] = previous
        if result.returncode != 0:
            return False
        try:
            payload = json.loads(result.stdout.strip())
        except (json.JSONDecodeError, TypeError):
            return False
        return all(payload.get(key) is True for key in ("file", "env", "proc", "socket"))


def network_isolation(sandbox: SandboxPaths) -> NetworkIsolation:
    """Resolve and negatively probe a strong platform sandbox or fail closed."""
    runtime_mounts = _runtime_mounts()
    python_path = _python_import_paths()
    system = platform.system()
    if system == "Darwin":
        launcher = trusted_executable("sandbox-exec", required=False)
        if launcher:
            isolation = _build_darwin_isolation(
                sandbox, launcher, runtime_mounts, python_path
            )
            if _probe_network_isolation(isolation, sandbox):
                return isolation
    elif system == "Linux":
        launcher = trusted_executable("bwrap", required=False)
        if launcher:
            isolation = build_linux_isolation(
                sandbox,
                launcher=launcher,
                runtime_mounts=runtime_mounts,
                python_path=python_path,
            )
            if _probe_network_isolation(isolation, sandbox):
                return isolation
    raise NetworkIsolationUnavailableError(
        f"no proven strong network/filesystem isolation capability for {system}"
    )


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
            return
        except PermissionError:
            process.kill()
        time.sleep(0.2)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (PermissionError, ProcessLookupError):
            pass
    else:  # pragma: no cover - Windows has no supported strong test sandbox
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
    isolate_network: bool = False,
    sandbox: SandboxPaths | None = None,
    env_overrides: dict[str, str] | None = None,
) -> BoundedProcessResult:
    """Run fixed argv with bounded capture and reap the group on every exit."""
    if not argv or any(not isinstance(token, str) for token in argv):
        raise ValueError("argv must contain strings")
    if timeout <= 0 or max_output_bytes < 0:
        raise ValueError("invalid subprocess bounds")
    original_argv = list(argv)
    launched_argv = original_argv
    merged_overrides = dict(env_overrides or {})
    if isolate_network:
        if sandbox is None:
            raise NetworkIsolationUnavailableError("private sandbox paths are required")
        isolation = network_isolation(sandbox)
        launched_argv = [*isolation.argv_prefix, *original_argv]
        merged_overrides.update(isolation.env_overrides)

    kwargs: dict[str, object] = {
        "cwd": Path(cwd),
        "env": minimal_subprocess_env(merged_overrides),
        "stdin": subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
    }
    if os.name == "posix":
        kwargs["start_new_session"] = True
    else:  # pragma: no cover
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    process = subprocess.Popen(launched_argv, **kwargs)  # type: ignore[arg-type]
    assert process.stdout is not None and process.stderr is not None
    stdout = bytearray()
    stderr = bytearray()
    budget = [max_output_bytes]
    lock = threading.Lock()
    overflow = threading.Event()
    readers = [
        threading.Thread(target=_reader, args=(process.stdout, stdout, budget, lock, overflow), daemon=True),
        threading.Thread(target=_reader, args=(process.stderr, stderr, budget, lock, overflow), daemon=True),
    ]
    for reader in readers:
        reader.start()
    writer: threading.Thread | None = None
    if input_text is not None:
        assert process.stdin is not None
        writer = threading.Thread(
            target=_write_input,
            args=(process.stdin, input_text.encode("utf-8")),
            daemon=True,
        )
        writer.start()

    deadline = time.monotonic() + timeout
    timed_out = False
    group_terminated = False
    while process.poll() is None:
        if overflow.is_set() or time.monotonic() >= deadline:
            timed_out = not overflow.is_set()
            _terminate_process_group(process)
            group_terminated = True
            break
        time.sleep(0.01)
    if not group_terminated:
        # A normal leader may leave descendants holding pipes or mutating the
        # snapshot. Always end the complete session/namespace before returning.
        _terminate_process_group(process)
    for reader in readers:
        reader.join(timeout=1)
    if writer is not None:
        writer.join(timeout=1)
    stdout_text = stdout.decode("utf-8", errors="replace")
    stderr_text = stderr.decode("utf-8", errors="replace")
    if overflow.is_set():
        raise ProcessOutputLimitError(stdout_text, stderr_text)
    if timed_out:
        raise ProcessTimeoutError(stdout_text, stderr_text)
    return BoundedProcessResult(
        argv=original_argv,
        returncode=int(process.returncode or 0),
        stdout=stdout_text,
        stderr=stderr_text,
    )
