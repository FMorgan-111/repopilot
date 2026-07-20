"""Credential-safe, bounded subprocess execution for repository tools."""

from __future__ import annotations

import os
import platform
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Sequence

_ENV_ALLOWLIST = frozenset(
    {
        "COMSPEC",
        "LANG",
        "LC_ALL",
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TZ",
        "WINDIR",
    }
)
_DARWIN_SANDBOX_PROFILE = "(version 1)(allow default)(deny network*)"


@dataclass(frozen=True)
class NetworkIsolation:
    """An explicit operating-system network isolation capability."""

    capability: str
    argv_prefix: tuple[str, ...]


@dataclass(frozen=True)
class BoundedProcessResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str


class NetworkIsolationUnavailableError(RuntimeError):
    """Raised instead of executing repository code without network isolation."""


NetworkIsolationUnavailable = NetworkIsolationUnavailableError


class ProcessOutputLimitError(RuntimeError):
    """Raised after the complete process group exceeds its streaming output budget."""

    def __init__(self, stdout: str, stderr: str) -> None:
        super().__init__("subprocess output exceeded the configured bound")
        self.stdout = stdout
        self.stderr = stderr


ProcessOutputLimitExceeded = ProcessOutputLimitError


class ProcessTimeoutError(RuntimeError):
    """Raised after the complete process group exceeds its runtime budget."""

    def __init__(self, stdout: str, stderr: str) -> None:
        super().__init__("subprocess exceeded the configured timeout")
        self.stdout = stdout
        self.stderr = stderr


def minimal_subprocess_env() -> dict[str, str]:
    """Copy only non-credential process settings required to launch local tools."""
    return {key: value for key, value in os.environ.items() if key in _ENV_ALLOWLIST}


def network_isolation() -> NetworkIsolation:
    """Resolve a supported launcher or fail before repository code executes."""
    system = platform.system()
    if system == "Darwin":
        launcher = shutil.which("sandbox-exec")
        if launcher:
            isolation = NetworkIsolation(
                capability="darwin-sandbox-exec",
                argv_prefix=(launcher, "-p", _DARWIN_SANDBOX_PROFILE),
            )
            if _probe_network_isolation(isolation):
                return isolation
    elif system == "Linux":
        bubblewrap = shutil.which("bwrap")
        if bubblewrap:
            isolation = NetworkIsolation(
                capability="linux-bwrap-unshare-net",
                argv_prefix=(
                    bubblewrap,
                    "--unshare-net",
                    "--die-with-parent",
                    "--bind",
                    "/",
                    "/",
                    "--",
                ),
            )
            if _probe_network_isolation(isolation):
                return isolation
        unshare = shutil.which("unshare")
        if unshare:
            isolation = NetworkIsolation(
                capability="linux-user-net-namespace",
                argv_prefix=(
                    unshare,
                    "--user",
                    "--map-root-user",
                    "--net",
                    "--",
                ),
            )
            if _probe_network_isolation(isolation):
                return isolation
    raise NetworkIsolationUnavailableError(
        f"no supported network isolation capability for {system}"
    )


def _probe_network_isolation(isolation: NetworkIsolation) -> bool:
    """Confirm the launcher can apply its policy before giving it repository code."""
    true_command = "/usr/bin/true" if Path("/usr/bin/true").is_file() else "true"
    try:
        result = run_bounded_process(
            [*isolation.argv_prefix, true_command],
            cwd=Path.cwd(),
            timeout=5,
            max_output_bytes=4_096,
        )
    except (OSError, RuntimeError, ValueError):
        return False
    return result.returncode == 0


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
                remaining = budget[0]
                accepted = chunk[:remaining]
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
        except (PermissionError, ProcessLookupError):
            return
        time.sleep(0.2)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (PermissionError, ProcessLookupError):
            pass
    else:  # pragma: no cover - exercised on Windows runners
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
            process.wait(timeout=0.2)
        except (OSError, subprocess.TimeoutExpired):
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
) -> BoundedProcessResult:
    """Run fixed argv with bounded streaming capture and whole-group termination."""
    if not argv or any(not isinstance(token, str) for token in argv):
        raise ValueError("argv must contain strings")
    if timeout <= 0 or max_output_bytes < 0:
        raise ValueError("invalid subprocess bounds")

    original_argv = list(argv)
    launched_argv = original_argv
    if isolate_network:
        isolation = network_isolation()
        launched_argv = [*isolation.argv_prefix, *original_argv]

    popen_kwargs: dict[str, object] = {
        "cwd": Path(cwd),
        "env": minimal_subprocess_env(),
        "stdin": subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
    }
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    else:  # pragma: no cover - exercised on Windows runners
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    process = subprocess.Popen(launched_argv, **popen_kwargs)  # type: ignore[arg-type]
    assert process.stdout is not None
    assert process.stderr is not None
    stdout = bytearray()
    stderr = bytearray()
    budget = [max_output_bytes]
    budget_lock = threading.Lock()
    overflow = threading.Event()
    readers = [
        threading.Thread(
            target=_reader,
            args=(process.stdout, stdout, budget, budget_lock, overflow),
            daemon=True,
        ),
        threading.Thread(
            target=_reader,
            args=(process.stderr, stderr, budget, budget_lock, overflow),
            daemon=True,
        ),
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
        if overflow.is_set():
            _terminate_process_group(process)
            group_terminated = True
            break
        if time.monotonic() >= deadline:
            timed_out = True
            _terminate_process_group(process)
            group_terminated = True
            break
        time.sleep(0.01)

    for reader in readers:
        reader.join(timeout=1)
    if writer is not None:
        writer.join(timeout=1)
    stdout_text = stdout.decode("utf-8", errors="replace")
    stderr_text = stderr.decode("utf-8", errors="replace")
    if overflow.is_set():
        # The leader can exit from SIGPIPE before the coordinator observes the
        # cap.  Its descendants still retain the process-group ID, so always
        # signal the group rather than relying on the leader's poll status.
        if not group_terminated:
            _terminate_process_group(process)
        raise ProcessOutputLimitError(stdout_text, stderr_text)
    if timed_out:
        raise ProcessTimeoutError(stdout_text, stderr_text)
    return BoundedProcessResult(
        argv=original_argv,
        returncode=int(process.returncode or 0),
        stdout=stdout_text,
        stderr=stderr_text,
    )
