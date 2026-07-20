"""Deterministic safety policy for model-selected local tools."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .state import AgentState, ToolAction

_COMMIT_RE = re.compile(r"[0-9a-fA-F]{40}")
_ENV_ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*")
_SHELL_META_RE = re.compile(r"[;&|><`\n\r]|\$\(")
_NETWORK_COMMANDS = frozenset(
    {"curl", "wget", "nc", "ncat", "netcat", "ssh", "scp", "ftp", "telnet"}
)
_SAFE_TEST_OPTION_RE = re.compile(
    r"(?:-[qvxs]+|--disable-warnings|--tb=(?:auto|long|short|line|native|no)|--maxfail=[0-9]+)"
)

_ARG_KEYS: dict[ToolAction, frozenset[str]] = {
    "search_symbol": frozenset({"symbol", "path"}),
    "search_text": frozenset({"text", "path"}),
    "read_symbol": frozenset({"path", "symbol"}),
    "read_range": frozenset({"path", "start_line", "end_line"}),
    "find_references": frozenset({"symbol", "path"}),
    "list_related_tests": frozenset({"path"}),
    "run_targeted_test": frozenset({"command"}),
    "inspect_git_diff": frozenset(),
    "validate_patch": frozenset(),
    "request_repair": frozenset(),
    "finish_investigation": frozenset(),
}
_REQUIRED_KEYS: dict[ToolAction, frozenset[str]] = {
    "search_symbol": frozenset({"symbol"}),
    "search_text": frozenset({"text"}),
    "read_symbol": frozenset({"path", "symbol"}),
    "read_range": frozenset({"path", "start_line", "end_line"}),
    "find_references": frozenset({"symbol"}),
    "list_related_tests": frozenset({"path"}),
    "run_targeted_test": frozenset({"command"}),
    "inspect_git_diff": frozenset(),
    "validate_patch": frozenset(),
    "request_repair": frozenset(),
    "finish_investigation": frozenset(),
}
_PATH_KEYS: dict[ToolAction, frozenset[str]] = {
    "search_symbol": frozenset({"path"}),
    "search_text": frozenset({"path"}),
    "read_symbol": frozenset({"path"}),
    "read_range": frozenset({"path"}),
    "find_references": frozenset({"path"}),
    "list_related_tests": frozenset({"path"}),
    "run_targeted_test": frozenset(),
    "inspect_git_diff": frozenset(),
    "validate_patch": frozenset(),
    "request_repair": frozenset(),
    "finish_investigation": frozenset(),
}


class ToolIntent(BaseModel):
    action: ToolAction
    args: dict[str, object] = Field(default_factory=dict)
    reason: str
    expected_evidence: str


class PolicyDecision(BaseModel):
    approved: bool
    status: Literal["approved", "rejected", "duplicate"]
    args_fingerprint: str
    normalized_args: dict[str, object] = Field(default_factory=dict)
    argv: list[str] = Field(default_factory=list)
    reason: str = ""


def _reject(fingerprint: str, reason: str, *, duplicate: bool = False) -> PolicyDecision:
    return PolicyDecision(
        approved=False,
        status="duplicate" if duplicate else "rejected",
        args_fingerprint=fingerprint,
        reason=reason,
    )


def _relative_path(root: Path, value: object, *, require_file: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("invalid_path")
    raw = value.strip().replace("\\", "/")
    if Path(raw).is_absolute() or re.match(r"^[A-Za-z]:/", raw):
        raise ValueError("absolute_path")
    if ".." in raw.split("/"):
        raise ValueError("path_traversal")
    resolved = (root / raw).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("path_escape")
    if require_file and not resolved.is_file():
        raise ValueError("path_not_file")
    return resolved.relative_to(root).as_posix() or "."


def _target_selector(root: Path, token: str) -> bool:
    path_part = token.split("::", 1)[0]
    if not path_part or path_part.startswith("-"):
        return False
    try:
        _relative_path(root, path_part, require_file=True)
    except ValueError:
        return False
    return True


def _safe_tokens(command: object) -> list[str]:
    if not isinstance(command, str) or not command.strip():
        raise ValueError("invalid_command")
    if _SHELL_META_RE.search(command):
        raise ValueError("shell_metacharacter")
    tokens = shlex.split(command, posix=True)
    if not tokens or any(_ENV_ASSIGNMENT_RE.fullmatch(token) for token in tokens):
        raise ValueError("environment_assignment")
    lowered = [token.lower() for token in tokens]
    if any(token in _NETWORK_COMMANDS for token in lowered) or any(
        token.startswith(("http://", "https://")) for token in lowered
    ):
        raise ValueError("network_command")
    return tokens


def _project_manifest_exists(root: Path, executable: str) -> bool:
    if executable in {"npm", "pnpm", "yarn", "bun"}:
        return (root / "package.json").is_file()
    if executable == "tox":
        return (root / "tox.ini").is_file() or (root / "pyproject.toml").is_file()
    if executable == "nox":
        return (root / "noxfile.py").is_file()
    if executable == "make":
        return (root / "Makefile").is_file()
    return False


def _is_project_test_command(root: Path, tokens: list[str]) -> bool:
    executable = tokens[0].lower() if tokens else ""
    if not _project_manifest_exists(root, executable):
        return False
    if executable in {"npm", "pnpm", "yarn", "bun"}:
        return len(tokens) >= 2 and (
            tokens[1] == "test"
            or (
                tokens[1] == "run"
                and len(tokens) >= 3
                and "test" in tokens[2].lower()
            )
        )
    if executable == "make":
        return len(tokens) >= 2 and "test" in tokens[1].lower()
    if executable == "nox":
        return any("test" in token.lower() for token in tokens[1:])
    return executable == "tox"


def _checkout_head(root: Path) -> str:
    process = subprocess.Popen(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        stdout, _ = process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, process.args)
    return stdout.strip()


def _test_argv(root: Path, state: AgentState, command: object) -> tuple[list[str], str]:
    tokens = _safe_tokens(command)
    if tokens[0] == "pytest":
        suffix = tokens[1:]
        argv = [sys.executable, "-m", "pytest", *suffix]
    elif tokens[:3] == ["python", "-m", "pytest"]:
        suffix = tokens[3:]
        argv = [sys.executable, "-m", "pytest", *suffix]
    else:
        configured = _safe_tokens(state.test_command) if state.test_command else []
        if (
            not configured
            or tokens[: len(configured)] != configured
            or not _is_project_test_command(root, configured)
        ):
            raise ValueError("test_command_not_allowlisted")
        suffix = tokens[len(configured) :]
        argv = tokens
    selectors = [token for token in suffix if _target_selector(root, token)]
    if not selectors:
        raise ValueError("target_selector_required")
    for token in suffix:
        if token == "--" or token in selectors or _SAFE_TEST_OPTION_RE.fullmatch(token):
            continue
        raise ValueError("non_selector_argument")
    return argv, " ".join(shlex.quote(token) for token in tokens)


class ToolPolicy:
    max_calls_per_round = 8
    max_calls_per_sample = 30

    def authorize(
        self,
        state: AgentState,
        intent: ToolIntent,
        *,
        calls_this_round: int,
    ) -> PolicyDecision:
        empty_fingerprint = hashlib.sha256(b"{}").hexdigest()
        if calls_this_round >= self.max_calls_per_round:
            return _reject(empty_fingerprint, "round_limit")
        if len(state.tool_history) >= self.max_calls_per_sample:
            return _reject(empty_fingerprint, "sample_limit")

        root = Path(state.repo_path).resolve() if state.repo_path else None
        if root is None or not root.is_dir() or not (root / ".git").exists():
            return _reject(empty_fingerprint, "invalid_repo")
        if not _COMMIT_RE.fullmatch(state.repo_ref):
            return _reject(empty_fingerprint, "invalid_repo_ref")
        if intent.action not in _ARG_KEYS:
            return _reject(empty_fingerprint, "unknown_action")
        keys = frozenset(intent.args)
        if not _REQUIRED_KEYS[intent.action].issubset(keys) or not keys.issubset(
            _ARG_KEYS[intent.action]
        ):
            return _reject(empty_fingerprint, "invalid_args")

        normalized = dict(intent.args)
        argv: list[str] = []
        try:
            for key in _PATH_KEYS[intent.action]:
                if key in normalized:
                    normalized[key] = _relative_path(
                        root,
                        normalized[key],
                        require_file=intent.action in {"read_symbol", "read_range", "list_related_tests"},
                    )
            for key in ("symbol", "text"):
                if key in normalized:
                    value = normalized[key]
                    if not isinstance(value, str) or not value.strip() or len(value) > 500:
                        raise ValueError("invalid_query")
                    normalized[key] = value.strip()
            if intent.action == "read_range":
                start = normalized["start_line"]
                end = normalized["end_line"]
                if (
                    not isinstance(start, int)
                    or isinstance(start, bool)
                    or not isinstance(end, int)
                    or isinstance(end, bool)
                    or start < 1
                    or end < start
                    or end - start + 1 > 400
                ):
                    raise ValueError("invalid_range")
            if intent.action == "run_targeted_test":
                argv, normalized_command = _test_argv(root, state, normalized["command"])
                normalized["command"] = normalized_command
            if intent.action not in {"request_repair", "finish_investigation"}:
                head = _checkout_head(root)
                if head != state.repo_ref.lower():
                    raise ValueError("repo_ref_mismatch")
        except (OSError, ValueError, subprocess.SubprocessError):
            return _reject(empty_fingerprint, "unsafe_args")

        payload = json.dumps(
            {"action": intent.action, "args": normalized},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if any(item.args_fingerprint == fingerprint for item in state.tool_history):
            return _reject(fingerprint, "duplicate_call", duplicate=True)
        return PolicyDecision(
            approved=True,
            status="approved",
            args_fingerprint=fingerprint,
            normalized_args=normalized,
            argv=argv,
        )
