"""Deterministic safety policy for model-selected local tools."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .local_search import is_sensitive_repo_path
from .safe_subprocess import run_bounded_process, trusted_executable, trusted_python
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
_PACKAGE_TEST_SCRIPT_RE = re.compile(r"test(?::[A-Za-z0-9_.-]+)?")
_PACKAGE_EXECUTABLES = frozenset({"npm", "pnpm", "yarn", "bun"})

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
    if is_sensitive_repo_path(raw):
        raise ValueError("sensitive_path")
    candidate = root / raw
    if candidate.is_symlink():
        raise ValueError("symlink_path")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("path_escape")
    if is_sensitive_repo_path(resolved.relative_to(root).as_posix()):
        raise ValueError("sensitive_path")
    if require_file and not resolved.is_file():
        raise ValueError("path_not_file")
    return resolved.relative_to(root).as_posix() or "."


def _git_result(root: Path, argv: list[str], *, max_output_bytes: int = 256_000) -> str:
    result = run_bounded_process(
        argv,
        cwd=root,
        timeout=10,
        max_output_bytes=max_output_bytes,
    )
    if result.returncode != 0:
        raise ValueError("git_query_failed")
    return result.stdout


def _is_test_path(path: str) -> bool:
    parts = Path(path).parts
    name = Path(path).name
    is_test_name = name.startswith("test_") and name.endswith(".py")
    is_test_name = is_test_name or name.endswith("_test.py")
    in_test_root = not parts[:-1] or any(
        part.lower() in {"test", "tests"} for part in parts[:-1]
    )
    return is_test_name and in_test_root


def _is_exact_base_member(root: Path, state: AgentState, path: str) -> bool:
    try:
        output = _git_result(
            root,
            ["git", "-C", str(root), "ls-tree", state.repo_ref, "--", path],
            max_output_bytes=8_192,
        )
    except (OSError, RuntimeError, ValueError):
        return False
    record = output.rstrip("\n")
    return bool(re.fullmatch(rf"100(?:644|755) blob [0-9a-f]{{40}}\t{re.escape(path)}", record))


def _is_approved_generated_test(root: Path, state: AgentState, path: str) -> bool:
    patch_approval = state.tool_patch_approval
    if patch_approval is None or patch_approval.base_ref.lower() != state.repo_ref.lower():
        return False
    patch_sha = hashlib.sha256(state.patch_content.encode("utf-8")).hexdigest()
    if patch_sha != patch_approval.patch_sha256:
        return False
    try:
        current = (root / path).read_bytes()
    except OSError:
        return False
    content_sha = hashlib.sha256(current).hexdigest()
    return any(
        approval.path == path
        and approval.content_sha256 == content_sha
        and approval.patch_gate_fingerprint == patch_approval.patch_gate_fingerprint
        for approval in state.generated_test_approvals
    )


def _target_selector(root: Path, state: AgentState, token: str) -> bool:
    path_part = token.split("::", 1)[0]
    if not path_part or path_part.startswith("-"):
        return False
    try:
        normalized = _relative_path(root, path_part, require_file=True)
    except ValueError:
        return False
    if not _is_test_path(normalized):
        return False
    return _is_exact_base_member(root, state, normalized) or _is_approved_generated_test(
        root, state, normalized
    )


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


def _committed_package_scripts(root: Path, state: AgentState) -> dict[str, object]:
    manifest = root / "package.json"
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError("unsafe_manifest")
    committed = _git_result(
        root,
        ["git", "-C", str(root), "show", f"{state.repo_ref}:package.json"],
    )
    try:
        working = manifest.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("unsafe_manifest") from exc
    if working != committed:
        raise ValueError("modified_manifest")
    try:
        value = json.loads(committed)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid_manifest") from exc
    scripts = value.get("scripts") if isinstance(value, dict) else None
    if not isinstance(scripts, dict):
        raise ValueError("missing_test_script")
    return scripts


def _project_test_prefix(
    root: Path, state: AgentState, tokens: list[str]
) -> str:
    if not tokens or tokens[0].lower() not in _PACKAGE_EXECUTABLES:
        raise ValueError("unsupported_project_runner")
    executable = tokens[0].lower()
    if len(tokens) == 2 and tokens[1] == "test":
        script = "test"
    elif len(tokens) == 3 and tokens[1] == "run":
        script = tokens[2]
    else:
        raise ValueError("unsafe_project_options")
    if not _PACKAGE_TEST_SCRIPT_RE.fullmatch(script):
        raise ValueError("non_test_script")
    scripts = _committed_package_scripts(root, state)
    if not isinstance(scripts.get(script), str) or not str(scripts[script]).strip():
        raise ValueError("missing_test_script")
    launcher = trusted_executable(executable, required=False)
    if launcher is None:
        raise ValueError("untrusted_project_runner")
    return launcher


def _checkout_head(root: Path) -> str:
    return _git_result(
        root,
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        max_output_bytes=4_096,
    ).strip()


def _test_argv(root: Path, state: AgentState, command: object) -> tuple[list[str], str]:
    tokens = _safe_tokens(command)
    if tokens[0] == "pytest":
        suffix = tokens[1:]
        argv = [trusted_python(), "-m", "pytest", *suffix]
    elif tokens[:3] == ["python", "-m", "pytest"]:
        suffix = tokens[3:]
        argv = [trusted_python(), "-m", "pytest", *suffix]
    else:
        configured = _safe_tokens(state.test_command) if state.test_command else []
        launcher = _project_test_prefix(root, state, configured) if configured else ""
        if not launcher or tokens[: len(configured)] != configured:
            raise ValueError("test_command_not_allowlisted")
        suffix = tokens[len(configured) :]
        if not suffix or suffix[0] != "--":
            raise ValueError("target_separator_required")
        suffix = suffix[1:]
        argv = [launcher, *tokens[1:]]
    selectors = [token for token in suffix if _target_selector(root, state, token)]
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
        except (OSError, RuntimeError, ValueError):
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
