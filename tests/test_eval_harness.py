from __future__ import annotations

import sys
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval import harness
from src.safe_subprocess import ProcessOutputLimitError, minimal_subprocess_env
from src.tool_policy import PYTEST_BOOTSTRAP

BASE_COMMIT = "a" * 40


def test_agent_v2_eval_wrapper_uses_five_transaction_default():
    assert (
        inspect.signature(harness.run_agent_v2_eval)
        .parameters["max_retries"]
        .default
        == 4
    )


def _sample(base_commit: object = BASE_COMMIT) -> dict:
    sample = {
        "id": "acme/widget#7:8",
        "repo": {"owner": "acme", "name": "widget"},
        "issue": {"title": "Fix widget", "body": "The widget is broken."},
        "patch": {"files": [{"path": "src/widget.py"}]},
        "signals": {"has_tests_changed": False},
    }
    if base_commit is not None:
        sample["base_commit"] = base_commit
    return sample


@pytest.mark.parametrize(
    ("api_opt_in", "environment_value"),
    [
        (False, None),
        (False, "1"),
        (True, None),
        (True, "true"),
    ],
)
async def test_run_eval_requires_api_and_exact_environment_opt_ins_before_load(
    monkeypatch, api_opt_in, environment_value
):
    if environment_value is None:
        monkeypatch.delenv(
            "REPOPILOT_UNSAFE_ALLOW_HOST_EXECUTION", raising=False
        )
    else:
        monkeypatch.setenv(
            "REPOPILOT_UNSAFE_ALLOW_HOST_EXECUTION", environment_value
        )

    def forbidden_load(*_args, **_kwargs):
        raise AssertionError("samples were loaded before authorization")

    monkeypatch.setattr(harness, "load_samples", forbidden_load)

    with pytest.raises(PermissionError, match="legacy host evaluation"):
        await harness.run_eval(unsafe_allow_host_execution=api_opt_in)


async def test_evaluate_sample_requires_api_and_environment_opt_ins_before_sample_access(
    monkeypatch,
):
    monkeypatch.setenv("REPOPILOT_UNSAFE_ALLOW_HOST_EXECUTION", "1")

    with pytest.raises(PermissionError, match="legacy host evaluation"):
        await harness.evaluate_sample(
            object(), 0, unsafe_allow_host_execution=False
        )


async def test_unauthorized_api_does_not_lazy_import_model_or_http_modules(monkeypatch):
    monkeypatch.setenv("REPOPILOT_UNSAFE_ALLOW_HOST_EXECUTION", "1")
    monkeypatch.setattr(
        harness.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(
            AssertionError(f"unauthorized API imported {name}")
        ),
    )

    with pytest.raises(PermissionError, match="legacy host evaluation"):
        await harness.run_eval(unsafe_allow_host_execution=False)


def test_cli_requires_an_explicit_evaluator_and_runs_nothing_by_default(monkeypatch):
    calls = []

    async def fake_legacy(**kwargs):
        calls.append(("legacy", kwargs))
        return []

    async def fake_agent_v2(**kwargs):
        calls.append(("agent-v2", kwargs))
        return []

    monkeypatch.setattr(harness, "run_eval", fake_legacy)
    monkeypatch.setattr(harness, "run_agent_v2_eval", fake_agent_v2)

    with pytest.raises(SystemExit):
        harness.main([])

    assert calls == []


def test_cli_legacy_selection_requires_exact_environment_opt_in(monkeypatch):
    calls = []

    async def fake_legacy(**kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(harness, "run_eval", fake_legacy)
    monkeypatch.setenv("REPOPILOT_UNSAFE_ALLOW_HOST_EXECUTION", "yes")

    with pytest.raises(PermissionError, match="legacy host evaluation"):
        harness.main(["--unsafe-legacy"])

    assert calls == []


def test_cli_legacy_selection_passes_explicit_api_opt_in(monkeypatch):
    calls = []

    async def fake_legacy(**kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(harness, "run_eval", fake_legacy)
    monkeypatch.setenv("REPOPILOT_UNSAFE_ALLOW_HOST_EXECUTION", "1")

    harness.main(["--unsafe-legacy", "--samples", "2"])

    assert calls == [
        {
            "n_samples": 2,
            "model": harness.DEFAULT_MODEL,
            "unsafe_allow_host_execution": True,
        }
    ]


@pytest.mark.parametrize(
    "base_commit",
    [
        None,
        "a" * 39,
        "a" * 41,
        "A" * 40,
        "g" * 40,
        123,
    ],
)
async def test_legacy_sample_rejects_missing_or_malformed_commit_before_side_effects(
    monkeypatch, base_commit
):
    monkeypatch.setenv("REPOPILOT_UNSAFE_ALLOW_HOST_EXECUTION", "1")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("side effect occurred before commit validation")

    monkeypatch.setattr(harness, "clone_repo", forbidden)
    monkeypatch.setattr(harness, "fetch_file_list", forbidden)
    monkeypatch.setattr(harness, "fetch_file_content", forbidden)
    monkeypatch.setattr(harness, "search_code_via_github", forbidden)

    with pytest.raises(ValueError, match="base_commit"):
        await harness.evaluate_sample(
            _sample(base_commit),
            0,
            unsafe_allow_host_execution=True,
        )


async def test_run_eval_validates_every_commit_before_creating_workspace_or_evaluating(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("REPOPILOT_UNSAFE_ALLOW_HOST_EXECUTION", "1")
    monkeypatch.setattr(
        harness,
        "load_samples",
        lambda _n: [_sample(), _sample(None)],
    )
    results_path = tmp_path / "must-not-exist.json"
    monkeypatch.setattr(harness, "RESULTS_PATH", results_path)

    def forbidden_workspace(*_args, **_kwargs):
        raise AssertionError("workspace created before all samples were validated")

    monkeypatch.setattr(harness.tempfile, "TemporaryDirectory", forbidden_workspace)

    async def forbidden_evaluate(*_args, **_kwargs):
        raise AssertionError("evaluation started before all samples were validated")

    monkeypatch.setattr(harness, "evaluate_sample", forbidden_evaluate)

    with pytest.raises(ValueError, match="base_commit"):
        await harness.run_eval(unsafe_allow_host_execution=True)

    assert not results_path.exists()


def test_clone_fetches_and_verifies_exact_detached_commit(monkeypatch, tmp_path):
    target = tmp_path / "checkout"
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[-3:] == ["symbolic-ref", "-q", "HEAD"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        stdout = BASE_COMMIT + "\n" if command[-2:] == ["rev-parse", "HEAD"] else ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(harness.subprocess, "run", fake_run)

    assert harness.clone_repo(
        "acme", "widget", BASE_COMMIT, target, eval_root=tmp_path
    )

    commands = [command for command, _kwargs in calls]
    assert any(command[-2:] == ["init", str(target)] for command in commands)
    assert any(
        command[-6:]
        == [
            "-C",
            str(target),
            "remote",
            "add",
            "origin",
            "https://github.com/acme/widget.git",
        ]
        for command in commands
    )
    assert any(
        command[-8:]
        == [
            "-C",
            str(target),
            "fetch",
            "--depth",
            "1",
            "--no-tags",
            "origin",
            BASE_COMMIT,
        ]
        for command in commands
    )
    assert any(
        command[-5:] == ["-C", str(target), "checkout", "--detach", BASE_COMMIT]
        for command in commands
    )
    assert any(
        command[-4:] == ["-C", str(target), "rev-parse", "HEAD"]
        for command in commands
    )
    assert any(
        command[-5:] == ["-C", str(target), "symbolic-ref", "-q", "HEAD"]
        for command in commands
    )
    for command in commands:
        assert "credential.helper=" in command
        assert "credential.interactive=never" in command
        assert "core.hooksPath=/dev/null" in command
    assert not any("clone" in command for command in commands)
    for _command, kwargs in calls:
        assert kwargs["env"]["HOME"] != str(Path.home())
        assert "GITHUB_TOKEN" not in kwargs["env"]
        assert "LLM_API_KEY" not in kwargs["env"]


def test_clone_rejects_malformed_commit_before_filesystem_or_subprocess(
    monkeypatch, tmp_path
):
    target = tmp_path / "must-not-exist"
    monkeypatch.setattr(
        harness.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("subprocess called for malformed commit")
        ),
    )

    with pytest.raises(ValueError, match="base_commit"):
        harness.clone_repo(
            "acme", "widget", "main", target, eval_root=tmp_path
        )

    assert not target.exists()


def test_clone_rejects_checkout_outside_exact_eval_root_before_removal(
    monkeypatch, tmp_path
):
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(
        harness.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("subprocess called for escaped checkout")
        ),
    )

    with pytest.raises(ValueError, match="checkout.*eval root"):
        harness.clone_repo(
            "acme",
            "widget",
            BASE_COMMIT,
            eval_root / ".." / "outside",
            eval_root=eval_root,
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize(
    "sample_id",
    ["..", "../outside", "./.", "a/b", "a\\b", ".#.:.."],
)
async def test_sample_id_never_controls_checkout_or_cleanup_path(
    monkeypatch, tmp_path, sample_id
):
    monkeypatch.setenv("REPOPILOT_UNSAFE_ALLOW_HOST_EXECUTION", "1")
    eval_root = tmp_path / "eval"
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    seen = []

    def fake_clone(_owner, _repo, _commit, target, *, eval_root, **_kwargs):
        seen.append((target, eval_root))
        return False

    monkeypatch.setattr(harness, "clone_repo", fake_clone)
    sample = _sample()
    sample["id"] = sample_id

    result = await harness.evaluate_sample(
        sample,
        0,
        unsafe_allow_host_execution=True,
        _eval_root=eval_root,
    )

    assert result["error"] == "clone_failed"
    assert seen == [(eval_root / "checkout", eval_root)]
    assert sentinel.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize(
    ("head", "detached_returncode"),
    [("b" * 40, 1), (BASE_COMMIT, 0)],
)
def test_clone_rejects_head_mismatch_or_attached_checkout(
    monkeypatch, tmp_path, head, detached_returncode
):
    def fake_run(command, **_kwargs):
        if command[-2:] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout=head + "\n", stderr="")
        if command[-3:] == ["symbolic-ref", "-q", "HEAD"]:
            return SimpleNamespace(
                returncode=detached_returncode, stdout="refs/heads/main\n", stderr=""
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(harness.subprocess, "run", fake_run)

    assert not harness.clone_repo(
        "acme",
        "widget",
        BASE_COMMIT,
        tmp_path / "checkout",
        eval_root=tmp_path,
    )


def test_clone_removes_partial_checkout_when_subprocess_raises_baseexception(
    monkeypatch, tmp_path
):
    target = tmp_path / "checkout"
    calls = 0

    def fake_run(_command, **_kwargs):
        nonlocal calls
        calls += 1
        target.mkdir(parents=True, exist_ok=True)
        if calls == 2:
            raise KeyboardInterrupt("cancelled")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(harness.subprocess, "run", fake_run)

    with pytest.raises(KeyboardInterrupt, match="cancelled"):
        harness.clone_repo(
            "acme", "widget", BASE_COMMIT, target, eval_root=tmp_path
        )

    assert not target.exists()
    assert not (tmp_path / ".checkout-git-home").exists()


def test_github_tree_and_content_requests_and_caches_bind_exact_commit(monkeypatch):
    commits = ("a" * 40, "b" * 40)
    urls = []

    def fake_get(url, timeout=30):
        urls.append((url, timeout))
        if "/git/trees/" in url:
            return {"tree": [{"type": "blob", "path": url[-40:]}]}
        return {"content": "Y29udGVudA=="}

    harness._gh_file_cache.clear()
    harness._gh_content_cache.clear()
    monkeypatch.setattr(harness, "_gh_get", fake_get)

    assert harness.fetch_file_list("acme", "widget", commits[0]) != harness.fetch_file_list(
        "acme", "widget", commits[1]
    )
    assert harness.fetch_file_content(
        "acme", "widget", "src/widget.py", commits[0]
    ) == "content"
    assert harness.fetch_file_content(
        "acme", "widget", "src/widget.py", commits[1]
    ) == "content"

    requested = [url for url, _timeout in urls]
    assert any(f"/git/trees/{commits[0]}?" in url for url in requested)
    assert any(f"/git/trees/{commits[1]}?" in url for url in requested)
    assert any(f"ref={commits[0]}" in url for url in requested)
    assert any(f"ref={commits[1]}" in url for url in requested)
    assert len(requested) == 4


def test_github_content_helper_rejects_mutable_ref_before_network(monkeypatch):
    monkeypatch.setattr(
        harness,
        "_gh_get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("network called for mutable ref")
        ),
    )

    with pytest.raises(ValueError, match="base_commit"):
        harness.fetch_file_content("acme", "widget", "src/widget.py", "main")


async def test_legacy_evaluation_uses_exact_checkout_local_search_not_github_search(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("REPOPILOT_UNSAFE_ALLOW_HOST_EXECUTION", "1")
    monkeypatch.setattr(harness, "clone_repo", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(harness, "grep_repo", lambda *_args, **_kwargs: ["src/widget.py"])
    monkeypatch.setattr(
        harness, "read_file_content", lambda *_args, **_kwargs: "def widget(): pass"
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("mutable GitHub search/content path was used")

    monkeypatch.setattr(harness, "fetch_file_list", forbidden)
    monkeypatch.setattr(harness, "fetch_file_content", forbidden)
    monkeypatch.setattr(harness, "search_code_via_github", forbidden)
    monkeypatch.setattr(
        harness,
        "run_bounded_process",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="src/widget.py\0", stderr=""
        ),
    )

    async def fake_locate(*_args, **_kwargs):
        return ["src/widget.py"], 0, 0

    async def fake_patch(*_args, **_kwargs):
        return "", 0, 0

    monkeypatch.setattr(harness, "locate_files_phase", fake_locate)
    monkeypatch.setattr(harness, "generate_patch_phase", fake_patch)

    result = await harness.evaluate_sample(
        _sample(),
        0,
        unsafe_allow_host_execution=True,
        _eval_root=tmp_path,
    )

    assert result["error"] is None


def test_every_legacy_subprocess_uses_production_minimal_environment(
    monkeypatch, tmp_path
):
    sentinels = {
        "LLM_API_KEY": "model-secret",
        "GITHUB_TOKEN": "github-secret",
        "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "actions-secret",
        "UNRELATED_SECRET": "ambient-secret",
    }
    for key, value in sentinels.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("PATH", str(tmp_path / "hostile-bin"))

    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[-3:] == ["symbolic-ref", "-q", "HEAD"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        if command[-2:] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(
                returncode=0, stdout=BASE_COMMIT + "\n", stderr=""
            )
        if "apply" in command or command[0] == "patch":
            return SimpleNamespace(returncode=1, stdout="", stderr="rejected")
        if command[0] == "rg":
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        if "pytest" in command:
            return SimpleNamespace(returncode=1, stdout="failed", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    repo_path = tmp_path / "checkout"

    assert harness.clone_repo(
        "acme", "widget", BASE_COMMIT, repo_path, eval_root=tmp_path
    )
    harness.apply_patch(repo_path, "not a patch")
    harness.run_tests(repo_path)

    assert calls
    for _command, kwargs in calls:
        expected = minimal_subprocess_env()
        if "HOME" in kwargs["env"]:
            expected = minimal_subprocess_env({"HOME": kwargs["env"]["HOME"]})
            assert Path(kwargs["env"]["HOME"]).name.endswith("-git-home")
        assert kwargs["env"] == expected
        assert not sentinels.keys() & kwargs["env"].keys()
        assert kwargs["env"]["PATH"] != str(tmp_path / "hostile-bin")


def test_run_tests_uses_version_stable_trusted_pytest_bootstrap(monkeypatch, tmp_path):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="1 passed", stderr="")

    monkeypatch.setattr(harness.subprocess, "run", fake_run)

    success, output = harness.run_tests(tmp_path)

    assert success is True
    assert "1 passed" in output
    assert captured["command"] == [
        sys.executable,
        "-I",
        "-c",
        PYTEST_BOOTSTRAP,
        "-x",
        "-q",
        "--tb=short",
    ]
    assert captured["cwd"] == str(tmp_path)
    assert captured["env"] == minimal_subprocess_env()


def test_tracked_search_excludes_git_metadata_and_untracked_files(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "src").mkdir()
    (repo / ".git" / "config").write_text("needle", encoding="utf-8")
    (repo / "untracked.py").write_text("needle", encoding="utf-8")
    (repo / "src" / "tracked.py").write_text("needle", encoding="utf-8")

    results = harness.grep_repo(
        repo,
        (".git/config", "src/tracked.py"),
        frozenset({".git/config", "src/tracked.py"}),
        ["needle"],
    )

    assert results == ["src/tracked.py"]


def test_tracked_search_scans_once_with_aggregate_byte_budget(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    for name in ("one.py", "two.py", "three.py"):
        (repo / name).write_text("x" * 100 + " needle", encoding="utf-8")

    results = harness.grep_repo(
        repo,
        ("one.py", "two.py", "three.py"),
        frozenset({"one.py", "two.py", "three.py"}),
        ["needle", "other", "third"],
        max_total_bytes=150,
    )

    assert results == ["one.py"]


async def test_internal_ls_files_uses_bounded_output(monkeypatch, tmp_path):
    monkeypatch.setenv("REPOPILOT_UNSAFE_ALLOW_HOST_EXECUTION", "1")
    monkeypatch.setenv("LLM_API_KEY", "model-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "github-secret")
    seen = []
    monkeypatch.setattr(harness, "clone_repo", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(harness, "grep_repo", lambda *_args, **_kwargs: [])

    def fake_run(command, **kwargs):
        seen.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="src/widget.py\0", stderr="")

    monkeypatch.setattr(harness, "run_bounded_process", fake_run)

    async def fake_locate(*_args, **_kwargs):
        return [], 0, 0

    async def fake_patch(*_args, **_kwargs):
        return "", 0, 0

    monkeypatch.setattr(harness, "locate_files_phase", fake_locate)
    monkeypatch.setattr(harness, "generate_patch_phase", fake_patch)

    await harness.evaluate_sample(
        _sample(),
        0,
        unsafe_allow_host_execution=True,
        _eval_root=tmp_path,
    )

    ls_call = next(call for call in seen if call[0] == ["git", "ls-files", "-z"])
    assert ls_call[1]["max_output_bytes"] == harness.MAX_TRACKED_LIST_BYTES
    assert ls_call[1]["decode_errors"] == "strict"
    assert "env_overrides" not in ls_call[1]


def test_tracked_file_list_rejects_too_many_entries(monkeypatch, tmp_path):
    output = "x\0" * (harness.MAX_TRACKED_FILES + 1)
    monkeypatch.setattr(
        harness,
        "run_bounded_process",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=output, stderr=""
        ),
    )

    with pytest.raises(ValueError, match="too many tracked files"):
        harness._load_tracked_files(tmp_path)


@pytest.mark.parametrize(
    "output",
    [
        "unterminated",
        "dup.py\0dup.py\0",
        "../escape.py\0",
        ".git/config\0",
        ("x" * 1_025) + "\0",
    ],
)
def test_tracked_file_list_rejects_malformed_paths(monkeypatch, tmp_path, output):
    monkeypatch.setattr(
        harness,
        "run_bounded_process",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=output, stderr=""
        ),
    )

    with pytest.raises(ValueError, match="tracked file list"):
        harness._load_tracked_files(tmp_path)


async def test_tracked_file_output_overflow_fails_before_llm(monkeypatch, tmp_path):
    monkeypatch.setenv("REPOPILOT_UNSAFE_ALLOW_HOST_EXECUTION", "1")
    monkeypatch.setattr(harness, "clone_repo", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        harness,
        "run_bounded_process",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ProcessOutputLimitError("truncated", "")
        ),
    )

    async def forbidden_llm(*_args, **_kwargs):
        raise AssertionError("LLM called after tracked-file output overflow")

    monkeypatch.setattr(harness, "locate_files_phase", forbidden_llm)
    monkeypatch.setattr(harness, "generate_patch_phase", forbidden_llm)

    result = await harness.evaluate_sample(
        _sample(),
        0,
        unsafe_allow_host_execution=True,
        _eval_root=tmp_path,
    )

    assert result["error"].startswith("ProcessOutputLimitError:")


def test_read_file_content_requires_tracked_safe_path(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("host secret", encoding="utf-8")
    (repo / "link.txt").symlink_to(outside)
    (repo / ".git").mkdir()
    (repo / ".git" / "config").write_text("git secret", encoding="utf-8")
    (repo / "alias").symlink_to(repo / ".git", target_is_directory=True)
    (repo / "untracked.txt").write_text("untracked", encoding="utf-8")
    (repo / "tracked.txt").write_text("tracked", encoding="utf-8")
    oversized = repo / "oversized.txt"
    oversized.write_bytes(b"x" * (harness.MAX_SOURCE_FILE_BYTES + 1))
    tracked = frozenset(
        {
            "../outside.txt",
            "link.txt",
            ".git/config",
            "alias/config",
            "tracked.txt",
            "oversized.txt",
        }
    )

    assert harness.read_file_content(repo, "../outside.txt", tracked_files=tracked) == ""
    assert harness.read_file_content(repo, "link.txt", tracked_files=tracked) == ""
    assert harness.read_file_content(repo, ".git/config", tracked_files=tracked) == ""
    assert harness.read_file_content(repo, "alias/config", tracked_files=tracked) == ""
    assert harness.read_file_content(repo, "untracked.txt", tracked_files=tracked) == ""
    assert harness.read_file_content(repo, "oversized.txt", tracked_files=tracked) == ""
    assert (
        harness.read_file_content(repo, "tracked.txt", tracked_files=tracked)
        == "tracked"
    )


def test_apply_patch_uses_only_strict_git_apply(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(harness.subprocess, "run", fake_run)

    assert harness.apply_patch(tmp_path, "not a patch") == (True, "")
    assert commands == [["git", "apply", "--check"], ["git", "apply"]]
