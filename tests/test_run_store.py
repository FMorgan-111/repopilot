import json
import multiprocessing
import os
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from src import new_agent, run_store
from src.repair_rounds import (
    begin_repair_round,
    freeze_authorized_repair_round,
    record_failed_repair_round,
)
from src.state import (
    Evidence,
    FixAttempt,
    ModelInvocation,
    NoProgressEvent,
    PatchEdit,
    Phase,
    ToolPatchApproval,
)


def _claim_in_subprocess(
    root_dir: str,
    run_id: str,
    expected_payload: dict,
    start,
    results,
) -> None:
    expected = new_agent.AgentState.model_validate(expected_payload)
    start.wait()
    try:
        run_store.claim_run_for_resume(
            run_id,
            expected,
            root_dir=Path(root_dir),
        )
    except run_store.ResumeConflictError:
        results.put("conflict")
    except BaseException as exc:
        results.put(f"error:{type(exc).__name__}")
    else:
        results.put("claimed")


def paused_state(trace_id: str = "abc123def456"):
    frame = new_agent.DecisionFrame(
        frame_id="df_0001",
        stage="plan",
        summary="Need user approval before patching.",
        hypotheses=[
            new_agent.Hypothesis(
                id="H1",
                claim="The change is too risky to ship without confirmation.",
                evidence=["The issue requests a breaking API change."],
                score=0.88,
            )
        ],
        selected_hypothesis_id="H1",
        next_checks=["Confirm whether breaking changes are allowed."],
        recommended_action="ask_user",
        confidence=0.88,
        risk="high",
    )
    return new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        trace_id=trace_id,
        current_phase=new_agent.Phase.WAITING_FOR_USER,
        pending_human_input=True,
        human_input_request={
            "frame_id": "df_0001",
            "stage": "plan",
            "question": "Confirm whether breaking changes are allowed.",
            "summary": "Need user approval before patching.",
            "risk": "high",
            "confidence": 0.88,
        },
        decision_frame=frame,
        frame_history=[frame],
        route_decisions=[
            {
                "source": "decision_frame",
                "current_phase": "PLAN",
                "selected_phase": "WAITING_FOR_USER",
                "route": "WAITING_FOR_USER",
                "frame_id": "df_0001",
                "recommended_action": "ask_user",
            }
        ],
    )


def replay_state(trace_id: str = "abc123def456"):
    plan_frame = new_agent.DecisionFrame(
        frame_id="df_0001",
        stage="plan",
        summary="Need user approval before patching.",
        hypotheses=[
            new_agent.Hypothesis(
                id="H1",
                claim="The API change may be breaking.",
                evidence=["The issue asks to remove an existing field."],
                score=0.82,
                why_selected="It explains the compatibility risk.",
            )
        ],
        selected_hypothesis_id="H1",
        evidence=["Existing clients depend on the field."],
        next_checks=["Confirm whether breaking changes are allowed."],
        recommended_action="ask_user",
        confidence=0.82,
        risk="high",
        trace_notes="Planner stopped before patching.",
    )
    reflect_frame = new_agent.DecisionFrame(
        frame_id="df_0002",
        stage="reflect",
        summary="Previous patch failed because tests expect compatibility.",
        selected_hypothesis_id="H2",
        evidence=["Regression test failed on missing field."],
        recommended_action="plan",
        confidence=0.74,
        risk="medium",
        parent_frame_id="df_0001",
    )
    return new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        trace_id=trace_id,
        current_phase=new_agent.Phase.WAITING_FOR_USER,
        pending_human_input=True,
        human_input_request={
            "frame_id": "df_0001",
            "stage": "plan",
            "question": "Confirm whether breaking changes are allowed.",
            "summary": "Need user approval before patching.",
            "risk": "high",
            "confidence": 0.82,
        },
        decision_frame=plan_frame,
        frame_history=[plan_frame, reflect_frame],
        decision_warnings=[
            {
                "frame_id": "df_0001",
                "recommended_action": "ask_user",
                "expected_phase": "WAITING_FOR_USER",
                "actual_phase": "PLAN",
            }
        ],
        route_decisions=[
            {
                "source": "decision_frame",
                "current_phase": "PLAN",
                "selected_phase": "WAITING_FOR_USER",
                "route": "__end__",
                "frame_id": "df_0001",
                "recommended_action": "ask_user",
            },
            {
                "source": "current_phase",
                "current_phase": "PLAN",
                "selected_phase": "PLAN",
                "route": "plan_fix",
                "fallback_reason": "already_consumed",
            },
        ],
    )


def diagnostic_state(trace_id: str = "diag123"):
    state = replay_state(trace_id)
    state.node_diagnostics.append(
        {
            "node": "plan_fix",
            "event": "phase",
            "status": "timeout",
            "elapsed_seconds": 90.0,
            "error_type": "TimeoutError",
            "error": "TimeoutError",
            "phase_timeout_seconds": 90.0,
        }
    )
    return state


@pytest.mark.parametrize(
    "run_id",
    [
        "../escape",
        "/absolute",
        "nested/name",
        r"nested\\name",
        ".hidden",
        "-leading",
        "a" * 65,
        "white space",
        "unicode-运行",
        "",
    ],
)
def test_run_store_rejects_unsafe_run_ids_before_path_use(tmp_path, run_id):
    with pytest.raises(ValueError, match="run_id"):
        run_store.run_path(run_id, root_dir=tmp_path / "missing-root")

    assert not (tmp_path / "missing-root").exists()


@pytest.mark.parametrize("run_id", ["a", "A0", "run-1", "run_2", "z" * 64])
def test_run_store_accepts_exact_safe_run_id_grammar(tmp_path, run_id):
    assert run_store.run_path(run_id, root_dir=tmp_path) == (
        tmp_path / "runs" / f"{run_id}.json"
    )


def test_run_store_rejects_symlinked_runs_directory(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    (root / "runs").symlink_to(real, target_is_directory=True)

    with pytest.raises(ValueError, match="runs directory"):
        run_store.save_run(paused_state("safe-id"), root_dir=root)


def test_run_store_rejects_symlinked_root_directory(tmp_path):
    real = tmp_path / "real-root"
    real.mkdir()
    root = tmp_path / "root"
    root.symlink_to(real, target_is_directory=True)

    with pytest.raises(ValueError, match="root directory"):
        run_store.save_run(paused_state("safe-id"), root_dir=root)


def test_run_store_creates_missing_nested_root_through_anchored_walk(tmp_path):
    root = tmp_path / "missing" / "nested" / "root"

    saved = run_store.save_run(paused_state("safe-id"), root_dir=root)

    assert saved == root / "runs" / "safe-id.json"
    assert run_store.load_run("safe-id", root_dir=root).trace_id == "safe-id"


def test_run_store_rejects_symlinked_intermediate_root_component(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(ValueError, match="root directory components"):
        run_store.save_run(
            paused_state("safe-id"), root_dir=linked / "nested" / "root"
        )


def test_run_store_rejects_runs_directory_swap_during_open(tmp_path, monkeypatch):
    root = tmp_path / "root"
    runs = root / "runs"
    runs.mkdir(parents=True)
    original_open = run_store.os.open
    swapped = False

    def swapping_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == "runs" and kwargs.get("dir_fd") is not None and not swapped:
            swapped = True
            runs.rename(root / "old-runs")
            runs.mkdir()
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(run_store.os, "open", swapping_open)

    with pytest.raises(ValueError, match="changed while it was opened"):
        run_store.save_run(paused_state("safe-id"), root_dir=root)


def test_list_runs_rejects_dangling_runs_symlink(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "runs").symlink_to(root / "missing", target_is_directory=True)

    with pytest.raises(ValueError, match="runs directory"):
        run_store.list_runs(root_dir=root)


def test_first_runs_directory_creation_fsyncs_root(tmp_path, monkeypatch):
    root = tmp_path / "root"
    synced_identities = []
    original_fsync = run_store.os.fsync

    def recording_fsync(descriptor):
        info = run_store.os.fstat(descriptor)
        synced_identities.append((info.st_dev, info.st_ino, info.st_mode))
        return original_fsync(descriptor)

    monkeypatch.setattr(run_store.os, "fsync", recording_fsync)
    run_store.save_run(paused_state("safe-id"), root_dir=root)

    parent_info = root.parent.stat()
    assert (
        parent_info.st_dev,
        parent_info.st_ino,
    ) in {(device, inode) for device, inode, _mode in synced_identities}
    assert any(stat.S_ISDIR(mode) for _device, _inode, mode in synced_identities)


def test_list_runs_uses_descriptor_anchored_timestamp(tmp_path, monkeypatch):
    root = tmp_path / "root"
    run_store.save_run(paused_state("safe-id"), root_dir=root)

    monkeypatch.setattr(
        run_store,
        "_updated_at",
        lambda _path: pytest.fail("list_runs must not stat an unanchored path"),
    )

    [summary] = run_store.list_runs(root_dir=root)
    inspected = run_store.inspect_run("safe-id", root_dir=root)

    assert summary["run_id"] == "safe-id"
    assert summary["updated_at"].endswith("+00:00")
    assert inspected["updated_at"].endswith("+00:00")


@pytest.mark.parametrize("failure", ["resolve", "stat", "open", "fstat"])
def test_run_store_closes_descriptors_on_directory_validation_failure(
    tmp_path, monkeypatch, failure
):
    root = tmp_path / "root"
    (root / "runs").mkdir(parents=True)
    before = len(os.listdir("/dev/fd"))

    if failure == "resolve":
        original = run_store.Path.resolve

        def fail_resolve(self, *args, **kwargs):
            if self == root:
                raise OSError("injected resolve failure")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(run_store.Path, "resolve", fail_resolve)
    elif failure == "stat":
        original = run_store.os.stat

        def fail_stat(path, *args, **kwargs):
            if path == root:
                raise OSError("injected stat failure")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(run_store.os, "stat", fail_stat)
    elif failure == "open":
        original = run_store.os.open

        def fail_open(path, flags, *args, **kwargs):
            if path == "runs":
                raise OSError("injected open failure")
            return original(path, flags, *args, **kwargs)

        monkeypatch.setattr(run_store.os, "open", fail_open)
    else:
        original = run_store.os.fstat
        calls = 0

        def fail_fstat(descriptor):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected fstat failure")
            return original(descriptor)

        monkeypatch.setattr(run_store.os, "fstat", fail_fstat)

    with pytest.raises((OSError, ValueError), match="injected|opened safely"):
        run_store.save_run(paused_state("safe-id"), root_dir=root)

    assert len(os.listdir("/dev/fd")) == before


def test_run_store_rejects_non_directory_runs_path(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "runs").write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match="runs directory"):
        run_store.save_run(paused_state("safe-id"), root_dir=root)


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_run_store_load_rejects_nonregular_run_file(tmp_path, kind):
    root = tmp_path / "root"
    directory = root / "runs"
    directory.mkdir(parents=True)
    target = directory / "safe-id.json"
    if kind == "symlink":
        outside = tmp_path / "outside.json"
        outside.write_text("{}", encoding="utf-8")
        target.symlink_to(outside)
    else:
        os.mkfifo(target)

    with pytest.raises(ValueError, match="regular file"):
        run_store.load_run("safe-id", root_dir=root)


def test_save_run_is_atomic_and_uses_mode_0600(tmp_path):
    root = tmp_path / "root"
    saved = run_store.save_run(paused_state("safe-id"), root_dir=root)

    assert stat.S_IMODE(saved.stat().st_mode) == 0o600
    assert list(saved.parent.glob("*.tmp")) == []


@pytest.mark.parametrize(
    "kind",
    ["symlink", "fifo", "directory", "hardlink"],
)
def test_save_run_rejects_unsafe_per_run_lock(tmp_path, kind):
    root = tmp_path / "root"
    runs = root / "runs"
    runs.mkdir(parents=True)
    lock_path = runs / ".safe-id.lock"
    if kind == "symlink":
        outside = tmp_path / "outside.lock"
        outside.write_text("", encoding="utf-8")
        lock_path.symlink_to(outside)
    elif kind == "fifo":
        os.mkfifo(lock_path)
    elif kind == "directory":
        lock_path.mkdir()
    else:
        outside = tmp_path / "outside.lock"
        outside.write_text("", encoding="utf-8")
        os.link(outside, lock_path)

    with pytest.raises(ValueError, match="run lock must be a regular file"):
        run_store.save_run(paused_state("safe-id"), root_dir=root)


@pytest.mark.parametrize("mode", [0o644, 0o666])
def test_save_run_rejects_per_run_lock_without_exact_permissions(
    tmp_path, mode
):
    root = tmp_path / "root"
    runs = root / "runs"
    runs.mkdir(parents=True)
    lock_path = runs / ".safe-id.lock"
    lock_path.write_text("", encoding="utf-8")
    lock_path.chmod(mode)

    with pytest.raises(ValueError, match="run lock must be a regular file"):
        run_store.save_run(paused_state("safe-id"), root_dir=root)


def test_save_run_replace_failure_preserves_previous_file_and_cleans_temp(
    tmp_path, monkeypatch
):
    root = tmp_path / "root"
    state = paused_state("safe-id")
    saved = run_store.save_run(state, root_dir=root)
    original = saved.read_bytes()
    state.issue_url = "https://github.com/acme/widget/issues/999"

    def fail_replace(*_args, **_kwargs):
        raise OSError("injected replace failure")

    monkeypatch.setattr(run_store.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failure"):
        run_store.save_run(state, root_dir=root)

    assert saved.read_bytes() == original
    assert list(saved.parent.glob("*.tmp")) == []


def test_run_store_rejects_oversized_regular_file(tmp_path):
    root = tmp_path / "root"
    directory = root / "runs"
    directory.mkdir(parents=True)
    target = directory / "safe-id.json"
    with target.open("wb") as handle:
        handle.truncate(run_store.MAX_RUN_FILE_BYTES + 1)

    with pytest.raises(ValueError, match="size limit"):
        run_store.load_run("safe-id", root_dir=root)


def test_save_and_load_paused_run_preserves_pause_state(tmp_path):
    root_dir = tmp_path / ".repopilot"
    state = paused_state()

    saved_path = run_store.save_run(state, root_dir=root_dir)
    loaded = run_store.load_run(state.trace_id, root_dir=root_dir)

    assert saved_path == root_dir / "runs" / f"{state.trace_id}.json"
    assert loaded.trace_id == state.trace_id
    assert loaded.current_phase == new_agent.Phase.WAITING_FOR_USER
    assert loaded.pending_human_input is True
    assert loaded.human_input_request == state.human_input_request
    assert loaded.frame_history == state.frame_history
    assert loaded.route_decisions == state.route_decisions


def test_claim_run_for_resume_atomically_consumes_paused_request(tmp_path):
    root_dir = tmp_path / ".repopilot"
    original = paused_state("claim-once")
    run_store.save_run(original, root_dir=root_dir)
    expected = run_store.load_run(original.trace_id, root_dir=root_dir)
    expected_before = expected.model_copy(deep=True)

    claimed = run_store.claim_run_for_resume(
        original.trace_id,
        expected,
        root_dir=root_dir,
    )
    durable = run_store.load_run(original.trace_id, root_dir=root_dir)

    assert expected == expected_before
    assert claimed == durable
    assert claimed is not expected
    assert claimed.resume_in_progress is True
    assert claimed.current_phase == new_agent.Phase.PLAN
    assert claimed.pending_human_input is False
    assert claimed.human_input_request == {}


def test_claim_run_for_resume_rejects_stale_expected_state(tmp_path):
    root_dir = tmp_path / ".repopilot"
    original = paused_state("stale-claim")
    run_store.save_run(original, root_dir=root_dir)
    stale = run_store.load_run(original.trace_id, root_dir=root_dir)
    updated = stale.model_copy(deep=True)
    updated.human_input_request["question"] = (
        "Durably updated after authorization"
    )
    run_store.save_run(updated, root_dir=root_dir)

    with pytest.raises(run_store.ResumeConflictError):
        run_store.claim_run_for_resume(
            original.trace_id,
            stale,
            root_dir=root_dir,
        )

    durable = run_store.load_run(original.trace_id, root_dir=root_dir)
    assert durable == updated
    assert durable.resume_in_progress is False


def test_claim_run_for_resume_requires_matching_trace_id(tmp_path):
    root_dir = tmp_path / ".repopilot"
    original = paused_state("trace-match")
    run_store.save_run(original, root_dir=root_dir)
    expected = run_store.load_run(original.trace_id, root_dir=root_dir)
    expected.trace_id = "different-run"

    with pytest.raises(run_store.ResumeConflictError):
        run_store.claim_run_for_resume(
            original.trace_id,
            expected,
            root_dir=root_dir,
        )

    assert run_store.load_run(
        original.trace_id, root_dir=root_dir
    ).resume_in_progress is False


def test_claim_run_for_resume_maps_missing_durable_state_to_conflict(tmp_path):
    root_dir = tmp_path / ".repopilot"
    original = paused_state("missing-after-authorization")
    path = run_store.save_run(original, root_dir=root_dir)
    expected = run_store.load_run(original.trace_id, root_dir=root_dir)
    path.unlink()

    with pytest.raises(run_store.ResumeConflictError):
        run_store.claim_run_for_resume(
            original.trace_id,
            expected,
            root_dir=root_dir,
        )


def test_claim_write_failure_preserves_unclaimed_pause(tmp_path, monkeypatch):
    root_dir = tmp_path / ".repopilot"
    original = paused_state("claim-write-failure")
    run_store.save_run(original, root_dir=root_dir)
    expected = run_store.load_run(original.trace_id, root_dir=root_dir)

    def fail_write(*_args, **_kwargs):
        raise OSError("injected claim write failure")

    monkeypatch.setattr(run_store, "_atomic_write_run_at", fail_write)

    with pytest.raises(OSError, match="claim write failure"):
        run_store.claim_run_for_resume(
            original.trace_id,
            expected,
            root_dir=root_dir,
        )

    assert run_store.load_run(original.trace_id, root_dir=root_dir) == original
    assert expected == original


def test_simultaneous_claims_for_same_run_succeed_exactly_once(tmp_path):
    root_dir = tmp_path / ".repopilot"
    original = paused_state("simultaneous-claim")
    run_store.save_run(original, root_dir=root_dir)
    expected = run_store.load_run(original.trace_id, root_dir=root_dir)
    start = threading.Barrier(2)

    def attempt_claim():
        start.wait(timeout=5)
        try:
            run_store.claim_run_for_resume(
                original.trace_id,
                expected,
                root_dir=root_dir,
            )
        except run_store.ResumeConflictError:
            return "conflict"
        return "claimed"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: attempt_claim(), range(2)))

    assert sorted(results) == ["claimed", "conflict"]
    assert run_store.load_run(
        original.trace_id, root_dir=root_dir
    ).resume_in_progress is True


def test_cross_process_claims_for_same_run_succeed_exactly_once(tmp_path):
    root_dir = tmp_path / ".repopilot"
    original = paused_state("process-claim")
    run_store.save_run(original, root_dir=root_dir)
    expected = run_store.load_run(original.trace_id, root_dir=root_dir)
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_claim_in_subprocess,
            args=(
                str(root_dir),
                original.trace_id,
                expected.model_dump(mode="json"),
                start,
                results,
            ),
        )
        for _index in range(2)
    ]

    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    assert all(process.exitcode == 0 for process in processes)
    assert sorted(results.get(timeout=2) for _index in range(2)) == [
        "claimed",
        "conflict",
    ]


def test_different_run_ids_use_independent_save_locks(tmp_path, monkeypatch):
    root_dir = tmp_path / ".repopilot"
    states = [paused_state("run-a"), paused_state("run-b")]
    writes_started = threading.Barrier(2)
    original_atomic_write = run_store._atomic_write_run_at

    def synchronized_write(run_id, payload, directory_fd, path, **kwargs):
        writes_started.wait(timeout=5)
        return original_atomic_write(
            run_id,
            payload,
            directory_fd,
            path,
            **kwargs,
        )

    monkeypatch.setattr(
        run_store,
        "_atomic_write_run_at",
        synchronized_write,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        saved = list(
            executor.map(
                lambda state: run_store.save_run(state, root_dir=root_dir),
                states,
            )
        )

    assert {path.name for path in saved} == {"run-a.json", "run-b.json"}


def test_save_and_load_preserves_model_routing_state(tmp_path):
    root_dir = tmp_path / ".repopilot"
    state = paused_state("routing-state")
    state.active_model = "claude-opus-4-8:stable"
    state.active_provider = "escalation"
    state.escalated = True
    state.escalation_reason = "repeated_edit"
    state.no_progress_rounds = 2
    state.last_plan_signature = "plan-sha"
    state.last_context_fingerprint = "context-sha"
    state.last_test_failure_signature = "test-sha"
    state.model_history = [
        ModelInvocation(
            model="gemini-3.5-flash:stable",
            provider="primary",
            node="plan_fix",
            elapsed_seconds=1.25,
            input_tokens=120,
            output_tokens=35,
            status="invalid_response",
            error_class="ValidationError: secret response body",
        )
    ]
    state.no_progress_history = [
        NoProgressEvent(
            kind="repeated_edit", fingerprint="edit-sha", node="plan_fix"
        )
    ]

    run_store.save_run(state, root_dir=root_dir)
    loaded = run_store.load_run(state.trace_id, root_dir=root_dir)

    assert loaded.active_model == state.active_model
    assert loaded.active_provider == state.active_provider
    assert loaded.escalated is True
    assert loaded.escalation_reason == "repeated_edit"
    assert loaded.no_progress_rounds == 2
    assert loaded.last_plan_signature == "plan-sha"
    assert loaded.last_context_fingerprint == "context-sha"
    assert loaded.last_test_failure_signature == "test-sha"
    assert loaded.model_history == state.model_history
    assert loaded.model_history[0].error_class == "ValidationError"
    assert loaded.no_progress_history == state.no_progress_history


def _repair_gate_approval() -> ToolPatchApproval:
    return ToolPatchApproval(
        base_ref="a" * 40,
        patch_sha256="b" * 64,
        patch_gate_fingerprint="c" * 64,
        changed_manifest=(),
        manifest_fingerprint="d" * 64,
    )


def test_same_round_failure_is_idempotent_after_run_store_roundtrip(
    tmp_path, monkeypatch
):
    root_dir = tmp_path / ".repopilot"
    monkeypatch.setattr(
        "src.model_policy.escalation_is_configured", lambda: False
    )
    state = paused_state("repair-round-idempotency")
    state.current_phase = Phase.PLAN
    first = begin_repair_round(state)
    record_failed_repair_round(
        state,
        round_id=first,
        provider="primary",
        model="gemini-3.5-flash:stable",
        failure_reason="tests_failed",
        retry_phase=Phase.REFLECT,
    )
    second = begin_repair_round(state)
    state.patch_content = "diff --git a/widget.py b/widget.py"
    state.patch_edits = [
        PatchEdit(file_path="widget.py", search="old", replace="new")
    ]
    state.tool_patch_approval = _repair_gate_approval()
    freeze_authorized_repair_round(state)
    state.fix_attempts.append(
        FixAttempt(
            failure_kind="tests_failed",
            repair_round_id=first,
            repair_provider="primary",
            repair_model="gemini-3.5-flash:stable",
        )
    )

    run_store.save_run(state, root_dir=root_dir)
    loaded = run_store.load_run(state.trace_id, root_dir=root_dir)
    before = loaded.model_copy(deep=True)
    duplicate = record_failed_repair_round(
        loaded,
        round_id=first,
        provider="primary",
        model="gemini-3.5-flash:stable",
        failure_reason="tests_failed",
        retry_phase=Phase.REFLECT,
    )

    assert second == 2
    assert loaded == before
    assert duplicate.counted is False
    assert loaded.retry_count == 1
    assert loaded.last_counted_repair_round_id == 1
    assert loaded.current_repair_round_id == 2
    assert loaded.authorized_repair_round_id == 2
    assert loaded.fix_attempts[-1].repair_round_id == 1


def test_terminal_roundtrip_with_cleared_current_id_rejects_sixth_round(
    tmp_path, monkeypatch
):
    root_dir = tmp_path / ".repopilot"
    monkeypatch.setattr(
        "src.model_policy.escalation_is_configured", lambda: False
    )
    state = paused_state("terminal-repair-ledger")
    state.max_retries = 4
    state.current_phase = Phase.PLAN

    for _index in range(5):
        round_id = begin_repair_round(state)
        record_failed_repair_round(
            state,
            round_id=round_id,
            provider="primary",
            model="gemini-3.5-flash:stable",
            failure_reason="tests_failed",
            retry_phase=Phase.REFLECT,
        )

    state.current_repair_round_id = 0
    run_store.save_run(state, root_dir=root_dir)
    loaded = run_store.load_run(state.trace_id, root_dir=root_dir)

    with pytest.raises(ValueError, match="budget is exhausted"):
        begin_repair_round(loaded)

    assert loaded.repair_round_sequence == 5
    assert loaded.last_counted_repair_round_id == 5
    assert loaded.retry_count == 4


def test_legacy_saved_state_loads_model_routing_defaults(tmp_path):
    root_dir = tmp_path / ".repopilot"
    path = run_store.run_path("legacy", root_dir=root_dir)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "issue_url": "https://github.com/acme/widget/issues/7",
                "trace_id": "legacy",
                "current_phase": "PLAN",
            }
        ),
        encoding="utf-8",
    )

    loaded = run_store.load_run("legacy", root_dir=root_dir)

    assert loaded.active_model == "gemini-3.5-flash:stable"
    assert loaded.active_provider == "primary"
    assert loaded.escalated is False
    assert loaded.escalation_reason == ""
    assert loaded.no_progress_rounds == 0
    assert loaded.model_history == []
    assert loaded.no_progress_history == []
    assert loaded.resume_in_progress is False


def test_save_load_and_replay_preserve_outcome_summary_state(tmp_path):
    root_dir = tmp_path / ".repopilot"
    state = paused_state("outcome-summary")
    state.attempt_outcome_summary = "Guard moved before unsafe submit."
    state.summary_token_usage = 47

    run_store.save_run(state, root_dir=root_dir)
    loaded = run_store.load_run(state.trace_id, root_dir=root_dir)
    replay = run_store.replay_run(state.trace_id, root_dir=root_dir)

    assert loaded.attempt_outcome_summary == state.attempt_outcome_summary
    assert loaded.summary_token_usage == 47
    assert replay["attempt_outcome_summary"] == state.attempt_outcome_summary
    assert replay["summary_token_usage"] == 47


def test_run_store_sanitizes_hostile_summary_even_when_state_validation_is_bypassed(
    tmp_path,
):
    root_dir = tmp_path / ".repopilot"
    state = paused_state("hostile-outcome-summary")
    hostile = (
        "safe persisted prefix Authorization: Bearer sk-run-store-sentinel "
        "FAIL_TO_PASS HTTP/1.1 tests/generated_run_store.py"
    )
    object.__setattr__(state, "attempt_outcome_summary", hostile)

    saved_path = run_store.save_run(state, root_dir=root_dir)
    saved_text = saved_path.read_text(encoding="utf-8")
    loaded = run_store.load_run(state.trace_id, root_dir=root_dir)
    replay = run_store.replay_run(state.trace_id, root_dir=root_dir)

    for rendered in (
        saved_text,
        loaded.attempt_outcome_summary,
        replay["attempt_outcome_summary"],
    ):
        assert "sk-run-store-sentinel" not in rendered
        assert "FAIL_TO_PASS" not in rendered
        assert "HTTP/1.1" not in rendered
        assert "tests/generated_run_store.py" not in rendered


def test_run_store_load_and_replay_sanitize_hostile_file(tmp_path):
    root_dir = tmp_path / ".repopilot"
    path = run_store.run_path("hostile-summary-file", root_dir=root_dir)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "issue_url": "https://github.com/acme/widget/issues/7",
                "trace_id": "hostile-summary-file",
                "current_phase": "PLAN",
                "attempt_outcome_summary": (
                    "safe loaded prefix raw HTTP response body: "
                    "raw-load-sentinel"
                ),
            }
        ),
        encoding="utf-8",
    )

    loaded = run_store.load_run("hostile-summary-file", root_dir=root_dir)
    replay = run_store.replay_run("hostile-summary-file", root_dir=root_dir)

    assert loaded.attempt_outcome_summary == "safe loaded prefix"
    assert replay["attempt_outcome_summary"] == "safe loaded prefix"


def test_run_store_clears_legacy_evaluator_contaminated_patch_state(tmp_path):
    root_dir = tmp_path / ".repopilot"
    sentinel = "legacy-evaluator-patch-sentinel-8139"
    path = run_store.run_path("hostile-legacy-patch", root_dir=root_dir)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "issue_url": "https://github.com/acme/widget/issues/7",
                "trace_id": "hostile-legacy-patch",
                "current_phase": "PLAN",
                "patch_content": f"gold_patch={sentinel}",
                "patch_edits": [
                    {
                        "file_path": "tests/test_safe.py",
                        "search": "value = 1",
                        "replace": f"FAIL_TO_PASS: {sentinel}",
                    }
                ],
                "fix_attempts": [
                    {
                        "patch_content": f"test_patch={sentinel}",
                        "test_result": "passed",
                        "success": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = run_store.load_run("hostile-legacy-patch", root_dir=root_dir)
    saved = run_store.save_run(loaded, root_dir=root_dir).read_text(encoding="utf-8")

    assert loaded.patch_content == ""
    assert loaded.patch_edits == []
    assert loaded.fix_attempts[0].patch_content == ""
    assert loaded.fix_attempts[0].success is False
    assert sentinel not in saved
    assert "gold_patch" not in saved.casefold()
    assert "fail_to_pass" not in saved.casefold()


def test_legacy_saved_state_loads_outcome_summary_defaults(tmp_path):
    root_dir = tmp_path / ".repopilot"
    path = run_store.run_path("legacy-summary", root_dir=root_dir)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "issue_url": "https://github.com/acme/widget/issues/7",
                "trace_id": "legacy-summary",
                "current_phase": "PLAN",
            }
        ),
        encoding="utf-8",
    )

    loaded = run_store.load_run("legacy-summary", root_dir=root_dir)

    assert loaded.attempt_outcome_summary == ""
    assert loaded.summary_token_usage == 0


def test_save_and_load_preserves_evidence_and_legacy_state_defaults_to_empty(tmp_path):
    root_dir = tmp_path / ".repopilot"
    state = paused_state("evidence-state")
    state.evidence = [
        Evidence(
            evidence_id="ev_abc123",
            tool="read_file",
            file_path="src/widget.py",
            symbol="Widget.render",
            summary="Rendered widget source.",
            content="def render(): pass",
            fingerprint="abc123",
        )
    ]

    run_store.save_run(state, root_dir=root_dir)
    loaded = run_store.load_run(state.trace_id, root_dir=root_dir)

    assert loaded.evidence == state.evidence

    legacy_path = run_store.run_path("legacy-evidence", root_dir=root_dir)
    legacy_path.write_text(
        json.dumps(
            {
                "issue_url": "https://github.com/acme/widget/issues/7",
                "trace_id": "legacy-evidence",
                "current_phase": "PLAN",
            }
        ),
        encoding="utf-8",
    )

    assert run_store.load_run("legacy-evidence", root_dir=root_dir).evidence == []


def test_save_run_uses_repopilot_home_by_default(tmp_path, monkeypatch):
    repopilot_home = tmp_path / "custom-repopilot-home"
    state = paused_state()
    monkeypatch.setenv("REPOPILOT_HOME", str(repopilot_home))

    saved_path = run_store.save_run(state)

    assert run_store.default_runs_dir() == repopilot_home
    assert saved_path == repopilot_home / "runs" / f"{state.trace_id}.json"
    assert saved_path.exists()


def test_inspect_run_returns_stable_summary(tmp_path):
    root_dir = tmp_path / ".repopilot"
    state = paused_state()
    run_store.save_run(state, root_dir=root_dir)

    summary = run_store.inspect_run(state.trace_id, root_dir=root_dir)

    assert summary["run_id"] == "abc123def456"
    assert summary["issue_url"] == "https://github.com/acme/widget/issues/7"
    assert summary["current_phase"] == "WAITING_FOR_USER"
    assert summary["pending_human_input"] is True
    assert summary["human_input_question"] == "Confirm whether breaking changes are allowed."
    assert summary["latest_decision_frame"]["frame_id"] == "df_0001"
    assert summary["latest_decision_frame"]["recommended_action"] == "ask_user"
    assert summary["updated_at"].endswith("+00:00")


def test_list_runs_returns_saved_run_summaries_sorted_by_run_id(tmp_path):
    root_dir = tmp_path / ".repopilot"
    run_store.save_run(paused_state("run-b"), root_dir=root_dir)
    run_store.save_run(paused_state("run-a"), root_dir=root_dir)

    summaries = run_store.list_runs(root_dir=root_dir)

    assert [summary["run_id"] for summary in summaries] == ["run-a", "run-b"]
    assert all(summary["current_phase"] == "WAITING_FOR_USER" for summary in summaries)


def test_replay_run_returns_white_box_timeline(tmp_path):
    root_dir = tmp_path / ".repopilot"
    state = replay_state()
    run_store.save_run(state, root_dir=root_dir)

    replay = run_store.replay_run(state.trace_id, root_dir=root_dir)

    assert replay["run_id"] == "abc123def456"
    assert replay["issue_url"] == "https://github.com/acme/widget/issues/7"
    assert replay["current_phase"] == "WAITING_FOR_USER"
    assert replay["pause"]["question"] == "Confirm whether breaking changes are allowed."
    assert replay["timeline"] == [
        {
            "index": 1,
            "type": "decision_frame",
            "frame_id": "df_0001",
            "stage": "plan",
            "summary": "Need user approval before patching.",
            "selected_hypothesis_id": "H1",
            "selected_hypothesis": {
                "id": "H1",
                "claim": "The API change may be breaking.",
                "evidence": ["The issue asks to remove an existing field."],
                "score": 0.82,
                "why_selected": "It explains the compatibility risk.",
                "why_not_selected": "",
            },
            "recommended_action": "ask_user",
            "risk": "high",
            "confidence": 0.82,
            "route": {
                "source": "decision_frame",
                "current_phase": "PLAN",
                "selected_phase": "WAITING_FOR_USER",
                "route": "__end__",
                "frame_id": "df_0001",
                "recommended_action": "ask_user",
            },
            "warnings": [
                {
                    "frame_id": "df_0001",
                    "recommended_action": "ask_user",
                    "expected_phase": "WAITING_FOR_USER",
                    "actual_phase": "PLAN",
                }
            ],
            "next_checks": ["Confirm whether breaking changes are allowed."],
            "trace_notes": "Planner stopped before patching.",
        },
        {
            "index": 2,
            "type": "decision_frame",
            "frame_id": "df_0002",
            "stage": "reflect",
            "summary": "Previous patch failed because tests expect compatibility.",
            "selected_hypothesis_id": "H2",
            "selected_hypothesis": None,
            "recommended_action": "plan",
            "risk": "medium",
            "confidence": 0.74,
            "route": None,
            "warnings": [],
            "next_checks": [],
            "trace_notes": "",
        },
        {
            "index": 3,
            "type": "route_decision",
            "route": {
                "source": "current_phase",
                "current_phase": "PLAN",
                "selected_phase": "PLAN",
                "route": "plan_fix",
                "fallback_reason": "already_consumed",
            },
        },
    ]


def test_replay_run_includes_node_diagnostics(tmp_path):
    root_dir = tmp_path / ".repopilot"
    state = diagnostic_state()
    run_store.save_run(state, root_dir=root_dir)

    replay = run_store.replay_run(state.trace_id, root_dir=root_dir)

    assert replay["timeline"][-1] == {
        "index": 4,
        "type": "node_diagnostic",
        "diagnostic": {
            "node": "plan_fix",
            "event": "phase",
            "status": "timeout",
            "elapsed_seconds": 90.0,
            "error_type": "TimeoutError",
            "error": "TimeoutError",
            "phase_timeout_seconds": 90.0,
        },
    }


def test_format_replay_markdown_summarizes_timeline():
    replay = run_store.summarize_replay(replay_state())

    markdown = run_store.format_replay_markdown(replay)

    assert markdown == "\n".join(
        [
            "# RepoPilot Replay: abc123def456",
            "",
            "- Issue: https://github.com/acme/widget/issues/7",
            "- Final phase: WAITING_FOR_USER",
            "- Pending human input: yes",
            "- Question: Confirm whether breaking changes are allowed.",
            "",
            "## Timeline",
            "",
            "### 1. PLAN df_0001",
            "",
            "Need user approval before patching.",
            "",
            "- Selected hypothesis: H1",
            "- Hypothesis claim: The API change may be breaking.",
            "- Recommended action: ask_user",
            "- Risk: high",
            "- Confidence: 0.82",
            "- Route: __end__",
            "- Warning: expected WAITING_FOR_USER but actual PLAN",
            "- Next check: Confirm whether breaking changes are allowed.",
            "- Trace notes: Planner stopped before patching.",
            "",
            "### 2. REFLECT df_0002",
            "",
            "Previous patch failed because tests expect compatibility.",
            "",
            "- Selected hypothesis: H2",
            "- Recommended action: plan",
            "- Risk: medium",
            "- Confidence: 0.74",
            "",
            "### 3. Route Decision",
            "",
            "- Route: plan_fix",
            "- Source: current_phase",
            "- Fallback reason: already_consumed",
        ]
    )


def test_format_replay_markdown_includes_node_diagnostics():
    replay = run_store.summarize_replay(diagnostic_state())

    markdown = run_store.format_replay_markdown(replay)

    assert "### 4. Node Diagnostic" in markdown
    assert "- Node: plan_fix" in markdown
    assert "- Event: phase" in markdown
    assert "- Status: timeout" in markdown
    assert "- Error: TimeoutError" in markdown
    assert "- Phase timeout seconds: 90.0" in markdown
