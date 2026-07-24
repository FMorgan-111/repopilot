from __future__ import annotations

import asyncio

import pytest

from src.async_safety import (
    CancellationDrainError,
    drain_task,
    wait_for_phase,
)
from src.nodes.commit import (
    PRCancellationCleanupError,
    PRCancellationTransactionError,
)
from src import timeout_diagnostics
from src.timeout_diagnostics import (
    TimeoutCleanupEvidence,
    extract_timeout_cleanup_evidence,
)

pytestmark = pytest.mark.asyncio


async def test_wait_for_phase_returns_completed_result():
    async def complete():
        await asyncio.sleep(0)
        return "done"

    assert await wait_for_phase(complete(), timeout=1) == "done"


async def test_wait_for_phase_drains_child_and_rejects_late_success():
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    child_finished = asyncio.Event()

    async def suppress_cancellation():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleanup_started.set()
            await release_cleanup.wait()
            child_finished.set()
            return "late success"

    phase = asyncio.create_task(
        wait_for_phase(suppress_cancellation(), timeout=0.01)
    )
    await asyncio.wait_for(cleanup_started.wait(), timeout=1)
    assert not phase.done()
    release_cleanup.set()

    with pytest.raises(asyncio.TimeoutError):
        await phase
    assert child_finished.is_set()


async def test_wait_for_phase_chains_terminal_cleanup_failure_from_timeout():
    cleanup_error = OSError("cleanup failed")
    captured: dict[str, asyncio.CancelledError] = {}

    async def fail_cleanup():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as cancellation:
            captured["cancellation"] = cancellation
            raise CancellationDrainError(
                "child cleanup", cancellation, cleanup_error
            ) from cleanup_error

    with pytest.raises(asyncio.TimeoutError) as caught:
        await wait_for_phase(fail_cleanup(), timeout=0.01)

    terminal = caught.value.__cause__
    assert isinstance(terminal, CancellationDrainError)
    assert terminal.cancellation is captured["cancellation"]
    assert terminal.cleanup_error is cleanup_error
    assert terminal.__cause__ is cleanup_error


async def test_wait_for_phase_external_cancellation_drains_child():
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    child_finished = asyncio.Event()

    async def clean_up():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleanup_started.set()
            await release_cleanup.wait()
            child_finished.set()
            raise

    phase = asyncio.create_task(wait_for_phase(clean_up(), timeout=10))
    await asyncio.sleep(0)
    phase.cancel("external cancellation")
    await asyncio.wait_for(cleanup_started.wait(), timeout=1)
    assert not phase.done()
    release_cleanup.set()

    with pytest.raises(asyncio.CancelledError):
        await phase
    assert phase.cancelled()
    assert child_finished.is_set()


async def test_wait_for_phase_external_cleanup_failure_preserves_first_cancel():
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_error = OSError("cleanup failed")
    child_cancellation: dict[str, asyncio.CancelledError] = {}

    async def fail_cleanup():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as cancellation:
            child_cancellation["value"] = cancellation
            cleanup_started.set()
            await release_cleanup.wait()
            raise CancellationDrainError(
                "child cleanup", cancellation, cleanup_error
            ) from cleanup_error

    phase = asyncio.create_task(wait_for_phase(fail_cleanup(), timeout=10))
    await asyncio.sleep(0)
    phase.cancel("first cancellation")
    await asyncio.wait_for(cleanup_started.wait(), timeout=1)
    phase.cancel("second cancellation")
    await asyncio.sleep(0)
    assert not phase.done()
    release_cleanup.set()

    with pytest.raises(CancellationDrainError) as caught:
        await phase

    error = caught.value
    assert error.operation == "phase execution"
    assert error.cancellation.args == ("first cancellation",)
    terminal = error.cleanup_error
    assert isinstance(terminal, CancellationDrainError)
    assert terminal.cancellation is child_cancellation["value"]
    assert error.__cause__ is terminal


async def test_drain_task_keeps_first_delayed_cancellation():
    worker_started = asyncio.Event()
    release_worker = asyncio.Event()

    async def worker():
        worker_started.set()
        await release_worker.wait()
        return "drained"

    async def orchestrate():
        task = asyncio.create_task(worker())
        return await drain_task(task)

    draining = asyncio.create_task(orchestrate())
    await worker_started.wait()
    draining.cancel("first cancellation")
    await asyncio.sleep(0)
    draining.cancel("second cancellation")
    await asyncio.sleep(0)
    assert not draining.done()
    release_worker.set()

    outcome = await draining
    assert outcome.result == "drained"
    assert outcome.error is None
    assert outcome.delayed_cancellation is not None
    assert outcome.delayed_cancellation.args == ("first cancellation",)


async def test_drain_task_keeps_cancellation_when_target_settles_same_turn():
    worker_started = asyncio.Event()
    release_worker = asyncio.Event()

    async def worker():
        worker_started.set()
        await release_worker.wait()
        return "settled"

    async def orchestrate():
        task = asyncio.create_task(worker())
        return await drain_task(task)

    draining = asyncio.create_task(orchestrate())
    await worker_started.wait()
    release_worker.set()
    draining.cancel("same-turn cancellation")

    outcome = await draining
    assert outcome.result == "settled"
    assert outcome.error is None
    assert outcome.delayed_cancellation is not None
    assert outcome.delayed_cancellation.args == ("same-turn cancellation",)


async def test_timeout_cleanup_same_turn_external_cancel_beats_late_success():
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def suppress_timeout_cancellation():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleanup_started.set()
            await release_cleanup.wait()
            return "late success"

    phase = asyncio.create_task(
        wait_for_phase(suppress_timeout_cancellation(), timeout=0.01)
    )
    await asyncio.wait_for(cleanup_started.wait(), timeout=1)
    release_cleanup.set()
    phase.cancel("same-turn external cancellation")

    with pytest.raises(asyncio.CancelledError):
        await phase
    assert phase.cancelled()


async def test_timeout_cleanup_same_turn_external_cancel_keeps_terminal_failure():
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_error = OSError("cleanup failed")
    child_cancellation: dict[str, asyncio.CancelledError] = {}

    async def fail_timeout_cleanup():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as cancellation:
            child_cancellation["value"] = cancellation
            cleanup_started.set()
            await release_cleanup.wait()
            raise CancellationDrainError(
                "child cleanup", cancellation, cleanup_error
            ) from cleanup_error

    phase = asyncio.create_task(
        wait_for_phase(fail_timeout_cleanup(), timeout=0.01)
    )
    await asyncio.wait_for(cleanup_started.wait(), timeout=1)
    release_cleanup.set()
    phase.cancel("same-turn external cancellation")

    with pytest.raises(CancellationDrainError) as caught:
        await phase

    error = caught.value
    assert error.cancellation.args == ("same-turn external cancellation",)
    terminal = error.cleanup_error
    assert isinstance(terminal, CancellationDrainError)
    assert terminal.cancellation is child_cancellation["value"]
    assert error.__cause__ is terminal


async def test_cancellation_drain_error_preserves_objects_by_identity():
    cancellation = asyncio.CancelledError("original cancellation")
    cleanup_error = RuntimeError("worker failed")

    error = CancellationDrainError(
        "store write", cancellation, cleanup_error
    )

    assert error.operation == "store write"
    assert error.cancellation is cancellation
    assert error.cleanup_error is cleanup_error


async def test_generic_oci_drain_evidence_has_exact_safe_interface():
    cancellation = asyncio.CancelledError("do not inspect cancellation args")
    cleanup_error = OSError(" socket   cleanup\nfailed ")
    drain = CancellationDrainError(
        " OCI   container\ncleanup ", cancellation, cleanup_error
    )

    evidence = extract_timeout_cleanup_evidence(drain)

    assert evidence == TimeoutCleanupEvidence(
        failure_kind="generic_drain",
        cause_type="CancellationDrainError",
        cleanup_error_type="OSError",
        cleanup_error="socket cleanup failed",
        operation="OCI container cleanup",
    )
    assert list(TimeoutCleanupEvidence.__dataclass_fields__) == [
        "failure_kind",
        "cause_type",
        "cleanup_error_type",
        "cleanup_error",
        "operation",
        "pr_number",
    ]
    assert evidence.diagnostic_details() == {
        "timeout_cleanup_kind": "generic_drain",
        "timeout_cause_type": "CancellationDrainError",
        "cleanup_error_type": "OSError",
        "cleanup_error": "socket cleanup failed",
        "cleanup_operation": "OCI container cleanup",
    }
    assert evidence.summary().startswith(
        "Cancellation cleanup failed during OCI container cleanup"
    )


@pytest.mark.parametrize(
    "failure_kind, pr_number",
    [
        ("pr_cleanup", True),
        ("pr_transaction", 0),
        ("pr_cleanup", -1),
        ("pr_transaction", "19"),
    ],
)
async def test_direct_pr_evidence_rejects_invalid_pr_number_at_render_boundary(
    failure_kind,
    pr_number,
):
    evidence = TimeoutCleanupEvidence(
        failure_kind=failure_kind,
        cause_type="CancellationDrainError",
        cleanup_error_type="OSError",
        cleanup_error="failed",
        pr_number=pr_number,
    )

    assert "for an unknown pull request" in evidence.summary()
    assert "cleanup_pr_number" not in evidence.diagnostic_details()


async def test_direct_generic_evidence_never_renders_pr_number():
    evidence = TimeoutCleanupEvidence(
        failure_kind="generic_drain",
        cause_type="CancellationDrainError",
        cleanup_error_type="OSError",
        cleanup_error="failed",
        operation="OCI cleanup",
        pr_number=71,
    )

    assert evidence.summary().startswith(
        "Cancellation cleanup failed during OCI cleanup"
    )
    assert "71" not in evidence.summary()
    assert "cleanup_pr_number" not in evidence.diagnostic_details()


async def test_direct_evidence_collapses_oversized_exception_classes():
    oversized = "Oversized" * 20 + "Error"
    evidence = TimeoutCleanupEvidence(
        failure_kind="generic_drain",
        cause_type=oversized,
        cleanup_error_type=oversized,
        cleanup_error="failed",
        operation="OCI cleanup",
    )

    details = evidence.diagnostic_details()
    assert details["timeout_cause_type"] == "BaseException"
    assert details["cleanup_error_type"] == "BaseException"
    assert len(details["timeout_cause_type"]) <= 120
    assert len(details["cleanup_error_type"]) <= 120
    assert oversized not in evidence.summary()
    assert "(BaseException: failed)" in evidence.summary()


async def test_extractor_collapses_oversized_exception_classes():
    drain_class = type(
        "OversizedDrain" * 20 + "Error",
        (CancellationDrainError,),
        {},
    )
    cleanup_class = type(
        "OversizedCleanup" * 20 + "Error",
        (RuntimeError,),
        {},
    )
    drain = drain_class(
        "OCI cleanup",
        asyncio.CancelledError("cancel"),
        cleanup_class("failed"),
    )

    evidence = extract_timeout_cleanup_evidence(drain)

    assert evidence is not None
    assert evidence.cause_type == "BaseException"
    assert evidence.cleanup_error_type == "BaseException"
    assert evidence.diagnostic_details()["timeout_cause_type"] == "BaseException"
    assert evidence.diagnostic_details()["cleanup_error_type"] == "BaseException"


async def test_pr_transaction_evidence_uses_transaction_error():
    cancellation = asyncio.CancelledError("cancel transaction")
    transaction_error = ValueError(" transaction   failed ")
    drain = PRCancellationTransactionError(
        19, cancellation, transaction_error
    )
    drain.cleanup_error = AssertionError("must not be reported as cleanup")

    evidence = extract_timeout_cleanup_evidence(drain)

    assert evidence == TimeoutCleanupEvidence(
        failure_kind="pr_transaction",
        cause_type="PRCancellationTransactionError",
        cleanup_error_type="ValueError",
        cleanup_error="transaction failed",
        pr_number=19,
    )
    assert evidence.diagnostic_details() == {
        "timeout_cleanup_kind": "pr_transaction",
        "timeout_cause_type": "PRCancellationTransactionError",
        "cleanup_error_type": "ValueError",
        "cleanup_error": "transaction failed",
        "cleanup_pr_number": 19,
    }
    assert evidence.summary().startswith(
        "PR cancellation transaction failed for pull request 19"
    )
    assert "cleanup failed" not in evidence.summary()


async def test_breadth_first_search_prefers_first_pr_drain_over_generic():
    cancellation = asyncio.CancelledError("cancel")
    deep_pr = PRCancellationCleanupError(
        42, cancellation, OSError("deep cleanup")
    )
    generic = CancellationDrainError(
        "generic cleanup", cancellation, deep_pr
    )
    shallow_pr = PRCancellationCleanupError(
        41, cancellation, RuntimeError("shallow cleanup")
    )
    root = asyncio.TimeoutError()
    root.__cause__ = generic
    root.__context__ = shallow_pr

    evidence = extract_timeout_cleanup_evidence(root)

    assert evidence is not None
    assert evidence.failure_kind == "pr_cleanup"
    assert evidence.pr_number == 41
    assert evidence.cleanup_error == "shallow cleanup"


async def test_timeout_evidence_identity_seen_does_not_spend_depth_on_diamond_cycle():
    cancellation = asyncio.CancelledError("cancel")
    expected = PRCancellationCleanupError(
        53, cancellation, OSError("bounded cleanup")
    )
    root = RuntimeError("root")
    left = RuntimeError("left")
    right = RuntimeError("right")
    shared = RuntimeError("shared")
    left_context = RuntimeError("left context")
    right_context = RuntimeError("right context")
    cycle = RuntimeError("cycle")
    root.__cause__, root.__context__ = left, right
    left.__cause__, left.__context__ = shared, left_context
    right.__cause__, right.__context__ = shared, right_context
    shared.__cause__, shared.__context__ = cycle, expected
    cycle.__cause__ = root

    evidence = extract_timeout_cleanup_evidence(root)

    assert evidence is not None
    assert evidence.pr_number == 53


async def test_timeout_evidence_never_visits_beyond_maximum_depth():
    cancellation = asyncio.CancelledError("cancel")
    hidden = PRCancellationCleanupError(
        59, cancellation, OSError("too deep")
    )
    chain = [
        RuntimeError(f"level {index}")
        for index in range(timeout_diagnostics._MAX_CAUSE_DEPTH)
    ]
    for current, following in zip(chain, chain[1:]):
        current.__cause__ = following
    chain[-1].__cause__ = hidden

    assert extract_timeout_cleanup_evidence(chain[0]) is None


@pytest.mark.parametrize("pr_number", [True, 0, -1])
async def test_pr_cleanup_evidence_rejects_non_positive_strict_int(pr_number):
    drain = PRCancellationCleanupError(
        pr_number,
        asyncio.CancelledError("cancel"),
        OSError("cleanup failed"),
    )

    evidence = extract_timeout_cleanup_evidence(drain)

    assert evidence is not None
    assert evidence.pr_number is None
    assert "cleanup_pr_number" not in evidence.diagnostic_details()
    assert evidence.summary().startswith(
        "PR cancellation cleanup failed for an unknown pull request"
    )


async def test_generic_evidence_redacts_and_bounds_all_rendered_text():
    secret = "sk-timeout-evidence-secret-123456789"
    long_error_type = type("X" * 500 + "Error", (RuntimeError,), {})
    cleanup_error = long_error_type(
        f" Authorization: Bearer {secret} " + " failure" * 100
    )
    cancellation = asyncio.CancelledError(f"cancellation token={secret}")
    drain = CancellationDrainError(
        f" OCI cleanup token={secret} " + " operation" * 100,
        cancellation,
        cleanup_error,
    )

    evidence = extract_timeout_cleanup_evidence(drain)

    assert evidence is not None
    assert len(evidence.operation) <= timeout_diagnostics._MAX_OPERATION
    assert len(evidence.cleanup_error) <= timeout_diagnostics._MAX_ERROR_SUMMARY
    assert len(evidence.summary()) <= timeout_diagnostics._MAX_EVIDENCE_SUMMARY
    rendered = str(evidence.diagnostic_details()) + evidence.summary()
    assert secret not in rendered
    assert "[REDACTED]" in rendered
    assert "  " not in evidence.operation
    assert "\n" not in evidence.cleanup_error
