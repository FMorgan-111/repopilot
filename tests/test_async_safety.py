from __future__ import annotations

import asyncio

import pytest

from src.async_safety import (
    CancellationDrainError,
    drain_task,
    wait_for_phase,
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
