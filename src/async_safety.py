"""Version-stable asyncio cancellation and timeout helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


class CancellationDrainError(RuntimeError):
    """Cancellation won, but draining the operation exposed a failure."""

    def __init__(
        self,
        operation: str,
        cancellation: asyncio.CancelledError,
        cleanup_error: BaseException,
    ) -> None:
        super().__init__(f"{operation} failed while draining cancellation")
        self.operation = operation
        self.cancellation = cancellation
        self.cleanup_error = cleanup_error


@dataclass(frozen=True)
class TaskDrainOutcome(Generic[T]):
    """Terminal task state plus cancellation delayed while it was drained."""

    result: T | None
    error: BaseException | None
    delayed_cancellation: asyncio.CancelledError | None


async def drain_task(task: asyncio.Future[T]) -> TaskDrainOutcome[T]:
    """Wait for *task* to finish while retaining the first caller cancellation."""
    delayed_cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as cancellation:
            if task.done():
                break
            if delayed_cancellation is None:
                delayed_cancellation = cancellation
        except BaseException:
            break
    try:
        result = task.result()
    except BaseException as error:
        return TaskDrainOutcome(None, error, delayed_cancellation)
    return TaskDrainOutcome(result, None, delayed_cancellation)


def _raise_external_cancellation(
    cancellation: asyncio.CancelledError,
    outcome: TaskDrainOutcome[object],
) -> None:
    error = outcome.error
    if error is not None and not isinstance(error, asyncio.CancelledError):
        raise CancellationDrainError(
            "phase execution", cancellation, error
        ) from error
    raise cancellation


async def wait_for_phase(awaitable: Awaitable[T], timeout: float | None) -> T:
    """Apply a phase deadline without relying on version-specific wait_for cleanup."""
    child = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.wait_for(asyncio.shield(child), timeout=timeout)
    except asyncio.TimeoutError as timeout_error:
        if not child.done():
            child.cancel()
        outcome = await drain_task(child)
        if outcome.delayed_cancellation is not None:
            _raise_external_cancellation(
                outcome.delayed_cancellation,
                outcome,
            )
        if outcome.error is not None:
            raise asyncio.TimeoutError() from outcome.error
        raise asyncio.TimeoutError() from timeout_error
    except asyncio.CancelledError as cancellation:
        if not child.done():
            child.cancel()
        outcome = await drain_task(child)
        _raise_external_cancellation(cancellation, outcome)
        raise AssertionError("unreachable")
