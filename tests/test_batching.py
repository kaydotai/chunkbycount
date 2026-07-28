from __future__ import annotations

import asyncio

import pytest

from chunkbycount.utils.batching import BatchExecutionError, run_batched


@pytest.mark.asyncio
async def test_run_batched_is_a_sliding_concurrency_limit():
    active = 0
    maximum_active = 0
    started: list[int] = []
    started_before_slow_task_finished: list[int] = []
    slow_task_finished = False

    async def _task(index: int) -> int:
        nonlocal active, maximum_active, slow_task_finished
        started.append(index)
        if index >= 3 and not slow_task_finished:
            started_before_slow_task_finished.append(index)
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0.03 if index == 0 else 0.001)
        if index == 0:
            slow_task_finished = True
        active -= 1
        return index

    results = await run_batched([_task(index) for index in range(8)], batch_size=3)

    assert results == list(range(8))
    assert maximum_active == 3
    assert started_before_slow_task_finished


@pytest.mark.asyncio
async def test_run_batched_stops_scheduling_after_failure():
    started: list[int] = []

    async def _task(index: int) -> int:
        started.append(index)
        if index == 0:
            raise RuntimeError("bad key")
        await asyncio.sleep(1)
        return index

    with pytest.raises(BatchExecutionError) as exc_info:
        await run_batched(
            [_task(index) for index in range(50)],
            batch_size=5,
            log_prefix="count",
        )

    assert exc_info.value.stage == "count"
    assert set(exc_info.value.failures) == {0}
    assert set(started) == set(range(5))


@pytest.mark.asyncio
async def test_run_batched_propagates_task_cancellation():
    async def _cancel_self() -> None:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await run_batched([_cancel_self()], batch_size=1, log_prefix="count")
