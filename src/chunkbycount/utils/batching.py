import asyncio
from collections.abc import Awaitable, Sequence
import inspect
from typing import Any

from chunkbycount.utils.log import logger


class BatchExecutionError(RuntimeError):
    """Raised when one or more tasks in a batch fail."""

    def __init__(self, stage: str, failures: dict[int, BaseException]) -> None:
        self.stage = stage
        self.failures = failures
        indexes = ", ".join(str(index) for index in failures)
        super().__init__(f"{stage} failed for task indexes: {indexes}")


async def run_batched(
    tasks: Sequence[Awaitable[Any]],
    batch_size: int,
    *,
    log_prefix: str | None = None,
) -> list[Any]:
    """Run awaitables with bounded concurrency, preserving result order."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if not tasks:
        return []

    pending_inputs = iter(enumerate(tasks))
    running: dict[asyncio.Future[Any], int] = {}
    missing = object()
    results: list[Any] = [missing] * len(tasks)

    def schedule_next() -> bool:
        try:
            index, awaitable = next(pending_inputs)
        except StopIteration:
            return False
        future = asyncio.ensure_future(awaitable)
        running[future] = index
        if log_prefix:
            logger.debug(
                "%s task %s started",
                log_prefix,
                index,
                extra={"stage": log_prefix},
            )
        return True

    def close_unscheduled() -> None:
        for _index, awaitable in pending_inputs:
            if inspect.iscoroutine(awaitable):
                awaitable.close()
            elif isinstance(awaitable, asyncio.Future):
                awaitable.cancel()

    async def cancel_running() -> None:
        for future in running:
            future.cancel()
        if running:
            await asyncio.gather(*running, return_exceptions=True)
        running.clear()

    try:
        for _ in range(min(batch_size, len(tasks))):
            schedule_next()

        while running:
            done, _pending = await asyncio.wait(
                running,
                return_when=asyncio.FIRST_COMPLETED,
            )
            failures: dict[int, BaseException] = {}
            for future in done:
                index = running.pop(future)
                try:
                    results[index] = future.result()
                except Exception as exc:
                    failures[index] = exc
                    results[index] = exc
                    logger.error(
                        "Task failed in %s at index %s: %s",
                        log_prefix or "batch",
                        index,
                        exc,
                        extra={"stage": log_prefix, "chunk": index},
                    )

            if failures:
                await cancel_running()
                close_unscheduled()
                error = BatchExecutionError(log_prefix or "batch", failures)
                raise error from next(iter(failures.values()))

            for _ in range(len(done)):
                if not schedule_next():
                    break
    except BaseException:
        await cancel_running()
        close_unscheduled()
        raise

    return list(results)
