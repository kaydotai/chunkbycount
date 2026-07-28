import asyncio
from collections.abc import Awaitable
import inspect
import math
import os
from typing import TypeVar

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


def _get_env_int(
    name: str,
    default: int,
    *,
    minimum: int | None = 1,
    maximum: int | None = None,
) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}, got {value}")
    return value


def _get_env_float(
    name: str,
    default: float,
    *,
    minimum: float | None = 0.0,
    maximum: float | None = None,
    minimum_inclusive: bool = False,
) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {raw!r}")
    if minimum is not None:
        below_minimum = value < minimum if minimum_inclusive else value <= minimum
        if below_minimum:
            relation = "at least" if minimum_inclusive else "greater than"
            raise ValueError(f"{name} must be {relation} {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}, got {value}")
    return value


def _get_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of true/false, 1/0, yes/no, or on/off; got {raw!r}"
    )


retry_on_llm_call = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type(
        (
            TimeoutError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.InternalServerError,
        )
    ),
    reraise=True,
)

_T = TypeVar("_T")


def get_runner_timeout_s(default: float = 300.0) -> float:
    """Resolve timeout for `Runner.run(...)` calls from environment."""
    return _get_env_float("CHUNKBYCOUNT_RUNNER_TIMEOUT_S", default)


async def run_with_timeout(task: Awaitable[_T], timeout_s: float | None = None) -> _T:
    """Await *task* with a bounded timeout to prevent indefinite stalls."""
    try:
        if timeout_s is not None and timeout_s <= 0:
            raise ValueError("timeout_s must be greater than zero")
        resolved_timeout = (
            timeout_s if timeout_s is not None else get_runner_timeout_s()
        )
    except Exception:
        if inspect.iscoroutine(task):
            task.close()
        elif isinstance(task, asyncio.Future):
            task.cancel()
        raise

    return await asyncio.wait_for(
        task,
        timeout=resolved_timeout,
    )
