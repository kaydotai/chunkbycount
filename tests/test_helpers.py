from __future__ import annotations

import pytest

from chunkbycount.helpers import _get_env_float, _get_env_int, run_with_timeout


@pytest.mark.parametrize("value", ["not-a-number", "0", "-5"])
def test_positive_float_environment_values_fail_loudly(monkeypatch, value):
    monkeypatch.setenv("CHUNKBYCOUNT_RUNNER_TIMEOUT_S", value)

    with pytest.raises(ValueError, match="CHUNKBYCOUNT_RUNNER_TIMEOUT_S"):
        _get_env_float("CHUNKBYCOUNT_RUNNER_TIMEOUT_S", 300.0)


def test_zero_is_valid_when_environment_range_allows_it(monkeypatch):
    monkeypatch.setenv("CHUNKBYCOUNT_DEDUP_SIMILARITY_THRESHOLD", "0")
    monkeypatch.setenv("CHUNKBYCOUNT_EMBED_MAX_RETRIES", "0")

    assert (
        _get_env_float(
            "CHUNKBYCOUNT_DEDUP_SIMILARITY_THRESHOLD",
            0.95,
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=True,
        )
        == 0
    )
    assert (
        _get_env_int(
            "CHUNKBYCOUNT_EMBED_MAX_RETRIES",
            2,
            minimum=0,
        )
        == 0
    )


@pytest.mark.asyncio
async def test_run_with_timeout_closes_coroutine_when_environment_is_invalid(
    monkeypatch,
):
    async def _inner() -> None:
        pass

    task = _inner()
    monkeypatch.setenv("CHUNKBYCOUNT_RUNNER_TIMEOUT_S", "30O")

    with pytest.raises(ValueError, match="CHUNKBYCOUNT_RUNNER_TIMEOUT_S"):
        await run_with_timeout(task)

    assert task.cr_frame is None
