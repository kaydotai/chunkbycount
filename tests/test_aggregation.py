from __future__ import annotations

from pydantic import BaseModel

import pytest

from chunkbycount.aggregation import aggregation_task


class Items(BaseModel):
    items: list[str]


@pytest.mark.asyncio
async def test_aggregation_does_not_mutate_or_accumulate_instructions(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[list[object]] = []
    expected_result = object()

    async def _fake_run(agent, inputs, *, run_config):
        calls.append(inputs)
        return expected_result

    monkeypatch.setattr("chunkbycount.aggregation.Runner.run", _fake_run)
    original_inputs = [{"role": "user", "content": "existing result"}]

    first = await aggregation_task(
        inputs=original_inputs,
        aggregation_model="ignored",
        system_prompt="system",
        user_prompt="source",
        output_type=Items,
    )
    second = await aggregation_task(
        inputs=original_inputs,
        aggregation_model="ignored",
        system_prompt="system",
        user_prompt="source",
        output_type=Items,
    )

    assert first is expected_result
    assert second is expected_result
    assert original_inputs == [{"role": "user", "content": "existing result"}]
    assert [len(call) for call in calls] == [2, 2]
    assert all(
        call[-1]["content"].count("Verify the current structured result") == 1
        for call in calls
    )
