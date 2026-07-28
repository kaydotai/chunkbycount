from __future__ import annotations

import pytest

from chunkbycount.chunking import CountEstimateWarning, chunk_by_count
from chunkbycount.counting import EntityCount
from chunkbycount.utils.batching import BatchExecutionError


@pytest.mark.asyncio
async def test_chunk_by_count_aggregates_by_entity_count(
    monkeypatch: pytest.MonkeyPatch,
):
    async def _fake_count_entities(
        *, extraction_prompt: str, model_name: str
    ) -> EntityCount:
        return EntityCount(total_count=1)

    def _fake_chunk_by_token_count(
        text: str, model_name: str, max_tokens: int = 8000, overlap: int = 500
    ) -> list[str]:
        return ["aa", "bb", "cc"]

    monkeypatch.setattr("chunkbycount.chunking.count_entities", _fake_count_entities)
    monkeypatch.setattr(
        "chunkbycount.chunking.chunk_by_token_count", _fake_chunk_by_token_count
    )

    chunks = await chunk_by_count(
        "ignored",
        prompt_builder=lambda s: s,
        model_name="ignored",
        max_count_per_chunk=2,
        token_per_chunk=10,
        batch_size=50,
    )

    assert chunks == ["aabb", "cc"]


@pytest.mark.asyncio
async def test_chunk_by_count_recursively_splits_over_budget_blocks(
    monkeypatch: pytest.MonkeyPatch,
):
    async def _fake_count_entities(
        *, extraction_prompt: str, model_name: str
    ) -> EntityCount:
        return EntityCount(total_count=len(extraction_prompt))

    def _fake_chunk_by_token_count(
        text: str, model_name: str, max_tokens: int = 8000, overlap: int = 500
    ) -> list[str]:
        if len(text) <= max_tokens:
            return [text]
        midpoint = len(text) // 2
        return [text[:midpoint], text[midpoint:]]

    monkeypatch.setattr("chunkbycount.chunking.count_entities", _fake_count_entities)
    monkeypatch.setattr(
        "chunkbycount.chunking.chunk_by_token_count", _fake_chunk_by_token_count
    )
    monkeypatch.setattr(
        "chunkbycount.chunking._token_count",
        lambda text, model_name: len(text),
    )

    chunks = await chunk_by_count(
        "abcd",
        prompt_builder=lambda text: text,
        model_name="ignored",
        max_count_per_chunk=2,
        token_per_chunk=4,
        batch_size=2,
    )

    assert chunks == ["ab", "cd"]


@pytest.mark.asyncio
async def test_chunk_by_count_propagates_count_failures(
    monkeypatch: pytest.MonkeyPatch,
):
    async def _fake_count_entities(
        *, extraction_prompt: str, model_name: str
    ) -> EntityCount:
        if extraction_prompt == "bad":
            raise RuntimeError("count failed")
        return EntityCount(total_count=1)

    monkeypatch.setattr("chunkbycount.chunking.count_entities", _fake_count_entities)
    monkeypatch.setattr(
        "chunkbycount.chunking.chunk_by_token_count",
        lambda **_: ["good", "bad"],
    )

    with pytest.raises(BatchExecutionError) as exc_info:
        await chunk_by_count(
            "ignored",
            prompt_builder=lambda text: text,
            model_name="ignored",
            batch_size=2,
        )

    assert exc_info.value.stage == "count"
    assert set(exc_info.value.failures) == {1}


@pytest.mark.asyncio
async def test_chunk_by_count_isolates_implausible_high_estimate(
    monkeypatch: pytest.MonkeyPatch,
):
    async def _fake_count_entities(
        *, extraction_prompt: str, model_name: str
    ) -> EntityCount:
        return EntityCount(total_count=5)

    monkeypatch.setattr("chunkbycount.chunking.count_entities", _fake_count_entities)
    monkeypatch.setattr(
        "chunkbycount.chunking.chunk_by_token_count",
        lambda *args, **kwargs: ["ab"],
    )
    monkeypatch.setattr(
        "chunkbycount.chunking._token_count",
        lambda text, model_name: 2,
    )

    with pytest.warns(CountEstimateWarning, match="implausible density"):
        chunks = await chunk_by_count(
            "ab",
            prompt_builder=lambda text: text,
            model_name="ignored",
            max_count_per_chunk=2,
        )

    assert chunks == ["ab"]


@pytest.mark.asyncio
async def test_chunk_by_count_isolates_plausible_estimate_that_cannot_split_safely(
    monkeypatch: pytest.MonkeyPatch,
):
    text = "漢字漢字漢字漢字漢字漢字漢字漢字"

    async def _fake_count_entities(
        *, extraction_prompt: str, model_name: str
    ) -> EntityCount:
        return EntityCount(total_count=235)

    monkeypatch.setattr("chunkbycount.chunking.count_entities", _fake_count_entities)

    with pytest.warns(CountEstimateWarning, match="could not be subdivided safely"):
        chunks = await chunk_by_count(
            text,
            prompt_builder=lambda value: value,
            model_name="ignored",
            tokenizer_model_name="gpt-3.5-turbo",
            max_count_per_chunk=10,
            token_per_chunk=250,
        )

    assert chunks == [text]


@pytest.mark.asyncio
async def test_chunk_by_count_uses_small_blocks_when_all_estimates_are_zero(
    monkeypatch: pytest.MonkeyPatch,
):
    async def _fake_count_entities(
        *, extraction_prompt: str, model_name: str
    ) -> EntityCount:
        return EntityCount(total_count=0)

    monkeypatch.setattr("chunkbycount.chunking.count_entities", _fake_count_entities)
    monkeypatch.setattr(
        "chunkbycount.chunking.chunk_by_token_count",
        lambda *args, **kwargs: ["aa", "bb", "cc"],
    )

    with pytest.warns(CountEstimateWarning, match="zero for every"):
        chunks = await chunk_by_count(
            "ignored",
            prompt_builder=lambda text: text,
            model_name="ignored",
        )

    assert chunks == ["aa", "bb", "cc"]
