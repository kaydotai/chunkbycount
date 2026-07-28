from __future__ import annotations

import pytest
import tiktoken

from chunkbycount.chunking import (
    ChunkPlanningError,
    chunk_by_count,
    chunk_by_token_count,
)


def test_token_chunks_round_trip_unicode_without_overlap():
    text = "Start 👋🏽 café 漢字 end"

    chunks = chunk_by_token_count(
        text,
        model_name="gpt-4o-mini",
        max_tokens=2,
        overlap=0,
    )

    assert "".join(chunks) == text


@pytest.mark.parametrize(
    ("max_tokens", "overlap"),
    [(0, 0), (10, -1), (10, 10), (10, 11)],
)
def test_token_chunk_validation(max_tokens: int, overlap: int):
    with pytest.raises(ValueError):
        chunk_by_token_count(
            "text",
            model_name="gpt-4o-mini",
            max_tokens=max_tokens,
            overlap=overlap,
        )


@pytest.mark.asyncio
async def test_count_chunk_rejects_initial_block_larger_than_window():
    with pytest.raises(ValueError, match="token_per_chunk"):
        await chunk_by_count(
            "text",
            prompt_builder=lambda text: text,
            model_name="gpt-4o-mini",
            token_per_chunk=101,
            max_tokens_per_window=100,
        )


def test_empty_text_has_no_chunks():
    assert (
        chunk_by_token_count(
            "",
            model_name="gpt-4o-mini",
            max_tokens=10,
            overlap=0,
        )
        == []
    )


def test_small_token_limit_uses_zero_overlap_by_default():
    chunks = chunk_by_token_count(
        "one two three",
        model_name="gpt-4o-mini",
        max_tokens=1,
    )

    assert "".join(chunks) == "one two three"


def test_token_chunks_prefer_record_boundaries():
    text = (
        "PRODUCT P-001\nName: First\n\n"
        "PRODUCT P-002\nName: Second\n\n"
        "PRODUCT P-003\nName: Third\n"
    )

    chunks = chunk_by_token_count(
        text,
        model_name="gpt-4o-mini",
        max_tokens=18,
    )

    assert "".join(chunks) == text
    assert all(chunk.lstrip().startswith("PRODUCT") for chunk in chunks)
    encoder = tiktoken.encoding_for_model("gpt-4o-mini")
    assert all(
        len(encoder.encode(chunk, disallowed_special=())) <= 18 for chunk in chunks
    )


def test_token_chunks_recheck_standalone_unicode_token_count():
    text = "🇧🇬👋🏽"
    encoder = tiktoken.encoding_for_model("gpt-4o-mini")

    chunks = chunk_by_token_count(text, "gpt-4o-mini", max_tokens=7)

    assert "".join(chunks) == text
    assert all(
        len(encoder.encode(chunk, disallowed_special=())) <= 7 for chunk in chunks
    )


def test_token_chunks_reject_an_indivisible_sequence_over_budget():
    with pytest.raises(ChunkPlanningError, match="indivisible Unicode"):
        chunk_by_token_count("👋", "gpt-4o-mini", max_tokens=1)
