from __future__ import annotations

from dataclasses import dataclass
from pydantic import BaseModel, Field

import pytest

from chunkbycount.extraction import StructuredListExtractor
from chunkbycount.utils.batching import BatchExecutionError


class Item(BaseModel):
    id: int


class ItemList(BaseModel):
    items: list[Item] = Field(default_factory=list)


class OptionalItemList(BaseModel):
    items: list[Item] | None = None


class ItemSummary(BaseModel):
    items: list[Item] = Field(default_factory=list)
    total: int = 0


class MultipleLists(BaseModel):
    items: list[Item] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@dataclass
class _AggResult:
    final_output: ItemList


@pytest.mark.asyncio
async def test_extractor_detects_list_field_name():
    extractor = StructuredListExtractor(
        model_class=ItemList,
        extraction_model_name="ignored",
        deduplication_model_name="ignored",
        extraction_system_prompt="ignored",
        extraction_prompt_builder=lambda s: s,
    )

    assert extractor._list_field_name == "items"


@pytest.mark.asyncio
async def test_extractor_detects_optional_list_field_name():
    extractor = StructuredListExtractor(
        model_class=OptionalItemList,
        extraction_model_name="ignored",
        deduplication_model_name="ignored",
        extraction_system_prompt="ignored",
        extraction_prompt_builder=lambda s: s,
    )

    assert extractor._list_field_name == "items"


def test_extractor_requires_prompt_builder():
    with pytest.raises(ValueError, match="extraction_prompt_builder is required"):
        StructuredListExtractor(
            model_class=ItemList,
            extraction_model_name="ignored",
            deduplication_model_name="ignored",
            extraction_system_prompt="ignored",
        )


def test_extractor_validates_runtime_environment(monkeypatch):
    monkeypatch.setenv("CHUNKBYCOUNT_RUNNER_TIMEOUT_S", "30O")

    with pytest.raises(ValueError, match="CHUNKBYCOUNT_RUNNER_TIMEOUT_S"):
        StructuredListExtractor(
            model_class=ItemList,
            extraction_model_name="ignored",
            extraction_system_prompt="ignored",
            extraction_prompt_builder=lambda text: text,
        )


def test_extractor_rejects_unknown_list_field_name():
    with pytest.raises(ValueError, match="Unknown list_field_name"):
        StructuredListExtractor(
            model_class=ItemList,
            extraction_model_name="ignored",
            deduplication_model_name="ignored",
            extraction_system_prompt="ignored",
            extraction_prompt_builder=lambda s: s,
            list_field_name="missing",
        )


def test_extractor_rejects_non_list_list_field_name():
    with pytest.raises(ValueError, match="is not list-typed"):
        StructuredListExtractor(
            model_class=ItemSummary,
            extraction_model_name="ignored",
            deduplication_model_name="ignored",
            extraction_system_prompt="ignored",
            extraction_prompt_builder=lambda s: s,
            list_field_name="total",
        )


def test_extractor_requires_list_field_name_for_ambiguous_models():
    with pytest.raises(ValueError, match="multiple list-typed fields"):
        StructuredListExtractor(
            model_class=MultipleLists,
            extraction_model_name="ignored",
            extraction_system_prompt="ignored",
            extraction_prompt_builder=lambda text: text,
        )


@pytest.mark.asyncio
async def test_aextract_propagates_extraction_failures(
    monkeypatch: pytest.MonkeyPatch,
):
    extractor = StructuredListExtractor(
        model_class=ItemList,
        extraction_model_name="ignored",
        deduplication_model_name="ignored",
        extraction_system_prompt="ignored",
        extraction_prompt_builder=lambda s: s,
    )

    async def _fake_chunk_by_count(
        *, text: str, max_count_per_chunk: int, token_per_chunk: int
    ) -> list[str]:
        return ["x"]

    async def _fake_extract_chunks(chunks: list[str]):
        raise BatchExecutionError("extract", {0: RuntimeError("failed")})

    monkeypatch.setattr(extractor, "_chunk_by_count", _fake_chunk_by_count)
    monkeypatch.setattr(extractor, "_extract_chunks", _fake_extract_chunks)

    with pytest.raises(BatchExecutionError, match="extract failed"):
        await extractor.aextract("ignored")


@pytest.mark.asyncio
async def test_aextract_returns_empty_model_for_empty_input(
    monkeypatch: pytest.MonkeyPatch,
):
    extractor = StructuredListExtractor(
        model_class=ItemList,
        extraction_model_name="ignored",
        extraction_system_prompt="ignored",
        extraction_prompt_builder=lambda text: text,
    )

    async def _unexpected_call(**kwargs):
        pytest.fail("empty input should not invoke the counting model")

    monkeypatch.setattr(extractor, "_chunk_by_count", _unexpected_call)

    result = await extractor.aextract("  ")

    assert result == ItemList(items=[])


@pytest.mark.asyncio
async def test_aextract_sets_deduplicated_items(monkeypatch: pytest.MonkeyPatch):
    extractor = StructuredListExtractor(
        model_class=ItemList,
        extraction_model_name="ignored",
        deduplication_model_name="ignored",
        extraction_system_prompt="ignored",
        extraction_prompt_builder=lambda s: s,
    )

    async def _fake_chunk_by_count(
        *, text: str, max_count_per_chunk: int, token_per_chunk: int
    ) -> list[str]:
        return ["x", "y"]

    async def _fake_extract_chunks(chunks: list[str]):
        return [object(), object()]

    async def _fake_aggregate(chunks, extraction_results, *, full_text=None):
        return [
            _AggResult(ItemList(items=[Item(id=1), Item(id=2)])),
            _AggResult(ItemList(items=[Item(id=2), Item(id=3)])),
        ]

    async def _fake_deduplicate(*, items):
        return [Item(id=1), Item(id=2), Item(id=3)]

    monkeypatch.setattr(extractor, "_chunk_by_count", _fake_chunk_by_count)
    monkeypatch.setattr(extractor, "_extract_chunks", _fake_extract_chunks)
    monkeypatch.setattr(extractor, "_aggregate", _fake_aggregate)
    monkeypatch.setattr(extractor, "_deduplicate", _fake_deduplicate)

    result = await extractor.aextract("ignored")
    assert [x.id for x in result.items] == [1, 2, 3]


def test_optional_llm_stages_are_disabled_by_default():
    extractor = StructuredListExtractor(
        model_class=ItemList,
        extraction_model_name="extract",
        counting_model_name="count",
        extraction_system_prompt="prompt",
        extraction_prompt_builder=lambda text: text,
    )

    assert extractor.counting_model_name == "count"
    assert extractor.use_aggregation is False
    assert extractor.use_guardrails is False
    assert extractor.semantic_deduplication is False


def test_optional_stage_configuration_enables_stage_when_flag_is_omitted():
    extractor = StructuredListExtractor(
        model_class=ItemList,
        extraction_model_name="extract",
        extraction_system_prompt="prompt",
        extraction_prompt_builder=lambda text: text,
        aggregation_model="aggregate",
        guardrail_prompt="verify",
    )

    assert extractor.use_aggregation is True
    assert extractor.use_guardrails is True


def test_explicit_optional_stage_opt_out_wins_over_configuration():
    extractor = StructuredListExtractor(
        model_class=ItemList,
        extraction_model_name="extract",
        extraction_system_prompt="prompt",
        extraction_prompt_builder=lambda text: text,
        aggregation_model="aggregate",
        guardrail_prompt="verify",
        use_aggregation=False,
        use_guardrails=False,
    )

    assert extractor.use_aggregation is False
    assert extractor.use_guardrails is False


def test_extractor_rejects_conflicting_deduplication_options():
    with pytest.raises(ValueError, match="deduplicate_items=True"):
        StructuredListExtractor(
            model_class=ItemList,
            extraction_model_name="ignored",
            extraction_system_prompt="ignored",
            extraction_prompt_builder=lambda text: text,
            semantic_deduplication=True,
            deduplicate_items=False,
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("batch_size", 0),
        ("max_count_per_chunk", 0),
        ("token_per_chunk", 0),
        ("max_tokens_per_window", 0),
    ],
)
def test_extractor_rejects_non_positive_limits(name: str, value: int):
    kwargs = {
        "model_class": ItemList,
        "extraction_model_name": "ignored",
        "extraction_system_prompt": "ignored",
        "extraction_prompt_builder": lambda text: text,
        name: value,
    }

    with pytest.raises(ValueError, match=name):
        StructuredListExtractor(**kwargs)


@pytest.mark.asyncio
async def test_extractor_validates_document_input():
    extractor = StructuredListExtractor(
        model_class=ItemList,
        extraction_model_name="ignored",
        extraction_system_prompt="ignored",
        extraction_prompt_builder=lambda text: text,
    )

    with pytest.raises(ValueError, match="document_text"):
        await extractor.aextract([{"name": "missing text"}])


@pytest.mark.asyncio
async def test_extractor_can_preserve_identical_occurrences():
    extractor = StructuredListExtractor(
        model_class=ItemList,
        extraction_model_name="ignored",
        extraction_system_prompt="ignored",
        extraction_prompt_builder=lambda text: text,
        deduplicate_items=False,
    )
    items = [Item(id=1), Item(id=1)]

    assert await extractor._deduplicate(items) == items


@pytest.mark.asyncio
async def test_sync_extract_rejects_an_active_event_loop():
    extractor = StructuredListExtractor(
        model_class=ItemList,
        extraction_model_name="ignored",
        extraction_system_prompt="ignored",
        extraction_prompt_builder=lambda text: text,
    )

    with pytest.raises(RuntimeError, match="await aextract"):
        extractor.extract("text")
