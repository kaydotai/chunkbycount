from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Sequence
import json
import os
from typing import Any

from openai import AsyncOpenAI

from chunkbycount.helpers import _get_env_float, _get_env_int
from chunkbycount.llm_deduplication import deduplicate_until_converged
from chunkbycount.utils.log import logger


def _jsonify_item(item: Any) -> str:
    """Convert an item to a stable JSON representation where possible."""
    if hasattr(item, "model_dump"):
        item = item.model_dump(mode="json", exclude_none=True)
    try:
        return json.dumps(
            item,
            default=str,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError):
        return repr(item)


def deduplicate_exact(
    items: Iterable[Any],
    *,
    key: Callable[[Any], Hashable] | None = None,
) -> list[Any]:
    """Remove exact duplicates while preserving the first occurrence and order."""
    deduplicated: list[Any] = []
    seen: set[Hashable] = set()

    for item in items:
        marker: Hashable = key(item) if key is not None else _jsonify_item(item)
        try:
            already_seen = marker in seen
        except TypeError as exc:
            raise TypeError("deduplication_key must return a hashable value") from exc
        if already_seen:
            continue
        seen.add(marker)
        deduplicated.append(item)

    return deduplicated


def _batched(iterable: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for idx in range(0, len(iterable), batch_size):
        yield iterable[idx : idx + batch_size]


async def _get_embeddings_async(
    texts: Sequence[str],
    model: str = "text-embedding-3-small",
    batch: int = 512,
) -> list[list[float]]:
    """Fetch embeddings in bounded concurrent batches."""
    if batch <= 0:
        raise ValueError("embedding batch size must be greater than zero")

    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    client_kwargs: dict[str, Any] = {
        "max_retries": _get_env_int(
            "CHUNKBYCOUNT_EMBED_MAX_RETRIES",
            2,
            minimum=0,
        ),
        "timeout": _get_env_float(
            "CHUNKBYCOUNT_EMBED_TIMEOUT_S",
            120.0,
        ),
    }
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is required for Gemini semantic deduplication"
            )
        client_kwargs.update(
            api_key=api_key,
            base_url=os.getenv(
                "OPENAI_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta/openai/",
            ),
        )
        batch = min(batch, 100)
        if model == "text-embedding-3-small":
            model = "gemini-embedding-001"

    client = AsyncOpenAI(**client_kwargs)
    semaphore = asyncio.Semaphore(5)

    async def _embed_one(batch_texts: Sequence[str]) -> list[list[float]]:
        async with semaphore:
            response = await client.embeddings.create(
                model=model,
                input=list(batch_texts),
            )
            return [record.embedding for record in response.data]

    try:
        results = await asyncio.gather(
            *(_embed_one(batch_texts) for batch_texts in _batched(texts, batch))
        )
    finally:
        await client.close()
    return [embedding for result in results for embedding in result]


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, index: int) -> int:
        if self.parent[index] != index:
            self.parent[index] = self.find(self.parent[index])
        return self.parent[index]

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            self.parent[left_root] = right_root
        elif self.rank[left_root] > self.rank[right_root]:
            self.parent[right_root] = left_root
        else:
            self.parent[right_root] = left_root
            self.rank[left_root] += 1


async def deduplication_task(
    items: list[Any],
    model: str,
    deduplication_instructions: str | None = None,
    embedding_model: str = "text-embedding-3-small",
    similarity_threshold: float | None = None,
    exact_key: Callable[[Any], Hashable] | None = None,
) -> list[Any]:
    """Run optional embedding- and LLM-assisted semantic deduplication."""
    remaining_items = deduplicate_exact(items, key=exact_key)
    if len(remaining_items) < 2:
        return remaining_items

    if similarity_threshold is None:
        similarity_threshold = _get_env_float(
            "CHUNKBYCOUNT_DEDUP_SIMILARITY_THRESHOLD",
            0.95,
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=True,
        )
    if not 0 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be between 0 and 1")

    try:
        import numpy as np
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:
        raise ImportError(
            "semantic deduplication requires the optional dependency group: "
            "pip install 'chunkbycount[semantic-dedup]'"
        ) from exc

    canonical_texts = [_jsonify_item(item) for item in remaining_items]
    embeddings = await _get_embeddings_async(
        canonical_texts,
        model=embedding_model,
    )
    logger.debug("Obtained embeddings for %s items", len(canonical_texts))

    embedding_matrix = np.array(embeddings, dtype=np.float32)
    neighbours = NearestNeighbors(
        radius=1.0 - similarity_threshold,
        metric="cosine",
        algorithm="auto",
    )
    neighbours.fit(embedding_matrix)
    _distances, indices = neighbours.radius_neighbors(
        embedding_matrix,
        return_distance=True,
    )

    union_find = _UnionFind(len(remaining_items))
    for source_index, neighbour_indices in enumerate(indices):
        for neighbour_index in neighbour_indices:
            if neighbour_index != source_index:
                union_find.union(source_index, int(neighbour_index))

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(remaining_items)):
        groups[union_find.find(index)].append(index)

    kept: list[tuple[int, Any]] = []
    for member_indices in groups.values():
        if len(member_indices) == 1:
            index = member_indices[0]
            kept.append((index, remaining_items[index]))
            continue

        group_items = [remaining_items[index] for index in member_indices]
        cleaned = await deduplicate_until_converged(
            items=group_items,
            model=model,
            deduplication_instructions=deduplication_instructions,
        )
        cleaned_ids = {id(item) for item in cleaned}
        kept.extend(
            (index, remaining_items[index])
            for index in member_indices
            if id(remaining_items[index]) in cleaned_ids
        )

    kept.sort(key=lambda pair: pair[0])
    return [item for _, item in kept]
