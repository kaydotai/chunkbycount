from bisect import bisect_left
from collections.abc import Callable
from functools import cache
import math
import warnings

import tiktoken
from tiktoken.core import Encoding

from chunkbycount.counting import count_entities
from chunkbycount.utils.batching import run_batched
from chunkbycount.utils.log import logger


class ChunkPlanningError(RuntimeError):
    """Raised when source text cannot be split losslessly within token bounds."""


class CountEstimateWarning(UserWarning):
    """Warn that count estimates were anomalous and a safe fallback was used."""


def _warn_count_fallback(message: str) -> None:
    logger.warning(message)
    warnings.warn(message, CountEstimateWarning, stacklevel=3)


@cache
def _encoding_for_model(model_name: str) -> Encoding:
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        logger.warning(
            "tiktoken has no encoder mapping for %s; using cl100k_base",
            model_name,
        )
        return tiktoken.get_encoding("cl100k_base")


def _token_count(text: str, model_name: str) -> int:
    return len(_encoding_for_model(model_name).encode(text, disallowed_special=()))


def _safe_token_boundary(offsets: list[int], boundary: int, start: int) -> int:
    """Move a token boundary away from the middle of a UTF-8 character."""
    original = boundary
    while boundary > start and offsets[boundary] == offsets[boundary - 1]:
        boundary -= 1
    if boundary > start:
        return boundary

    boundary = original
    while boundary < len(offsets) and offsets[boundary] == offsets[boundary - 1]:
        boundary += 1
    return boundary


def _preferred_text_boundary(
    text: str,
    offsets: list[int],
    *,
    start: int,
    end: int,
) -> int:
    """Prefer a nearby paragraph or line break over a hard token cut."""
    if end - start < 2:
        return end

    minimum_token = start + max(1, (end - start) // 2)
    minimum_char = offsets[minimum_token]
    end_char = offsets[end] if end < len(offsets) else len(text)
    for separator in ("\n\n", "\n"):
        separator_char = text.rfind(separator, minimum_char, end_char)
        if separator_char < 0:
            continue
        candidate = bisect_left(offsets, separator_char, start + 1, end)
        candidate = _safe_token_boundary(offsets, candidate, start)
        if start < candidate < end:
            return candidate
    return end


def _fit_token_limit(
    text: str,
    offsets: list[int],
    *,
    encoder: Encoding,
    start: int,
    end: int,
    max_tokens: int,
) -> int:
    """Shrink a safe boundary if its standalone text retokenizes over budget."""
    while end > start:
        start_char = offsets[start]
        end_char = offsets[end] if end < len(offsets) else len(text)
        if (
            len(
                encoder.encode(
                    text[start_char:end_char],
                    disallowed_special=(),
                )
            )
            <= max_tokens
        ):
            return end

        candidate = _safe_token_boundary(offsets, end - 1, start)
        if candidate >= end:
            break
        end = candidate

    raise ChunkPlanningError(
        "an indivisible Unicode sequence exceeds max_tokens when encoded"
    )


def chunk_by_token_count(
    text: str,
    model_name: str,
    max_tokens: int = 8000,
    overlap: int = 0,
) -> list[str]:
    """Split text into token-bounded chunks without corrupting Unicode text."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")
    if overlap < 0 or overlap >= max_tokens:
        raise ValueError("overlap must satisfy 0 <= overlap < max_tokens")
    if not text:
        return []

    encoder = _encoding_for_model(model_name)
    tokens = encoder.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return [text]

    decoded, offsets = encoder.decode_with_offsets(tokens)
    if decoded != text:
        raise ChunkPlanningError("tokenizer did not round-trip the input text")

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        if end < len(tokens):
            end = _safe_token_boundary(offsets, end, start)
            end = _preferred_text_boundary(
                text,
                offsets,
                start=start,
                end=end,
            )
        end = _fit_token_limit(
            text,
            offsets,
            encoder=encoder,
            start=start,
            end=end,
            max_tokens=max_tokens,
        )
        if end <= start:
            raise ChunkPlanningError("could not find a safe token boundary")

        start_char = offsets[start]
        end_char = offsets[end] if end < len(tokens) else len(text)
        chunks.append(text[start_char:end_char])

        if end == len(tokens):
            break

        next_start = max(0, end - overlap)
        if next_start:
            next_start = _safe_token_boundary(offsets, next_start, start)
        start = next_start if next_start > start else end

    return chunks


def merge_model_fields(target, source) -> None:
    """Concatenate matching list fields from source into target."""
    if type(source) is not type(target):
        raise TypeError(
            "cannot merge models of different types: "
            f"{type(source).__name__} != {type(target).__name__}"
        )

    for field_name, source_value in source.__dict__.items():
        if field_name.startswith("_") or not hasattr(target, field_name):
            continue
        target_value = getattr(target, field_name)
        if isinstance(target_value, list) and isinstance(source_value, list):
            setattr(target, field_name, target_value + source_value)
        elif target_value is None and isinstance(source_value, list):
            setattr(target, field_name, list(source_value))


async def _count_chunks(
    chunks: list[str],
    *,
    prompt_builder: Callable[[str], str],
    model_name: str,
    batch_size: int,
) -> list[tuple[str, int]]:
    prompts = [prompt_builder(chunk) for chunk in chunks]
    if not all(isinstance(prompt, str) for prompt in prompts):
        raise TypeError("prompt_builder must return a string")
    tasks = [
        count_entities(
            extraction_prompt=prompt,
            model_name=model_name,
        )
        for prompt in prompts
    ]
    counts = await run_batched(tasks, batch_size=batch_size, log_prefix="count")
    return [
        (chunk, count.total_count) for chunk, count in zip(chunks, counts, strict=True)
    ]


async def _split_oversized_chunk(
    chunk: str,
    count: int,
    *,
    prompt_builder: Callable[[str], str],
    model_name: str,
    tokenizer_model_name: str,
    max_count_per_chunk: int,
    batch_size: int,
) -> list[tuple[str, int]]:
    if count <= max_count_per_chunk:
        return [(chunk, count)]

    token_count = _token_count(chunk, tokenizer_model_name)
    if token_count <= 1:
        _warn_count_fallback(
            f"count probe estimated {count} items in a single-token block; "
            "isolating the block because the estimate cannot be subdivided"
        )
        return [(chunk, max_count_per_chunk)]
    if count > max_count_per_chunk * token_count:
        _warn_count_fallback(
            f"count probe estimated an implausible density ({count} items in "
            f"{token_count} tokens); isolating the block instead of aborting"
        )
        return [(chunk, max_count_per_chunk)]

    target_parts = min(
        token_count,
        max(2, math.ceil(count / max_count_per_chunk)),
    )
    try:
        children = chunk_by_token_count(
            chunk,
            model_name=tokenizer_model_name,
            max_tokens=max(1, math.ceil(token_count / target_parts)),
            overlap=0,
        )
    except ChunkPlanningError as exc:
        _warn_count_fallback(
            f"count probe estimated {count} items, but the block could not be "
            f"subdivided safely ({exc}); isolating the block instead"
        )
        return [(chunk, max_count_per_chunk)]
    if len(children) <= 1:
        _warn_count_fallback(
            f"count probe estimated {count} items, but the token-bounded block "
            "could not be subdivided; isolating the block instead"
        )
        return [(chunk, max_count_per_chunk)]

    counted_children = await _count_chunks(
        children,
        prompt_builder=prompt_builder,
        model_name=model_name,
        batch_size=batch_size,
    )
    result: list[tuple[str, int]] = []
    for child, child_count in counted_children:
        result.extend(
            await _split_oversized_chunk(
                child,
                child_count,
                prompt_builder=prompt_builder,
                model_name=model_name,
                tokenizer_model_name=tokenizer_model_name,
                max_count_per_chunk=max_count_per_chunk,
                batch_size=batch_size,
            )
        )
    return result


async def chunk_by_count(
    text: str,
    *,
    prompt_builder: Callable[[str], str],
    model_name: str,
    tokenizer_model_name: str | None = None,
    token_per_chunk: int = 250,
    max_count_per_chunk: int = 10,
    max_tokens_per_window: int = 8000,
    batch_size: int = 20,
) -> list[str]:
    """Pack consecutive token blocks into predicted item-bounded windows."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not callable(prompt_builder):
        raise TypeError("prompt_builder must be callable")
    if not model_name:
        raise ValueError("model_name is required")
    if max_count_per_chunk <= 0:
        raise ValueError("max_count_per_chunk must be greater than zero")
    if token_per_chunk <= 0:
        raise ValueError("token_per_chunk must be greater than zero")
    if max_tokens_per_window <= 0:
        raise ValueError("max_tokens_per_window must be greater than zero")
    if token_per_chunk > max_tokens_per_window:
        raise ValueError(
            "token_per_chunk must be less than or equal to max_tokens_per_window"
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if not text:
        return []
    tokenizer_model_name = tokenizer_model_name or model_name

    token_chunks = chunk_by_token_count(
        text=text,
        model_name=tokenizer_model_name,
        max_tokens=token_per_chunk,
        overlap=0,
    )
    logger.debug("Created %s initial token blocks", len(token_chunks))

    counted = await _count_chunks(
        token_chunks,
        prompt_builder=prompt_builder,
        model_name=model_name,
        batch_size=batch_size,
    )
    if len(counted) > 1 and all(count == 0 for _chunk, count in counted):
        _warn_count_fallback(
            "count probe returned zero for every non-empty block; using the "
            f"{len(token_chunks)} original token-bounded blocks as extraction "
            "windows, which can substantially increase extraction calls"
        )
        return token_chunks

    bounded: list[tuple[str, int]] = []
    for chunk, count in counted:
        bounded.extend(
            await _split_oversized_chunk(
                chunk,
                count,
                prompt_builder=prompt_builder,
                model_name=model_name,
                tokenizer_model_name=tokenizer_model_name,
                max_count_per_chunk=max_count_per_chunk,
                batch_size=batch_size,
            )
        )

    aggregated: list[str] = []
    buffer: list[str] = []
    item_total = 0
    token_total = 0
    for chunk, count in bounded:
        chunk_tokens = _token_count(chunk, tokenizer_model_name)
        if chunk_tokens > max_tokens_per_window:
            raise ChunkPlanningError(
                "a Unicode-safe block exceeds max_tokens_per_window; increase "
                "the window token budget"
            )
        exceeds_item_budget = buffer and item_total + count > max_count_per_chunk
        exceeds_token_budget = (
            buffer and token_total + chunk_tokens > max_tokens_per_window
        )
        if exceeds_item_budget or exceeds_token_budget:
            aggregated.append("".join(buffer))
            buffer = []
            item_total = 0
            token_total = 0

        buffer.append(chunk)
        item_total += count
        token_total += chunk_tokens

    if buffer:
        aggregated.append("".join(buffer))

    return aggregated
