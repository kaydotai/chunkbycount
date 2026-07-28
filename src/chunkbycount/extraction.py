from __future__ import annotations

import asyncio
from collections.abc import Callable, Hashable, Sequence
import json
from pydantic import BaseModel
from typing import Any, get_args, get_origin

from agents import Agent, Runner

from chunkbycount.aggregation import aggregation_task
from chunkbycount.chunking import chunk_by_count, merge_model_fields
from chunkbycount.deduplication import deduplicate_exact, deduplication_task
from chunkbycount.helpers import (
    get_runner_timeout_s,
    retry_on_llm_call,
    run_with_timeout,
)
from chunkbycount.utils.batching import run_batched
from chunkbycount.utils.log import logger
from chunkbycount.utils.model_settings import build_run_config


class StructuredListExtractor:
    """Extract a list-typed Pydantic model from long document text."""

    def __init__(
        self,
        *,
        model_class: type[BaseModel],
        extraction_model_name: str,
        extraction_system_prompt: str,
        extraction_prompt_builder: Callable[[str], str] | None = None,
        counting_model_name: str | None = None,
        aggregation_model: str | None = None,
        deduplication_model_name: str | None = None,
        aggregation_prompt: str | None = None,
        guardrail_prompt: str | None = None,
        deduplication_instructions: str | None = None,
        use_aggregation: bool | None = None,
        use_guardrails: bool | None = None,
        semantic_deduplication: bool = False,
        deduplicate_items: bool = True,
        deduplication_key: Callable[[Any], Hashable] | None = None,
        batch_size: int = 20,
        max_count_per_chunk: int = 10,
        token_per_chunk: int = 250,
        max_tokens_per_window: int = 8000,
        aggregate_full_context: bool = False,
        relevance_filter: Callable[[dict[str, Any]], bool] | None = None,
        list_field_name: str | None = None,
    ) -> None:
        if not isinstance(model_class, type) or not issubclass(model_class, BaseModel):
            raise TypeError("model_class must be a Pydantic BaseModel class")
        if not extraction_model_name:
            raise ValueError("extraction_model_name is required")
        if not extraction_system_prompt:
            raise ValueError("extraction_system_prompt is required")
        if extraction_prompt_builder is None:
            raise ValueError("extraction_prompt_builder is required")
        if not callable(extraction_prompt_builder):
            raise TypeError("extraction_prompt_builder must be callable")
        if deduplication_key is not None and not callable(deduplication_key):
            raise TypeError("deduplication_key must be callable")
        if relevance_filter is not None and not callable(relevance_filter):
            raise TypeError("relevance_filter must be callable")
        if semantic_deduplication and not deduplicate_items:
            raise ValueError("semantic_deduplication requires deduplicate_items=True")
        for name, value in (
            ("batch_size", batch_size),
            ("max_count_per_chunk", max_count_per_chunk),
            ("token_per_chunk", token_per_chunk),
            ("max_tokens_per_window", max_tokens_per_window),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if token_per_chunk > max_tokens_per_window:
            raise ValueError(
                "token_per_chunk must be less than or equal to max_tokens_per_window"
            )

        get_runner_timeout_s()
        build_run_config()

        self.model_class = model_class
        self.extraction_model_name = extraction_model_name
        self.counting_model_name = counting_model_name or extraction_model_name
        self.aggregation_model = aggregation_model or extraction_model_name
        self.deduplication_model_name = (
            deduplication_model_name or extraction_model_name
        )
        self.extraction_system_prompt = extraction_system_prompt
        self.extraction_prompt_builder = extraction_prompt_builder
        self.aggregation_prompt = aggregation_prompt
        self.guardrail_prompt = guardrail_prompt
        self.deduplication_instructions = deduplication_instructions

        aggregation_configured = (
            aggregation_model is not None
            or aggregation_prompt is not None
            or aggregate_full_context
        )
        self.use_aggregation = (
            aggregation_configured if use_aggregation is None else use_aggregation
        )
        self.use_guardrails = (
            guardrail_prompt is not None if use_guardrails is None else use_guardrails
        )
        self.semantic_deduplication = semantic_deduplication
        self.deduplicate_items = deduplicate_items
        self.deduplication_key = deduplication_key
        self.batch_size = batch_size
        self.max_count_per_chunk = max_count_per_chunk
        self.token_per_chunk = token_per_chunk
        self.max_tokens_per_window = max_tokens_per_window
        self.aggregate_full_context = aggregate_full_context
        self._relevance_filter = relevance_filter

        if list_field_name is not None:
            field_info = self.model_class.model_fields.get(list_field_name)
            if field_info is None:
                raise ValueError(
                    f"Unknown list_field_name '{list_field_name}' for "
                    f"{self.model_class.__name__}"
                )
            if not self._is_list_annotation(field_info.annotation):
                raise ValueError(
                    f"Field '{list_field_name}' on {self.model_class.__name__} "
                    "is not list-typed"
                )
            self._list_field_name = list_field_name
        else:
            self._list_field_name = self._detect_list_field()

    @staticmethod
    def _is_list_annotation(annotation: Any) -> bool:
        origin = get_origin(annotation)
        if origin is list:
            return True
        return any(get_origin(argument) is list for argument in get_args(annotation))

    @staticmethod
    def _detect_list_field_for(model_class: type[BaseModel]) -> str | None:
        names = [
            name
            for name, field_info in model_class.model_fields.items()
            if StructuredListExtractor._is_list_annotation(field_info.annotation)
        ]
        if len(names) > 1:
            raise ValueError(
                f"{model_class.__name__} has multiple list-typed fields; "
                "pass list_field_name explicitly"
            )
        return names[0] if names else None

    def _detect_list_field(self) -> str:
        name = self._detect_list_field_for(self.model_class)
        if name is None:
            raise ValueError(
                f"{self.model_class.__name__} has no list-typed field. "
                "Pass list_field_name explicitly."
            )
        return name

    def _empty_result(self) -> BaseModel:
        return self.model_class.model_validate({self._list_field_name: []})

    def _build_prompt(self, text: str) -> str:
        prompt = self.extraction_prompt_builder(text)
        if not isinstance(prompt, str):
            raise TypeError("extraction_prompt_builder must return a string")
        return prompt

    @staticmethod
    def _normalise_input(
        raw_input: str | bytes | bytearray | list[dict[str, Any]],
        relevance_filter: Callable[[dict[str, Any]], bool] | None,
    ) -> str:
        if isinstance(raw_input, (bytes, bytearray)):
            raw_input = raw_input.decode("utf-8")
        if isinstance(raw_input, str):
            return raw_input
        if not isinstance(raw_input, list):
            raise TypeError(
                "raw_input must be text, UTF-8 bytes, or a list of document "
                "dictionaries"
            )

        documents: list[dict[str, Any]] = []
        for index, document in enumerate(raw_input):
            if not isinstance(document, dict):
                raise TypeError(f"document at index {index} must be a dictionary")
            if "document_text" not in document:
                raise ValueError(
                    f"document at index {index} is missing 'document_text'"
                )
            if not isinstance(document["document_text"], str):
                raise TypeError(f"document_text at index {index} must be a string")
            if relevance_filter is None or relevance_filter(document):
                documents.append(document)
        return "\n\n".join(document["document_text"] for document in documents)

    def extract(
        self,
        raw_input: str | bytes | bytearray | list[dict[str, Any]],
    ) -> BaseModel:
        """Run extraction from synchronous code."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "extract() cannot run inside an active event loop; "
                "use 'await aextract(...)' instead"
            )
        return asyncio.run(self.aextract(raw_input))

    async def aextract(
        self,
        raw_input: str | bytes | bytearray | list[dict[str, Any]],
    ) -> BaseModel:
        """Run extraction and return an instance of ``model_class``."""
        text = self._normalise_input(raw_input, self._relevance_filter)
        if not text.strip():
            return self._empty_result()

        chunks = await self._chunk_by_count(
            text=text,
            max_count_per_chunk=self.max_count_per_chunk,
            token_per_chunk=self.token_per_chunk,
        )
        logger.info(
            "Prepared %s chunks (predicted item limit=%s)",
            len(chunks),
            self.max_count_per_chunk,
        )
        if not chunks:
            return self._empty_result()

        extraction_results = await self._extract_chunks(chunks)
        if self.use_guardrails:
            extraction_results = list(await self._guardrail(extraction_results, chunks))

        final_results = await self._aggregate(
            chunks,
            extraction_results,
            full_text=text,
        )
        if not final_results:
            return self._empty_result()

        combined_model = final_results[0].final_output
        for result in final_results[1:]:
            merge_model_fields(combined_model, result.final_output)

        items = getattr(combined_model, self._list_field_name)
        deduplicated = await self._deduplicate(items=items or [])
        setattr(combined_model, self._list_field_name, deduplicated)
        return combined_model

    async def _aggregate(
        self,
        chunks: Sequence[str],
        extraction_results: Sequence[Any],
        *,
        full_text: str | None = None,
    ) -> Sequence[Any]:
        """Optionally verify and enrich each extracted chunk."""
        if not self.use_aggregation or not extraction_results:
            return extraction_results

        include_all_context = self.aggregate_full_context and full_text is not None
        system_prompt = self.extraction_system_prompt
        if self.aggregation_prompt:
            system_prompt += "\n\n" + self.aggregation_prompt

        requests: list[tuple[list[Any], str]] = []
        for index, (chunk, result) in enumerate(
            zip(chunks, extraction_results, strict=True)
        ):
            if include_all_context:
                context = full_text
            else:
                next_chunk = chunks[index + 1] if index + 1 < len(chunks) else ""
                context = chunk + next_chunk
            requests.append((result.to_input_list(), self._build_prompt(context)))

        tasks = [
            aggregation_task(
                inputs=inputs,
                aggregation_model=self.aggregation_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_type=self.model_class,
            )
            for inputs, user_prompt in requests
        ]

        return await run_batched(
            tasks=tasks,
            batch_size=self.batch_size,
            log_prefix="aggregate",
        )

    async def _deduplicate(self, items: Sequence[Any]) -> list[Any]:
        if not self.deduplicate_items:
            return list(items)
        if not self.semantic_deduplication:
            return deduplicate_exact(items, key=self.deduplication_key)
        return await deduplication_task(
            items=list(items),
            model=self.deduplication_model_name,
            deduplication_instructions=self.deduplication_instructions,
            exact_key=self.deduplication_key,
        )

    async def _chunk_by_count(
        self,
        *,
        text: str,
        max_count_per_chunk: int,
        token_per_chunk: int,
    ) -> list[str]:
        return await chunk_by_count(
            text,
            prompt_builder=self._build_prompt,
            model_name=self.counting_model_name,
            tokenizer_model_name=self.extraction_model_name,
            token_per_chunk=token_per_chunk,
            max_count_per_chunk=max_count_per_chunk,
            max_tokens_per_window=self.max_tokens_per_window,
            batch_size=self.batch_size,
        )

    async def _extract_chunks(self, chunks: Sequence[str]) -> list[Any]:
        """Run extraction tasks in bounded batches."""

        @retry_on_llm_call
        async def _run_single(chunk_text: str) -> Any:
            agent = Agent(
                name="Extraction Agent",
                model=self.extraction_model_name,
                instructions=self.extraction_system_prompt,
                output_type=self.model_class,
            )
            return await run_with_timeout(
                Runner.run(
                    agent,
                    [
                        {
                            "role": "user",
                            "content": self._build_prompt(chunk_text),
                        }
                    ],
                    run_config=build_run_config(),
                )
            )

        return await run_batched(
            [_run_single(chunk) for chunk in chunks],
            batch_size=self.batch_size,
            log_prefix="extract",
        )

    async def _guardrail(
        self,
        results: Sequence[Any],
        chunks: Sequence[str],
    ) -> Sequence[Any]:
        """Optionally verify extracted results against their source chunks."""

        @retry_on_llm_call
        async def _run_guardrail(agent: Agent, inputs: list[Any]) -> Any:
            return await run_with_timeout(
                Runner.run(agent, inputs, run_config=build_run_config())
            )

        requests: list[tuple[Agent, list[Any]]] = []
        for chunk, result in zip(chunks, results, strict=True):
            prompt = (
                "Verify every extracted item against the source text. Remove or "
                "correct unsupported information and return the complete result."
            )
            if self.guardrail_prompt:
                prompt += "\n\n" + self.guardrail_prompt

            inputs = result.to_input_list()
            inputs.append(
                {
                    "role": "user",
                    "content": (
                        f"{prompt}\n\n## Text\n{chunk}"
                        "\n\n## Extracted result\n"
                        + json.dumps(
                            result.final_output.model_dump(mode="json"),
                            ensure_ascii=False,
                        )
                    ),
                }
            )
            agent = Agent(
                name="Guardrail Agent",
                model=self.extraction_model_name,
                instructions=self.extraction_system_prompt,
                output_type=self.model_class,
            )
            requests.append((agent, inputs))

        return await run_batched(
            [_run_guardrail(agent, inputs) for agent, inputs in requests],
            batch_size=self.batch_size,
            log_prefix="guardrail",
        )
