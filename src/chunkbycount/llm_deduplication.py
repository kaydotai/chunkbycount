import asyncio
import json
from pydantic import BaseModel, Field

from agents import Agent, Runner

from chunkbycount.helpers import _get_env_float, retry_on_llm_call
from chunkbycount.utils.model_settings import build_run_config


class DeduplicationInstructions(BaseModel):
    """Model the minimal instruction set returned by the LLM for deduplication."""

    deleted_lines: list[int] = Field(
        default_factory=list, description="Line numbers that should be deleted"
    )


MAX_ITERATIONS = 5  # Safety-guard to prevent infinite loops


@retry_on_llm_call
async def _deduplicate_once(
    items: list,
    model,
    deduplication_instructions: str | None = None,
    document_context: str | None = None,
):
    """Run a *single* pass of the LLM-based deduplication over ``items``.

    Iteration until convergence is handled by ``deduplicate_until_converged``.
    """
    # Build numbered JSONL
    numbered_jsonl: list[str] = []
    for idx, item in enumerate(items, start=1):
        if hasattr(item, "model_dump_json"):
            numbered_jsonl.append(f"{idx}\t{item.model_dump_json(exclude_none=True)}")
        else:
            numbered_jsonl.append(f"{idx}\t{json.dumps(item, default=str)}")

    jsonl_text = "\n".join(numbered_jsonl)

    system_prompt = (
        "You are provided with a JSONL list of items. "
        "Each line begins with a 1-based line number, followed by a tab and the JSON object. "
        "Identify duplicate entries and return only the lines that should be deleted. "
        "If a deduplication metric is supplied and two or more similar items are found, retain the item with the most complete information and delete the others.\n"
    )
    if deduplication_instructions:
        system_prompt += f"Specific instructions: {deduplication_instructions}."

    agent = Agent(
        name="Deduplication Agent",
        model=model,
        instructions=system_prompt,
        output_type=DeduplicationInstructions,
    )

    inputs = [
        {
            "role": "user",
            "content": jsonl_text,
        }
    ]
    # If additional document context is provided, append it as a separate message part so the
    # LLM can make better informed decisions when picking the most complete item.
    if document_context:
        inputs[0]["content"] += (
            "\n---\nAdditional document context (may be helpful for choosing the most "
            "complete entry):\n" + document_context
        )

    dedup_timeout_s = _get_env_float("CHUNKBYCOUNT_DEDUP_TIMEOUT_S", 120.0)
    result = await asyncio.wait_for(
        Runner.run(agent, inputs, run_config=build_run_config()),
        timeout=dedup_timeout_s,
    )
    instructions = result.final_output

    deleted_lines: set[int] = set(instructions.deleted_lines)

    # Apply deletions
    final_items = [
        obj for i, obj in enumerate(items, start=1) if i not in deleted_lines
    ]
    return final_items


async def deduplicate_until_converged(
    items: list,
    model,
    deduplication_instructions: str | None = None,
    document_context: str | None = None,
):
    """Run ``_deduplicate_once`` repeatedly until the list length stabilises.

    A hard ``MAX_ITERATIONS`` limit is applied as a safety-guard to avoid
    pathological looping in case the LLM behaves unexpectedly.
    """
    previous_len = len(items)
    for _ in range(MAX_ITERATIONS):
        items = await _deduplicate_once(
            items,
            model,
            deduplication_instructions,
            document_context,
        )
        if len(items) == previous_len:
            break
        previous_len = len(items)
    return items
