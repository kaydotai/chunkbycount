from pydantic import BaseModel, Field

from agents import Agent, Runner

from chunkbycount.helpers import retry_on_llm_call, run_with_timeout
from chunkbycount.utils.model_settings import build_run_config


class EntityCount(BaseModel):
    """How many target entities were found in the document."""

    total_count: int = Field(..., ge=0, description="Non-negative integer")


async def count_entities(
    extraction_prompt: str,
    model_name: str,
) -> EntityCount:
    """Let an LLM read the extraction prompt (which defines the target entity) and
    count **all occurrences** of that entity.
    Returns an EntityCount object.
    """
    # Default system / agent instructions
    default_instructions = (
        "You are the **Entity-Counting Agent**.\n"
        "The user gives you:\n"
        "  - An *extraction prompt* that describes exactly which entity type to find.\n"
        "  - The full text chunk.\n\n"
        "Count target entities only in the source-text portion of the prompt, "
        "not in instructions or examples. Count partial boundary records rather "
        "than omitting them. Preserve legitimate repeated occurrences unless "
        "the extraction instructions explicitly say to collapse them. If no "
        "target entity appears, return 0."
    )

    # User prompt
    count_prompt = (
        "## Extraction prompt (defines the entity to count)\n"
        f"{extraction_prompt}\n\n"  # Text is included in the prompt
        "## Task\n"
        "Count occurrences of the entity defined above and respond with the "
        "required JSON object. Follow any inclusion, exclusion, and identity "
        "rules in the extraction prompt."
    )

    @retry_on_llm_call
    async def _task():
        agent = Agent(
            name="Entity-Counting Agent",
            model=model_name,
            instructions=default_instructions,
            output_type=EntityCount,
        )
        return await run_with_timeout(
            Runner.run(
                agent,
                [
                    {
                        "role": "user",
                        "content": count_prompt,
                    }
                ],
                run_config=build_run_config(),
            )
        )

    result = await _task()
    return result.final_output
