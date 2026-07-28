from agents import Agent, Runner

from chunkbycount.helpers import retry_on_llm_call, run_with_timeout
from chunkbycount.utils.model_settings import build_run_config


@retry_on_llm_call
async def aggregation_task(
    inputs: list,
    aggregation_model,
    system_prompt,
    user_prompt,
    output_type,
):
    agent = Agent(
        name="Aggregation Agent",
        model=aggregation_model,
        instructions=system_prompt,
        output_type=output_type,
    )

    aggregation_instruction = {
        "role": "user",
        "content": (
            "Verify the current structured result against the "
            "source context below. Correct or enrich existing "
            "items only. Do not add, remove, or reorder list items; "
            "return the same number of items. If nothing needs to "
            "change, return the result as is.\n\n"
            f"{user_prompt}"
        ),
    }

    aggregation_result = await run_with_timeout(
        Runner.run(
            agent, [*inputs, aggregation_instruction], run_config=build_run_config()
        )
    )
    return aggregation_result
