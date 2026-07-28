import os

from agents import (
    AsyncOpenAI,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from dotenv import load_dotenv


def configure_from_environment() -> None:
    """Load ``.env`` and configure the Agents SDK for Gemini when selected."""
    load_dotenv()

    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider not in {"openai", "gemini"}:
        raise ValueError("LLM_PROVIDER must be either 'openai' or 'gemini'")
    if provider == "openai":
        return

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")

    client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url=os.getenv(
            "OPENAI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        ),
    )
    set_default_openai_client(client, use_for_tracing=False)
    set_default_openai_api("chat_completions")
    set_tracing_disabled(True)


def load_environment_variables() -> None:
    """Backward-compatible alias for :func:`configure_from_environment`."""
    configure_from_environment()


def set_environment_variables(*args, **kwargs) -> None:
    """Backward-compatible alias for :func:`configure_from_environment`."""
    configure_from_environment()


if __name__ == "__main__":
    configure_from_environment()
