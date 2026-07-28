from __future__ import annotations

from agents import RunConfig

from chunkbycount.helpers import _get_env_bool


def build_run_config() -> RunConfig:
    """Disable Agents SDK tracing unless the caller explicitly opts in."""
    tracing_enabled = _get_env_bool("CHUNKBYCOUNT_ENABLE_TRACING", False)
    return RunConfig(tracing_disabled=not tracing_enabled)
