from __future__ import annotations

import pytest

from chunkbycount.utils import set_openai_api_key


def test_gemini_configuration_requires_a_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(set_openai_api_key, "load_dotenv", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        set_openai_api_key.configure_from_environment()


def test_configuration_rejects_unknown_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(set_openai_api_key, "load_dotenv", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "gemni")

    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        set_openai_api_key.configure_from_environment()


def test_gemini_configuration_uses_public_sdk_hooks(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: dict[str, object] = {}
    sentinel_client = object()

    monkeypatch.setattr(set_openai_api_key, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        set_openai_api_key,
        "AsyncOpenAI",
        lambda **kwargs: calls.update(client_kwargs=kwargs) or sentinel_client,
    )
    monkeypatch.setattr(
        set_openai_api_key,
        "set_default_openai_client",
        lambda client, **kwargs: calls.update(
            default_client=client,
            default_client_kwargs=kwargs,
        ),
    )
    monkeypatch.setattr(
        set_openai_api_key,
        "set_default_openai_api",
        lambda api: calls.update(default_api=api),
    )
    monkeypatch.setattr(
        set_openai_api_key,
        "set_tracing_disabled",
        lambda disabled: calls.update(tracing_disabled=disabled),
    )
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    set_openai_api_key.configure_from_environment()

    assert calls["client_kwargs"] == {
        "api_key": "test-key",  # pragma: allowlist secret
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    }
    assert calls["default_client"] is sentinel_client
    assert calls["default_client_kwargs"] == {"use_for_tracing": False}
    assert calls["default_api"] == "chat_completions"
    assert calls["tracing_disabled"] is True
