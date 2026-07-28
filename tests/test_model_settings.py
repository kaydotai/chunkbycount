from __future__ import annotations

import pytest

from chunkbycount.utils.model_settings import build_run_config


def test_tracing_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("CHUNKBYCOUNT_ENABLE_TRACING", raising=False)

    assert build_run_config().tracing_disabled is True


def test_tracing_can_be_enabled_explicitly(monkeypatch):
    monkeypatch.setenv("CHUNKBYCOUNT_ENABLE_TRACING", "true")

    assert build_run_config().tracing_disabled is False


def test_invalid_tracing_value_is_rejected(monkeypatch):
    monkeypatch.setenv("CHUNKBYCOUNT_ENABLE_TRACING", "sometimes")

    with pytest.raises(ValueError, match="CHUNKBYCOUNT_ENABLE_TRACING"):
        build_run_config()
