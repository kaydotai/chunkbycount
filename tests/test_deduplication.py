from __future__ import annotations

from pydantic import BaseModel

import pytest

from chunkbycount.deduplication import deduplicate_exact


class Record(BaseModel):
    code: str
    label: str


def test_exact_dedup_preserves_distinct_punctuation():
    first = Record(code="AB-1", label="Example")
    punctuation_variant = Record(code="AB1", label="Example")
    duplicate = Record(code="AB-1", label="Example")

    result = deduplicate_exact([first, punctuation_variant, duplicate])

    assert result == [first, punctuation_variant]


def test_exact_dedup_canonicalises_dictionary_key_order():
    result = deduplicate_exact(
        [
            {"code": "A", "label": "One"},
            {"label": "One", "code": "A"},
        ]
    )

    assert result == [{"code": "A", "label": "One"}]


def test_exact_dedup_accepts_an_explicit_key():
    items = [
        Record(code="A", label="first"),
        Record(code="a", label="second"),
    ]

    result = deduplicate_exact(items, key=lambda item: item.code.casefold())

    assert result == [items[0]]


def test_exact_dedup_rejects_unhashable_keys():
    with pytest.raises(TypeError, match="hashable"):
        deduplicate_exact([{"id": 1}], key=lambda item: item)
