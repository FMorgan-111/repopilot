
import pytest
from src.llm import _extract_json


def test_extract_json_raw():
    assert _extract_json('{"x":"y"}') == {"x": "y"}


def test_extract_json_code_fence():
    assert _extract_json('```json\n{"a":1}\n```') == {"a": 1}


def test_extract_json_nested():
    assert _extract_json('text {"a":{"b":2}} more') == {"a": {"b": 2}}


def test_extract_json_invalid():
    with pytest.raises(ValueError):
        _extract_json("no json here")
