"""JSON parsing tolerance for chat-model responses (the pandas-style failure:
model emits slightly-malformed JSON or buries it in prose).

These are env-independent: the pure extractor is tested directly, and the
llm_call retry is driven by monkeypatching llm_request (no HTTP / no gateway
URL), so they stay green regardless of the configured provider endpoint.
"""

import pytest

import src.llm as llm
from src.llm import _extract_json, llm_call


def test_extract_json_strict_still_works():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_trailing_comma():
    assert _extract_json('{"a": 1, "b": [1, 2,],}') == {"a": 1, "b": [1, 2]}


def test_extract_json_smart_quotes():
    assert _extract_json('{“a”: “x”}') == {"a": "x"}


def test_extract_json_fenced_with_trailing_comma():
    text = 'here you go:\n```json\n{"a": 1,}\n```\n'
    assert _extract_json(text) == {"a": 1}


def test_extract_json_prose_wrapped_object():
    text = 'Sure! {"a": 1, "b": 2,} That is the fix.'
    assert _extract_json(text) == {"a": 1, "b": 2}


def test_extract_json_raises_on_garbage():
    with pytest.raises(ValueError):
        _extract_json("no json here at all")


async def test_llm_call_retries_once_on_unparseable(monkeypatch):
    calls = []

    async def fake_llm_request(messages, model=None):
        calls.append(messages)
        content = "I cannot help with that." if len(calls) == 1 else '{"ok": true}'
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr(llm, "llm_request", fake_llm_request)
    result = await llm_call("sys", "user")

    assert result == {"ok": True}
    assert len(calls) == 2
    # The retry carries an explicit "valid JSON" instruction.
    assert "valid JSON" in calls[1][-1]["content"]


async def test_llm_call_no_retry_when_first_parses(monkeypatch):
    calls = []

    async def fake_llm_request(messages, model=None):
        calls.append(messages)
        return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    monkeypatch.setattr(llm, "llm_request", fake_llm_request)
    result = await llm_call("sys", "user")

    assert result == {"ok": True}
    assert len(calls) == 1
