import json
import logging
import re

from src.tracer import Tracer


def test_trace_id_is_generated():
    tracer = Tracer()

    assert re.fullmatch(r"[0-9a-f]{12}", tracer.trace_id)
    assert tracer.steps == []


def test_log_appends_step_and_prints_jsonl(caplog):
    caplog.set_level(logging.INFO, logger="repopilot.tracer")
    tracer = Tracer()

    tracer.log("read_issue", {"number": 1}, {"title": "Bug"})

    assert len(tracer.steps) == 1
    entry = tracer.steps[0]
    assert entry["trace_id"] == tracer.trace_id
    assert entry["step"] == "read_issue"
    assert entry["input"] == {"number": 1}
    assert entry["output"] == {"title": "Bug"}
    assert "ts" in entry
    assert "error" not in entry

    assert len(caplog.records) == 1
    printed = json.loads(caplog.records[0].message)
    assert printed == entry


def test_log_includes_error_field(caplog):
    caplog.set_level(logging.INFO, logger="repopilot.tracer")
    tracer = Tracer()

    tracer.log("search_code", {"query": "auth"}, {}, error="HTTP 500")

    assert tracer.steps[0]["error"] == "HTTP 500"
    assert len(caplog.records) == 1
    printed = json.loads(caplog.records[0].message)
    assert printed["error"] == "HTTP 500"


def test_multiple_steps_accumulate(caplog):
    """Logging multiple steps accumulates them in order."""
    caplog.set_level(logging.INFO, logger="repopilot.tracer")
    tracer = Tracer()

    tracer.log("step1", {"a": 1}, {"b": 2})
    tracer.log("step2", {"c": 3}, {"d": 4})
    tracer.log("step3", {"e": 5}, {"f": 6})

    assert len(tracer.steps) == 3
    assert tracer.steps[0]["step"] == "step1"
    assert tracer.steps[1]["step"] == "step2"
    assert tracer.steps[2]["step"] == "step3"
    assert len(caplog.records) == 3


def test_trace_ids_are_unique():
    """Different Tracer instances should have different trace IDs."""
    t1 = Tracer()
    t2 = Tracer()
    t3 = Tracer()

    ids = {t1.trace_id, t2.trace_id, t3.trace_id}
    assert len(ids) == 3  # all unique


def test_log_with_empty_input_output(caplog):
    """Log handles empty dicts for input/output."""
    caplog.set_level(logging.INFO, logger="repopilot.tracer")
    tracer = Tracer()

    tracer.log("empty_step", {}, {})

    assert tracer.steps[0]["input"] == {}
    assert tracer.steps[0]["output"] == {}


def test_log_error_none_excluded(caplog):
    """When error is explicitly None, it should be excluded from the entry."""
    caplog.set_level(logging.INFO, logger="repopilot.tracer")
    tracer = Tracer()

    tracer.log("no_error", {"x": 1}, {"y": 2}, error=None)

    assert "error" not in tracer.steps[0]


def test_log_timestamps_are_iso8601(caplog):
    """The ts field should be ISO 8601 format."""
    caplog.set_level(logging.INFO, logger="repopilot.tracer")
    tracer = Tracer()

    tracer.log("ts_test", {}, {})

    ts = tracer.steps[0]["ts"]
    # Should contain T (ISO separator) and end with Z or +00:00
    assert "T" in ts
    assert ts.endswith("Z") or "+" in ts
