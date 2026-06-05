import json
import re

from src.tracer import Tracer


def test_trace_id_is_generated():
    tracer = Tracer()

    assert re.fullmatch(r"[0-9a-f]{12}", tracer.trace_id)
    assert tracer.steps == []


def test_log_appends_step_and_prints_jsonl(capsys):
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

    out = capsys.readouterr().out.strip()
    printed = json.loads(out)
    assert printed == entry


def test_log_includes_error_field(capsys):
    tracer = Tracer()

    tracer.log("search_code", {"query": "auth"}, {}, error="HTTP 500")

    assert tracer.steps[0]["error"] == "HTTP 500"
    printed = json.loads(capsys.readouterr().out)
    assert printed["error"] == "HTTP 500"
