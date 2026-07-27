import json

from src.evidence import EvidenceStore
from src.state import AgentState


def make_state() -> AgentState:
    return AgentState(issue_url="https://github.com/acme/widget/issues/7")


def test_add_assigns_stable_id_and_ignores_summary_in_fingerprint():
    first = EvidenceStore(make_state()).add(
        tool="read_file",
        summary="First wording",
        content="line one\r\nline two\r\n",
        file_path="src/../src/widget.py",
        symbol="Widget.render",
    )
    second = EvidenceStore(make_state()).add(
        tool="read_file",
        summary="Entirely different wording",
        content="line one\nline two\n",
        file_path="src/widget.py",
        symbol="Widget.render",
    )

    assert first.added is True
    assert first.evidence.evidence_id.startswith("ev_")
    assert first.evidence.evidence_id == second.evidence.evidence_id
    assert first.evidence.fingerprint == second.evidence.fingerprint


def test_add_suppresses_duplicate_fingerprint_without_appending():
    state = make_state()
    store = EvidenceStore(state)
    original = store.add(
        tool="read_file",
        summary="Original summary",
        content="same source",
        file_path="src/widget.py",
    )
    duplicate = store.add(
        tool="read_file",
        summary="Changed summary does not matter",
        content="same source",
        file_path="src/widget.py",
    )

    assert original.added is True
    assert duplicate.added is False
    assert duplicate.evidence == original.evidence
    assert state.evidence == [original.evidence]


def test_add_bounds_items_and_redacted_content():
    state = make_state()
    store = EvidenceStore(state, max_items=2, max_content_chars=32)
    first = store.add(tool="search", summary="one", content="a" * 100)
    second = store.add(tool="search", summary="two", content="b")
    rejected = store.add(tool="search", summary="three", content="c")

    assert first.added is True
    assert len(first.evidence.content) == 32
    assert second.added is True
    assert rejected.added is False
    assert rejected.disposition == "capacity"
    assert rejected.evidence is None
    assert [item.summary for item in state.evidence] == ["one", "two"]


def test_duplicate_and_capacity_have_distinct_result_contracts():
    state = make_state()
    store = EvidenceStore(state, max_items=1)
    original = store.add(tool="search", summary="one", content="same")

    duplicate = store.add(tool="search", summary="again", content="same")
    capacity = store.add(tool="search", summary="two", content="different")

    assert original.disposition == "added"
    assert duplicate.disposition == "duplicate"
    assert duplicate.evidence == original.evidence
    assert capacity.disposition == "capacity"
    assert capacity.evidence is None


def test_add_bounds_persisted_summary_as_well_as_content():
    evidence = EvidenceStore(
        make_state(),
        max_content_chars=32,
        max_summary_chars=12,
    ).add(
        tool="search",
        summary="a summary that must not be persisted without a bound",
        content="short content",
    ).evidence

    assert evidence.summary == "a summary th"
    assert len(evidence.summary) == 12


def test_add_redacts_secrets_and_normalizes_traversal_path():
    evidence = EvidenceStore(make_state(), max_content_chars=200).add(
        tool="read_file",
        summary="Authorization: Bearer summary-secret",
        content=(
            "Authorization: Bearer content-secret\n"
            "https://example.test/?api_key=query-secret\n"
        ),
        file_path="../../src/../private/./settings.py",
    ).evidence

    assert "summary-secret" not in evidence.summary
    assert "content-secret" not in evidence.content
    assert "query-secret" not in evidence.content
    assert evidence.file_path == "private/settings.py"
    assert ".." not in evidence.file_path


def test_select_keeps_requested_order_and_total_content_bound():
    state = make_state()
    store = EvidenceStore(state)
    first = store.add(tool="search", summary="first", content="12345").evidence
    second = store.add(tool="search", summary="second", content="67890").evidence
    third = store.add(tool="search", summary="third", content="abcde").evidence
    budget = len(store.render_for_prompt([third, first]))

    selected = store.select(
        [third.evidence_id, "missing", first.evidence_id, second.evidence_id],
        max_total_chars=budget,
    )

    assert selected == [third, first]
    assert len(store.render_for_prompt(selected)) <= budget


def test_select_bounds_complete_prompt_payload_and_deduplicates_requested_ids():
    state = make_state()
    store = EvidenceStore(state)
    first = store.add(
        tool="search",
        summary="summary that must count toward the prompt budget",
        content="a",
    ).evidence
    second = store.add(tool="search", summary="second", content="b").evidence
    first_payload = json.dumps(
        first.model_dump(), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )

    too_small = store.select([first.evidence_id], max_total_chars=len(first.content))
    selected = store.select(
        [first.evidence_id, first.evidence_id, second.evidence_id],
        max_total_chars=len(first_payload),
    )

    assert too_small == []
    assert selected == [first]
    assert store.render_for_prompt(selected) == first_payload
    assert len(store.render_for_prompt(selected)) <= len(first_payload)
