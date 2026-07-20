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
    assert [item.summary for item in state.evidence] == ["one", "two"]


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

    selected = store.select(
        [third.evidence_id, "missing", first.evidence_id, second.evidence_id],
        max_total_chars=10,
    )

    assert selected == [third, first]
    assert sum(len(item.content) for item in selected) <= 10
