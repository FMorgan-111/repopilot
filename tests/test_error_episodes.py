"""Tests for cross-repo semantic episode recall (P0 architecture upgrade).

Uses a deterministic fake embedder (token-hash bag-of-words) so no model
download is required; real embedding is exercised only in production.
"""

import hashlib
import json
import sqlite3

import numpy as np
import pytest

from src import new_agent
from src.memory import error_episode_store as eps
from src.memory import vector_backend
from src.memory.error_episode_store import ErrorEpisodeStore
from src.memory.keyframe import extract_keyframe
from src.memory.numpy_vector_index import NumpyVectorIndex
from src.memory.sqlite_vec_index import SqliteVecIndex
from src.nodes import plan as plan_node
from src.nodes import verify as verify_node
from src.patch_gate import validate_patch_batch
from src.repair_rounds import begin_repair_round, bind_repair_round_author
from src.state import RepairPlan, VerifiedEdit, VerifiedEditBatch


class FakeEmbedder:
    dim = 384

    def embed(self, text: str) -> list[float]:
        v = np.zeros(self.dim, dtype="float32")
        for tok in text.lower().split():
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = float(np.linalg.norm(v))
        return (v / n if n else v).tolist()


def _seeded_store():
    store = ErrorEpisodeStore(db_path=":memory:", embedder=FakeEmbedder())
    store.record(
        owner="django",
        repo="django",
        issue_url="u1",
        issue_title="request middleware crashes on missing header",
        issue_body="KeyError raised when header missing in middleware",
        error_log='Traceback\n  File "m.py", line 10, in process\n    x=h["k"]\nKeyError: k',
        patch="guard the header lookup",
        success=True,
    )
    store.record(
        owner="pallets",
        repo="flask",
        issue_url="u2",
        issue_title="json serialization of set fails",
        issue_body="TypeError serializing a set to json",
        error_log="TypeError: set is not JSON serializable",
        patch="add set encoder",
        success=False,
    )
    return store


def _base_state(**kw):
    return new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        issue_title="crash in request middleware on missing header",
        issue_body="KeyError when a header is missing in the request middleware",
        current_phase=new_agent.Phase.PLAN,
        owner="acme",
        repo="widget",
        **kw,
    )


def _exact_plan_state(exact_repair_state):
    exact_repair_state.issue_title = "crash in request middleware on missing header"
    exact_repair_state.issue_body = (
        "KeyError when a header is missing in the request middleware"
    )
    return exact_repair_state


def _execute_response(
    file_path="src/widget.py",
    search="return 'old-sentinel'",
    replace="return 'new-sentinel'",
):
    return json.dumps(
        {
            "plan": "fix",
            "patch": "",
            "patch_edits": [{"file": file_path, "search": search, "replace": replace}],
            "files": [file_path],
            "test_command": "pytest",
            "decision_frame": {
                "stage": "plan",
                "summary": "fix",
                "recommended_action": "execute",
                "risk": "low",
                "confidence": 0.7,
            },
        }
    )


# ---- keyframe ---------------------------------------------------------------


def test_extract_keyframe_captures_exception_and_frames():
    log = (
        "Traceback (most recent call last):\n"
        '  File "/repo/app/router.py", line 42, in handle\n'
        "    return self.dispatch(req)\n"
        '  File "/repo/app/core.py", line 88, in dispatch\n'
        "    return route[key]\n"
        "KeyError: 'missing'\n"
    )
    kf = extract_keyframe(log)
    assert "KeyError" in kf
    assert "core.py:88 in dispatch" in kf
    assert len(kf) <= 2000


def test_extract_keyframe_empty_and_fallback():
    assert extract_keyframe("") == ""
    plain = "some non-traceback output that just failed"
    assert extract_keyframe(plain)  # falls back to tail, non-empty


# ---- vector index -----------------------------------------------------------


def test_sqlite_vec_index_cosine_knn():
    try:
        idx = SqliteVecIndex(sqlite3.connect(":memory:"), dim=4)
    except AttributeError:
        pytest.skip("this Python build cannot load SQLite extensions")
    idx.add(1, [1, 0, 0, 0])
    idx.add(2, [0, 1, 0, 0])
    idx.add(3, [0.9, 0.1, 0, 0])
    hits = idx.search([1, 0, 0, 0], k=2)
    assert hits[0].rowid == 1
    assert hits[1].rowid == 3


def test_numpy_vector_index_cosine_knn_and_persistence(tmp_path):
    db_path = tmp_path / "numpy-vectors.db"
    first_conn = sqlite3.connect(db_path)
    first_index = NumpyVectorIndex(first_conn, dim=3)
    first_index.add(1, [1, 0, 0])
    first_index.add(2, [0, 1, 0])
    first_index.add(3, [0.9, 0.1, 0])
    first_conn.commit()
    first_conn.close()

    second_conn = sqlite3.connect(db_path)
    second_index = NumpyVectorIndex(second_conn, dim=3)
    hits = second_index.search([1, 0, 0], k=2)
    second_conn.close()

    assert [hit.rowid for hit in hits] == [1, 3]
    assert hits[0].distance == pytest.approx(0.0)


def test_numpy_vector_index_validates_dimensions():
    idx = NumpyVectorIndex(sqlite3.connect(":memory:"), dim=3)

    with pytest.raises(ValueError, match="expected dim 3, got 2"):
        idx.add(1, [1, 0])
    with pytest.raises(ValueError, match="expected dim 3, got 2"):
        idx.search([1, 0], k=1)


def test_backend_factory_falls_back_when_sqlite_vec_is_unavailable(monkeypatch):
    def broken_sqlite_vec_index(conn, dim):
        raise AttributeError("extension loading disabled")

    monkeypatch.setattr(
        vector_backend, "_create_sqlite_vec_index", broken_sqlite_vec_index
    )
    monkeypatch.setattr(vector_backend, "_fallback_warning_emitted", False)

    with pytest.warns(RuntimeWarning, match="falling back to NumPy"):
        index, backend_name = vector_backend.create_vector_index(
            sqlite3.connect(":memory:"), 3
        )

    assert isinstance(index, NumpyVectorIndex)
    assert backend_name == "numpy"


def test_episode_store_exposes_selected_backend():
    store = ErrorEpisodeStore(db_path=":memory:", embedder=FakeEmbedder())

    assert store.backend_name in {"sqlite_vec", "numpy"}


# ---- episode store ----------------------------------------------------------


def test_recall_surfaces_cross_repo_episode():
    store = _seeded_store()
    res = store.recall(
        issue_title="middleware crash on missing header",
        issue_body="KeyError missing header in request middleware",
        k=2,
    )
    assert res
    assert res[0].repo == "django"
    assert res[0].success is True


def test_recall_excludes_self_issue():
    store = _seeded_store()
    res = store.recall(
        issue_title="json serialization of set fails",
        issue_body="TypeError serializing a set to json",
        k=3,
        exclude_issue_url="u2",
    )
    assert (
        all(r.issue_url != "u2" for r in res) if hasattr(res[0], "issue_url") else True
    )
    assert all(r.patch != "add set encoder" for r in res)


# ---- PLAN recall injection --------------------------------------------------


async def test_plan_prompt_includes_recalled_episodes(exact_repair_state, monkeypatch):
    store = _seeded_store()
    monkeypatch.setattr(eps, "get_episode_store", lambda: store)
    captured = {}

    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        captured["user"] = user
        return _execute_response()

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    await plan_node.plan_fix(_exact_plan_state(exact_repair_state))

    assert "RELATED PAST FIX EPISODES" in captured["user"]
    assert "✅ SUCCESS" in captured["user"]
    assert "django/django" in captured["user"]


async def test_plan_recall_is_best_effort_on_error(exact_repair_state, monkeypatch):
    def boom():
        raise RuntimeError("store down")

    monkeypatch.setattr(eps, "get_episode_store", boom)
    captured = {}

    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        captured["user"] = user
        return _execute_response()

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    next_state = await plan_node.plan_fix(_exact_plan_state(exact_repair_state))

    assert next_state.current_phase == new_agent.Phase.EXECUTE
    assert "RELATED PAST FIX EPISODES" not in captured["user"]


async def test_plan_recall_disabled_by_default(exact_repair_state, monkeypatch):
    # Episodes are opt-in; without REPOPILOT_ENABLE_EPISODES the store is off.
    monkeypatch.delenv("REPOPILOT_ENABLE_EPISODES", raising=False)
    eps.reset_episode_store()
    captured = {}

    async def fake_llm_call(system, user, model=None, *, provider="primary", **_kwargs):
        captured["user"] = user
        return _execute_response()

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    await plan_node.plan_fix(_exact_plan_state(exact_repair_state))

    assert eps.get_episode_store() is None
    assert "RELATED PAST FIX EPISODES" not in captured["user"]


# ---- VERIFY records episodes ------------------------------------------------


async def test_verify_records_episode(exact_repair_state, monkeypatch):
    store = ErrorEpisodeStore(db_path=":memory:", embedder=FakeEmbedder())
    monkeypatch.setattr(eps, "get_episode_store", lambda: store)
    state = _exact_plan_state(exact_repair_state)
    state.skip_commit = True
    plan = RepairPlan(
        root_cause="widget returns the old sentinel",
        target_files=["src/widget.py"],
        target_symbols=["widget"],
        required_behavior="widget returns the new sentinel",
        regression_test_strategy="run the focused widget test",
    )
    state.active_repair_plan = plan
    begin_repair_round(state)
    bind_repair_round_author(state)
    assert validate_patch_batch(
        state,
        plan,
        VerifiedEditBatch(
            edits=[
                VerifiedEdit(
                    file_path="src/widget.py",
                    search="return 'old-sentinel'",
                    replace="return 'new-sentinel'",
                    intent="return the new sentinel",
                )
            ]
        ),
    ).accepted
    assert state.tool_patch_approval is not None
    state.fix_attempts.append(
        new_agent.FixAttempt(
            patch_content=state.patch_content,
            patch_edits=[edit.model_copy(deep=True) for edit in state.patch_edits],
            test_result="passed",
            success=True,
            patch_gate_fingerprint=(state.tool_patch_approval.patch_gate_fingerprint),
            repair_provider=state.authorized_repair_provider,
            repair_model=state.authorized_repair_model,
            repair_round_id=state.authorized_repair_round_id,
        )
    )
    state.current_phase = new_agent.Phase.VERIFY
    await verify_node.verify_fix(state)

    recalled = store.recall(
        issue_title=state.issue_title, issue_body=state.issue_body, k=1
    )
    assert recalled
    assert recalled[0].success is True
    assert recalled[0].patch == state.patch_content
