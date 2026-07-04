"""Tests for cross-repo semantic episode recall (P0 architecture upgrade).

Uses a deterministic fake embedder (token-hash bag-of-words) so no model
download is required; real embedding is exercised only in production.
"""

import hashlib
import json

import numpy as np

from src import new_agent
from src.memory import error_episode_store as eps
from src.memory.error_episode_store import ErrorEpisodeStore
from src.memory.keyframe import extract_keyframe
from src.memory.sqlite_vec_index import SqliteVecIndex
from src.nodes import plan as plan_node
from src.nodes import verify as verify_node


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
        owner="django", repo="django", issue_url="u1",
        issue_title="request middleware crashes on missing header",
        issue_body="KeyError raised when header missing in middleware",
        error_log='Traceback\n  File "m.py", line 10, in process\n    x=h["k"]\nKeyError: k',
        patch="guard the header lookup", success=True,
    )
    store.record(
        owner="pallets", repo="flask", issue_url="u2",
        issue_title="json serialization of set fails",
        issue_body="TypeError serializing a set to json",
        error_log="TypeError: set is not JSON serializable",
        patch="add set encoder", success=False,
    )
    return store


def _base_state(**kw):
    return new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        issue_title="crash in request middleware on missing header",
        issue_body="KeyError when a header is missing in the request middleware",
        current_phase=new_agent.Phase.PLAN,
        owner="acme", repo="widget",
        **kw,
    )


def _execute_response(file_path="a.py", search="a", replace="b"):
    return json.dumps(
        {
            "plan": "fix", "patch": "",
            "patch_edits": [{"file": file_path, "search": search, "replace": replace}],
            "files": [file_path], "test_command": "pytest",
            "decision_frame": {
                "stage": "plan", "summary": "fix",
                "recommended_action": "execute", "risk": "low", "confidence": 0.7,
            },
        }
    )


# ---- keyframe ---------------------------------------------------------------

def test_extract_keyframe_captures_exception_and_frames():
    log = (
        'Traceback (most recent call last):\n'
        '  File "/repo/app/router.py", line 42, in handle\n'
        '    return self.dispatch(req)\n'
        '  File "/repo/app/core.py", line 88, in dispatch\n'
        '    return route[key]\n'
        'KeyError: \'missing\'\n'
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
    import sqlite3

    idx = SqliteVecIndex(sqlite3.connect(":memory:"), dim=4)
    idx.add(1, [1, 0, 0, 0])
    idx.add(2, [0, 1, 0, 0])
    idx.add(3, [0.9, 0.1, 0, 0])
    hits = idx.search([1, 0, 0, 0], k=2)
    assert hits[0].rowid == 1
    assert hits[1].rowid == 3


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
    assert all(r.issue_url != "u2" for r in res) if hasattr(res[0], "issue_url") else True
    assert all(r.patch != "add set encoder" for r in res)


# ---- PLAN recall injection --------------------------------------------------

async def test_plan_prompt_includes_recalled_episodes(monkeypatch):
    store = _seeded_store()
    monkeypatch.setattr(eps, "get_episode_store", lambda: store)
    captured = {}

    async def fake_llm_call(system, user):
        captured["user"] = user
        return _execute_response()

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    await plan_node.plan_fix(_base_state())

    assert "RELATED PAST FIX EPISODES" in captured["user"]
    assert "✅ SUCCESS" in captured["user"]
    assert "django/django" in captured["user"]


async def test_plan_recall_is_best_effort_on_error(monkeypatch):
    def boom():
        raise RuntimeError("store down")

    monkeypatch.setattr(eps, "get_episode_store", boom)
    captured = {}

    async def fake_llm_call(system, user):
        captured["user"] = user
        return _execute_response()

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    next_state = await plan_node.plan_fix(_base_state())

    assert next_state.current_phase == new_agent.Phase.EXECUTE
    assert "RELATED PAST FIX EPISODES" not in captured["user"]


async def test_plan_recall_disabled_by_default(monkeypatch):
    # Episodes are opt-in; without REPOPILOT_ENABLE_EPISODES the store is off.
    monkeypatch.delenv("REPOPILOT_ENABLE_EPISODES", raising=False)
    eps.reset_episode_store()
    captured = {}

    async def fake_llm_call(system, user):
        captured["user"] = user
        return _execute_response()

    monkeypatch.setattr(plan_node, "llm_call", fake_llm_call)
    await plan_node.plan_fix(_base_state())

    assert eps.get_episode_store() is None
    assert "RELATED PAST FIX EPISODES" not in captured["user"]


# ---- VERIFY records episodes ------------------------------------------------

async def test_verify_records_episode(monkeypatch):
    store = ErrorEpisodeStore(db_path=":memory:", embedder=FakeEmbedder())
    monkeypatch.setattr(eps, "get_episode_store", lambda: store)
    state = _base_state(
        skip_commit=True,
        fix_attempts=[
            new_agent.FixAttempt(
                patch_content="the winning patch",
                test_result="passed",
                success=True,
            )
        ],
    )
    await verify_node.verify_fix(state)

    recalled = store.recall(
        issue_title=state.issue_title, issue_body=state.issue_body, k=1
    )
    assert recalled
    assert recalled[0].success is True
    assert recalled[0].patch == "the winning patch"
