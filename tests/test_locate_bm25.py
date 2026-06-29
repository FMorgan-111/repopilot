from src import new_agent
from src.nodes import locate as locate_node


class EmptyMemoryStore:
    async def get_file_index(self, owner, repo, limit=8):
        return []


async def test_locate_code_records_bm25_rerank_and_reorders_hydrated_files(
    monkeypatch,
):
    async def fake_search_code(query, owner, repo):
        return [
            {"path": "src/unrelated.py", "sha": "sha-noise"},
            {"path": "src/tox/tox_env/api.py", "sha": "sha-target"},
        ]

    async def fake_read_file(owner, repo, path):
        content_by_path = {
            "src/unrelated.py": "def render_dashboard():\n    return template\n",
            "src/tox/tox_env/api.py": (
                "class ToxEnv:\n"
                "    def envpython(self):\n"
                "        return self.environment_python\n"
                "    def reuse_environment(self):\n"
                "        return self.name\n"
            ),
        }
        return {"content": content_by_path[path], "sha": f"read-{path}"}

    monkeypatch.setenv("REPOPILOT_DISABLE_PARALLEL", "1")
    monkeypatch.setattr(locate_node, "get_store", lambda: EmptyMemoryStore())
    monkeypatch.setattr(locate_node, "search_code", fake_search_code)
    monkeypatch.setattr(locate_node, "read_file", fake_read_file)
    state = new_agent.AgentState(
        issue_url="https://github.com/tox-dev/tox/issues/3075",
        owner="tox-dev",
        repo="tox",
        issue_title="envpython reuses wrong tox environment",
        issue_body="tox should not reuse the black environment for tip-black.",
        current_phase=new_agent.Phase.LOCATE,
    )

    next_state = await locate_node.locate_code(state)

    assert next_state.current_phase == new_agent.Phase.PLAN
    assert next_state.relevant_files[0].path == "src/tox/tox_env/api.py"
    bm25_call = next(
        call for call in next_state.tool_calls if call.tool_name == "bm25_rerank"
    )
    assert bm25_call.args["candidate_count"] == 2
    assert "envpython reuses wrong tox environment" in bm25_call.args["query"]
    assert bm25_call.result["ranked"][0]["path"] == "src/tox/tox_env/api.py"
    assert bm25_call.result["ranked"][0]["bm25_score"] > 0


async def test_locate_code_preserves_hydrated_order_when_bm25_has_no_signal(
    monkeypatch,
):
    async def fake_search_code(query, owner, repo):
        return [
            {"path": "src/high.py", "sha": "sha-high"},
            {"path": "src/low.py", "sha": "sha-low"},
        ]

    async def fake_read_file(owner, repo, path):
        content_by_path = {
            "src/high.py": "alpha beta gamma",
            "src/low.py": "delta epsilon zeta",
        }
        return {"content": content_by_path[path], "sha": f"read-{path}"}

    monkeypatch.setenv("REPOPILOT_DISABLE_PARALLEL", "1")
    monkeypatch.setattr(locate_node, "get_store", lambda: EmptyMemoryStore())
    monkeypatch.setattr(locate_node, "search_code", fake_search_code)
    monkeypatch.setattr(locate_node, "read_file", fake_read_file)
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        owner="acme",
        repo="widget",
        issue_title="unmatchedquery",
        issue_body="unmatchedquery",
        current_phase=new_agent.Phase.LOCATE,
    )

    next_state = await locate_node.locate_code(state)

    assert [file.path for file in next_state.relevant_files] == [
        "src/high.py",
        "src/low.py",
    ]
    bm25_call = next(
        call for call in next_state.tool_calls if call.tool_name == "bm25_rerank"
    )
    assert bm25_call.result["applied"] is False
