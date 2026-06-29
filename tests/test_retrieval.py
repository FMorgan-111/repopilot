from src.retrieval import bm25_rerank
from src.state import FileInfo


def test_bm25_rerank_prioritizes_file_with_issue_terms():
    files = [
        FileInfo(
            path="src/unrelated.py",
            relevance_score=0.6,
            reason="source file",
            content="def render_dashboard():\n    return template\n",
        ),
        FileInfo(
            path="src/tox/tox_env/api.py",
            relevance_score=0.4,
            reason="source file",
            content=(
                "class ToxEnv:\n"
                "    def envpython(self):\n"
                "        return self.environment_python\n"
                "    def reuse_environment(self):\n"
                "        return self.name\n"
            ),
        ),
    ]

    ranked = bm25_rerank(
        "tox envpython should not reuse the wrong environment",
        files,
    )

    assert ranked[0].path == "src/tox/tox_env/api.py"
    assert ranked[0].relevance_score > ranked[1].relevance_score
    assert "bm25 rerank score=" in ranked[0].reason
    assert "matched issue terms:" in ranked[0].reason


def test_bm25_rerank_preserves_order_when_no_useful_matches():
    files = [
        FileInfo(
            path="src/high.py",
            relevance_score=0.9,
            reason="high heuristic score",
            content="alpha beta gamma",
        ),
        FileInfo(
            path="src/low.py",
            relevance_score=0.2,
            reason="low heuristic score",
            content="delta epsilon zeta",
        ),
    ]

    ranked = bm25_rerank("unmatchedquery", files)

    assert [file.path for file in ranked] == ["src/high.py", "src/low.py"]
    assert [file.relevance_score for file in ranked] == [0.9, 0.2]
    assert [file.reason for file in ranked] == [
        "high heuristic score",
        "low heuristic score",
    ]


def test_bm25_rerank_preserves_order_when_content_is_empty():
    files = [
        FileInfo(
            path="src/high.py",
            relevance_score=0.9,
            reason="high heuristic score",
            content="",
        ),
        FileInfo(
            path="src/envpython.py",
            relevance_score=0.2,
            reason="low heuristic score",
            content="",
        ),
    ]

    ranked = bm25_rerank("envpython", files)

    assert [file.path for file in ranked] == ["src/high.py", "src/envpython.py"]
    assert [file.relevance_score for file in ranked] == [0.9, 0.2]
