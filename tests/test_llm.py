import json

from src.llm import classify_issue, generate_fix_plan, llm_call, rank_files


def deepseek_response(content):
    return {"choices": [{"message": {"content": json.dumps(content)}}]}


async def test_llm_call_success_returns_json(httpx_mock, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    httpx_mock.add_response(
        method="POST",
        url="https://api.deepseek.com/v1/chat/completions",
        json=deepseek_response({"answer": "ok"}),
    )

    result = await llm_call("system", "user")

    assert result == {"answer": "ok"}
    request = httpx_mock.requests[0]
    assert request.headers["authorization"] == "Bearer test-key"
    assert json.loads(request.content)["response_format"] == {"type": "json_object"}


async def test_classify_issue_returns_type_severity_confidence(httpx_mock):
    httpx_mock.add_response(
        method="POST",
        url="https://api.deepseek.com/v1/chat/completions",
        json=deepseek_response(
            {
                "type": "bug",
                "severity": "high",
                "confidence": 0.92,
                "reasoning": "Crashes on login.",
            }
        ),
    )

    result = await classify_issue("Login crash", "App crashes after submit.")

    assert result["type"] == "bug"
    assert result["severity"] == "high"
    assert result["confidence"] == 0.92


async def test_rank_files_returns_ordered_file_list(httpx_mock):
    httpx_mock.add_response(
        method="POST",
        url="https://api.deepseek.com/v1/chat/completions",
        json=deepseek_response(
            {
                "files": [
                    {
                        "path": "src/auth.py",
                        "relevance_score": 0.95,
                        "reason": "Handles login.",
                    },
                    {
                        "path": "src/main.py",
                        "relevance_score": 0.4,
                        "reason": "Routes requests.",
                    },
                ]
            }
        ),
    )

    result = await rank_files(
        "Login crash",
        "Crash on submit",
        [{"path": "src/main.py"}, {"path": "src/auth.py"}],
    )

    assert [item["path"] for item in result] == ["src/auth.py", "src/main.py"]
    assert result[0]["relevance_score"] > result[1]["relevance_score"]


async def test_generate_fix_plan_returns_expected_keys(httpx_mock):
    httpx_mock.add_response(
        method="POST",
        url="https://api.deepseek.com/v1/chat/completions",
        json=deepseek_response(
            {
                "fix_plan": "Patch auth validation.",
                "risk_level": "medium",
                "test_suggestions": ["Add regression test."],
            }
        ),
    )

    result = await generate_fix_plan(
        "Login crash",
        "Crash on submit",
        {"type": "bug", "severity": "high", "confidence": 0.9},
        [{"path": "src/auth.py", "relevance_score": 0.95, "reason": "Relevant"}],
    )

    assert result["fix_plan"] == "Patch auth validation."
    assert result["risk_level"] == "medium"
    assert result["test_suggestions"] == ["Add regression test."]
