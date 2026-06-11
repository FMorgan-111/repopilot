import json

import pytest

from src import cli


def test_cli_resume_json_calls_resume_agent(monkeypatch, capsys):
    calls = []

    async def fake_resume_agent_v2(run_id, human_answer):
        calls.append({"run_id": run_id, "human_answer": human_answer})
        return {
            "success": True,
            "waiting_for_user": False,
            "final_phase": "DONE",
            "run_id": run_id,
            "trace_id": run_id,
            "error": None,
        }

    monkeypatch.setattr(cli, "resume_agent_v2", fake_resume_agent_v2)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["resume", "abc123def456", "Breaking changes are not allowed.", "--json"])

    assert exc_info.value.code == 0
    assert calls == [
        {
            "run_id": "abc123def456",
            "human_answer": "Breaking changes are not allowed.",
        }
    ]
    assert json.loads(capsys.readouterr().out) == {
        "success": True,
        "waiting_for_user": False,
        "final_phase": "DONE",
        "run_id": "abc123def456",
        "trace_id": "abc123def456",
        "error": None,
    }


def test_cli_issue_url_path_remains_supported(monkeypatch, capsys):
    calls = []

    async def fake_agent_v2(issue_url, max_retries=3, token_budget=50000):
        calls.append(
            {
                "issue_url": issue_url,
                "max_retries": max_retries,
                "token_budget": token_budget,
            }
        )
        return {
            "success": True,
            "final_phase": "DONE",
            "error": None,
            "turns_taken": 1,
            "token_used": 10,
        }

    monkeypatch.setattr(cli, "agent_v2", fake_agent_v2)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "https://github.com/acme/widget/issues/42",
                "--max-retries",
                "2",
                "--token-budget",
                "5000",
            ]
        )

    assert exc_info.value.code == 0
    assert calls == [
        {
            "issue_url": "https://github.com/acme/widget/issues/42",
            "max_retries": 2,
            "token_budget": 5000,
        }
    ]
    assert "RepoPilot analyzing https://github.com/acme/widget/issues/42" in capsys.readouterr().out


def test_cli_runs_json_lists_saved_runs(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "list_runs",
        lambda: [
            {
                "run_id": "abc123def456",
                "issue_url": "https://github.com/acme/widget/issues/7",
                "current_phase": "WAITING_FOR_USER",
                "pending_human_input": True,
                "human_input_question": "Confirm whether breaking changes are allowed.",
                "latest_decision_frame": {"frame_id": "df_0001"},
                "updated_at": "2026-06-11T12:00:00+00:00",
            }
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["runs", "--json"])

    assert exc_info.value.code == 0
    assert json.loads(capsys.readouterr().out) == [
        {
            "run_id": "abc123def456",
            "issue_url": "https://github.com/acme/widget/issues/7",
            "current_phase": "WAITING_FOR_USER",
            "pending_human_input": True,
            "human_input_question": "Confirm whether breaking changes are allowed.",
            "latest_decision_frame": {"frame_id": "df_0001"},
            "updated_at": "2026-06-11T12:00:00+00:00",
        }
    ]


def test_cli_inspect_json_returns_saved_run_summary(monkeypatch, capsys):
    calls = []

    def fake_inspect_run(run_id):
        calls.append(run_id)
        return {
            "run_id": run_id,
            "issue_url": "https://github.com/acme/widget/issues/7",
            "current_phase": "WAITING_FOR_USER",
            "pending_human_input": True,
            "human_input_question": "Confirm whether breaking changes are allowed.",
            "latest_decision_frame": {"frame_id": "df_0001"},
            "updated_at": "2026-06-11T12:00:00+00:00",
        }

    monkeypatch.setattr(cli, "inspect_run", fake_inspect_run)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["inspect", "abc123def456", "--json"])

    assert exc_info.value.code == 0
    assert calls == ["abc123def456"]
    assert json.loads(capsys.readouterr().out) == {
        "run_id": "abc123def456",
        "issue_url": "https://github.com/acme/widget/issues/7",
        "current_phase": "WAITING_FOR_USER",
        "pending_human_input": True,
        "human_input_question": "Confirm whether breaking changes are allowed.",
        "latest_decision_frame": {"frame_id": "df_0001"},
        "updated_at": "2026-06-11T12:00:00+00:00",
    }


def test_cli_replay_json_returns_trace_replay(monkeypatch, capsys):
    calls = []

    def fake_replay_run(run_id):
        calls.append(run_id)
        return {
            "run_id": run_id,
            "issue_url": "https://github.com/acme/widget/issues/7",
            "current_phase": "WAITING_FOR_USER",
            "pause": {
                "pending_human_input": True,
                "question": "Confirm whether breaking changes are allowed.",
            },
            "timeline": [
                {
                    "index": 1,
                    "type": "decision_frame",
                    "frame_id": "df_0001",
                    "stage": "plan",
                    "summary": "Need user approval before patching.",
                    "recommended_action": "ask_user",
                    "route": {"route": "__end__"},
                    "warnings": [],
                }
            ],
        }

    monkeypatch.setattr(cli, "replay_run", fake_replay_run)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["replay", "abc123def456", "--json"])

    assert exc_info.value.code == 0
    assert calls == ["abc123def456"]
    assert json.loads(capsys.readouterr().out) == {
        "run_id": "abc123def456",
        "issue_url": "https://github.com/acme/widget/issues/7",
        "current_phase": "WAITING_FOR_USER",
        "pause": {
            "pending_human_input": True,
            "question": "Confirm whether breaking changes are allowed.",
        },
        "timeline": [
            {
                "index": 1,
                "type": "decision_frame",
                "frame_id": "df_0001",
                "stage": "plan",
                "summary": "Need user approval before patching.",
                "recommended_action": "ask_user",
                "route": {"route": "__end__"},
                "warnings": [],
            }
        ],
    }


def test_cli_replay_markdown_prints_formatted_replay(monkeypatch, capsys):
    replay = {
        "run_id": "abc123def456",
        "timeline": [],
    }

    monkeypatch.setattr(cli, "replay_run", lambda run_id: replay)
    monkeypatch.setattr(
        cli,
        "format_replay_markdown",
        lambda replay_data: f"# Replay {replay_data['run_id']}",
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["replay", "abc123def456", "--markdown"])

    assert exc_info.value.code == 0
    assert capsys.readouterr().out == "# Replay abc123def456\n"
