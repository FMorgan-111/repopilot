from src import new_agent


async def test_agent_v2_state_machine_transitions_to_done(monkeypatch):
    visited = []

    async def understand(state):
        visited.append(state.current_phase.value)
        state.issue_title = "Login crash"
        state.current_phase = new_agent.Phase.LOCATE
        return state

    async def locate(state):
        visited.append(state.current_phase.value)
        state.current_phase = new_agent.Phase.PLAN
        return state

    async def plan(state):
        visited.append(state.current_phase.value)
        state.current_phase = new_agent.Phase.EXECUTE
        return state

    async def execute(state):
        visited.append(state.current_phase.value)
        state.current_phase = new_agent.Phase.VERIFY
        return state

    async def verify(state):
        visited.append(state.current_phase.value)
        state.current_phase = new_agent.Phase.COMMIT
        return state

    async def commit(state):
        visited.append(state.current_phase.value)
        state.pr_url = "https://github.com/acme/widget/pull/42"
        state.current_phase = new_agent.Phase.DONE
        return state

    async def failure(state):
        visited.append(state.current_phase.value)
        state.current_phase = new_agent.Phase.FAILED
        return state

    monkeypatch.setattr(new_agent, "understand_issue", understand)
    monkeypatch.setattr(new_agent, "locate_code", locate)
    monkeypatch.setattr(new_agent, "plan_fix", plan)
    monkeypatch.setattr(new_agent, "execute_fix", execute)
    monkeypatch.setattr(new_agent, "verify_fix", verify)
    monkeypatch.setattr(new_agent, "commit_fix", commit)
    monkeypatch.setattr(new_agent, "handle_failure", failure)

    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        max_retries=2,
        token_budget=5000,
    )

    final_state = await new_agent.run_graph(new_agent.build_agent_graph(), state)

    assert visited == ["UNDERSTAND", "LOCATE", "PLAN", "EXECUTE", "VERIFY", "COMMIT"]
    assert final_state.current_phase == new_agent.Phase.DONE
    assert final_state.pr_url == "https://github.com/acme/widget/pull/42"


async def test_verify_fix_replans_failed_attempt_once():
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        max_retries=2,
    )
    state.current_phase = new_agent.Phase.VERIFY
    state.fix_attempts.append(
        new_agent.FixAttempt(
            patch_content="diff --git a/src/auth.py b/src/auth.py",
            file_path="src/auth.py",
            test_result="failed",
            error_log="assert False",
            success=False,
        )
    )

    next_state = await new_agent.verify_fix(state)

    assert next_state.current_phase == new_agent.Phase.REFLECT
    assert next_state.retry_count == 1
