import src.new_agent as new_agent


async def test_patch_apply_failure_routes_to_failure_when_budget_exhausted():
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        max_retries=1,
        retry_count=1,
        current_phase=new_agent.Phase.VERIFY,
        fix_attempts=[
            new_agent.FixAttempt(
                patch_content="malformed diff 1",
                test_result="patch_apply_failed",
                failure_kind="patch_apply_failed",
                error_log="error: first patch apply failed",
                success=False,
            ),
            new_agent.FixAttempt(
                patch_content="malformed diff 2",
                test_result="patch_apply_failed",
                failure_kind="patch_apply_failed",
                error_log="error: second patch apply failed",
                success=False,
            ),
        ],
    )

    next_state = await new_agent.verify_fix(state)

    assert next_state.current_phase == new_agent.Phase.FAILURE
    assert next_state.retry_count == 1
    assert next_state.failure_reason == "Maximum retries reached: 1."


async def test_consecutive_preflight_failures_do_not_consume_semantic_retry():
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        max_retries=1,
        retry_count=0,
        current_phase=new_agent.Phase.VERIFY,
        fix_attempts=[
            new_agent.FixAttempt(
                patch_content="malformed diff 1",
                test_result="patch_apply_failed",
                failure_kind="patch_apply_failed",
                error_log="Patch preflight check failed:\nerror: No valid patches in input",
                success=False,
            ),
            new_agent.FixAttempt(
                patch_content="malformed diff 2",
                test_result="patch_apply_failed",
                failure_kind="patch_apply_failed",
                error_log="Patch preflight check failed:\nerror: corrupt patch at line 3",
                success=False,
            ),
        ],
    )

    next_state = await new_agent.verify_fix(state)

    assert next_state.current_phase == new_agent.Phase.REFLECT
    assert next_state.retry_count == 0


async def test_consecutive_search_replace_failures_do_not_consume_semantic_retry():
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        max_retries=1,
        retry_count=0,
        current_phase=new_agent.Phase.VERIFY,
        fix_attempts=[
            new_agent.FixAttempt(
                patch_edits=[
                    new_agent.PatchEdit(
                        file_path="src/auth.py",
                        search="missing old block\n",
                        replace="new block\n",
                    )
                ],
                test_result="patch_apply_failed",
                failure_kind="patch_apply_failed",
                error_log=(
                    "Search/replace edit failed: edit 1 search block was not found "
                    "in src/auth.py."
                ),
                success=False,
            ),
            new_agent.FixAttempt(
                patch_edits=[
                    new_agent.PatchEdit(
                        file_path="src/auth.py",
                        search="still missing old block\n",
                        replace="new block\n",
                    )
                ],
                test_result="patch_apply_failed",
                failure_kind="patch_apply_failed",
                error_log=(
                    "Search/replace edit failed: edit 1 search block was not found "
                    "in src/auth.py."
                ),
                success=False,
            ),
        ],
    )

    next_state = await new_agent.verify_fix(state)

    assert next_state.current_phase == new_agent.Phase.REFLECT
    assert next_state.retry_count == 0


async def test_preflight_repair_failures_are_bounded_without_semantic_retry():
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        max_retries=1,
        retry_count=0,
        current_phase=new_agent.Phase.VERIFY,
        fix_attempts=[
            new_agent.FixAttempt(
                patch_content="malformed diff 1",
                test_result="patch_apply_failed",
                failure_kind="patch_apply_failed",
                error_log="Patch preflight check failed:\nerror: No valid patches in input",
                success=False,
            ),
            new_agent.FixAttempt(
                patch_content="malformed diff 2",
                test_result="patch_apply_failed",
                failure_kind="patch_apply_failed",
                error_log="Patch preflight check failed:\nerror: corrupt patch at line 3",
                success=False,
            ),
            new_agent.FixAttempt(
                patch_content="malformed diff 3",
                test_result="patch_apply_failed",
                failure_kind="patch_apply_failed",
                error_log="Patch preflight check failed:\nerror: corrupt patch at line 5",
                success=False,
            ),
        ],
    )

    next_state = await new_agent.verify_fix(state)

    assert next_state.current_phase == new_agent.Phase.FAILURE
    assert next_state.retry_count == 0
    assert next_state.failure_reason == (
        "Patch repair budget exhausted after 3 failures."
    )


async def test_search_replace_repair_failures_are_bounded_without_semantic_retry():
    state = new_agent.AgentState(
        issue_url="https://github.com/acme/widget/issues/7",
        max_retries=1,
        retry_count=0,
        current_phase=new_agent.Phase.VERIFY,
        fix_attempts=[
            new_agent.FixAttempt(
                patch_edits=[
                    new_agent.PatchEdit(
                        file_path="src/auth.py",
                        search=f"missing old block {idx}\n",
                        replace="new block\n",
                    )
                ],
                test_result="patch_apply_failed",
                failure_kind="patch_apply_failed",
                error_log=(
                    "Search/replace edit failed: edit 1 search block was not found "
                    "in src/auth.py."
                ),
                success=False,
            )
            for idx in range(3)
        ],
    )

    next_state = await new_agent.verify_fix(state)

    assert next_state.current_phase == new_agent.Phase.FAILURE
    assert next_state.retry_count == 0
    assert next_state.failure_reason == (
        "Patch repair budget exhausted after 3 failures."
    )
