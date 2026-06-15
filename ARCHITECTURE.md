# RepoPilot 架构说明

_最后更新：2026-06-15_

RepoPilot 是一个把 GitHub Issue 转成修复 PR 的 agent。当前实现围绕强类型的 Pydantic 状态、可审计的决策帧，以及白盒的 `plan -> reflect -> execute` 循环展开，因此每一次运行都可以回放、恢复和检查。

## 系统总览

```text
GitHub issue / CLI / API
    -> AgentState + Tracer
    -> LangGraph（或 fallback runner）
    -> UNDERSTAND -> LOCATE -> PLAN -> EXECUTE -> VERIFY
    -> REFLECT -> PLAN 循环，或 COMMIT / FAILURE
    -> JSON payload + trace log + 可选的磁盘暂停态
```

## 仓库结构

| 文件 / 模块 | 作用 |
|---|---|
| `src/main.py` | FastAPI 接口：`/analyze`、`/agent`、`/agent/v2`、`/agent/v2/resume` 和 replay 接口 |
| `src/cli.py` | CLI 入口：issue 运行、resume、run 列表、inspect、replay |
| `src/new_agent.py` | 编排层：构建 graph、运行 graph、整理 API payload |
| `src/graph.py` | 路由逻辑、fallback graph runner、超时控制、决策帧消费 |
| `src/state.py` | 核心状态模型、枚举、辅助函数、可追踪的决策帧记录 |
| `src/schemas.py` | 结构化 LLM 输出的校验 schema（plan / reflect） |
| `src/nodes/` | 各阶段节点实现：understand、locate、plan、execute、verify、reflect、commit、failure |
| `src/run_store.py` | 暂停运行的持久化存储，以及 inspect / replay 辅助函数 |
| `src/tracer.py` | 结构化 trace 事件采集 |

## 核心运行模型

### `AgentState`

`AgentState` 是一次运行的唯一事实来源，承载：

- issue 元数据和 repo 元数据
- 当前 `Phase`
- 排序后的文件、修复尝试、工具调用、对话历史
- token 使用量、重试计数、失败状态
- 白盒决策数据，例如 `decision_frame`、`frame_history`、`route_decisions` 和 `decision_warnings`
- 暂停态：`pending_human_input` 和 `human_input_request`

### 决策帧

[`src/state.py`](src/state.py) 定义了规划和反思共享的 reasoning 快照：

- `DecisionFrame`
- `Hypothesis`
- `PatchEdit`
- `FixAttempt`
- `ToolCall`
- `FinalReport`

`DecisionFrame` 是节点之间可审计的协议，记录：

- `stage`（`diagnose`、`plan`、`reflect`）
- `summary`
- `hypotheses`
- `selected_hypothesis_id`
- `evidence`
- `next_checks`
- `recommended_action`
- `risk`
- `confidence`
- `parent_frame_id`
- `trace_notes`

## 结构化 LLM 输出

[`src/schemas.py`](src/schemas.py) 校验的是完整的 LLM 输出，而不只是嵌入的 frame。

- `PlanDecision` 包装完整的 PLAN 输出
- `ReflectDecision` 包装完整的 REFLECT 输出
- 两者都兼容旧版扁平 JSON 格式
- 两者都要求嵌入的 `decision_frame.stage` 与所在节点匹配

这样可以把 LLM 合约显式化，同时在 prompt 格式演进时保持向后兼容。

## Graph 与路由

RepoPilot 在安装了 LangGraph 时使用它；如果不可用，则退回到一个纯 Python 的 fallback graph。两条路径共享同一套 state 合约。

[`src/new_agent.py`](src/new_agent.py) 里的 `build_agent_graph()` 负责 wiring 节点，[`src/graph.py`](src/graph.py) 负责路由。

### 执行流程

1. `agent_v2()` 创建新的 `AgentState` 和 trace id。
2. graph 运行 `UNDERSTAND -> LOCATE -> PLAN -> EXECUTE -> VERIFY`。
3. verify 失败后路由到 `REFLECT`，然后回到 `PLAN`。
4. verify 成功后路由到 `COMMIT`，除非启用了 eval 模式下的 `skip_commit`。
5. 严重失败路由到 `FAILURE`，然后终止。

### 路由规则

路由器会先尝试消费最新的 `DecisionFrame`。如果这个 frame 是新鲜且受支持的，就由它的 `recommended_action` 决定下一步。

支持的 action -> phase 映射：

- `collect_more_context` -> `LOCATE`
- `plan` -> `PLAN`
- `execute` -> `EXECUTE`
- `reflect` -> `REFLECT`
- `stop` -> `FAILURE`
- `ask_user` -> `WAITING_FOR_USER`

如果 frame 已过期、缺失、已经消费过，或者 action 不受支持，路由就回退到 `current_phase`。这些情况都会记录到 `route_decisions` 和 `decision_warnings`，而不是悄悄改控制流。

### 人工介入暂停

`ask_user` 会把运行切成一个可持久化暂停态：

- `pending_human_input = True`
- `current_phase = WAITING_FOR_USER`
- `human_input_request` 从 frame 中生成，优先使用 `next_checks[0]`，否则回退到 frame summary
- 运行会被保存，之后可以恢复

`resume_agent_v2(run_id, human_answer)` 会重新加载已保存的运行，追加人类回答，清空暂停态，把 phase 重置为 `PLAN`，然后继续执行。

## 各阶段职责

| 节点 | 职责 |
|---|---|
| `understand_issue` | 读取 GitHub Issue，分类任务，提取有效信号 |
| `locate_code` | 搜索并排序最可能相关的仓库文件 |
| `plan_fix` | 生成面向 patch 的计划和结构化 decision frame |
| `execute_fix` | 应用 patch 并运行目标测试命令 |
| `verify_fix` | 解释测试输出，决定成功、反思还是失败 |
| `reflect_on_failure` | 分析上一次尝试为什么失败，以及下一步该改什么 |
| `commit_fix` | 发布修复并打开 draft PR |
| `handle_failure` | 汇报已取得的部分进展并正常退出 |

## 持久化与回放

[`src/run_store.py`](src/run_store.py) 默认把暂停运行保存在 `~/.repopilot/runs`，也支持通过环境变量 `REPOPILOT_HOME` 改变根目录。

保存的运行支持：

- `list_runs`
- `inspect_run`
- `replay_run`
- Markdown 格式的回放输出

`agent_v2()` 返回的 API payload 会包含完整的白盒数据面：

- `decision_frame`
- `frame_history`
- `decision_warnings`
- `route_decisions`
- `node_diagnostics`
- `human_input_request`
- `waiting_for_user`
- `run_id`

这让运行结果可以从 CLI 和 HTTP API 两个入口都直接检查。

## 可观测性

[`src/tracer.py`](src/tracer.py) 负责采集结构化 trace 事件。除此之外，保存的 state 和 replay 输出也会保留路由与诊断元数据，便于事后分析失败。

重要的可观测性产物：

- `Tracer` 事件，用于 run 级别 trace
- `route_decisions`，用于重建路由
- `decision_warnings`，用于查看 frame/router 不一致
- `node_diagnostics`，用于 timeout 和 crash
- 已保存的 paused-run JSON，用于后续 resume 或 replay

## 兼容性接口

- `/agent/v2` 是主运行入口
- `/agent/v2/resume` 用于恢复暂停运行
- `/agent/v2/runs/{run_id}/replay` 提供 JSON 或 Markdown 回放
- `/analyze` 和 `/agent` 保留用于向后兼容
- `intelligent_analyze_issue()` 只是 `agent_v2()` 的别名

## 运行边界

RepoPilot 的设计刻意保持收敛：

- 面向维护真实仓库的专业开发者
- 强调可审计的 reasoning，而不是黑盒自动化
- 使用有上限的重试和 token budget，而不是无限循环
- 对不支持的控制动作走审计型 fallback
- 支持 eval 模式下的 `skip_commit`，在验证通过后可以不打开 PR 直接结束

## 测试覆盖

当前测试主要围绕白盒运行时：

- schema 校验：`tests/test_decision_schemas.py`
- decision frame 路由和暂停 / 恢复：`tests/test_decision_frame.py`
- graph 行为和 agent 流程：`tests/test_new_agent.py`
- 暂停运行存储：`tests/test_run_store.py`
- HTTP 接口：`tests/test_main.py`

## 相关文档

- [`docs/CODEX_CONTEXT.md`](docs/CODEX_CONTEXT.md)
- [`docs/PRODUCTION_PLAN.md`](docs/PRODUCTION_PLAN.md)
- [`docs/MEMORY_DESIGN_V2.md`](docs/MEMORY_DESIGN_V2.md)
- [`docs/RESUME_STRATEGY.md`](docs/RESUME_STRATEGY.md)
- [`docs/TECH_DESIGN_AND_INTERVIEW.md`](docs/TECH_DESIGN_AND_INTERVIEW.md)
