# CLAUDE.md — RepoPilot

RepoPilot 是把 GitHub Issue 自动修成 PR 的 agent,基于 LangGraph 的白盒状态机(UNDERSTAND → LOCATE → PLAN → EXECUTE → VERIFY →(失败则 REFLECT 回 PLAN))。默认分支 `master`。分阶段 handoff 见 `docs/CODEX_CONTEXT.md`。

## 环境与命令

- 用 `.venv`(uv 创建)。若报 `ModuleNotFoundError`,venv 依赖不全,补装:
  `uv pip install --python .venv/bin/python -r requirements.txt aiosqlite tenacity`
  (`requirements.txt` 未列全 `aiosqlite`/`tenacity`,需手动补。)
- 测试:`.venv/bin/python -m pytest -q`(异步测试经 `tests/conftest.py` 自动跑,无需 mark)。
- 全量 eval(会真实 clone 仓库 + 跑测试 + 打 LLM API):
  `.venv/bin/python eval/harness.py --agent-v2 --samples <N> --max-retries 2 --token-budget 100000`
  样本集:`data/samples/issues_fixes.jsonl`(现 150 条);更大的池子在 `data/dataset-merged.jsonl`(1493 条)。
- `.env` 需 `LLM_API_KEY` / `GITHUB_TOKEN`。模型经 `LLM_MODEL`(默认 `claude-sonnet-5:stable`)、端点经 `OPENAI_BASE_URL` 可配（默认 `https://linoapi.com.cn/v1`）；`llm_call(system,user,model=)` 支持按节点传模型。

## 关键模块

- `src/nodes/`:各阶段节点(understand/locate/plan/execute/verify/reflect/failure)。
- `src/state.py`:`AgentState`、`DecisionFrame`、`Hypothesis`、`FixAttempt`、`PatchEdit`;预算检查 `_is_budget_exceeded`。
- `src/graph.py`:状态机路由。`src/llm.py` + `src/http_client.py`:LLM 调用(OpenAI 兼容,单轮阻塞)。
- `src/memory/repo_store.py`:按 `owner/repo` 分库的 SQLite 记忆。`src/retrieval.py`:单 repo BM25 词法重排。
- `eval/harness.py`:eval 入口(`SAMPLES_PATH` 硬编码 `data/samples/issues_fixes.jsonl`,`load_samples` 取前 N)。

## 已知架构缺口(2026-07-04 核实)

- **跨仓库学习为零**:memory 按 owner/repo 物理分库,检索是单 repo BM25,无 embedding/跨 repo 迁移。→ 正在做 PLAN 阶段跨 repo 语义召回(sqlite-vec)。
- **跨仓库学习**:`src/memory/error_episode_store.py` 全局 `episodes.db` + sqlite-vec 余弦检索,PLAN 阶段召回相似历史修复(✅成功当模板/❌失败当避坑),VERIFY 后写 episode。**opt-in**:置 `REPOPILOT_ENABLE_EPISODES=1` 才启用(默认关,因首次 embed 要从 HuggingFace 下 bge-small,离线/CI 会卡)。embedding 用 fastembed(ONNX,~130MB,384维),`src/memory/embedding.py` 懒加载。
- **无流式**:LLM 调用阻塞式取完整 content,无 SSE/增量解析。
- **REFLECT→PLAN 不共享上下文**:`llm_call` 单轮 system+user;`conversation_history` 被记录但从不作为 messages 回传 LLM。
- **trace_notes 半结构化**:自由文本,缺 `failure_category`/`root_cause_type`/`wrong_file`;`FixAttempt.failure_kind` 只有粗粒度机械分类。
- **无模型 provider 抽象**:无 provider 接口/注册表/fallback 链;换非 OpenAI 兼容协议需改 `http_client.py`/`llm.py`。

## 约定

- eval 会在 `~/.repopilot/repos/<owner-repo>-work` 复用每 repo 的工作树 + venv(样本间 `git reset --hard`)。若引入样本级并行,需按 repo 分片避免抢同一目录。
- `eval/eval_results.json`、`eval_summary.md` 是每跑必变的再生产物,已 gitignore。
