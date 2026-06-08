# RepoPilot 多层 Memory 架构设计

> 设计者：Claude Opus (claude-opus-4-20250514)
> 日期：2026-06-08
> 版本：v1.0
>
> 本文档为 RepoPilot 设计一套四层记忆架构，解决当前进程中全部记忆随 FastAPI 重启丢失、跨 issue 零知识复用、反思不累积等致命问题。

---

## 1. 用户使用场景

从 RepoPilot 用户的实际使用出发，列出 7 个场景。每个场景标注对应的记忆层级需求。

### 场景 A：「丢一个 issue，让它修」— 一次性任务

用户在 GitHub 上看到一个 bug，把 issue URL 粘贴到 RepoPilot，期望 agent 自动分析、生成修复、跑测试、开 PR。

- 需要的记忆：当前 issue 的代码搜索结果、fix attempt 历史、LLM 对话上下文（工作记忆）
- 不需要跨 issue 或跨 session 记忆
- 最简单的场景，当前 v2 架构勉强覆盖（但不稳定，因为 conversation_history 从未被 LLM 读取）

### 场景 B：「同一 repo 连续给多个 issue」— 批量修复

用户在半天内给了 numpy 5 个 issue。期望 RepoPilot 在修第 4 个 issue 时已经知道 numpy 的目录结构、核心模块位置、哪些文件经常被改、上次修类似 bug 用了什么模式。

- 需要跨 issue 记忆：repo 知识积累（Layer 1）
- 关键痛点：当前架构对 numpy#12345 和 numpy#12346 完全独立处理，代码搜索从零开始
- 收益：第 4 个 issue 的 LOCATE 阶段可能从 6 次 API 调用降到 2 次

### 场景 C：「上次没修好，过几天再试」— 跨 session 续接

用户给了 issue A，RepairPilot 执行到 PLAN 阶段后发现测试环境有问题（pytest 依赖缺失），返回 FAILURE。三天后用户修好测试环境，重新提交同一个 issue。

- 需要跨 session 记忆：之前的 fix_attempts、reflection_notes、代码搜索结果（Layer 2）
- 当前架构：FastAPI 重启后一切丢失，只能从零再来
- 理想状态：恢复上次的 state，直接跳到 PLAN 或 EXECUTE，避免重复 API 调用

### 场景 D：「修了 30 个 issue 后开始变聪明」— 经验积累

用户持续使用 RepoPilot 几个月，修了上百个 issue。期望 agent 能学会：
- 「Python 项目的类型错误通常改函数签名+调用点，不只是改一处」
- 「Django 的 migration 问题需要同时检查 models.py 和最新 migration 文件」
- 「pandas 项目的测试命名惯例是 tests/test_<module>.py」

- 需要跨 repo、跨 session 的策略记忆（Layer 2 + Layer 3）
- 这是当前架构完全无法触及的目标

### 场景 E：「团队多人共享一个 agent 实例」— 多租户

一个 5 人团队共用同一个 RepoPilot 实例。每个人修不同的 repo。需要隔离每个人的 memory（至少 repo 级别的），同时又能共享通用的修复策略。

- 需要：repo 级别的 memory 隔离 + 全局共享的策略记忆
- 关键挑战：用户 A 的 repo 私有信息不能泄露给用户 B 的 session

### 场景 F：「修了一半手动干预」— 人机协作

Agent 在 EXECUTE 阶段跑测试时，用户看了 patch 觉得不对，手动修改后再让 agent 继续。Agent 需要理解「用户改了什么、为什么改」，而不是盲目覆盖。

- 需要：能并入人工反馈的记忆结构（Layer 0 需要支持 human-in-the-loop 插入）
- 当前架构：没有接入点，AgentState 只能被节点函数修改

### 场景 G：「CI 上自动触发」— 无头模式

GitHub Actions 在新 issue 打上 "good first issue" 标签时自动调用 RepoPilot。无人值守，期望 agent 在遇到不确定的情况时保守处理（开 draft PR 而不是直接 merge），并把决策依据写入记忆供事后审计。

- 需要：审计 trail（tool_call 历史）+ 决策依据持久化
- 所有记忆层都需要持久化到磁盘/数据库

---

## 2. 记忆分层设计

### 架构总览

```
┌─────────────────────────────────────────────────────────┐
│                     RepoPilot Agent                     │
├─────────────────────────────────────────────────────────┤
│  Layer 0: Working Memory    │  进程内存 + Redis 备份    │
│  (当前 issue 的即时上下文)   │  生命周期: 单次请求       │
├─────────────────────────────────────────────────────────┤
│  Layer 1: Execution Memory  │  SQLite per-repo          │
│  (repo 级别的知识)           │  生命周期: 跨 issue       │
├─────────────────────────────────────────────────────────┤
│  Layer 2: Reflection Memory │  SQLite global            │
│  (跨 repo 的经验学习)        │  生命周期: 跨 session     │
├─────────────────────────────────────────────────────────┤
│  Layer 3: Meta Memory       │  SQLite global + 向量库   │
│  (使用模式与自我优化)        │  生命周期: 永久            │
└─────────────────────────────────────────────────────────┘
```

---

### Layer 0：Working Memory（工作记忆）

当前 issue 处理过程中的所有即时状态。这是唯一一个直接注入 LLM prompt 的记忆层。

#### 存储内容

| 子区域 | 存储内容 | 来源 |
|--------|---------|------|
| `conversation_context` | LLM 对话历史（最近的 N 轮，含 system prompt） | UNDERSTAND, PLAN, REFLECT 阶段 |
| `code_cache` | 已读取的源文件内容（路径 → 内容 + sha） | LOCATE 阶段 |
| `attempt_trail` | 所有的 FixAttempt（patch + 测试结果 + 错误日志） | EXECUTE + VERIFY |
| `current_plan` | 当前激活的 fix_plan + patch_content | PLAN 阶段 |
| `budget_state` | 已用 token / 剩余 token / 重试次数 | 全局追踪 |
| `tool_audit` | 最近 N 次工具调用（名称、参数、结果摘要） | 所有阶段 |

#### 数据结构

```python
class WorkingMemory(BaseModel):
    # Conversation — 直接喂给 LLM
    conversation_turns: list[ConversationTurn]  # max 20 turns
    system_prompt_hash: str  # 用于判断 system prompt 是否变化

    # Code cache — 避免重复 read_file API 调用
    file_cache: dict[str, FileInfo]  # path → (content, sha, relevance)

    # Execution trail
    fix_attempts: list[FixAttempt]  # 有序列表，完整保留（不截断）
    current_patch: str | None
    current_plan: str | None

    # Budget tracking
    tokens_used: int
    tokens_budget: int
    retry_count: int
    max_retries: int
```

#### 存储介质与生命周期

- **主存储**：进程内存（AgentState 内嵌）
- **备份存储**：Redis（可选，key = `repopilot:wm:{issue_url_hash}`，TTL = 1h）
- **生命周期**：单次 issue 处理请求。请求结束后转为 Layer 1 的快照。
- **大小限制**：conversation_turns ≤ 20 轮（约 16K tokens），file_cache ≤ 50 个文件

#### 读写方式

- **写**：每个节点函数 `_remember()` / `_record_tool()` 写入
- **读**：`plan_fix()` 和 `reflect_on_failure()` 读取 conversation + attempt_trail 构造 LLM prompt
- **修复当前 bug**：`plan_fix` 已经读取 `fix_attempts` 和 `reflection_notes`，但**从未读取 `conversation_history`**。这是 P0 级 bug。

#### 淘汰策略

- conversation_turns：FIFO 滑动窗口，最多 20 轮。超过后丢弃最旧的轮次（但保留 system prompt）。
- file_cache：LRU，最多 50 个文件。当 relevance_score 更高的文件出现时淘汰低分文件。
- fix_attempts：保留全部（最多 `max_retries * 2` ≈ 6 个，token 开销极小）。

---

### Layer 1：Execution Memory（执行记忆 — Repo 级别）

一个 repo 在多次 issue 修复中积累的结构性知识。让第 N 个 issue 的修复能利用前 N-1 个 issue 的发现。

#### 存储内容

| 子区域 | 存储内容 | 用途 |
|--------|---------|------|
| `file_index` | 文件路径 → {topics, 修改频率, 最后修改时间} | 加速 LOCATE |
| `module_graph` | 模块间的 import 关系图（简化版） | PLAN 时评估修改影响范围 |
| `issue_log` | issue_id → {分类, 涉及文件, 修复策略, 是否成功} | 相似 issue 匹配 |
| `test_patterns` | 测试文件命名惯例、测试命令、pytest 配置路径 | EXECUTE 阶段 |
| `project_conventions` | 代码风格、PR 模板、commit message 格式、base branch 名 | COMMIT 阶段 |

#### 数据结构

```python
class RepoMemory(BaseModel):
    repo_id: str  # "owner/repo"

    # 文件索引: path → 元数据
    files: dict[str, RepoFileEntry]  # max 500 entries

    # Issue 历史: 最近的 issue 处理记录
    recent_issues: list[IssueRecord]  # max 50, 按时间倒序

    # 测试知识
    test_patterns: TestKnowledge

    # 项目约定
    conventions: ProjectConventions

    # 元数据
    last_accessed: datetime
    created_at: datetime
    version: int = 1

class RepoFileEntry(BaseModel):
    path: str
    topics: list[str]  # ["authentication", "login", "JWT"]
    fix_count: int  # 这个文件被修改了多少次
    last_modified_issue: str | None  # 最后一次修改来自哪个 issue
    relevance_persistence: float  # 0-1, 随时间和未使用衰减

class IssueRecord(BaseModel):
    issue_number: int
    issue_title: str
    issue_type: str  # bug / feature / refactor
    files_modified: list[str]
    fix_strategy: str  # "type_signature_change" / "null_check" / "config_update"
    success: bool
    tokens_consumed: int
    completed_at: datetime

class TestKnowledge(BaseModel):
    test_framework: str  # "pytest" / "unittest" / "jest"
    test_directory: str  # "tests/" / "test/" / "__tests__/"
    test_command: str  # "python -m pytest -q" / "pytest tests/"
    file_pattern: str  # "test_*.py" / "*_test.py"
```

#### 存储介质与生命周期

- **存储**：每个 repo 一个 SQLite 文件：`~/.repopilot/memory/repos/{owner}___{repo}.db`
- **生命周期**：跨 issue，跨 session。只要 repo 还在用，数据就保留。
- **大小限制**：单文件 ≤ 10MB。file_index 最多 500 个条目，recent_issues 最多 50 条。

#### 读写方式

- **写**：
  - LOCATE 阶段完成后 → 更新 `files` 索引（paths + topics）
  - 修复成功/失败后 → 追加 `recent_issues` 记录
  - 测试跑完后 → 更新 `test_patterns`
  - COMMIT 完成后 → 更新 `conventions`
- **读**：
  - LOCATE 阶段 → 先查 `files` 索引中有没有直接匹配的路径，有则跳过 code search API
  - PLAN 阶段 → 查 `recent_issues` 中是否有相似 issue，复用修复策略
  - EXECUTE 阶段 → 读 `test_patterns` 获取测试命令和目录

#### 淘汰策略

- **files**：LRU + 衰减。`relevance_persistence` 每日衰减 5%。低于 0.1 且超过 30 天未使用的条目被清理。
- **recent_issues**：FIFO，最多 50 条。新的进来，最旧的出去。
- **全库清理**：每次写入后检查总大小 > 10MB → 触发压缩（清理衰减分低于 0.05 的 files 条目 + 合并相似的 issue 记录）。

---

### Layer 2：Reflection Memory（反思记忆 — 跨 Repo 经验）

从所有 repo 的所有 issue 中提取的通用修复策略。这是 RepoPilot "边用边学"的核心。

#### 存储内容

| 子区域 | 存储内容 | 用途 |
|--------|---------|------|
| `strategy_catalog` | 修复策略类型 → 成功率、适用条件、失败模式 | PLAN 时选择策略 |
| `failure_patterns` | 常见失败模式 → 根因分析、规避方法 | REFLECT 时避免重复错误 |
| `tool_effectiveness` | 工具使用模式 → 哪种工具组合更有效 | 优化 LOCATE 策略 |

#### 数据结构

```python
class StrategyEntry(BaseModel):
    strategy_id: str  # uuid
    strategy_name: str  # "add_null_check", "update_type_signature", ...
    description: str  # 自然语言描述
    applicable_conditions: list[str]  # ["Python", "TypeError", "function_call"]
    success_count: int
    failure_count: int
    avg_tokens_consumed: float
    repo_types: list[str]  # 在哪些类型的项目中有效
    examples: list[StrategyExample]  # 最多 3 个代表性例子
    last_used: datetime
    confidence: float  # 基于贝叶斯更新的信心值 (0-1)

class StrategyExample(BaseModel):
    issue_url: str
    repo: str
    patch_snippet: str  # 精简后的 patch（前 500 字符）

class FailurePattern(BaseModel):
    pattern_id: str
    pattern_name: str  # "pytest_import_error", "patch_does_not_apply", ...
    root_cause_category: str  # "missing_dependency", "wrong_file_path", ...
    detection_signature: str  # 用于匹配的字符串模式（错误日志片段）
    avoidance_strategy: str
    occurrence_count: int
    last_seen: datetime
```

#### 存储介质与生命周期

- **存储**：全局 SQLite 文件：`~/.repopilot/memory/reflections.db`
- **生命周期**：永久。跨 repo，跨 session，跨 FastAPI 重启。
- **大小限制**：strategy_catalog ≤ 200 条，failure_patterns ≤ 100 条。

#### 读写方式

- **写**（异步，不阻塞主流程）：
  - 每次 REFLECT 完成后 → 提取失败模式（如果 error_log 匹配已知 pattern → 计数+1；否则 → 创建新 pattern）
  - 每次 DONE/FAILED → 更新所用策略的 success/failure 计数
  - 策略名称由 LLM 在 PLAN 阶段生成（在 response JSON 中要求 `strategy` 字段）
- **读**：
  - PLAN 阶段 → 读取 strategy_catalog 中相同条件的前 3 个高成功率策略，注入 LLM system prompt
  - REFLECT 阶段 → 读取 failure_patterns 中匹配的 pattern，辅助分析

#### 淘汰策略

- **StrategyEntry**：基于 `confidence` 的软淘汰。confidence 低于 0.2 的条目不再注入 prompt，但保留数据。手动或定期清理（30 天低于 0.1 的条目删除）。
- **FailurePattern**：FIFO + 频率加权。最近 7 天内出现过 3 次以上的 pattern 优先级最高。超过 60 天未出现且 occurrence_count < 3 → 删除。
- **贝叶斯更新**：confidence = (success_count + 1) / (success_count + failure_count + 2)。冷启动 confidence = 0.5。

---

### Layer 3：Meta Memory（元记忆 — 自我优化）

观察 RepoPilot 自身的使用模式，优化资源分配和默认行为。

#### 存储内容

| 子区域 | 存储内容 | 用途 |
|--------|---------|------|
| `usage_stats` | 每日/每周使用量、按 repo 分组 | 资源预分配 |
| `token_efficiency` | 每 repo 的 token 消耗趋势 | 预算动态调整 |
| `user_preferences` | PR 风格偏好（draft/ready、详细/简洁）、issue 类型偏好 | 个性化输出 |
| `retry_optimization` | 不同 max_retries 设置下的最终成功率 | 动态调整重试次数 |

#### 数据结构

```python
class UsageStats(BaseModel):
    total_issues_processed: int
    total_tokens_consumed: int
    total_prs_created: int
    overall_success_rate: float
    per_repo_stats: dict[str, RepoStats]
    daily_stats: list[DailyStats]  # 最近 90 天

class RepoStats(BaseModel):
    issues_processed: int
    success_rate: float
    avg_tokens_per_issue: float
    avg_retries_per_issue: float
    most_common_issue_type: str

class DailyStats(BaseModel):
    date: str  # "2026-06-08"
    issues_processed: int
    tokens_consumed: int
    prs_created: int
    success_count: int
    failure_count: int

class RetryOptimization(BaseModel):
    max_retries_tested: dict[int, RetryBucket]  # 1 → {...}, 2 → {...}, 3 → {...}

class RetryBucket(BaseModel):
    total_attempts: int
    resolved_after_retry_1: int
    resolved_after_retry_2: int
    resolved_after_retry_3: int
    never_resolved: int
```

#### 存储介质与生命周期

- **存储**：全局 SQLite 文件：`~/.repopilot/memory/meta.db`
- **可选增强**：向量数据库（ChromaDB / Qdrant）用于对 reflections 做语义检索
- **生命周期**：永久。聚合数据无限保留，detail 数据定期压缩。

#### 读写方式

- **写**：
  - 每次 `agent_v2()` 完成后 → 更新 usage_stats + token_efficiency
  - 在 `handle_failure()` 和 `commit_fix()` 中更新 retry_optimization
- **读**：
  - agent_v2 入口处 → 从 per_repo_stats 获取历史成功率，动态调整 max_retries 和 token_budget
  - COMMIT 阶段 → 从 user_preferences 读取 PR 风格

#### 淘汰策略

- daily_stats：保留最近 90 天。超过 90 天的聚合到 monthly_stats。
- per_repo_stats：永久保留（每个 repo 就一行记录）。
- retry_optimization：永久保留，数据量极小。

---

## 3. 架构缺陷

诚实列出这套设计在什么情况下会失效。

### 3.1 Layer 0 的缺陷

**冷启动 LLM context 污染**：
- conversation_turns 超过 20 轮时最早的上下文被丢弃。如果 issue 很复杂（需要 > 20 轮交互），模型可能丢失初始的分析推理链。
- 缓解：引入摘要机制 —— 每 10 轮对前 10 轮做一次 LLM 摘要，替代原始对话。

**file_cache 的 staleness**：
- 文件在 agent 运行期间被其他 PR 修改。sha 校验只能检测到，但无法 merge。当前架构假设 agent 独占操作。
- 场景：两个 issue 同时触发 agent（并发场景），操作同一个文件。

### 3.2 Layer 1 的缺陷

**Repo 重命名 / 迁移**：
- repo_id 变化（owner 改名、repo 迁移）时，已积累的记忆全部失联。需要支持 repo_id alias。
- 目前没有设计别名机制。

**过时的文件索引**：
- repo 重构（文件移动、重命名）后，Layer 1 的 file_index 指向旧路径。LOCATE 阶段查旧路径会 miss。
- 场景：repo 从单体架构迁移到微服务，目录结构全变。

**敏感信息泄露**：
- 如果 issue 中包含 API key 或 token，会随着 `issue_log` 被持久化到 SQLite。虽然每个 repo 一个 db 文件提供了基本隔离，但没有内容过滤。
- 场景：用户误贴了包含 credentials 的 error log。

**冷启动**：
- 新 repo 的 Layer 1 是空的，完全退化为当前架构。第一次修复需要接受全套 API 调用的代价。

### 3.3 Layer 2 的缺陷

**跨语言泛化失败**：
- "add_null_check" 策略在 Python 和 TypeScript 中可能有效，但在 Rust（Option type）中语法完全不同。策略的 `applicable_conditions` 如果不够精确，会在不适用的语境中被推荐。
- 场景：用户在 Rust 项目里收到 "add_null_check" 建议，LLM 尝试生成 Rust 语法却失败。

**策略命名歧义**：
- LLM 在 PLAN 阶段自行命名策略（`strategy` 字段）。两个实质上相同的策略可能被命名为 "null_guard" 和 "add_null_check"，导致策略碎片化、信心值分散。
- 缓解：需要策略去重/合并机制（基于 patch 相似度聚类），但这引入了额外复杂度。

**负迁移**：
- 从一个 repo 学到的"成功策略"在另一个 repo 中重复使用但失败，降低了该策略的 confidence。但如果失败原因是 repo 特殊配置而非策略本身错误，则 confidence 被错误降低。
- 场景：pytest 测试命令在 repo A 是 `pytest`，repo B 需要 `pytest --config=...`。策略本身正确，但因测试命令问题被标记为 failure。

### 3.4 Layer 3 的缺陷

**隐私风险**：
- usage_stats 记录了所有 issue URL、repo 名称、处理时间。如果 RepoPilot 实例被共享，这些元数据可能暴露用户的开发活动。
- 场景：公司内部 server 上的 RepoPilot，meta.db 包含所有私有 repo 的 issue 处理记录。

**冷启动 max_retries 优化不可靠**：
- 需要至少几百次 issue 处理才能让 retry_optimization 的统计显著。小样本下的动态调整可能适得其反（把 max_retries 调到 1 导致更多失败）。

### 3.5 通用缺陷

**这整套记忆对哪类 issue 帮不上忙？**

1. **依赖外部系统的 issue**：涉及数据库迁移、Kubernetes 配置、CI/CD pipeline 的 issue，因为修复依赖于 agent 无法访问的外部系统状态。
2. **需要领域知识的 issue**：密码学实现 bug、机器学习模型精度问题等，agent 毫无专业背景知识。
3. **超大规模的 issue**：涉及 50+ 个文件的架构重构，Layer 0 的 token budget 和 Layer 1 的文件索引都覆盖不到。
4. **刻意混淆的 issue**：恶意提交的 issue（代码注入、社会工程），记忆系统反而可能"学习"并复现恶意模式。

**内存与延迟成本**：
- Layer 1 的 SQLite 读写每次约 5-20ms，在主流程中同步读写会增加延迟。
- Layer 2 的策略检索如果做语义匹配（需要向量数据库），延迟可能到 50-200ms。
- 生产建议：Layer 1/2 的写操作全部异步化，读操作中对延迟敏感的（LOCATE 阶段的 file_index 查询）做同步，其他的异步。

**数据一致性风险**：
- Layer 0（内存）和 Layer 1（SQLite）之间存在时间窗口：LOCATE 完成后内存有 file_cache，但 Layer 1 的 file_index 在异步写入完成前可能不一致。
- 两个并发请求操作同一个 repo 时，Layer 1 的 file_index 和 recent_issues 可能冲突（SQLite 的 WAL 模式可以缓解写冲突，但业务层面的数据版本冲突仍然存在）。

---

## 4. 实现建议

### 4.1 优先级分类

#### P0 — 立刻可以改（不改基础设施）

这些改动在当前代码基础上直接做，不需要数据库或外部服务。

1. **让 `plan_fix` 读取 `conversation_history`**
   - 当前 `plan_fix` 只传了 `files_context` + `previous_failures` + `reflection_context`
   - 需要追加 `conversation_context`（最近 6 轮对话），让 LLM 知道之前的分析推理
   - 改动量：`plan_fix()` 中增加约 10 行代码

2. **reflection_notes 从字符串改为列表**
   - 当前 `state.reflection_notes: str` — 第二次 reflect 覆盖第一次
   - 改为 `state.reflection_notes: list[ReflectionNote]`，累积所有反思
   - `plan_fix` 读取所有反思（而不仅是最新的）
   - 改动量：新增 `ReflectionNote` 模型（3 个字段），修改 `reflect_on_failure()` 和 `plan_fix()`

3. **AgentState 拆分**
   - 当前 25+ 个字段全部平铺在一个 Pydantic model 里
   - 拆分为三层嵌套结构：
     ```python
     class AgentState(BaseModel):
         # 核心标识
         issue_url: str
         issue_number: int
         owner: str
         repo: str

         # Layer 0: 工作记忆（嵌入）
         working_memory: WorkingMemory

         # 状态机控制
         current_phase: Phase
         retry_count: int
         max_retries: int

         # Layer 1 接口（延迟加载）
         repo_memory_id: str | None  # 指向 SQLite

         # Layer 2 接口（延迟加载）
         active_strategies: list[str]  # 策略 ID 列表

         # 审计
         trace_id: str
     ```
   - 改动量：新建 `WorkingMemory` model，重构 AgentState，适配所有节点函数

4. **`conversation_history` 添加摘要压缩**
   - 当 conversation_turns 超过 15 轮时，对前 10 轮调用 LLM 做一次摘要
   - 摘要替换前 10 轮，释放 token 空间给后续推理
   - 改动量：`_remember()` 中增加触发逻辑（约 20 行）

#### P1 — 需要 SQLite 基础设施

5. **实现 RepoMemoryStore（Layer 1）**

   ```python
   class RepoMemoryStore:
       """Per-repo SQLite memory."""

       def __init__(self, base_path: Path = Path("~/.repopilot/memory/repos")):
           self.base_path = base_path.expanduser()

       def _db_path(self, owner: str, repo: str) -> Path:
           return self.base_path / f"{owner}___{repo}.db"

       async def get_or_create(self, owner: str, repo: str) -> RepoMemory:
           """加载 repo memory，不存在则创建空记录。"""

       async def update_file_index(self, owner: str, repo: str,
                                    files: list[RepoFileEntry]) -> None:
           """LOCATE 阶段后更新文件索引。"""

       async def record_issue(self, owner: str, repo: str,
                               record: IssueRecord) -> None:
           """修复完成后记录 issue 结果。"""

       async def query_similar_issues(self, owner: str, repo: str,
                                       issue_type: str, keywords: list[str],
                                       limit: int = 5) -> list[IssueRecord]:
           """PLAN 阶段查询相似 issue。"""

       async def get_test_knowledge(self, owner: str, repo: str) -> TestKnowledge | None:
           """EXECUTE 阶段获取测试配置。"""

       async def compact(self, owner: str, repo: str) -> None:
           """清理衰减条目，压缩数据库。"""
   ```

   关键实现细节：
   - 使用 aiosqlite 做异步访问
   - 每个 repo 一个 .db 文件，避免锁竞争
   - WAL 模式启用（允许并发读）
   - file_index 使用 FTS5 全文索引加速 topic 搜索

6. **实现 ReflectionStore（Layer 2）**

   ```python
   class ReflectionStore:
       """Global cross-repo reflection/strategy memory."""

       def __init__(self, db_path: Path = Path("~/.repopilot/memory/reflections.db")):
           self.db_path = db_path.expanduser()

       async def record_strategy_outcome(self, strategy_name: str,
                                          success: bool, repo: str,
                                          issue_type: str) -> None:
           """更新策略的 success/failure 计数。"""

       async def get_top_strategies(self, conditions: list[str],
                                     limit: int = 3) -> list[StrategyEntry]:
           """获取针对特定条件的最优策略。"""

       async def record_failure_pattern(self, error_log: str,
                                         root_cause: str,
                                         avoidance: str) -> None:
           """记录或更新失败模式。"""

       async def match_failure_pattern(self, error_log: str) -> list[FailurePattern]:
           """给定错误日志，匹配已知失败模式。"""
   ```

#### P2 — 需要向量数据库或更复杂的基础设施

7. **语义策略检索**（Layer 2 升级）
   - 当前设计用条件匹配（字符串 `applicable_conditions`）
   - 更好的方案：用 embedding 对 strategy 做语义索引
   - 在 PLAN 阶段，对当前 issue 描述做 embedding，检索 top-3 最相似的策略
   - 需要：ChromaDB / Qdrant / pgvector，以及 embedding API 调用成本

8. **Meta Memory 面板**（Layer 3 UI）
   - 一个简单的 Web 面板显示 usage_stats、token_efficiency、策略信心值
   - 用户可手动调高/调低特定策略的信心
   - 这不是核心功能，但对长期用户有价值

### 4.2 与 LangGraph 状态机的对接

当前 LangGraph 的核心循环：

```
UNDERSTAND → LOCATE → PLAN → EXECUTE → VERIFY → COMMIT → DONE
                                  ↑         ↓
                                  └─ retry ─┘ (失败→REFLECT→PLAN)
                                             FAILURE (issue comment)
```

各节点需要修改的地方：

| 节点 | 写入记忆 | 读取记忆 |
|------|---------|---------|
| `understand_issue` | L0: conversation_context | — |
| `locate_code` | L0: file_cache; L1: file_index (async) | L1: file_index (优先查已有索引) |
| `plan_fix` | L0: conversation_context, current_plan; L2: strategy (async) | L0: conversation_context, file_cache; L1: similar_issues; L2: top_strategies |
| `execute_fix` | L0: fix_attempts | L0: current_patch; L1: test_patterns |
| `verify_fix` | — | L0: fix_attempts |
| `reflect_on_failure` | L0: reflection_notes (list); L2: failure_pattern (async) | L2: known_failure_patterns |
| `commit_fix` | L1: issue_record (async); L2: strategy_outcome (async); L3: usage_stats (async) | L3: user_preferences |
| `handle_failure` | L1: issue_record (async); L2: strategy_outcome (async); L3: usage_stats (async) | — |

所有标记为 `(async)` 的写操作应在后台 fire-and-forget，不阻塞状态机流转。可以使用 `asyncio.create_task()` 或一个轻量的 background worker queue。

### 4.3 AgentState 最终结构提案

```python
class AgentState(BaseModel):
    # ── 身份标识 ──
    issue_url: str
    owner: str = ""
    repo: str = ""
    issue_number: int = 0

    # ── 状态机控制 ──
    current_phase: Phase = Phase.UNDERSTAND
    retry_count: int = 0
    max_retries: int = 3

    # ── Layer 0: 工作记忆（内嵌，跟随状态流转）──
    working_memory: WorkingMemory = Field(default_factory=WorkingMemory)

    # ── Layer 1-3: 持久化记忆的引用（不跟状态流转）──
    repo_memory_id: str | None = None  # 延迟加载
    active_strategy_ids: list[str] = Field(default_factory=list)

    # ── 输出字段（COMMIT/DONE 阶段填充）──
    pr_url: str | None = None
    branch_name: str = ""
    base_branch: str = "main"
    failure_reason: str = ""

    # ── 审计 ──
    trace_id: str = ""
```

`WorkingMemory` 内嵌在 AgentState 中随状态图流转，所以 LangGraph 的序列化机制（checkpointer）可以自动持久化 Layer 0。Layer 1-3 通过独立的 Store 类访问，不进入状态流转。

### 4.4 新增依赖

```
aiosqlite>=0.20.0      # 异步 SQLite（Layer 1, 2, 3）
chromadb>=0.5.0         # 可选，向量检索（Layer 2 升级）
redis>=5.0.0            # 可选，工作记忆备份
```

### 4.5 迁移路径

1. **Phase 1**（本周）：P0 的 4 项改动 — 修复 conversation_history 不读取的 bug，reflection 列表化，AgentState 拆分，摘要压缩。
2. **Phase 2**（2 周内）：实现 RepoMemoryStore（Layer 1 SQLite），集成到 locate_code 和 plan_fix。
3. **Phase 3**（1 个月内）：实现 ReflectionStore（Layer 2 SQLite），策略记录和检索。
4. **Phase 4**（长期）：向量检索升级 + Meta Memory 面板。

---

## 附录 A：与当前实现的对比

| 维度 | 当前 v2 (new_agent.py) | 本设计 |
|------|----------------------|--------|
| 跨 issue 记忆 | 无 | Layer 1: repo 级别知识 |
| 跨 session 记忆 | 无（进程内存） | Layer 2: 反思 + Layer 3: 元数据 |
| 反思累积 | 覆盖 | 列表累积 + 策略提取 |
| conversation 使用 | 写但不读 | 写且读，摘要压缩 |
| 失败模式学习 | 无 | Layer 2: failure_patterns |
| 策略优化 | 无 | Layer 2: 贝叶斯信心更新 |
| 并发安全 | 无 | SQLite WAL + per-repo 隔离 |
| 隐私控制 | 无 | per-repo 文件隔离 |

## 附录 B：关键设计决策记录

1. **为什么用 SQLite 而不是 PostgreSQL？** — RepoPilot 目前是一个单机 FastAPI 应用。SQLite 零运维、零配置，且 WAL 模式足以支持轻量并发。未来如果需要多实例部署，再迁移到 PostgreSQL。

2. **为什么 Layer 1 用 per-repo 文件而不是单一大表？** — 隔离性更好：删除一个 repo 的数据就是删一个文件。并发隔离：两个 repo 的写入不会互斥。备份简单。

3. **为什么 Layer 2 的策略命名交给 LLM 而不是预定义枚举？** — 预定义枚举无法覆盖所有修复策略类型。LLM 自由命名更灵活，代价是策略碎片化。通过 embedding 聚类可以在 Phase 4 解决碎片化。

4. **为什么 reflection 写操作全部异步？** — 主状态机的延迟对用户体验影响很大。reflection 写入是"优化未来"而非"完成当前任务"，不应阻塞当前 issue 的处理。

5. **为什么不引入向量数据库作为 Phase 1 的一部分？** — 向量数据库增加运维复杂度（持久化、备份、版本兼容）。Phase 1-3 的字符串匹配 + FTS5 已经覆盖 80% 的场景。Phase 4 的语义检索是锦上添花。
