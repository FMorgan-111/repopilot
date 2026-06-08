# ARCHITECTURE_V2.md — RepoPilot 工业级架构设计

## 1. 架构选型决策

### 选型：单主图 + 两个子图（LangGraph）

不用 supervisor-worker（单 Issue 不需要多 Agent 协调），不用纯线性管道（typo 和架构级 bug 不能同一条路径）。

**结论：单主图控制路由 + 搜索子图 + 修复子图。** 子图各自封装，可独立测试升级。


## 2. 完整图结构

### 主图

```
[issue_url]
     │
     ▼
┌────────────┐
│ issue_fetch│  纯代码：GitHub API，提取 title/body/labels/stack_trace
└─────┬──────┘
      │
      ▼
┌────────────┐
│   triage   │  LLM flash：分类 + 复杂度评估
└─────┬──────┘
      │
      ├── complexity=trivial ──────────────────────────────┐
      │                                                    │
      ▼                                                    ▼
┌────────────┐                                   ┌────────────────┐
│  SEARCH    │                                   │  fast_path_fix │
│  SUBGRAPH  │                                   └───────┬────────┘
└─────┬──────┘                                          │
      │                                                 │
      ▼                                                 │
┌────────────┐                                          │
│  context   │  纯代码：合并结果，读文件，裁剪 token    │
│  assembly  │                                          │
└─────┬──────┘                                          │
      │                                                 │
      ▼                                                 │
┌────────────┐                                          │
│  planner   │  LLM pro：定位根因，确定修改位置         │
└─────┬──────┘                                          │
      │                                                 │
      ▼                                                 │
┌────────────┐                                          │
│    FIX     │                                          │
│  SUBGRAPH  │                                          │
└─────┬──────┘                                          │
      └──────────────────┬──────────────────────────────┘
                         │
                         ▼
                  ┌────────────┐
                  │ summarizer │  LLM flash：格式化输出
                  └─────┬──────┘
                        │
                   [AgentResult]
```

### 搜索子图

```
[issue 信息]
      │
      ▼
┌─────────────┐
│query_builder│  LLM flash：提取 symbols/keywords/file_patterns
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────────────┐
│           三层并行搜索（同时触发）             │
│                                              │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
│  │ layer_1  │  │ layer_2  │  │ layer_3   │  │
│  │ 符号搜索 │  │ 语义搜索 │  │ 依赖图    │  │
│  │(AST/grep)│  │(embedding│  │(import    │  │
│  │          │  │)         │  │ graph)    │  │
│  └────┬─────┘  └────┬─────┘  └─────┬─────┘  │
└───────┼─────────────┼──────────────┼─────────┘
        └─────────────┴──────────────┘
                      │
                      ▼
             ┌────────────────┐
             │ result_merger  │  纯代码：去重、合并、初步评分
             └───────┬────────┘
                     │
                     ▼
             ┌────────────────┐
             │relevance_ranker│  LLM flash：排名 + 决定是否再搜
             └───────┬────────┘
                     │
           ┌─────────┴─────────┐
           │  需要继续搜索？    │  （最多 2 轮）
           ▼                [继续搜索]
        [输出: 排名文件列表]
```

### 修复子图

```
[planner 输出 + 文件内容]
           │
           ▼
  ┌────────────────┐
  │ diff_generator │  LLM pro：生成 unified diff（逐文件）
  └───────┬────────┘
          │
          ▼
  ┌────────────────┐
  │  quality_eval  │  级联验证：语法 → 静态分析 → LLM 审查（可选）
  └───────┬────────┘
          │
          ├── pass ──────── [输出: diff + 报告]
          ├── warn ──────── [输出: diff + 警告]
          └── fail ──── reflector（LLM pro）→ diff_generator（最多 2 次）
```


## 3. 模型策略

### 节点模型分配

| 节点 | 模型 | 理由 |
|------|------|------|
| `triage` | flash | 分类规则明确 |
| `query_builder` | flash | 关键词提取，无需深度推理 |
| `relevance_ranker` | flash | 排序任务 |
| `summarizer` | flash | 格式化输出 |
| `planner` | **pro** | 理解代码逻辑、定位根因 |
| `diff_generator` | **pro** | 生成语法正确代码，错了就失败 |
| `reflector` | **pro** | 修复失败后的深度反思 |

**成本**：flash ~12K tokens × $0.0014/1K ≈ $0.017；pro ~16K tokens × $0.0027/1K ≈ $0.043。单次约 **$0.06**。

### 微调结论

**当前不微调。** 等积累 500+ (issue, diff) 数据对后，优先微调 `diff_generator`——让 flash 学会生成可 apply 的 diff，降低 80% 成本。

### DeepSeek JSON 处理升级

两处改进：

**1. System prompt 后缀：**
```
CRITICAL: Respond with a single valid JSON object only.
No text before or after. No markdown fences.
Start with { and end with }.
```

**2. 区分错误类型的重试：**
- `json.JSONDecodeError` → 告诉模型 JSON 格式不对，重试
- `pydantic.ValidationError` → 告诉模型哪个字段不合规，重试

### 本地模型（Ollama）预留

环境变量 `LLM_BACKEND=ollama` 切换，节点代码零修改：

```python
def _config(tier="pro"):
    if os.getenv("LLM_BACKEND") == "ollama":
        models = {"pro": "qwen2.5-coder:32b", "flash": "qwen2.5-coder:7b"}
        return "", "http://localhost:11434/v1", models[tier]
    # deepseek 逻辑不变
```


## 4. 三层搜索策略

| 层 | 方法 | 适用 | 实现 |
|----|------|------|------|
| L1 符号 | GitHub Search API `repo:owner/repo symbol in:file` | 已知类名/函数名 | 精确，仅找已知符号 |
| L2 语义 | sentence-transformers 本地 embedding | 模糊关键词 | Issue embedding vs 文件摘要 cosine 相似度 |
| L3 依赖图 | 解析 Python import 构建有向图 | 复杂跨文件 | 从 L1/L2 结果各扩展 1 跳 |

Agent 决策而非全部自动触发：
- L1 结果 ≥3 个 → 跳过 L2（符号搜索已足够）
- complexity = "trivial" → 跳过 L3（不需要依赖图）


## 5. 修复质量评估器

三级级联：

**Level 1 — 语法验证（纯代码，零成本）**
- diff 格式合法（python-patch 能解析）
- apply 后代码通过 `ast.parse()`

**Level 2 — 静态分析（纯代码，低成本）**
- 对 patched 代码运行 `ruff check`

**Level 3 — LLM 语义审查（flash，仅 `risk_level=="high"`）**
- 问 LLM：这个 diff 逻辑上是否解决了 Issue？

测试运行（clone → apply → pytest）放 v2，MVP 不包含。


## 6. 流式输出设计

用户终端看到：

```
🔍 读取 Issue #42...
   ✓ "NullPointerException in PaymentService.process()"

🏷️  分类: bug · high · moderate complexity

🔎 搜索代码库...
   [符号] PaymentService, process() → payment/service.py
   [语义] utils/validator.py (0.89)
   [依赖] payment/service.py → utils/validator.py ✓

📖 读取文件: payment/service.py, utils/validator.py

🧠 分析根因...
   validator.validate_amount() 未处理 None，service.process() 未做 guard

💡 修复方案 (risk: low)...
   • payment/service.py:134 — 添加 None guard
   • utils/validator.py:89  — 接受 Optional[float]

✅ 质量检查: 语法 ✓  静态分析 ✓

📋 Diff: [unified diff]

⏱  8.3s  |  trace: .repopilot/traces/issue-42.json
```

使用 `graph.astream_events()` 监听节点事件，不流式打印 LLM token（信噪比太低）。


## 7. 文件结构

```
repopilot/
├── src/
│   ├── graph/
│   │   ├── master.py              # 主图：节点连接 + 条件路由
│   │   ├── search_subgraph.py
│   │   └── fix_subgraph.py
│   │
│   ├── nodes/                     # 每个节点一个文件
│   │   ├── issue_fetch.py
│   │   ├── triage.py
│   │   ├── query_builder.py
│   │   ├── context_assembly.py
│   │   ├── planner.py
│   │   ├── diff_generator.py
│   │   ├── quality_eval.py
│   │   ├── reflector.py
│   │   └── summarizer.py
│   │
│   ├── tools/
│   │   ├── github.py              # 现有 tools.py
│   │   ├── search_symbol.py       # Layer 1
│   │   ├── search_semantic.py     # Layer 2
│   │   └── search_deps.py         # Layer 3
│   │
│   ├── state.py                   # LangGraph State
│   ├── llm.py                     # 升级版（pro/flash + retry）
│   ├── schemas.py                 # 扩展 Pydantic
│   ├── tracer.py
│   └── main.py
│
└── tests/
```


## 8. 实现优先级

### MVP（Week 1-2）— 发布 HN 前

- issue_fetch + triage（含快速路径）
- query_builder + Layer 1 符号搜索
- context_assembly + planner + diff_generator
- Level 1 质量验证（语法）
- 基础流式输出（print）

**不包含**：LangGraph 图（先纯函数）、Layer 2/3、reflector、astream_events

### v1.0（Week 2-4）

- 迁移到 LangGraph 主图 + 搜索子图
- Layer 2 语义搜索
- Level 2 静态分析（ruff）
- reflector 节点
- astream_events 流式输出

### v1.5（Week 4-8）

- Layer 3 依赖图遍历
- GitHub Action 集成
- 支持 JS/TS
- Ollama 本地模型


## 9. 关键决策总结

| 决策 | 选择 | 理由 |
|------|------|------|
| 图框架 | LangGraph | 可观测推理链是核心差异化 |
| 图结构 | 单主图 + 2 子图 | 避免过度工程，保留扩展性 |
| MVP 实现 | 先函数调用，后迁移 | 不为框架延误发布 |
| 模型分配 | pro 仅 planner/diff/reflector | 质量和成本平衡 |
| 微调 | 暂缓等 500+ 数据 | 现在投入回报率低 |
| JSON 处理 | prompt suffix + 区分错误重试 | 不依赖 API 能力 |
| 搜索策略 | 三层并行 + agent 决策深度 | 比单层精确，比全跑高效 |
| 质量验证 | 级联（语法→静态→LLM） | 轻量优先，重量按需 |
| 流式输出 | 节点粒度，非 token 粒度 | 信噪比更高 |
