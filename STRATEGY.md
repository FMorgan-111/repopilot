# STRATEGY.md — RepoPilot 开源增长战略


## 1. 竞品分析与差异化定位

### 竞品地图

| 产品 | 核心定位 | 局限 |
|------|----------|------|
| **Sweep** | AI 自动写 PR，处理简单 Issue | 需要 GitHub App 安装，黑盒操作，复杂 Issue 失败率高 |
| **Cody (Sourcegraph)** | 代码搜索 + 问答，企业级 | 回答问题，不生成修复；需要 Sourcegraph 基础设施 |
| **Devin** | 全自动软件工程师 | $500/月，闭源，慢，不透明 |
| **Cursor** | IDE 内 AI 辅助编辑 | 本地 IDE，非 Agent，不处理 Issue 工作流 |
| **Aider** | 终端 LLM 编程助手 | 需要人工描述任务，不理解 Issue 上下文 |
| **Copilot** | 行级代码补全 | 补全工具，不具备代码库理解和任务规划能力 |

### RepoPilot 的独特定位

**"给 Issue URL，拿修复方案"** — 这是其他工具没有做好的精确场景。

差异化三角：

```
透明可控（你看到每一步推理）
        △
        │
可审查 ←┼→ 轻量部署
（输出 diff，不直接 merge） （pip install，无需 IDE 插件）
```

- **对比 Sweep**：RepoPilot 输出可审查的推理链 + diff，不黑盒自动 merge
- **对比 Devin**：开源、可本地运行、成本低 100 倍
- **对比 Aider**：无需人工描述，直接消化 Issue 的自然语言上下文
- **核心卡点**：LangGraph 驱动的多步搜索推理，可观测、可 debug、可扩展


## 2. 产品定义

### 目标用户（优先级排序）

1. **独立开发者 / 小团队**：积压 Issue 多，无专职 QA，时间是瓶颈
2. **开源维护者**：处理贡献者报 bug，快速 triage 和定位
3. **安全研究员 / CTF 选手**：快速定位漏洞相关代码路径
4. **学习者**：看 AI 怎么阅读陌生代码库，作为学习工具

### 核心场景

```
$ repopilot https://github.com/org/repo/issues/42

🔍 读取 Issue #42: "NullPointerException in PaymentService.process()"
📂 搜索代码库: PaymentService → process() → 调用链
📖 读取相关文件: payment/service.py:L134, utils/validator.py:L89
🧠 分析根因: validator 未处理 None 输入，service 未做 guard
💡 修复方案:
   - payment/service.py:134 — 添加 None check
   - utils/validator.py:89  — 验证输入非空
📋 生成 diff（可直接 git apply）
```

**30 秒 Demo 亮点**：
录制一个 gif：左边 GitHub Issue（真实知名项目的 bug），右边终端实时输出搜索路径 → 文件读取 → 推理 → diff。整个过程 30 秒内完成。选择有名的开源项目（如 FastAPI、Django 的历史 Issue）作为演示，可信度立刻拉满。


## 3. 技术壁垒

### 短期壁垒（6个月内可构建）

**1. Issue 理解管道**
- Issue 文本 → 结构化故障信息（错误类型、复现路径、影响范围）
- 比简单 RAG 多一层语义解析，减少无关文件的检索噪音

**2. 分层代码搜索策略**
```
符号搜索（AST）→ 语义搜索（embedding）→ 依赖图遍历
```
三层混合，比单纯向量搜索精确，比纯关键词搜索智能

**3. LangGraph 可观测推理链**
- 每一步搜索决策可记录、可回放、可评估
- 竞品是黑盒，RepoPilot 可以输出 `trace.json` 让用户看到 AI 为什么这么搜
- 这是社区信任的基础，也是研究者引用的理由

**4. 修复质量评估器**
- 自动验证生成的 diff 是否能通过现有测试
- "diff 可运行" 是比 "diff 看起来对" 更强的护城河

### 长期壁垒（需要数据积累）

- **Issue → Fix 配对数据集**：处理真实 Issue 积累的训练数据，形成飞轮
- **跨语言支持**：Python/JS/Go/Java 的代码搜索策略差异，需要大量调试


## 4. 开源运营

### README 结构（黄金标准）

```markdown
# RepoPilot

[一句话定位] GitHub Issue URL → 代码分析 → 修复方案，30 秒内

[Demo GIF — 必须是第一屏可见]

## 安装（3行以内）
pip install repopilot
export ANTHROPIC_API_KEY=***
repopilot https://github.com/org/repo/issues/42

## 工作原理（流程图，不超过5步）
## 支持的语言
## 与 Sweep/Aider 的区别（表格，诚实）
## 贡献指南
```

**关键原则**：
- Demo GIF 必须是真实项目的真实 Issue，不能是 hello world
- 安装步骤 ≤ 3 行，否则 50% 用户放弃
- 对比表格要诚实写出自己的不足，建立信任

### 发布时机

**不要等到"完美"再发布**，但要满足：
- Demo GIF 完成，能稳定跑通 3 个不同真实项目
- README 完整，包含对比表格
- 基础错误处理（API key 缺失、网络失败、无权限仓库给友好提示）
- CONTRIBUTING.md 存在

**最佳发布时机**：周二或周三上午（HN/Reddit 活跃时段，UTC）

### 传播策略

**第一波（发布当天）**：
- Hacker News `Show HN: RepoPilot – GitHub Issue URL → fix in 30s`
- Reddit: r/Python, r/MachineLearning, r/programming
- Twitter/X：找 3-5 个 AI coding 领域 KOL 私信，不是求转发，是求真实使用反馈

**第二波（发布后 2 周）**：
- 用 RepoPilot 处理知名开源项目（FastAPI、Pydantic、Requests）的历史 Issue，发博客文章
- 标题公式：`"I let AI fix a FastAPI bug — here's what it found in 30 seconds"`
- 提交到 dev.to、Medium、掘金（中文社区）

**持续传播**：
- 每月发一篇"RepoPilot 能做到 / 还不能做到"的诚实评测
- 诚实内容比营销内容传播更广

### 社区信任建立

- Issues 24 小时内回复（哪怕只是 "已确认，在排期"）
- ROADMAP.md 公开，让社区看到方向
- 明确标注 `good first issue`，降低贡献门槛
- 失败案例公开讨论，不删不藏


## 5. 增长路径

### 0 → 100 star（第 1-2 周）

**目标**：验证核心价值主张

- 核心动作：HN/Reddit 发布帖，Demo GIF 质量是关键
- 成功信号：有人在 Issues 里报 bug（说明真的在用）
- 里程碑功能：稳定支持 Python 仓库，输出可 apply 的 diff

### 100 → 1000 star（第 1-3 个月）

**目标**：建立 SEO 基础 + 口碑传播

- 发布 3 篇技术博客（真实案例 + 架构解析）
- 支持 JavaScript/TypeScript 仓库（覆盖最大用户群）
- 添加 GitHub Action 集成（`on: issues` 自动触发分析）
- **关键转折点**：GitHub Action 发布后，每个使用它的仓库都是自然传播
- 找 5 个中等规模开源项目（500-5000 star）的维护者合作，作为早期用户

### 1000 → 5000 star（第 3-12 个月）

**目标**：成为细分场景的默认工具

- **技术层面**：
  - 支持私有仓库（企业需求入口）
  - 本地模型支持（Ollama）—— 解锁隐私敏感用户
  - VS Code 插件（降低使用门槛）

- **内容层面**：
  - 建立 `awesome-repopilot` 案例库（社区贡献真实使用案例）
  - 在 AI coding 类 YouTube/播客上争取报道

- **社区层面**：
  - Discord/微信群（中文社区单独维护）
  - Hacktoberfest 参与，批量引入贡献者
  - 找 1-2 个核心贡献者共同维护

**5000 star 的判断标准**：
> 当有人在 Twitter 发 "遇到 bug 第一步先跑 repopilot" 时，说明心智占领成功了。


## 6. 商业化模式

### 原则：开源先行，商业化是结果不是目的

过早商业化会杀死社区信任。先做到 2000+ star，再考虑。

### 推荐路径：开源核心 + 托管服务

```
开源（MIT）                    商业（SaaS）
────────────────               ──────────────────────
CLI 工具                  →    Web Dashboard
本地运行                  →    托管 + 无需配置
单仓库                    →    多仓库 + 团队协作
手动触发                  →    Issue 自动触发 + Slack 通知
社区支持                  →    优先支持 + SLA
```

**定价参考**：
- 个人版：免费（使用自己的 API Key）
- 团队版：$29/月（托管 + 5 用户 + 私有仓库）
- 企业版：$199/月（本地部署 + 审计日志 + 自定义模型）

### 不推荐

- 双授权（GPL + 商业）：太复杂，适合基础设施工具不适合 AI 工具
- 纯咨询：无法规模化


## 执行优先级（接下来 4 周）

```
Week 1: Demo GIF + README 打磨 → 发布到 HN
Week 2: 处理早期反馈，修 10 个 Issue
Week 3: 发第一篇技术博客（真实案例）
Week 4: GitHub Action 集成 → 第二波传播
```

> **核心判断**：RepoPilot 的护城河不是模型能力（模型是商品），而是 **Issue 理解的精准度** + **搜索策略的可观测性** + **社区对这个工具的习惯依赖**。前两个靠工程，第三个靠时间和持续运营。
