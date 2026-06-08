# 数据采集脚本技术文档

> collect_dataset.py — GitHub Issue→Fix 数据集采集器
> 836 行 Python，836 行 + 77 行测试（4 tests）

## 架构总览

```
CLI 入口 → collect() 主循环 → 对每个仓库:
  ├─ 搜 Issue → 过滤 → 找 linked PR → 过滤 → 拿 diff → 写入 JSONL
  └─ 每步都写 SQLite state，中断可恢复
```

9 个模块，每层一个职责：

| 模块 | 行数 | 职责 |
|------|------|------|
| 配置常量 | 1-133 | 仓库列表、排除标签、正则模式 |
| 过滤管线 | 136-331 | 纯函数，层层筛选 Issue/PR 质量 |
| 数据构造 | 334-395 | 标准化输出格式 |
| StateStore | 398-467 | SQLite 断点续采 |
| RateLimiter | 470-481 | 异步限流器 |
| GitHubClient | 484-574 | HTTP 客户端 + 智能重试 |
| Progress | 577-607 | 终端进度条 |
| API 函数 | 610-703 | GitHub API 调用 |
| 主循环 | 714-808 | 采集编排 |

---

## 一、配置层

### DEFAULT_REPOS（50 个仓库）

三类各占约 1/3，依据 DATA_STRATEGY 的 "30% 明星 + 50% 中型 + 20% 小项目" 配比：

| 类型 | 仓库 | 特点 |
|------|------|------|
| Web 框架 | FastAPI, Django, Flask, Starlette, httpx, Tornado, aiohttp | Issue 质量高，PR 规范 |
| 数据科学 | numpy, pandas, scikit-learn, scipy, matplotlib, keras, tensorflow, transformers | 真实生产 bug |
| 工具/库 | poetry, black, ruff, mypy, rich, typer, click, jinja2, Pillow, boto3 | 中型项目，Issue→Fix 链路干净 |
| 平台 | home-assistant, airflow, prefect, superset, sentry-python | 大型项目，覆盖面广 |
| 其他 | cpython, sphinx, jupyter, ipython, langchain, openai-python, wagtail 等 | 多样性 |

支持 `--repo-list` 自定义列表（每行一个 `owner/repo` 或 JSON 数组）。

### CLOSING_REF_RE

匹配三种 Issue 引用格式（不需要额外 API 调用）：

```
fixes #123
fixes owner/repo#123
fixes https://github.com/owner/repo/issues/123
```

覆盖 fix/fixes/fixed/close/closes/closed/resolve/resolves/resolved 所有变体。

### BOT_HINTS

```
[bot], bot, dependabot, pre-commit-ci, github-actions, renovate, mergify
```

**不用 GitHub API 的 `type=Bot` 做唯一判断**——有些 bot 用个人 token 注册为 User 类型，但 login 名暴露身份。双重检查更稳。

---

## 二、过滤管线

设计原则：**每层一个门，通过或被挡掉。纯函数，独立可测。**

### Issue 过滤（should_keep_issue）

```
Issue 进来:
  ├─ has_bug_label()?         — 必须有 "bug" label
  ├─ has_excluded_labels()?   — 不能有 enhancement/feature/docs/dependencies/invalid/wontfix
  ├─ is_bot_actor()?          — 不能是 bot 创建的
  └─ is_meaningful_issue()?   — 标题 ≥4 字符，正文 ≥25 字符，≥5 个单词
```

### PR 过滤（should_keep_pr）

```
PR 进来:
  ├─ merged_at 非空?          — 必须是已合并的
  ├─ is_bot_actor()?          — 不能是 bot 创建的
  ├─ 正文非空?                — 空 PR body 说明自动化提交
  ├─ pr_body_references_issue()? — PR 正文必须引用目标 Issue
  ├─ is_probably_dependency_change()? — 排除依赖升级
  ├─ is_probably_formatting_only()?   — 排除纯格式化
  └─ should_keep_pr_files()?  — 文件数/行数/路径过滤
```

### 文件过滤（should_keep_pr_files）

- 文件数：**1-5 个**
- 变更行数：**5-300 行**（太小 = typo/import 微调，太大 = 重构）
- 排除纯重命名
- 排除 vendor/generated/lockfile

### 边界选择的理由

| 规则 | 下界 | 上界 | 理由 |
|------|------|------|------|
| 文件数 | 1 | 5 | >5 文件通常是重构 |
| 变更行 | 5 | 300 | <5 行太琐碎，>300 行通常是重构/新功能 |
| Issue 正文 | 25 字符 / 5 单词 | — | 过滤 "it doesn't work" 类无信息 Issue |

### 启发式过滤（is_probably_formatting_only）

看 PR 标题/正文是否含 format/black/ruff format/autopep8/isort/prettier，同时未改测试文件。

**用 "probably" 而非精确判断的原因**：完全精确需要解析 diff 内容看是否只改了空格/换行——成本太高。启发式过滤掉 90% 的格式化 PR，代价是可能误杀少量"修 bug 同时顺便格式化"的 PR——但这类修复本身就是噪音数据。

---

## 三、状态存储（StateStore）

### 为什么用 SQLite 而不是 JSON

采集 2000 条要扫 50 个仓库 × 300 个 Issue ≈ 15000 个 Issue。中断后重来 = 15000 次 API 调用白费。SQLite 做 checkpoint，`--resume` 秒级跳过。

### 表结构

```sql
processed_repos (repo TEXT PRIMARY KEY, completed_at TEXT)
processed_issues (repo, issue_number, status, pr_number, processed_at)
```

### 状态值颗粒度

不是 done/not_done 二元，而是 kept/skipped_issue/skipped_pr/error 四态。**数据治理需要知道"为什么被筛掉"**，方便后续分析过滤规则是否太严/太松。

### WAL 模式

采集长时间运行。WAL 允许读写并发——主循环写 JSONL 的同时可以查 SQLite 状态。比默认 rollback journal 更适合。

---

## 四、RateLimiter

### 限流策略

GitHub API 限流 5000 次/小时。0.3s 间隔 × 3600 次 ≈ 安全上限（< 5000）。

### 为什么用 asyncio.Lock

记录时间 → 计算等待 → 更新时间，三步必须原子。单协程当前不需要，但为并发预留。

---

## 五、GitHubClient

### 为什么用 aiohttp 而非 requests

采集是网络 IO 密集——大部分时间在等响应。同步模式下等 300ms 期间 CPU 空转。async 允许同一事件循环中切换。当前单协程，但架构支持未来开 3-5 个并发 worker。

### 三级重试策略

| 状态码 | 含义 | 策略 |
|--------|------|------|
| **429** | 被限流 | 等 `Retry-After` 头（无头则 min(60s, 2×attempt)） |
| **502/503/504** | GitHub 故障 | 指数退避（2s→4s→8s→最多 30s），最多 5 次 |
| **403** | 可能是 rate limit 打满 | 看 `X-RateLimit-Remaining`，0 则等 `X-RateLimit-Reset`。最多 3 次 |

**每次重试都有上限**：不无限重试。放弃→记录 error→跳到下一个 repo，比卡死强。

### paginate 智能终止

返回 < 100 条（per_page 大小）自动判断为最后一页，不空跑额外请求。

---

## 六、PR 查找策略（find_linked_prs）

### 双策略互补

**策略 1 — 关键词搜索**（search_prs_with_closing_keywords）：
```
搜 "#123" 在哪个 PR body 里出现
```
三种引用格式各搜一次。快，覆盖面广。

**策略 2 — Timeline API**（timeline_linked_prs）：
```
查 Issue 时间线，找 "cross-referenced" 事件
```
GitHub 自动记录 PR→Issue 引用，不管 PR body 有没有写 "fixes"。覆盖策略 1 的盲区。最多翻 2 页（200 条事件），老 Issue 时间线过长不追。

策略 1 先跑（快），策略 2 补漏。策略 2 找到的不重复记录。

---

## 七、主循环（collect）

### 异常策略

```
单个 Issue 失败（RuntimeError）    → warning + mark error + continue 下一个 Issue
整个 Repo 失败（RuntimeError）    → warning + continue 下一个 Repo
```

不因为一个 Issue 失败让整个采集中断。鲁棒性 > 完美性。

### flush 策略

`out.flush()` 每条写完后立即 flush。脚本被 kill 时，已 flush 的数据不丢。

### 输出格式

JSONL，每行一条完整 IssueFixExample。格式遵循 DATA_STRATEGY 定义的 schema：

```json
{
  "id": "owner/repo#issue:pr",
  "repo": {"owner", "name", "stars", "language"},
  "issue": {"number", "url", "title", "body", "labels", "created_at", "closed_at"},
  "pr": {"number", "url", "title", "body", "merged_at", "linked_by"},
  "patch": {"full_diff": "...", "files": [{"path", "status", "additions", "deletions", "patch"}]},
  "signals": {"has_tests_changed": bool, "fix_size_bucket": "small|medium|large"},
  "collected_at": "ISO8601"
}
```

---

## 八、当前局限

| 局限 | 原因 | 未来改进 |
|------|------|----------|
| 串行采集 | 先验证流程 | 加 `asyncio.Semaphore` 支持并发 worker |
| diff 不验证可 apply | 需 clone 仓库，成本高 | 加 `git apply --check` 作为质量信号 |
| 无增量更新 | 当前一次性扫完 | 加 `--since` 参数 |
| 无 quality_score 自动打分 | 需要更复杂的信号收集 | 加 CI status、review comments 分析 |

---

## 用法

```bash
# 试跑 50 条（不写文件，打印到 stdout）
python3 scripts/collect_dataset.py --output data/issues_fixes.jsonl --max-items 50 --dry-run

# 正式采集 2000 条
python3 scripts/collect_dataset.py --output data/issues_fixes.jsonl --max-items 2000

# 中断后续采
python3 scripts/collect_dataset.py --output data/issues_fixes.jsonl --max-items 2000 --resume

# 自定义仓库列表
python3 scripts/collect_dataset.py --output data/issues_fixes.jsonl --repo-list my_repos.txt
```

## 测试

```bash
pytest tests/test_collect_dataset.py -v
```

覆盖：引用解析、过滤逻辑、文件规则、记录构建（4 tests）。
