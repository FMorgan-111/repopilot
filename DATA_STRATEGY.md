# DATA_STRATEGY.md — RepoPilot 数据战略

> 核心原则：不是尽快堆量，而是做出**可复现、可评测、可持续更新、质量可解释**的 Issue→Fix 数据资产。

## 一、仓库选择

### 三层采集（Python only 起步）

| 类型 | 占比 | 目标 |
|------|------|------|
| 明星项目 10k+ stars | 30% | 高质量、真实生产问题 |
| 中型项目 500-10k stars | 50% | PR 更直接，链路更干净 |
| 小项目 50-500 stars | 20% | 简单 bug，适合学局部 patch |

**不要只采明星项目**。大项目 PR 巨大、讨论长、重构多。中型项目反而是训练主力。

多语言路线：V0 Python only → V1 Python+TS/JS+Go → V2 Rust/Java/C# → V3 跨语言 benchmark。

## 二、数据格式

每条不是简单 `{issue, diff}`，而是完整 `IssueFixExample`：

```json
{
  "id": "owner/repo#issue:pr",
  "repo": { "owner", "name", "stars", "language", "license" },
  "issue": { "number", "title", "body", "labels", "comments" },
  "pr": { "number", "title", "body", "merged_at", "linked_by" },
  "patch": {
    "full_diff": "...",
    "files": [{ "path", "status", "additions", "deletions", "patch" }]
  },
  "signals": { "ci_passed", "has_tests_changed", "fix_size_bucket", "link_confidence" },
  "splits": { "train_valid_test_group": "repo_level" }
}
```

- 同时保留 full_diff 和 files[]
- 分桶：small ≤100 lines, medium ≤500, large >500
- large 不进入核心训练集

## 三、数据清洗

### 高置信保留

- PR body 含 `fixes #123` / `closes #123` / `resolves #123`
- PR merged，不是 closed unmerged
- Issue 关闭时间接近 PR merge 时间
- 仅关联 1-2 个 Issue

### 排除

- enhancement / feature / question / discussion
- 文档 / 拼写 / 依赖升级 / 格式化 / lint-only
- 无正文 / 无复现的低信息 Issue
- bot 创建或 bot 修复
- PR 未合并 / reverted / hotfix
- generated files / lockfile-only / vendor

### 质量打分（gold / silver / bronze）

```
quality_score =
  link_confidence
  + issue_information_score
  + patch_locality_score
  + ci_signal_score
  + test_signal_score
  - noise_penalty
  - generated_file_penalty
```

- **gold**：高置信 + 小中型 patch + 测试/CI + 信息充分
- **silver**：链接明确但测试/CI 不足
- **bronze**：预训练/检索训练用，不进 benchmark

## 四、发布策略

双仓库 + HuggingFace：

- `repopilot/repopilot`：主项目，数据入口、示例、benchmark 说明
- `repopilot/issuefix-dataset`：独立数据集仓库（schema、脚本、规则）
- HuggingFace Datasets：`load_dataset()` 可直接加载

必须写 dataset card（来源、标准、偏差、license、引用格式）。

论文路线：V0 dataset card + leaderboard → V1 arXiv tech report → V2 学术 venue。

## 五、护城河

数据本身不是壁垒（任何人都能爬 GitHub）。护城河是：

1. **高质量链接判定** — 谁更准判断 Issue 和 PR 因果关系
2. **质量标签体系** — gold/silver/bronze、修复类型、难度
3. **持续更新管线** — 每月版本化，一次性 dataset 很快贬值
4. **评测闭环** — RepoPilot 自己用数据评测 → 用户反馈回补
5. **社区贡献机制** — 维护者可提交 verified pairs

采集脚本开源 80%（schema、爬虫、清洗、评测），保留内部质量打分权重。

## 六、落地路线

| 阶段 | 时间 | 目标 |
|------|------|------|
| Python Gold v0 | 第1月 | 采 200-500 仓库，筛 1k-3k gold，发布到 HuggingFace |
| Benchmark 化 | 第2月 | 人工审 500-1000 条，发布 3 个评测任务 + leaderboard |
| 社区化 | 第3月 | 独立数据集仓库，开放贡献，dataset card + tech report |
