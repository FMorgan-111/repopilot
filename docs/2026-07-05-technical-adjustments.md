# RepoPilot 技术调整记录 — 2026-07-05

> 一天的工作：22 个提交（`9767743` → `da34188`）。
> 主题：**从"盲目加功能"转向"用可信评估定位真瓶颈"**，中途剥掉三层假象、修掉一串真 bug。

---

## 0. 这一天的核心转变（为什么这么做）

**之前**：靠直觉判断"该修什么"，做了一堆补丁质量功能（search 模糊匹配、节点锚定、幻觉门…），但每次上完都不知道有没有用——eval 成功率始终在 0–2/10 抖动，看不出瓶颈。

**问题**：`FAILED` 是个黑盒。同一个 `FAILED` 背后可能是"文件路径错""补丁装不上""补丁装上但测试挂""clone 坏了"，混在一起，任何优化都是碰运气。

**这一天做的根本改变**：**先建可信的评估与可观测性，再让数据告诉我该修什么**。结果是连续三次"我以为的头号是幻影"，逐层剥离后才露出真瓶颈。**这套方法本身，比任何单个功能都重要。**

---

## 1. 让 eval 可信：离线 seed + 失败分类（评估基础设施）

### 1.1 离线 gold-file seeding（`4c30f03`）
- **之前**：eval 靠 GitHub 代码搜索定位文件，命中限流就返回 0 候选，大量样本死在 locate，根本到不了补丁阶段。
- **改动**：`--seed-gold-files` 用数据集已知的 gold 改动文件（`patch.files[].path`），经 GitHub Contents API（单文件、可靠）拉内容，直接 seed `relevant_files` 并从 PLAN 起跑，跳过 UNDERSTAND/LOCATE。
- **解决了什么**：把 GitHub 搜索限流这个 confounder 移出关键路径，让 eval 稳定测**补丁阶段**而非搜索运气。

### 1.2 失败分类子系统 `failure_taxonomy`（`c532e67`）
- **之前**：只有笼统的 `failure_kind` / `FAILED`，无法回答"到底在哪一步失败"。
- **改动**：读 `agent_payload.fix_attempts`（数据本就有，无需新埋点），把 error log 映射到细类——`wrong_file_path / invalid_diff / empty_patch / search_not_found / test_failed / infra / budget`，给出决定性失败 + per-attempt 分布，并支持**跨 run 对比**（`python -m eval.failure_taxonomy A.json B.json`）。
- **解决了什么**：把黑盒 `FAILED` 变成可量化的失败画像，这是后面所有"逐层剥离"的地基。

### 1.3 逐 attempt 失败埋点 + trace 不覆盖（`cad18d7`）
- **之前**：trace 固定写 `case_1.json`，每跑必覆盖，无法回溯；召回等内部行为不可观测。
- **改动**：每个 fix attempt 打 `[classify] kind=… err=…`；trace 文件名改 `trace_<trace_id>.json` 不再覆盖。
- **解决了什么**：任何一轮 eval 的失败原因，事后可逐帧复盘（后面定位路由 bug 全靠它）。

---

## 2. 补丁质量守卫（承接前一天，本日巩固）

- **A/B 收敛 + 居中窗口 + 幻觉门**（`9767743`、`4b6546d`）
  - **之前**：PLAN 只看文件**头部** 6000 字符，修复点常在 imports 之下被截断 → 模型编造 search 块；且 search 不匹配时白烧 4 轮。
  - **改动**：文件内容窗口**居中于 issue 关键词命中行**（同预算但让修复点入镜）；execute 前**校验 search 块是否真实存在**，不存在则回灌真实行、有界纠正。
  - **解决了什么**：降低 search 幻觉的发生与空转。

- **节点锚定 `node_target`（`f4d9ed1`）+ 尺寸闸门转换器（`c29fcdd`）**
  - **之前**：只有 search/replace，模型逐字记忆一错就锚不上。
  - **改动**：新增 `node_target`——模型给函数**点分名**，系统用 AST 定位整节点 span 替换（不 `unparse`、保格式）；search 应用失败时，若 replace 是完整函数且尺寸 ≥ 真实函数 60%，自动升级为整节点替换（**尺寸闸门防止用几行 stub 截断整个函数**）。
  - **负结果（诚实记录）**：eval 中 `node_target` 采用率 ≈ 0——模型不主动用；prompt 层强制（`0c5054d`）也无效、还拖慢 tox，已**回退**（`6942881`）。转换器保留（无害、有安全网），但当前触发≈0。**这是"漂亮工程 ≠ 有效"的第一个教训。**

---

## 3. Infra 硬化：让"能力问题"不被基础设施噪声淹没

### 3.1 clone 异步化 + blobless 修复（`a3705e7`、`acd5325`）
- **之前**：`git_clone` 用同步 `subprocess.run` 阻塞事件循环，阶段超时杀不掉挂起的 clone → 整个 run 挂死；缓存用 `--filter=blob:none` 建成 blobless，`git clone --local` 取不到 blob → exit 128。
- **改动**：clone 走 `asyncio` 子进程（超时/取消都 kill）；去掉 blob filter；缓存 clone 失败自愈重下。

### 3.2 LLM 读超时 → 流式（`acd5325`、`66d7875`）
- **之前**：网关非流式，缓冲完整响应才发首字节，大 prompt 生成 >60s 撞读超时，且 60s < wall-clock 导致慢调用**翻倍重试**。
- **改动**：改流式 SSE，读超时变**per-chunk idle 超时**——持续产出的长生成不再被砍；探明网关支持 SSE（首字节 ~2s）。
- **解决了什么**：tox 这类大 prompt 不再因超时失败。

### 3.3 gpt-5.5 薄适配（`d2032b5`）+ 控制变量实验
- **背景**：用户换模型 gpt-5.5，想测"是能力问题还是 harness 问题"。
- **实验设计**：同样本换模型，resolved 跳升=能力问题、不动=harness 问题；并识别出**超时、输出格式**两个 confounder 先隔离。
- **改动**：idle 超时 60→120（推理模型思考停顿）、wall-clock 240→300、phase 超时同步抬；PLAN prompt 从"prefer patch_edits"改**强制 patch_edits、禁 unified diff**。
- **结论**：薄 profile 就够用，**不需重做整套 harness**；但 gpt-5.5 在本 harness 上慢 10–20×、且不完全服从引导，当前不如 flash。切回 flash（`gemini-3.5-flash`）。

---

## 4. 长期记忆的对照实验（"漂亮工程 ≠ 有效"的实证）

- **之前**：跨 repo 语义记忆（embedding + sqlite-vec 向量检索 + 失败案例避坑）代码扎实，但**默认关闭、从未被任何数据验证**。文档还有 3 处与代码漂移（db 路径、表名、embedding 库）。
- **改动**：加 `[recall] N episode(s) injected` 可观测日志（`90fb381`），**确证召回真的在工作**（每次 plan 注入 2–3 条）；做开/关对照实验。
- **结论（诚实的负结果）**：开记忆后成功率 **1/10 → 0/10**、失败模式没变——召回了正确历史案例，模型照样犯同样的错。样本量小不能定论"有害"，但能确证：**它没带来可见收益，且此前从没被验证过**。

---

## 5. 四层剥离：找到真正的头号瓶颈（这一天的高潮）

用 `failure_taxonomy` 逐轮重跑基线，头号失败一路被**幻影**掩盖，靠逐帧取证剥开：

| 轮次 | 当时"头号" | 真相 | 修的 bug |
|---|---|---|---|
| v1 | `invalid_diff` 5/10 | **路由 bug 幻影** | 幻觉门/死补丁门清空补丁 + 改 `current_phase`，但**路由器读 `frame.recommended_action`（还是"execute"）** → 空补丁漏到 EXECUTE → 空 diff `git apply` → "No valid patches"。修：门改 phase 时同步改 recommended_action（`567a181`） |
| v2 | `wrong_file_path` 5/10 | **空 clone 幻影** | textual/ansible 工作树是 **0 文件/无 HEAD 的坏 clone**，`git_clone` 只判 `.git` 存在就复用、不验 checkout → 任何 apply 都 file-not-found。**模型其实第一轮全选对了 gold 文件**。修：`_worktree_is_healthy` 守卫复用，坏则删重建（`e8d855e`） |
| **v3** | **真相** | **`search_not_found` 5/10 头号** | 模型**记错代码细节**（如 scrapy `_ScrapyAgent.__init__` 真实签名有 `*,`、无默认值，模型写反）。这就是最早怀疑的 search 内容幻觉，之前被两层幻影压成 1/10 |

配套：`failure_taxonomy` 自己也修了 3 次（invalid_diff 误判→新增 `empty_patch`；门清空补丁的空 `fix_attempts` 从 `other` 归正 `search_not_found`）（`567a181`、`da34188`）。**评估工具本身也会撒谎，修了它三次才可信。**

### 最终真实分布（`eval/baseline_v3_10.json`，flash，10 样本）
```
resolved:          1/10
search_not_found:  5   ← 头号：模型 search 内容幻觉（记错代码）
test_failed:       3   ← 第二：补丁装上但测试挂（修复逻辑错）
infra:             1
```

**两个头号都是模型能力问题，不是机械问题。** 这个结论不是猜的，是剥掉三层假象后数据确证的。

---

## 6. 一句话总结这一天

> 没有加多少"能力"，主要做了一件事：**把评估变得可信**，然后让它连续否定我的三个假设，最终指向真瓶颈——模型在 search 块里记错代码（5/10）。**这一天最大的产出不是某个功能，而是"我现在能证明该修什么"。**

---

## 附：可复现命令
```bash
# 跑基线（记忆关，seeded）
.venv/bin/python eval/harness.py --agent-v2 --samples 10 --max-retries 2 \
  --token-budget 100000 --seed-gold-files

# 看失败分布
python -m eval.failure_taxonomy

# 跨 run 对比（如 记忆开 vs 关）
python -m eval.failure_taxonomy eval/baseline_v3_10.json eval/mem_runB_on.json
```
