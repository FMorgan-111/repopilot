# RepoPilot 技术设计与面试深挖文档

> 一个把 GitHub Issue 自动修成 PR 的 AI agent。LangGraph 状态机驱动,DeepSeek 推理。
> 本文从**基础设施层**到**推理层**详述设计,列出**技术难点**与**决策点**,并附**面试官追问 + 答案**。
>
> 现状(诚实):已在真实开源 bug `scrapy/scrapy#6195`(robots.txt UTF-8 BOM)上端到端跑通——
> 自动定位代码、生成补丁、建隔离环境、跑 pytest 验证通过(15 passed)。run `68285b8d86a2`,
> `final_phase=DONE`,19 turns / 11.8K token。目前 1/1,**不是统计成功率**。

---

## 目录
1. 项目总览与架构
2. 基础设施层设计
3. 执行层设计(clone / venv / install / patch / test)
4. 检索层设计(locate / BM25 / 上下文累积)
5. 推理层设计(plan / reflect / 决策帧 / 循环控制)
6. 技术难点全链(11 个根因诊断)
7. 关键技术决策点(含权衡)
8. 面试官可能追问 + 答案

---

## 1. 项目总览与架构

### 1.1 它做什么
输入一个 GitHub Issue URL,输出一个修复 PR(或在 eval 模式下产出经测试验证的补丁)。六阶段状态机:

```
UNDERSTAND -> LOCATE -> PLAN -> EXECUTE -> VERIFY -> COMMIT -> DONE
                ^                            |
                +--------- REFLECT <---------+  (测试失败时重试)
```

| 阶段 | 节点 | 职责 |
|---|---|---|
| UNDERSTAND | understand_issue | 拉取 issue、分类(bug/feature/security)、抽取严重度 |
| LOCATE | locate_code | GitHub 代码搜索 + 按相关性排序 + 读取候选文件 |
| PLAN | plan_fix | LLM 基于 issue + 文件内容生成 patch_edits + 决策帧 |
| EXECUTE | execute_fix | clone -> 建 venv -> 装包 -> apply patch -> 跑 pytest |
| VERIFY | verify_fix | 解析测试结果,路由到 COMMIT / REFLECT / FAILED |
| REFLECT | reflect_on_failure | LLM 分析失败根因,反馈给 PLAN |
| COMMIT | commit_fix | 推送变更、开 Draft PR(eval 模式跳过) |

### 1.2 核心差异化
- **决策可观测**:每阶段产出结构化 `DecisionFrame`(假设/证据/下一步/推荐动作),路由是纯函数 `route_from_state` 读帧决定走向,**不藏在 prompt 里**。对比 Devin/Sweep 的黑盒。
- **全程 Pydantic 状态**:整个 run 是一个 `AgentState`,可序列化、可 replay、可调试。
- **双图引擎**:装了 LangGraph 用 LangGraph,没装则回退到 30 行的 `FallbackCompiledGraph`,满足同一 `graph.ainvoke(state)` 契约——离线/CI 测试零额外依赖。

---

## 2. 基础设施层设计

### 2.1 状态模型(src/state.py)
`AgentState` 是单一 Pydantic 模型,承载整个 run。关键字段:
- `current_phase: Phase`(枚举驱动路由)
- `relevant_files: list[FileInfo]`(带 relevance_score、content、sha)
- `fix_attempts: list[FixAttempt]`(每次补丁 + 测试结果)
- `decision_frame` / `frame_history`(决策帧链)
- `token_usage` / `token_budget`(预算控制)
- `context_collection_count` / `last_locate_signature`(循环控制,后述)
- `skip_commit: bool`(eval 模式:验证通过即终态,不开 PR)

**设计细节**:`PatchEdit` 用 `@model_validator(mode="before")` 容忍 LLM 输出的 `file`/`path`/`file_path` 三种 key 别名。`Hypothesis.score` 把 LLM 误输出的 (1,10] 区间归一化到 0..1(LLM 常输出 9/9.0 而非 0.9)。这些是**和不确定的 LLM 输出打交道的边界防御**。

### 2.2 决策帧(DecisionFrame)
```
stage: diagnose | plan | reflect
summary, hypotheses[], selected_hypothesis_id, evidence[], next_checks[]
recommended_action: collect_more_context | plan | execute | reflect | stop | ask_user
confidence, risk, parent_frame_id, trace_notes
```
**为什么重要**:`recommended_action` 是 LLM 表达"我下一步想干什么"的结构化出口。路由层 `_RECOMMENDED_PHASES` 把它映射到 Phase。每个帧记 `parent_frame_id`,形成可追溯的推理链。

### 2.3 路由(纯函数 + 一次性消费)
`route_from_state` 的关键设计是**决策帧只消费一次**:用 `decision_route_checked_frame_id` 标记已消费的帧,避免同一帧被路由两次造成死循环。有完整的 fallback 语义(no_decision_frame / no_frame_id / stale_frame / already_consumed / unsupported_action),每个都记进 `route_decisions` 便于诊断。

### 2.4 HTTP 层与重试(src/http_client.py)
两套独立的重试策略:
- **GitHub**:429/502/503/504 + 网络错误,指数退避 1->2->4s,最多 3 重试,带令牌桶限流。
- **LLM**:502/503/504 + 网络错误,最多 1 重试(2 次尝试),退避上限 20s。

**关键设计(见难点 6.4)**:
- `LLM_REQUEST_TIMEOUT=60`:httpx 的 per-socket-operation 超时(连接/字节间隔),**不限制请求总时长**。
- `LLM_CALL_WALLCLOCK_TIMEOUT=200`:用 `asyncio.wait_for` 包裹的真·墙钟硬上限。超时抛 `asyncio.TimeoutError`,**故意不在重试白名单里**——慢调用快速失败不翻倍。
- `llm_retry_budget_seconds() = 200 + 20 = 220s`:一次慢尾被墙钟杀掉(不可重试),所以最坏是"一次瞬态快失败 + 退避 + 一次慢尾",不会两次慢尾叠加。

### 2.5 阶段超时(src/graph.py PHASE_TIMEOUTS)
```
understand:240  locate:180  plan:300  execute:600  verify:15  reflect:300  commit:600
```
plan/reflect 是 300s(不是 180):单次 LLM 调用可达 ~140s 重试路径,加上 DeepSeek 对 ~6K token 大 prompt 的延迟方差(实测 25-143s),180s 会被慢尾顶穿。

### 2.6 可观测性(Tracer)
每个 run 一个 trace_id,所有阶段转换、工具调用、节点诊断(`node_diagnostics`:elapsed_seconds、prompt/response token 估算)写进 trace JSON。**这是整个项目调试的数据基础**——所有根因都靠读 trace 数据定位,不靠猜。

### 2.7 记忆层(SQLite)
按 repo 的文件索引 + issue 历史。修过 N 次 bug 后,locate 优先搜历史修改过的文件(relevance 0.75)。WAL 模式、原子写、fire-and-forget(失败不阻断主流程)。

---

## 3. 执行层设计(src/nodes/execute.py)

这是从"补丁能否真正被验证"角度最关键的一层,也是踩坑最多的一层。

### 3.1 流程
```
clone(带缓存) -> 建隔离 venv -> pip install -e . -> apply patch_edits -> 跑 pytest
```

### 3.2 clone 策略
三级回退:`--depth 1 --filter=blob:none --single-branch`(最快)-> `--depth 1`(浅克隆)-> 全克隆。本地有缓存则 `--local --no-hardlinks` 从缓存克隆,避免重复远程拉取。

### 3.3 隔离 venv(难点 6.10)
```
python3 -m venv --system-site-packages <clone>-venv   # uv venv 作为 fallback
```
- `--system-site-packages`:复用系统已装的 pytest 等重依赖,只装缺的,省时间。
- venv **绕过 PEP 668**(系统 Python 是 externally-managed,系统 pip 拒装)。
- venv 建在 clone 旁边(`<clone>-venv`),不污染仓库、pytest 不会去收集它。
- 用 `.repopilot-ready` sentinel 标记:半成品 venv(缺 ensurepip)不会被 run_pytest 误用。

### 3.4 装包(best-effort,有界,失败容忍)
链式尝试 `pip install -e .[test]` -> `.[testing]` -> `.[dev]` -> `.`,240s 超时。**用 venv 的 pip,不是系统 pip**。失败不阻断(扁平布局/纯 stdlib 仓库照样能测)。每步记 `pip_install_editable` 工具调用便于诊断。

### 3.5 跑测试
venv ready 时,把测试命令改写为 `<venv-python> -m pytest ...`,并把 venv bin 加到 PATH、设 VIRTUAL_ENV——确保用 venv 解释器(editable install 可 import),而不是落到系统 pytest。无 venv 时回退系统 python3。

### 3.6 补丁应用
优先 `patch_edits`(确定性精确字符串替换:file + search + replace + replace_all),比 unified diff 鲁棒。有 patch_repair 模块在 diff apply 失败时尝试修复(路径/hunk context)。

---

## 4. 检索层设计(src/nodes/locate.py + src/retrieval.py)

这一层经历了 4 轮迭代(死循环 -> 不消费 next_checks -> 无状态丢上下文 -> 文档污染),每轮都是"规划器被喂了错的上下文"。

### 4.1 候选来源(优先级)
1. **记忆层**:历史修改过的文件(relevance 0.75)。
2. **规划器点名的文件**(relevance 0.9):从最近 plan 帧的 `trace_notes.files`(结构化)+ `next_checks`(正则抠路径)。这是让每轮真正拉到新上下文的关键。
3. **issue 文本搜索**:`_issue_search_terms` 抽 code terms(反引号)+ 标识符,GitHub 代码搜索。

### 4.2 跨轮累积(难点 6.7)
locate 原本无状态,每轮从零重建候选 -> 中途找到的好文件在决定轮被丢弃。改成:开头捕获上轮已 hydrate 的文件,跨轮保留;新文件叠加;token 只计新文件(避免重复计费)。

### 4.3 文档排除(难点 6.8)
`_is_doc_file` 排除 `.rst/.md/.txt` + `docs/` 目录。原因:累积放大了 BM25 偏好,11 万字符的 `config.rst` 等大文档霸占喂给规划器的 top-4,源码被挤出去。改源码 + 跑 pytest 的 agent,文档永远不该占名额。

### 4.4 BM25 重排(src/retrieval.py)
确定性词法重排,标准 BM25(k1=1.5, b=0.75, IDF 带 +0.5 平滑)。把 issue 标题+正文当 query,对 hydrated 文件打分。最终 relevance = `0.35 * 原始分 + 0.65 * 归一化BM25`(blend)。只在有词法信号时应用(`applied` 标记),无信号保持原序。文件路径也 tokenize 进文档(`tox/tox_env/api.py` -> `tox tox env api`)。

### 4.5 无进展刹车
hydrated 文件路径排序成签名,与上轮相同 -> 判定无进展、提前 FAILURE。配合 5.4 的硬上限,双层止血。

---

## 5. 推理层设计(src/nodes/plan.py + reflect.py)

### 5.1 plan_fix 的 prompt 结构
system(强约束 JSON schema:patch_edits 格式、decision_frame 必填字段)+ user(issue + 相关文件内容 + 之前的失败 + 反思 + 假设连续性 + **上下文压力** + 人类回答)。

### 5.2 文件上下文上限(难点 6.3)
`PLAN_MAX_FILES=4`,`PLAN_FILE_CONTENT_LIMIT=6000`,`PLAN_ISSUE_BODY_LIMIT=2500`。早期是 2x1200,导致规划器只看到文件前 3%(import 区),看不到函数体,永远喊"要更多上下文"。

### 5.3 上下文收集压力(难点 6.9)
按 `context_collection_count` 升级压力:
- count=0:无压力,正常侦查。
- count>=1:软压("已收集 N 次,强烈建议本轮出 patch_edits;要上下文必须点名具体未读文件;复用既有假设别重造")。
- count>=3(末轮):硬压("FINAL round,必须 execute + patch_edits,不许再 collect")。

这是为了治"规划器看着正确源码却每轮把假设推倒重来、攒不出交补丁的信念"。

### 5.4 循环硬上限
`MAX_CONTEXT_COLLECTION_ROUNDS=3`。每次 `collect_more_context` 计数 +1,超限强制 `stop` -> FAILURE(清晰失败原因)。前 3 次仍正常回 LOCATE,保留真正扩展上下文的机会。

### 5.5 假设连续性(reflect 后)
patch_apply 失败后,plan 不该漂移到新假设——`_preserve_patch_apply_hypothesis_anchor` 把上一个 plan 帧的 selected_hypothesis 锚定回来,prompt 里也注入"修补之前的补丁,别改语义"的指令。防止反思后假设跳变。

---

## 6. 技术难点全链(11 个根因)

> 这是项目的核心叙事。从"token 烧光连代码都没定位到"到"测试验证通过的正确补丁",
> 每一层都是**用真实 trace 数据定位 -> 改代码带测试 -> 重跑验证**,而非猜测。
> 关键模式:**每次先用数据证伪自己的假设再动手**。

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 1 | 7 轮 collect 烧光 50K token,没定位到代码 | locate 每轮用同一组搜索词,死循环无刹车 | 签名检测 + 硬上限(token 50K->26K) |
| 2 | locate 拿回同一批文件 | 不消费规划器的 next_checks | 读 trace_notes.files + next_checks 路径 |
| 3 | 规划器永远喊"要更多上下文" | 只看到文件前 3%(import 区) | 上下文 2x1200 -> 4x6000 |
| 4 | plan 阶段超时 | httpx timeout 不限请求总时长 | asyncio.wait_for 墙钟 200s |
| 5 | 决定轮丢失好文件,喂垃圾 | locate 无状态,每轮从零重建 | 跨轮累积上下文 |
| 6 | 大文档霸占 top-4 | BM25 偏爱关键词密集大文档 | 排除 docs/.rst 文件 |
| 7 | 看着正确源码不收敛 | 规划器每轮假设推倒重来 | 上下文收集压力(prompt) |
| 8 | tox not found | execute_fix 只 clone 不装包 | 换 scrapy 样本 + 加装包 |
| 9 | ModuleNotFoundError | 系统 pip 被 PEP 668 拒绝 | 隔离 venv(uv fallback) |
| 10 | 半成品 venv 被误用 | 缺 ensurepip 半创建失败 | .repopilot-ready sentinel |
| 11 | 最终 FAILED(404) | commit 对上游开 PR 无权限 | skip_commit:验证通过即 DONE |

### 6.4 详解:httpx timeout 不限制请求总时长(最硬核)
配置 `timeout=60` 但一次成功的 LLM 调用实测跑 142.7s。**怎么发现的**:读 trace 里每次调用的吞吐——725tok/25.7s=29tok/s,1022tok/48.2s=21tok/s,1194tok/142.7s=8.4tok/s,一条平滑退化曲线。如果是"超时+重试拼出来的"曲线不会平滑,说明单次生成端点在涓流。**根因**:httpx 裸 float timeout 展开成 connect/read/write/pool 四个**独立** per-operation 超时,read 是"两次收到字节的最大间隔"不是总时长,端点持续涓流则永不触发。**httpx 设计上没有"整个请求 <= N 秒"选项**。**修复**:asyncio.wait_for 包墙钟上限,且 asyncio.TimeoutError 不进重试白名单。

### 关键认知:基础设施 vs 推理
前 6 个根因是**确定性工程 bug**(能写测试复现),7 之后转向**LLM 非确定性推理质量**(prompt 工程 + 结构兜底)。能区分这两类问题是诊断的关键——确定性的写测试,非确定性的别指望测试,用 prompt 压力和硬上限兜底。

---

## 7. 关键技术决策点(含权衡)

### 7.1 LLM 慢尾:容忍 vs 快速失败
**决策**:墙钟 200s,容忍慢尾。**权衡**:宁可多等拿结果(当前目标是先跑通端到端),而非快速失败省 API 成本。如果转向成本敏感,降到 90s 但会杀死本可成功的慢调用。

### 7.2 venv:--system-site-packages vs 全新隔离
**决策**:复用系统 site-packages。**权衡**:省去重装 pytest/twisted 等重依赖的时间(装全套可能撞 240s 超时),代价是隔离性稍弱(系统包版本可能干扰)。对 eval 场景,速度优先。

### 7.3 patch_edits vs unified diff
**决策**:优先 patch_edits(精确字符串替换)。**权衡**:LLM 生成 unified diff 的 hunk context 经常错(行号/上下文),apply 失败率高;精确 search/replace 更鲁棒,代价是 LLM 必须逐字复制现有代码。

### 7.4 上下文上限:大 vs 省 token
**决策**:4x6000(从 2x1200 提高)。**权衡**:规划器需要看到函数体才能决策,3% 可见度导致无限循环、烧的 token 远比省的多。代价是 prompt 变大、延迟上升(进而触发了 6.4 的超时问题)。

### 7.5 文档排除:全排 vs 降权
**决策**:直接排除 docs/.rst。**权衡**:一个改源码 + 跑 pytest 的 agent 永远不会把文档当修复目标;降权不够(BM25 信号太强会反弹)。代价是放弃了"文档 bug"场景(超出 agent 范围)。

### 7.6 循环上限:3 轮硬停 vs 无限收集
**决策**:3 轮 + 末轮强制交补丁。**权衡**:trace 显示规划器置信度是**下降**的,多给几轮救不了被喂垃圾的它;硬停 + 压力比无限等待更早产出结果(哪怕错补丁也进了 execute->verify 循环)。

### 7.7 双图引擎
**决策**:LangGraph + 30 行 Fallback。**权衡**:多维护一套实现,换来离线/CI 零依赖测试 + 不被 LangGraph 版本绑死。

---

## 8. 面试官可能追问 + 答案

### Q1:"端到端只成功 1 个样本,这能算项目吗?"
A:项目的价值是**诊断深度和工程方法论**,不是 demo 成功率。我用真实 trace 数据把一条反复失败的执行链一层层定位到 11 个根因,每个都改代码带测试零回归。一个能讲清"为什么失败、怎么定位"的工程师,比只会展示 happy path 的更有说服力。而且这第 1 个绿是 trace 可复现的真实开源 bug,不是编造的。下一步就是跑更多样本拿统计 resolved%。

### Q2:"你怎么确定 httpx 的 60s 超时没生效,而不是别的原因?"
A:不是猜,是算吞吐。三次调用 29->21->8.4 tok/s 是一条平滑退化曲线。如果 142s 是"60s 超时 + 重试"拼的,会看到两段而非平滑曲线。然后查 httpx 文档确认:裸 float timeout 是四个 per-operation 超时,read 超时是字节间隔不是总时长。最后用 asyncio.wait_for 验证修复。每一步都有数据或文档支撑。

### Q3:"read timeout 到底是什么语义?为什么涓流能绕过它?"
A:httpx 的 read timeout 是"等待**下一个数据块**的最长时间",每收到一个 chunk 就重置。LLM 流式响应只要持续吐字节(哪怕很慢),read timeout 永远在 60s 内被重置,所以整个 142s 的生成全程没触发。httpx 没有"整个请求墙钟上限"这个选项——这是它的设计,不是 bug。所以必须在外层用 asyncio.wait_for 包。

### Q4:"为什么 asyncio.TimeoutError 不放进重试白名单?"
A:因为墙钟超时意味着"这次生成本身就慢",重试只会再等 200s 翻倍浪费,且大概率还是慢。而瞬态错误(502/网络)是毫秒级快失败,重试几乎不占时间、值得。所以墙钟超时快速失败、瞬态错误重试——两种失败模式区别对待。这也让重试预算可算:最坏 220s 而非 420s。

### Q5:"PEP 668 是什么?为什么 venv 能绕过?"
A:PEP 668 是 Python 的 externally-managed-environment 标记,系统 Python(尤其 Debian/Ubuntu)会放一个 EXTERNALLY-MANAGED 文件,让 `pip install` 直接拒绝,防止污染系统包。venv 是独立环境,没有这个标记,所以 venv 内的 pip 正常工作。我用 `--system-site-packages` 让 venv 复用系统已装的包(省下重装重依赖的时间),只把缺的装进 venv。

### Q6:"BM25 为什么会偏爱文档?你的 blend 0.35/0.65 怎么定的?"
A:BM25 是词频 x 逆文档频率。大文档(config.rst 11 万字符)塞满所有 config 术语,词频极高,分数碾压源码。虽然 BM25 有文档长度归一化(b=0.75),但关键词密度太高仍然占优。blend 里 BM25 占 0.65 是因为词法匹配比"文件名出现在 issue 里"这种启发式更可靠,但保留 0.35 给原始启发式(记忆层/规划器点名的文件应该有底分)。最终发现 blend 调参不如直接排除文档干净——这是个"与其调权重不如消除噪声源"的判断。

### Q7:"决策帧(DecisionFrame)解决了什么问题?和直接让 LLM 输出下一步有什么区别?"
A:区别是**路由逻辑在代码里还是在 prompt 里**。如果让 LLM 自由决定下一步,路由就藏在不可控的生成里,无法调试"为什么走了这条路"。DecisionFrame 把 LLM 的意图结构化成 `recommended_action`,路由是纯函数 `route_from_state` 读它决定 Phase。这样每个转换都可追溯(route_decisions)、可加守卫(一次性消费防死循环)、可加诊断警告(帧 stage 和预期 phase 不符时)。

### Q8:"上下文压力是 prompt 工程,它可靠吗?"
A:不可靠——它是行为微调不是确定性修复,效果取决于 LLM 听不听话。所以我配了硬兜底:末轮 prompt 哀求的同时,代码层有 `MAX_CONTEXT_COLLECTION_ROUNDS=3` 硬上限,超了强制 stop。prompt 压力负责"尽量让它早点收敛",硬上限负责"它不听话也不会无限烧 token"。两层配合,不把可靠性押在 LLM 单一行为上。

### Q9:"如果让你重做,哪里会不一样?"
A:三点。(1) locate 一开始就加"只给源码、源码优先于文档"的硬约束,能少走文档污染那一层。(2) eval 从第一天就跑多次取分布,而不是单次——我曾有一次侥幸 execute 没复现,差点误判能力。(3) execute_fix 的 venv/装包应该更早做,我前期在 src/ 布局仓库上反复撞 import 失败,其实根因一开始就在"不装包"。

### Q10:"这个 agent 和 SWE-bench 上的方案比怎么样?"
A:诚实说,我没在 SWE-bench 上跑过统计数字,目前是单样本验证。架构上我的差异点是"决策可观测"(LangGraph 显式状态机 + 决策帧),而很多 SWE-bench 方案是 ReAct 式的黑盒循环。下一步就是跑 SWE-bench Lite 子集报一个诚实的 resolved%。我不会夸大成"超过 X"——没数据的声称在面试里是减分项。

### Q11:"LLM 输出不稳定,你怎么保证补丁质量?"
A:分两层。**格式层**:Pydantic schema 强约束 + validate_or_retry(解析失败注入错误重试一次),还有别名容忍(file/path/file_path)、score 归一化(9->0.9)这些边界防御。**语义层**:补丁必须经 EXECUTE 真实 apply + VERIFY 跑 pytest 验证,测试不过就进 REFLECT 分析根因重试。所以"质量"不是信任 LLM,而是用真实测试做闭环——这也是为什么 execute_fix 的装包/venv 基础设施这么关键:没有能跑的测试,就没有质量信号。

### Q12:"turns=19、token=11.8K 是怎么构成的?哪里最贵?"
A:主要是 plan 调用——每次 ~6-8K token prompt(4 文件 x 6000 字符为主体),scrapy 这次收敛快所以总量低。早期 tox 那次烧到 50K 就是因为 plan 反复调用(collect 循环)。最贵的永远是 plan 阶段的文件上下文,所以 PLAN_MAX_FILES 和 FILE_CONTENT_LIMIT 是核心的成本/质量权衡旋钮。

---

## 附:关键常量速查
```
推理层  MAX_CONTEXT_COLLECTION_ROUNDS=3  PLAN_MAX_FILES=4  PLAN_FILE_CONTENT_LIMIT=6000  PLAN_ISSUE_BODY_LIMIT=2500
HTTP    LLM_REQUEST_TIMEOUT=60(per-op)  LLM_CALL_WALLCLOCK_TIMEOUT=200  LLM_MAX_ATTEMPTS=2  retry_budget=220s
阶段    plan=300 reflect=300 execute=600 understand=240 locate=180 verify=15 commit=600
执行    INSTALL_TIMEOUT=240  VENV_CREATE_TIMEOUT=120  venv: --system-site-packages, uv fallback
BM25    k1=1.5  b=0.75  blend = 0.35*原始 + 0.65*归一化BM25
```

*所有数据可回溯:examples/traces/case_1.json、eval/eval_results.json。首绿 run 68285b8d86a2。*
