# RepoPilot Bug 记录

## Bug 1：调试代码未删除（已修复 ✅）

**位置**：`src/agent.py:45-47`

**现象**：代码里残留了一句往 `/tmp/repopilot_errors.log` 写文件的调试代码，且只在 classify 这一步有，其他 5 步没有——明显是临时调试忘删的。`import sys` 导了但从未使用。

**影响**：面试官看到第一反应"这人 debug 完不收拾"。往系统 `/tmp` 写文件也不干净。

**解决**：删掉整个 `with open(...)` 块和死 `import sys`。错误已通过 `Tracer.log()` 记录。

---

## Bug 2：模型配置失效（已修复 ✅）

**位置**：`src/llm.py:41`、`src/llm.py:_config()`

**现象**：`.env.example` 里写了 `LLM_MODEL=deepseek-chat`，但代码从来没读这个环境变量。改 `.env` 想切换模型是无效的。

### 完整 Workflow

**Step 1 — 你在 `.env` 里改了模型**

```
LLM_MODEL=deepseek-v4-pro
```

心里想的是"以后所有 LLM 调用都用 v4-pro"。

**Step 2 — 但代码从来没看这个变量**

```python
# 旧 _config() — 只返回 2 个值
def _config() -> tuple[str, str]:
    api_key = os.getenv("DEEPSEEK_API_KEY") ...
    base_url = os.getenv("OPENAI_BASE_URL", ...)
    return api_key, base_url     # ← 没有 model！
```

`_config()` 根本没读 `LLM_MODEL`。它返回的是 `(api_key, base_url)`，没有 model 字段。

**Step 3 — llm_call 用了硬编码的默认值**

```python
async def llm_call(..., model: str = "deepseek-v4-flash") -> dict:
    api_key, base_url = _config()
    # model 参数来自函数签名默认值，不是来自 env
```

`model` 这个参数完全靠调用者传。如果调用者没传（`classify_issue`、`rank_files`、`generate_fix_plan` 都没传），就走默认值。而这个默认值是写在代码里的 `deepseek-v4-flash`——你在 `.env` 里改什么都没用。

**Step 4 — `.env.example` 写了，代码不读，完全误导**

```bash
# .env.example
LLM_MODEL=deepseek-chat   # ← 你觉得改了代码就会听？不会。
```

你（和其他开发者）看到这个注释，会自然地认为"改这个就能切模型"。实际上这个变量就是个死变量——定义了但没人读。Opus 管这叫"dead config knob"。

### 根因

`_config()` 只关心 API key 和 endpoint。模型名被当成"调用者自己决定的事"，散落在每个函数的默认参数里。环境和代码之间的契约断裂了——环境变量说"我有 LLM_MODEL"，代码说"我不看"。

### 影响

你永远在用代码里硬编码的默认模型。除非你手动改源代码里的函数签名，否则 `.env` 改了白改。

### 解决

让 `_config()` 读 `LLM_MODEL`，返回三元组：

```python
def _config() -> tuple[str, str, str]:
    api_key = os.getenv("DEEPSEEK_API_KEY") ...
    base_url = os.getenv("OPENAI_BASE_URL", ...)
    model = os.getenv("LLM_MODEL", "deepseek-v4-pro")  # ← 现在真的读了
    return api_key, base_url, model

async def llm_call(..., model: str = None):
    api_key, base_url, default_model = _config()
    if model is None:
        model = default_model            # ← 默认从 env 来，不是硬编码
```

`llm_call` 的默认值从 `"v4-flash"` 改成 `None`，意思是"我不做主了，听 `_config()` 的"。你在 `.env` 里改 `LLM_MODEL` 就真的能切模型了。

---

## Bug 3：错误返回 HTTP 200（已修复 ✅）

**位置**：`src/main.py:19-21`、`src/main.py:33-35`

**现象**：Issue URL 无效返回 200。GitHub API 挂了也返回 200。`curl`、监控、负载均衡全分不清成功还是失败。

**根因**：FastAPI 默认 HTTP 200，除非显式指定。`return {"status": "error", ...}` 没有设置状态码。

**解决**：用 `JSONResponse(content=..., status_code=status)` 替代裸 `return`。URL 格式错误 → 400，上游失败 → 502。

---

## Bug 4：代码搜索 query 太弱（已修复 ✅）

**位置**：`src/agent.py:52`

**现象**：拿 Issue 标题当 GitHub 搜索词，中文自然语言搜代码基本空结果，导致后续 ranking 和 fix plan 都是无源之水。

**完整 Workflow**：

1. 用户给 RepoPilot 一个 Issue URL
2. `parse_issue_url()` 解析 owner/repo/number
3. `read_issue()` 调 GitHub API 拿 title + body → 这一步正常
4. `classify_issue()` 调 DeepSeek 分类 → 这一步正常
5. `search_code(code_query, owner, repo)` 搜索代码 ← **这里出问题**

旧代码：
```python
query = issue["title"][:100]    # → "早报推送偶尔漏掉中午那期"
```

GitHub Code Search 不是语义搜索，是基于索引的关键词匹配。这行中文丢给 `/search/code` API 变成：

```
repo:FMorgan-111/ai-daily-brief 早报推送偶尔漏掉中午那期
```

仓库里没有任何文件包含这句中文 → 返回空列表。

**影响链**：搜索空 → `rank_files()` 没文件可排序 → `generate_fix_plan()` 拿不到代码上下文 → **对着空气写修复方案**。

**解决**（死代码层面）：
```python
raw = f"{issue['title']} {issue['body'][:200]}"
query = ' '.join(w for w in raw.replace('/', ' ').split() if len(w) > 1)[:200]
```
标题 + 正文前 200 字拼起来，过滤单字符噪音。

**最佳解法**（Agent 循环层面）：`src/agent_loop.py` 里把 `search_code` 暴露给 LLM 当工具。LLM 读完 Issue 后自己理解内容、自己提取关键词、自己决定搜什么：

```
LLM 读 Issue → 理解：这是 cron scheduler 问题
            → 生成 query: "cron scheduler concurrent fallback workflow"
            → 调 search_code(query="cron scheduler concurrent...")
            → 拿到 daily-brief.yml
            → 继续调用 read_file 细读文件内容
            → 信息够了，出修复方案
```

关键差别：死代码是"你替 LLM 搜"，Agent 是"LLM 自己决定搜什么"。

---

## Bug 5：JSON 提取只支持一层嵌套（已修复 ✅）

**位置**：`src/llm.py:25`

**现象**：`_extract_json()` 的备用正则只能处理一层 `{...}` 嵌套。遇到 `{"a":{"b":{"c":1}}}` 这种就挂了。

### 直观理解

**一层**（没嵌套）：
```json
{"name": "morgan"}
```
只有最外面一对 `{}`，里面没有 `{}`。

**两层**（值里面又套了 `{}`）：
```json
{"person": {"name": "morgan"}}
```
外面一对 `{}`，`"person"` 的值又是一个 `{}`。

**三层**：
```json
{"person": {"profile": {"name": "morgan"}}}
```
三对 `{}` 套在一起。

### 为什么正则搞不定

旧正则是 `\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}`——它只能数到一层括号。遇到两层嵌套时，正则误把里面的 `}` 当成最外层结束，提前截断了。

### 括号计数怎么做

换一种思路——扫描的时候边走边数：

```
{"person": {"name": "morgan"}}
↑                           ↑
从这里开始数                数到这，计数回到 0

字符 {         → 计数 = 1     ← 开始
字符 "person": → 计数 = 1
字符 {         → 计数 = 2     ← 里面的括号
字符 "name":"morgan" → 计数 = 2
字符 }         → 计数 = 1     ← 里面的括号关了
字符 }         → 计数 = 0     ← 回到零！切到这里
```

不管套了多少层，每个 `{` 必定跟一个 `}` 配对。**计数回到 0 就是最外层结束**。

### 代码

```python
start = text.find('{')
depth = 0
for i in range(start, len(text)):
    if text[i] == '{':
        depth += 1       # 遇到 { → 深度+1
    elif text[i] == '}':
        depth -= 1       # 遇到 } → 深度-1
        if depth == 0:   # 回到 0 = 最外层结束
            return json.loads(text[start:i+1])  # 切出来，解析

---

## 非 Bug 修复：Opus 误判模型名

Opus 看到 `deepseek-v4-flash` 这个命名不像标准 OpenAI 模型（`gpt-4`、`claude-3`），以为是 typo。实际上 DeepSeek 就是叫这个名字，能正常调用。

真正的修复是把 `LLM_MODEL` 环境变量激活，默认值设为 `deepseek-v4-pro`。

---

## 非 Bug 修复：环境问题

| # | 问题 | 解决 |
|---|------|------|
| 7 | `.venv/` 提交了空的虚拟环境，clone 后跑不了 | 从 git 移除 + gitignore |
| 8 | `requirements.txt` 未锁版本，不可复现 | 全部加 `==` 固定版本 |

---

## 改进：Pydantic 结构化输出验证（已实现 ✅）

### 问题：只验证 JSON 格式，不验证内容

之前 `_extract_json()` 只管"能解析出 JSON 就算赢"。LLM 返回了 `{"type": "bug", "severity": "invalid_value", "confidence": -5}` —— 格式是 JSON，但内容全错了：severity 必须是 low/medium/high，confidence 必须是 0-1。旧代码不会发现这些错误。

### Pydantic 是干什么的

Pydantic 是 Python 的数据校验库。你定义一个模型，它自动检查数据是不是符合规则：

```python
class Classification(BaseModel):
    type: str = Field(pattern="^(bug|feature|docs|test|security)$")
    severity: str = Field(pattern="^(low|medium|high)$")
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
```

- `type` 只能是 bug/feature/docs/test/security 之一
- `confidence` 必须是 0 到 1 之间的浮点数
- 缺字段、多了字段、类型不对——全部自动检测

### 完整 Workflow

**Step 1 — LLM 返回 JSON**

```json
{"type": "bug", "severity": "INVALID_VALUE", "confidence": 0.9}
```

JSON 解析成功，旧代码到此就结束了，直接当正确结果用。

**Step 2 — Pydantic 校验**

```python
Classification.model_validate(raw)
# → ValidationError: severity='INVALID_VALUE' does not match pattern
```

Pydantic 发现 `severity` 的值不在允许范围内，抛出详细的错误信息。

**Step 3 — 带错误信息重试**

```python
retry_user = (
    f"Your response did not match the schema. Errors: {errors}\n\n"
    f"Original request:\n{user}\n\n"
    "Return ONLY valid JSON matching the required keys and types."
)
raw2 = await llm_call(system, retry_user)
```

把具体错在哪里告诉 LLM，让它自己修正。不是模糊地说"重试"，而是说"severity 字段的值不符合 ^(low|medium|high)$ 这个规则"。

**Step 4 — 二次校验 + fallback**

```python
try:
    validated2 = schema.model_validate(raw2)
    return validated2.model_dump()
except ValidationError as e2:
    warnings.warn(f"schema failed after retry: {e2.errors()}")
    return raw2  # 还是失败？返回第二次的数据 + 打 warning
```

### 3 个关键决策

**1. 两次失败后不崩溃，返回 raw 数据打 warning。** Agent 不能因为校验失败就直接挂掉——返回不完美的结果比什么都不回强。`warnings.warn()` 让生产环境能监控"LLM 输出不合格"的频率。

**2. 错误提示说"schema 不匹配"，不说"JSON 无效"。** JSON 解析成功了（`_extract_json` 过了），问题出在字段内容不符合规则。说错原因会让 LLM 朝错误的方向修改。

**3. `confidence: 0` 改成 `0.0`。** Python 里 `0` 是 int，`0.0` 是 float。Pydantic 的 `float` 字段不认 int。一分钱的类型不匹配就是整个校验失败。

---

## 今日工作记录（2026-06-05）

### 1. 补写 Bug 2 完整 Workflow

**原因**：之前 Bug 2 只有 3 行根因+解决。用户问"为什么有 2"。

补充了 4 步完整路径：`.env` 改了模型 → 代码不读 `LLM_MODEL` → `llm_call` 硬编码默认值 → `.env.example` 写成死变量误导所有人。Opus 管这叫 dead config knob。

### 2. 新增 Pydantic 结构化输出章节

**原因**：用户问"Pydantic 是怎么回事"。按 BUGLOG 格式（Workflow 分步 + 根因 + 决策）写了完整说明。

### 3. 解释"invalid JSON" vs "schema 不匹配"

**问题**：旧重试提示说 "Your last response was invalid JSON"——但 JSON 解析成功了，错在 Pydantic 校验（字段值不符合规则）。

**为什么重要**：LLM 收到 "invalid JSON" 会去改格式（加引号、调括号），但真正的问题是内容不合规。说错原因 = 白重试一次。

### 4. 解释 fallback 为什么用 raw2 不用 raw

**决策**：第一次失败后已把具体错误喂给 LLM，第二次尝试的字段值整体更接近正确格式。不是"保证正确"，是"更大概率正确"。

**底线**：加上 `warnings.warn()` 亮信号，生产环境能监控 fallback 频率。

### 5. 解释为什么不多 retry 几次

**三个理由**：

1. **递减回报**：第一次已拿到具体错误反馈，第二次修不了 = 同模型同 prompt 下再多几次也一样
2. **累积延迟**：每个 retry 1-3 秒，Agent loop 里多轮调用会被放大
3. **错误分类**：只有"疏忽"能修（1 次够），"规则矛盾"和"能力边界"修不了（N 次也不行）

**替代方案**：不靠增加 retry 次数提高成功率，靠更准的错误提示、写死 prompt、换更强模型、或上 `response_format`。

### 6. Pydantic 校验基准测试

**原因**：用户质疑 "v4-pro 能力边界能不能搞定 schema"——诚实回答"不确定，没测过"。

**测试**：写 `tests/bench_pydantic.py` + `tests/quick_bench.py`，对 Classification / FileRanking / FixPlan 各跑多次。

**结果**：47/47 一次过，0 retry，0 fallback。

**结论**：retry/fallback 代码在 v4-pro 下是死路径，但保留作为模型切换 / API 变更 / prompt 改动时的防御层。

### 7. 确认模型配置

三重验证当前使用 v4-pro：代码默认值、bench 输出自报模型名、47 次全过（v4-flash 大概率会有 retry）。
