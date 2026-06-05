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

**根因**：`_config()` 只读 `DEEPSEEK_API_KEY` 和 `OPENAI_BASE_URL`，没读 `LLM_MODEL`。`llm_call()` 的 `model` 参数是硬编码的默认值。

**解决**：`_config()` 加 `model = os.getenv("LLM_MODEL", "deepseek-v4-pro")`，返回三元组。`llm_call()` 默认值改为 `None`，用 `_config()` 返回的 model。

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
