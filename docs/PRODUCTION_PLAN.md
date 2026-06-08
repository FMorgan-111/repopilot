# RepoPilot: Prototype → Production 实施计划

> 目标：将一个高质量原型（应届生前 10-15%）升级为"面试官会记住你"的 production agent 项目（前 5%）
>
> 创建日期：2026-06-08
> 当前版本：v0.1.0 (prototype, 距 production 差 3 分)

---

## 0. Production 的具体定义（针对本项目）

对于"一个应届生能拿出手面试的 production agent 项目"，production 不等于企业级生产环境。具体标准：

| 维度 | 当前状态 | Production 目标 |
|------|----------|------------------|
| **可运行性** | `pip install -e .` 后手动 uvicorn | `pip install repopilot && repopilot <url>` 一条命令 |
| **鲁棒性** | HTTP 无重试、无限速处理、print 代替 logging | 指数退避重试、429 限速处理、结构化 logging |
| **可验证性** | 零个真实 issue trace | ≥3 个真实开源项目 issue 的成功/失败 trace |
| **可演示性** | 无 demo 脚本、无录屏 | `demo.sh` 一键跑 3 个 case 输出结果 |
| **可部署性** | 仅本地 `pip install -e .` | PyPI 发布 + Dockerfile |
| **代码质量** | `new_agent.py` 999 行单文件、`chars//4` 估算 token | 拆分为 6 个模块文件、tiktoken 精确计数 |
| **可观测性** | `print()` 输出 JSONL | `logging` 模块 + 结构化日志 + Tracer 输出到文件 |
| **测试覆盖** | 46 tests，核心路径覆盖 | 50+ tests，含 HTTP mock 的异常路径 |
| **文档完整** | 6 份设计文档 + README | README 含 demo GIF、架构图、示例输出 |

**关键原则**：每一个改动都要能在面试中讲出来。改动代码量大但面试只能讲 30 秒的，降低优先级；改动小但能讲 3 分钟的，提升优先级。

---

## 1. 分阶段计划

### 阶段 0：预备 — 建立基线（0.5h）

**做什么**：记录当前状态，建立改动前基线

| 操作 | 文件 | 怎么做 |
|------|------|--------|
| 记录当前 test 全部通过 | `pytest tests/ -q -v` | 运行并保存输出到 `docs/baseline_tests.txt` |
| 记录当前代码行数统计 | `cloc src/ tests/` | 保存到 `docs/baseline_loc.txt` |
| 确认 CI 通过 | 检查 GitHub Actions | 截图保存 |

**验收标准**：
- [ ] `pytest tests/ -q` 全部通过（当前 46 tests）
- [ ] 基线数据已记录

**面试能讲多久**：0 秒（铺垫性工作）

---

### 阶段 1：鲁棒性基础 — HTTP 重试 + 限速处理 + logging（4h）

这是 production 的"地基"——没有它，任何生产运行都会随机崩溃。

#### 1.1 HTTP 重试 + 指数退避（2h）

**文件**：新建 `src/http_client.py`（约 120 行）

**具体改动**：

```python
# 新建 src/http_client.py
# 提取所有 httpx.AsyncClient 调用到一个带重试的客户端工厂

import httpx
import asyncio
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type
)

RETRYABLE_STATUS = {429, 502, 503, 504}
MAX_RETRIES = 3

def _should_retry(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS
    if isinstance(exc, (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException)):
        return True
    return False

@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry_error_callback=lambda retry_state: retry_state.outcome,
)
async def github_request(method: str, url: str, **kwargs) -> httpx.Response:
    """带重试的 GitHub API 请求。"""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.request(method, url, **kwargs)
    resp.raise_for_status()
    return resp

@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry_error_callback=lambda retry_state: retry_state.outcome,
)
async def llm_request(payload: dict, headers: dict, base_url: str) -> httpx.Response:
    """带重试的 LLM API 请求。"""
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{base_url}/chat/completions",
            json=payload, headers=headers
        )
    resp.raise_for_status()
    return resp
```

**改动上游文件**：

| 文件 | 改动位置 | 改动内容 |
|------|----------|----------|
| `src/tools.py` | L19, L37, L55 | `httpx.AsyncClient` → `github_request("GET", url, headers=_headers())` |
| `src/llm.py` | L72-74 | `httpx.AsyncClient` → `llm_request(payload, headers, base_url)` |
| `src/new_agent.py` | L623-625, L631-632, L639-640, L648-649, L658-659, L671-672, L683-684 | 所有 `httpx.AsyncClient` → `github_request(...)` |
| `pyproject.toml` | dependencies | 添加 `tenacity` |

#### 1.2 API 限速处理（1h）

**文件**：修改 `src/http_client.py`，新增 `src/rate_limiter.py`（约 80 行）

**具体改动**：

```python
# 新建 src/rate_limiter.py
import time
import asyncio
from collections import deque
from typing import Optional

class TokenBucketRateLimiter:
    """GitHub API 限速器 — 令牌桶算法。
    GitHub 限制：5000 req/hour (authenticated), 60 req/hour (unauthenticated).
    """
    def __init__(self, rate: float = 1.38, burst: int = 10):
        # rate=1.38 ≈ 5000/3600 requests per second
        self.rate = rate
        self.burst = burst
        self.tokens = float(burst)
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> float:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_refill = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return 0.0
            wait_time = (1.0 - self.tokens) / self.rate
            self.tokens = 0.0
            return wait_time

# 全局实例
_github_limiter = TokenBucketRateLimiter()

async def rate_limited_github_request(method: str, url: str, **kwargs) -> httpx.Response:
    wait = await _github_limiter.acquire()
    if wait > 0:
        await asyncio.sleep(wait)
    return await github_request(method, url, **kwargs)
```

#### 1.3 logging 替换 print（1h）

**文件**：修改 `src/tracer.py`（约 60 行）、新建 `src/logging_config.py`（约 30 行）

**具体改动**：

```python
# 新建 src/logging_config.py
import logging
import sys

def setup_logging(level: int = logging.INFO) -> None:
    """配置结构化 logging。"""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            '{"ts": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "msg": %(message)s}',
            datefmt="%Y-%m-%dT%H:%M:%S"
        )
    )
    root = logging.getLogger("repopilot")
    root.setLevel(level)
    root.handlers = [handler]  # 替换而非追加

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"repopilot.{name}")
```

```python
# 修改 src/tracer.py — print → logging + 可选文件输出
import json
import logging
from uuid import uuid4
from datetime import datetime, timezone
from pathlib import Path
from .logging_config import get_logger

logger = get_logger("tracer")

class Tracer:
    def __init__(self, output_path: str | None = None):
        self.trace_id = uuid4().hex[:12]
        self.steps: list[dict] = []
        self.output_path = Path(output_path) if output_path else None
        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def log(self, step: str, input: dict, output: dict, 
            error: str | None = None) -> None:
        entry = {
            "trace_id": self.trace_id,
            "step": step,
            "ts": datetime.now(timezone.utc).isoformat(),
            "input": input,
            "output": output,
        }
        if error is not None:
            entry["error"] = error
        self.steps.append(entry)
        # 结构化 logging 到 stderr（不干扰 stdout 的 JSON 输出）
        logger.info(json.dumps(entry))
        # 可选文件持久化
        if self.output_path:
            with open(self.output_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
```

#### 阶段 1 验收标准

- [ ] `pytest tests/ -q` 全部通过（可能需新增 3-5 个 HTTP mock 测试）
- [ ] 手动触发一个 503 错误场景，确认重试 3 次后优雅失败（不崩溃）
- [ ] 日志输出到 stderr 为 JSON 格式，stdout 干净
- [ ] `github_request` 在 `httpx.ConnectError` 时自动重试

**面试能讲多久**：**3 分钟**
- "我实现了指数退避重试，对 429/502/503/504 以及网络连接错误自动重试最多 3 次"
- "我用了令牌桶算法做 GitHub API 限速，确保不会因为短时间大量请求而被封"
- "我把 print 换成了结构化 logging，trace 可以同时输出到 stderr 和文件"

---

### 阶段 2：代码质量 — 拆文件 + tiktoken（3h）

#### 2.1 拆分 new_agent.py（2h）

**当前**：`src/new_agent.py` 999 行，包含所有 8 个阶段节点 + 辅助函数 + graph builder + Fallback 引擎 + GitHub helpers + 入口函数。

**目标**：拆分为 6 个文件，每个 100-250 行，职责单一。

| 新文件 | 从 new_agent.py 移入 | 行数估算 |
|--------|---------------------|----------|
| `src/state.py` | `AgentState`, `Phase`, `ConversationTurn`, `FileInfo`, `FixAttempt`, `ToolCall`, `FinalReport` 以及 `_as_state`, `_estimate_tokens`, `_remember`, `_record_tool`, `_is_budget_exceeded`, `_extract_json_object` | ~130 行 |
| `src/nodes/understand.py` | `understand_issue()`, `_issue_search_terms()` | ~80 行 |
| `src/nodes/locate.py` | `locate_code()`, `_rank_reason()` | ~90 行 |
| `src/nodes/plan.py` | `plan_fix()` | ~60 行 |
| `src/nodes/execute.py` | `execute_fix()`, `git_clone()`, `apply_patch()`, `run_pytest()`, `_primary_patch_file()` | ~140 行 |
| `src/nodes/verify.py` | `verify_fix()`, `_same_failure_seen_twice()` | ~60 行 |
| `src/nodes/reflect.py` | `reflect_on_failure()` | ~70 行 |
| `src/nodes/commit.py` | `commit_fix()`, `push_files()`, `create_pr()`, 以及所有 `_github_*` 辅助函数 | ~200 行 |
| `src/nodes/failure.py` | `handle_failure()` | ~40 行 |
| `src/graph.py` | `build_agent_graph()`, `route_from_state()`, `FallbackCompiledGraph`, `FallbackStateGraph`, `run_graph()` | ~80 行 |
| `src/new_agent.py` | 保留为兼容性 re-export，`from .nodes import *` + `agent_v2()`, `final_report_from_state()`, `intelligent_analyze_issue()` | ~50 行 |

**具体操作**：
1. 创建 `src/nodes/__init__.py` — 统一导出所有节点函数
2. 逐个创建 `src/nodes/*.py`
3. 更新 `src/new_agent.py` 为 thin wrapper
4. 运行 `pytest tests/ -q` 确认无回归

#### 2.2 tiktoken 替换 chars//4（1h）

**文件**：修改 `src/state.py` 中的 `_estimate_tokens()`

**当前**（`new_agent.py` L129-130）：
```python
def _estimate_tokens(*parts: str) -> int:
    return max(1, sum(len(part or "") for part in parts) // 4)
```

**改为**：
```python
import tiktoken

_ENCODER = None

def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        try:
            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # 回退到 chars//4（tiktoken 不可用时）
            return None
    return _ENCODER

def _estimate_tokens(*parts: str) -> int:
    """使用 tiktoken 精确估算 token 数，回退到 chars//4。"""
    text = "".join(part or "" for part in parts)
    encoder = _get_encoder()
    if encoder:
        return max(1, len(encoder.encode(text)))
    return max(1, len(text) // 4)
```

**同时更新** `pyproject.toml` dependencies 添加 `tiktoken`。

#### 阶段 2 验收标准

- [ ] `pytest tests/ -q` 全部通过
- [ ] `src/new_agent.py` 从 999 行缩减到 ~50 行（thin wrapper）
- [ ] 每个 `src/nodes/*.py` 文件 ≤250 行
- [ ] `import repopilot` 或 `from src.new_agent import agent_v2` 仍然可用（向后兼容）
- [ ] `_estimate_tokens("hello world")` 在安装 tiktoken 时返回精确值，未安装时回退到 chars//4
- [ ] 新增 2 个测试：`test_token_estimation_with_tiktoken`, `test_token_estimation_fallback`

**面试能讲多久**：**2.5 分钟**
- "我把 999 行的单文件拆分成了 6 个职责单一的模块——state、graph、以及每个 phase 一个 node 文件"
- "拆分后每个文件 ≤250 行，面试官可以直接打开 `src/nodes/reflect.py` 看到反思逻辑的完整实现"
- "token 估算从粗糙的 `chars//4` 升级到了 tiktoken 精确计数，并保留了 fallback"

---

### 阶段 3：可验证性 — 真实 trace（2h）

#### 3.1 Demo 脚本 + 3 个真实 case（1.5h）

**文件**：新建 `demo.sh`（约 50 行）、修改 `examples/` 目录

**具体改动**：

```bash
#!/bin/bash
# demo.sh — 一键运行 3 个真实 GitHub issue 的 RepoPilot 分析
# 用法: GITHUB_TOKEN=xxx LLM_API_KEY=xxx ./demo.sh

set -e
mkdir -p examples/traces

ISSUES=(
  "https://github.com/cookiecutter/cookiecutter/issues/1973"
  "https://github.com/Textualize/textual/issues/3996"
  "https://github.com/tiangolo/fastapi/issues/368"
)

for i in "${!ISSUES[@]}"; do
  url="${ISSUES[$i]}"
  echo "=== Case $((i+1)): $url ==="
  python -m src.cli "$url" --dry-run --token-budget 30000 --json \
    > "examples/traces/case_$((i+1)).json" 2>"examples/traces/case_$((i+1)).log"
  echo "  Done. Trace saved to examples/traces/case_$((i+1)).json"
done

echo ""
echo "All 3 cases complete. Summary:"
for i in 1 2 3; do
  success=$(python -c "import json; d=json.load(open('examples/traces/case_${i}.json')); print(d.get('success', False))")
  echo "  Case $i: success=$success"
done
```

**同时更新** `examples/candidate_issues.md` 添加每个 case 的实际运行结果摘要。

#### 3.2 Trace 分析脚本（0.5h）

**文件**：新建 `scripts/analyze_trace.py`（约 80 行）

解析 JSONL trace 文件，输出：
- 总 token 消耗
- 各 phase 耗时（通过 timestamp 差值估算）
- 重试次数
- 失败原因分类

#### 阶段 3 验收标准

- [ ] `demo.sh` 能成功跑完 3 个 case（至少 dry-run）
- [ ] 每个 case 有 JSON 输出 + 结构化日志
- [ ] `scripts/analyze_trace.py` 能解析 trace 输出可读摘要
- [ ] 3 个 case 中 ≥1 个成功（修复通过）
- [ ] `examples/` 下新增 `traces/` 目录含 3 个 trace 文件

**面试能讲多久**：**3 分钟**
- "我在 3 个真实开源项目上测试过——cookiecutter (25k stars)、Textual (36k stars)、FastAPI (99k stars)"
- "这是其中一个 case 的完整推理链（展示 JSON trace）——你可以看到它从 issue 文本提取关键词、搜索代码、定位文件、生成 diff 的全过程"
- "我写了一个 trace 分析脚本，可以自动统计 token 消耗、各阶段耗时、失败原因分布"

---

### 阶段 4：可部署性 — PyPI + Docker（2h）

#### 4.1 PyPI 发布准备（1h）

**文件**：修改 `pyproject.toml`（约 20 行新增）

**具体改动**：

```toml
# pyproject.toml 补充
[project]
# ... 已有内容保持不变 ...
readme = "README.md"
license = {text = "MIT"}
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Software Development :: Bug Tracking",
]
keywords = ["github", "issue", "ai-agent", "automation", "langgraph"]

[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-asyncio",
    "ruff",
    "tiktoken",
]

[project.urls]
Homepage = "https://github.com/FMorgan-111/repopilot"
Repository = "https://github.com/FMorgan-111/repopilot"
Issues = "https://github.com/FMorgan-111/repopilot/issues"
```

```bash
# 发布命令（实际执行需要 PyPI token）
pip install build twine
python -m build
twine upload dist/*
```

**注意**：实际上传 PyPI 需要 token，文档中给出命令，但不强求在计划时间内完成上传。

#### 4.2 Dockerfile（1h）

**文件**：新建 `Dockerfile`（约 40 行）

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# 安装 git（clone repo 需要）
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# 安装依赖
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# 复制源码
COPY src/ src/

# 入口
ENTRYPOINT ["python", "-m", "src.cli"]
```

**同时新建** `.dockerignore`（约 10 行）排除 git、缓存、测试数据等。

**新建** `docker-compose.yml`（可选，约 20 行）方便本地测试。

#### 阶段 4 验收标准

- [ ] `python -m build` 成功构建 wheel
- [ ] `pip install dist/*.whl` 在新 venv 中成功安装并可运行 `repopilot --help`
- [ ] `docker build -t repopilot .` 成功构建镜像
- [ ] `docker run --rm -e GITHUB_TOKEN=... -e LLM_API_KEY=... repopilot https://github.com/org/repo/issues/1 --dry-run` 能运行（至少到网络请求阶段）

**面试能讲多久**：**2 分钟**
- "项目从第一天就考虑了工程化——有 CI、有测试、PyPI 可安装"
- "我还做了 Docker 化，一条 `docker run` 就能在任意环境运行，不需要本地 Python 环境"
- "`pyproject.toml` 里有完整的 metadata、classifiers、依赖分组"

---

### 阶段 5：可演示性 — 录屏 + README 重写（2h）

#### 5.1 README 重写（1.5h）

**文件**：重写 `README.md`（目标 200-300 行）

**新增内容**：
1. **Demo GIF/截图区域**（顶部，替换 emoji 图标）
2. **架构图**（Mermaid/ASCII art，从 `ARCHITECTURE_V2.md` 精简）
3. **完整示例输出**（真实 case 的终端输出粘贴）
4. **技术亮点列表**（LangGraph 状态机、REFLECT 节点、Pydantic 校验、Fallback 引擎、Token Budget）
5. **快速开始** 更新为 `pip install repopilot` 后的实际命令
6. **目录结构** 展示拆分后的模块组织
7. **CI badge** 链接到 GitHub Actions
8. **PyPI badge** 链接到 PyPI 页面

#### 5.2 终端录屏方案（0.5h）

**文件**：新建 `scripts/record_demo.sh`（约 30 行）

使用 `script` 命令或 `asciinema`（如果安装）录制终端会话：

```bash
#!/bin/bash
# 录制 RepoPilot demo
# 需要安装: asciinema (pip install asciinema)

asciinema rec examples/demo.cast \
  -c "GITHUB_TOKEN=$GITHUB_TOKEN LLM_API_KEY=$LLM_API_KEY python -m src.cli https://github.com/cookiecutter/cookiecutter/issues/1973 --dry-run"

# 转换为 GIF（需要 agg 或 svg-term-cli）
# asciinema-agg examples/demo.cast examples/demo.gif
```

**如果不能录屏**，至少准备终端截图的文本版（ANSI 转义码保留颜色）。

#### 阶段 5 验收标准

- [ ] README 包含：架构图、示例输出、技术亮点、目录结构
- [ ] README 的 Quick Start 可以直接复制粘贴运行
- [ ] `scripts/record_demo.sh` 存在且可执行
- [ ] README 视觉上在 GitHub 上看起来"像回事"（badge、排版、代码高亮）

**面试能讲多久**：**1.5 分钟**
- "README 里有完整的架构图和示例输出——面试官不需要读代码就能理解项目做什么"
- "我还录了一个 asciinema demo，不过这里不方便播放，README 里有截图"

---

### 阶段 6：测试 + 异常路径覆盖（2h）

#### 6.1 HTTP mock 异常路径测试（1h）

**文件**：新增 `tests/test_http_client.py`（约 100 行）

覆盖：
- `github_request` 在 503 时重试 3 次后抛出异常
- `github_request` 在 429 时等待后重试
- `llm_request` 在连接超时时重试
- 令牌桶限速器不阻塞正常请求

#### 6.2 Tracer 行为测试（0.5h）

**文件**：修改 `tests/test_tracer.py`（约 50 行新增）

覆盖：
- Tracer 输出到文件
- Tracer 在 logging handler 存在时输出到 logger

#### 6.3 新增测试统计（0.5h）

目标是总测试数达到 50+ 并保持 100% 通过率。

#### 阶段 6 验收标准

- [ ] `pytest tests/ -q --tb=short` 全部通过，总数 ≥50
- [ ] 新增的 HTTP mock 测试不依赖真实网络
- [ ] `ruff check src/ tests/` 无新增警告

**面试能讲多久**：**0.5 分钟**
- "测试从 46 个增加到 50+，覆盖了异常路径——HTTP 重试、限速处理、logging 输出"

---

### 阶段 7（可选）：Layer 2 Memory 简化版（4-8h）

如果时间充裕（40h 预算），实现 MEMORY_DESIGN_V2.md 中的 Layer 1。

**文件**：新建 `src/memory.py` + `tests/test_memory.py`（约 250 行）

**功能**：
- SQLite 存储 per-repo 的 `(repo_full_name, file_path, fix_pattern, success_count, last_used)`
- 下次处理同一 repo 的 issue 时，优先搜索历史上成功改过的文件
- 简单的 LRU 淘汰（最多保留 1000 条记录）

**验收标准**：
- [ ] SQLite 文件在 `~/.repopilot/memory.db` 自动创建
- [ ] 连续处理 2 个同一 repo 的 issue 时，第二个能利用第一个的记忆
- [ ] 测试覆盖：写入、读取、LRU 淘汰、并发安全

**面试能讲多久**：**3 分钟**
- "我设计了一个四层记忆架构——目前实现了 Layer 1：per-repo 的 SQLite 执行历史"
- "agent 在处理同一个仓库的多个 issue 时能记住之前的修复模式，优先搜索历史上成功改过的文件"
- "核心挑战是并发安全性——我分析了 3 个 worker 进程同时写 SQLite 时的死锁和写竞争问题"

---

## 2. 时间预算分配

### 总时间：约 17.5h（阶段 0-6）+ 可选 6h（阶段 7）

| 阶段 | 内容 | 时间 | 累积 | 面试分钟 |
|------|------|------|------|----------|
| 0 | 基线建立 | 0.5h | 0.5h | 0 |
| 1 | HTTP 重试 + 限速 + logging | 4h | 4.5h | 3.0 |
| 2 | 拆文件 + tiktoken | 3h | 7.5h | 2.5 |
| 3 | 真实 trace + demo 脚本 | 2h | 9.5h | 3.0 |
| 4 | PyPI + Docker | 2h | 11.5h | 2.0 |
| 5 | README 重写 + 录屏 | 2h | 13.5h | 1.5 |
| 6 | 测试补充 | 2h | 15.5h | 0.5 |
| 7 | Layer 2 Memory（可选） | 6h | 21.5h | 3.0 |

### 三种预算下的取舍建议

#### 10h 预算（最小可行）

| 做 | 不做 | 理由 |
|----|------|------|
| ✅ 阶段 1（鲁棒性）| 精简到 2.5h：只做 `tenacity` 重试 + logging，跳过令牌桶 | 不重试 = 不可用 |
| ✅ 阶段 3（trace）| 全部 2h | 有真实 data 是最大加分项 |
| ✅ 阶段 5（README）| 全部 2h | 门面不能省 |
| ✅ 阶段 2（拆文件）| 精简到 1.5h：只做 `new_agent.py` 拆分为 3 个文件 | 不要 6 个文件，3 个够了 |
| ✅ 阶段 6（测试）| 精简到 1h：只加 4 个 HTTP mock 测试 | |
| ❌ 阶段 4（PyPI+Docker）| 不做 | 非面试核心 |
| ❌ 阶段 7（Memory）| 不做 | 时间不够 |
| **合计** | **约 9h** | |

**面试总时长**：能讲约 9 分钟（3+3+1.5+0.5+1），核心亮点都有。

#### 20h 预算（推荐）

| 做 | 不做 | 理由 |
|----|------|------|
| ✅ 阶段 0-6 全部 | | 四平八稳，无短板 |
| ❌ 阶段 7（Memory）| 不做 | 留到面试后 |
| **合计** | **约 15.5h** | 余 4.5h buffer |

**面试总时长**：能讲约 12.5 分钟，每个点都有实锤。

#### 40h 预算（豪华版）

| 做 | 额外做 | 理由 |
|----|--------|------|
| ✅ 阶段 0-6 全部 | | |
| ✅ 阶段 7（Memory）| | 面试能多讲 3 分钟 |
| ✅ 额外：改进搜索层 | AST 符号搜索（tree-sitter）+ embedding 语义搜索，约 12h | 差异化杀手锏 |
| ✅ 额外：技术博客 | "Building a Self-Reflective Coding Agent with LangGraph"，约 4h | 扩大影响力 |
| ✅ 额外：给开源项目提 PR | 用 RepoPilot 找到并修复的 bug，手动 review 后提交，约 3h | 有真实用户 |
| **合计** | **约 40.5h** | |

**面试总时长**：能讲约 21 分钟，可以从容挑选亮点来讲。

---

## 3. 面试话术按改动分类

### "面试能讲 30 秒" 的改动

| 改动 | 面试话术 |
|------|----------|
| PyPI 发布 | "项目发布到了 PyPI，`pip install repopilot` 就能用" |
| Dockerfile | "还做了 Docker 化，一行命令就能跑" |
| CI badge | "README 上有 CI badge，测试在 Python 3.10-3.12 上全部通过" |
| ruff lint | "代码通过了 ruff 静态检查" |
| `.dockerignore` | （不会提到） |

### "面试能讲 3 分钟" 的改动

| 改动 | 面试话术素材 |
|------|-------------|
| HTTP 重试 + 限速 | "我用了 `tenacity` 的指数退避重试——对于 429/502/503/504 和网络连接错误，自动重试最多 3 次。同时实现了令牌桶算法做 GitHub API 限速——GitHub 对未认证请求只有 60 req/hour，我的限速器确保不会因为短时间大量请求被封。这些都是生产环境必须考虑的——你不能因为一次网络抖动就让整个问题处理流程崩溃。" |
| 拆文件 | "原始的 agent 逻辑在单个 999 行的文件里。我把它拆成了职责单一的模块——`state.py` 管理状态机模型，`nodes/` 下每个 phase 一个文件，`graph.py` 管理 LangGraph 的构建和路由。这样面试官可以直接打开 `nodes/reflect.py` 看到反思逻辑的完整实现——不到 70 行代码，清晰易懂。" |
| 真实 trace | "我在 3 个真实开源项目上测试过——cookiecutter (25k stars)、Textual (36k stars)、FastAPI (99k stars)。这是其中一个 case 的完整推理链——你可以看到它从 issue 文本中提取关键词、通过 GitHub Code Search 定位到具体文件、用 LLM 生成 unified diff、git apply 后跑 pytest、最后创建 Draft PR。整个链路可追溯——每一步的输入输出都有 JSON trace。虽然目前只修简单 bug 比较稳，但这证明了这个 agent 架构是可行的。" |
| Layer 2 Memory | "我设计了一个四层记忆架构，目前实现了 Layer 1。每处理一个 issue，agent 会把 '哪个文件改了、什么模式、成功与否' 存到本地的 SQLite 里。下次处理同一个仓库的 issue 时，搜索阶段会优先查历史上成功改过的文件路径——这比从零开始搜索 GitHub API 快 10 倍。核心挑战是并发安全性——我分析了多个 worker 进程同时写 SQLite 时的死锁和写竞争问题，最终选择 WAL 模式 + 乐观重试。" |
| Logging 重构 | "我把代码里的 `print()` 全部换成了结构化 logging。trace 不再只是打印到 stdout——它会同时输出到 stderr 的 JSON 日志和文件。这样在生产环境里你可以把日志导入 ELK 或 Grafana，按 trace_id 串联整个请求链路。" |

---

## 4. 最终验收检查清单

完成所有阶段后，运行以下检查：

### 功能验收
- [ ] `pip install repopilot` 成功安装
- [ ] `repopilot --help` 显示帮助信息
- [ ] `repopilot <issue_url> --dry-run` 成功运行（至少到网络请求）
- [ ] `repopilot <issue_url> --json` 输出合法 JSON
- [ ] `demo.sh` 一键运行 3 个 case 输出结果

### 鲁棒性验收
- [ ] 网络断开时 HTTP 请求重试 3 次后优雅失败（不崩溃、不挂起）
- [ ] GitHub API 返回 429 时等待后重试
- [ ] 结构化日志输出到 stderr（不污染 stdout）
- [ ] `examples/traces/` 下有 3 个真实 case 的完整 trace

### 代码质量验收
- [ ] `ruff check src/ tests/` 零错误
- [ ] `pytest tests/ -q` ≥50 tests 全部通过
- [ ] `src/new_agent.py` ≤60 行（thin wrapper）
- [ ] 每个 `src/nodes/*.py` ≤250 行
- [ ] `_estimate_tokens()` 使用 tiktoken（有 fallback）

### 部署验收
- [ ] `python -m build` 成功构建 wheel
- [ ] `docker build -t repopilot .` 成功
- [ ] `docker run --rm repopilot --help` 正常输出

### 文档验收
- [ ] README 包含：架构图、demo 输出、技术亮点、Quick Start
- [ ] README 的 Quick Start 命令可以复制粘贴直接运行
- [ ] GitHub repo 页面显示 CI badge 为绿色
- [ ] `examples/candidate_issues.md` 包含实际运行结果

---

## 5. 附录：当前项目结构 → 目标结构

### 当前结构
```
repopilot/
├── src/
│   ├── __init__.py
│   ├── new_agent.py       # 999 行 — 所有逻辑
│   ├── agent.py            # 99 行 — v1 pipeline
│   ├── agent_loop.py       # 99 行 — v1 agent loop
│   ├── llm.py              # 155 行
│   ├── tools.py            # 65 行
│   ├── tracer.py           # 25 行 — print()
│   ├── schemas.py          # 38 行
│   ├── main.py             # 91 行 — FastAPI
│   └── cli.py              # 87 行
├── tests/                  # 12 个文件
├── docs/                   # 6 份设计文档
├── examples/
├── pyproject.toml
├── README.md
└── .github/workflows/ci.yml
```

### 目标结构（阶段 0-6 后）
```
repopilot/
├── src/
│   ├── __init__.py
│   ├── state.py            # ~130 行 — AgentState + 辅助
│   ├── graph.py            # ~80 行 — build_agent_graph + Fallback
│   ├── new_agent.py        # ~50 行 — thin wrapper (向后兼容)
│   ├── agent.py            # 99 行 — v1 pipeline (保持不变)
│   ├── agent_loop.py       # 99 行 — v1 agent loop (保持不变)
│   ├── llm.py              # ~160 行 — 使用 llm_request 替代裸 httpx
│   ├── tools.py            # ~70 行 — 使用 github_request 替代裸 httpx
│   ├── http_client.py      # ~120 行 — 带重试的 HTTP 客户端（新增）
│   ├── rate_limiter.py     # ~80 行 — 令牌桶限速器（新增）
│   ├── logging_config.py   # ~30 行 — 结构化 logging（新增）
│   ├── tracer.py           # ~60 行 — logging + 文件输出
│   ├── schemas.py          # 38 行 (保持不变)
│   ├── main.py             # 91 行 (保持不变)
│   ├── cli.py              # 87 行 (保持不变)
│   └── nodes/              # 新增目录
│       ├── __init__.py
│       ├── understand.py   # ~80 行
│       ├── locate.py       # ~90 行
│       ├── plan.py         # ~60 行
│       ├── execute.py      # ~140 行
│       ├── verify.py       # ~60 行
│       ├── reflect.py      # ~70 行
│       ├── commit.py       # ~200 行
│       └── failure.py      # ~40 行
├── tests/                  # 14 个文件 (+test_http_client.py, 更新的 test_tracer.py)
├── docs/                   # 7 份文档 (新增 PRODUCTION_PLAN.md)
├── examples/
│   ├── candidate_issues.md # 更新为含实际运行结果
│   └── traces/             # 新增：3 个真实 case 的 JSON trace
├── scripts/
│   ├── demo.sh             # 新增
│   ├── record_demo.sh      # 新增
│   └── analyze_trace.py    # 新增
├── pyproject.toml          # 更新：metadata + tiktoken + tenacity
├── Dockerfile              # 新增
├── .dockerignore           # 新增
├── README.md               # 重写
└── .github/workflows/ci.yml
```
