# RepoPilot 四层记忆架构：工业级并发可靠性评估与方案

> 评估者：Claude Opus (claude-opus-4-20250514)
> 日期：2026-06-08
> 版本：v2.0
>
> 本文档对 MEMORY_DESIGN.md v1.0 中的四层记忆架构进行工业级并发场景下的可靠性评估，
> 识别致命问题，并给出三种规模的推荐架构方案。

---

## 1. 评估范围与假设

### 1.1 评估的并发场景

| 场景 | 描述 | 关键挑战 |
|------|------|---------|
| S1 | 2 个请求同时操作同一个 repo | Layer 1 SQLite 并发写冲突 |
| S2 | 3 个 worker 进程同时更新 Layer 2 策略计数 | 全局 SQLite 跨进程写竞争 |
| S3 | 100 个 issue 在 1 小时内涌入 | 高吞吐写压力，fire-and-forget 可靠性 |
| S4 | Layer 0 file_cache 跨请求一致性 | 无共享内存下的缓存策略 |

### 1.2 当前架构基础

- **运行时**：FastAPI + uvicorn，当前单进程，目标支持 3 worker
- **状态管理**：LangGraph StateGraph，AgentState 在进程内存中
- **持久化**：无（当前 v2 完全无持久化，MEMORY_DESIGN.md v1.0 引入 SQLite）
- **依赖**：fastapi, httpx, uvicorn, pydantic, langgraph（无数据库驱动、无消息队列）

---

## 2. 逐层并发压力测试分析

### 2.1 Layer 0：Working Memory

#### 场景 S1：2 个请求同时操作同一个 repo

```
Request A: AgentState_A { file_cache: { "src/auth.py": ... } }
Request B: AgentState_B { file_cache: { "src/auth.py": ... } }
```

**结论：无冲突。** Layer 0 是纯进程内存，每个请求有独立的 AgentState 实例。两个请求各自维护自己的 file_cache 和 conversation_history，互不干扰。这是当前设计最可靠的一层。

#### 场景 S4：file_cache 跨请求一致性

**结论：不适用。** Layer 0 设计为单次请求生命周期，不存在"跨请求一致性"的概念。Redis 备份（可选）引入后可能出现问题（见 3.1），但核心设计安全。

**唯一风险**：LangGraph 的 checkpointer 机制如果被启用，会将 AgentState 序列化到外部存储（如 SQLite/Postgres）。此时两个请求如果共享同一个 thread_id，状态可能互相覆盖。当前代码未使用 checkpointer，风险为零。

### 2.2 Layer 1：Execution Memory（Per-Repo SQLite）

#### 场景 S1：2 个请求同时操作同一个 repo（核心压力场景）

假设两个请求同时处理 `numpy/numpy` 的 issue #12345 和 #12346：

```
时间线：
t0: Request A 完成 LOCATE → fire-and-forget 写 file_index to numpy___numpy.db
t0: Request B 完成 LOCATE → fire-and-forget 写 file_index to numpy___numpy.db
t1: Request A 完成 COMMIT → fire-and-forget 写 issue_record
t1: Request B 完成 COMMIT → fire-and-forget 写 issue_record
```

**SQLite WAL 模式下的行为**：

- 读操作：两个请求可以并发读取（WAL 支持多读一写）
- 写操作：同一时刻只有一个 writer。第二个 writer 尝试获取写锁时：

  ```
  SQLITE_BUSY 返回 → aiosqlite 默认行为取决于 busy_timeout 设置
  ```

**实际会发生什么**：

1. **如果设置了 busy_timeout（如 5000ms）**：SQLite 会等待最多 5 秒后重试。两个写操作最终都会成功，但有延迟。
2. **如果未设置 busy_timeout（默认 0）**：第二个写操作立即收到 SQLITE_BUSY，aiosqlite 抛出 `sqlite3.OperationalError: database is locked`。
3. **fire-and-forget 模式下**：异常被静默吞掉，数据悄无声息丢失。没有重试、没有告警、没有日志。

**file_index 更新的数据一致性问题**：

即使两个写操作都成功（通过 busy_timeout 重试），还存在业务层面的写后丢失（lost update）：

```
Request A 读取 file_index (100 条记录)
Request B 读取 file_index (100 条记录)
Request A 追加 3 条新文件 → 写入 (103 条)
Request B 追加 2 条新文件 → 写入 (102 条)  ← A 的 3 条丢失！
```

这是因为 v1.0 设计中的 `update_file_index` 接口接受的是完整列表（`list[RepoFileEntry]`），不是增量操作。调用方读取整个索引 → 本地修改 → 写回整个索引，这是典型的 read-modify-write 竞争。

**recent_issues 更新**：

如果使用 `INSERT INTO recent_issues ...`（追加操作），单个 INSERT 是原子的，不会丢数据。但如果先 SELECT 再决定是否 UPDATE（如去重），同样存在竞争。

**测试结果预估**：

| 并发数 | 成功率（无 busy_timeout） | 成功率（busy_timeout=5000） |
|--------|--------------------------|----------------------------|
| 2 请求同 repo | ~50%（一个失败） | ~100%（但写入延迟 +5s） |
| 5 请求同 repo | ~20% | ~80%（部分超时） |
| 10 请求同 repo | ~10% | ~40%（严重排队延迟） |

#### 场景 S3：100 个 issue 在 1 小时内涌入

假设 issue 均匀分布在 10 个不同 repo 上，每个 repo 10 个 issue。

- **per-repo 隔离的好处**：10 个 SQLite 文件各自独立，写锁不跨 repo 竞争。
- **瓶颈**：如果 100 个 issue 中 30 个集中在同一个热门 repo，该 repo 的 SQLite 成为瓶颈。30 个写操作在 1 小时内串行化 → 每个写操作平均排队等待 ~2 分钟。
- **fire-and-forget 的另一个问题**：主流程已完成并返回给用户，但 SQLite 写入在后台排队。如果 FastAPI 进程在写入完成前重启，数据永久丢失。

### 2.3 Layer 2：Reflection Memory（Global SQLite）

#### 场景 S2：3 个 worker 进程同时更新策略计数（最致命场景）

这是整个设计中最危险的并发场景。

```
Worker 1: 处理 repo-A issue → 策略 "add_null_check" 成功
Worker 2: 处理 repo-B issue → 策略 "add_null_check" 成功
Worker 3: 处理 repo-C issue → 策略 "add_null_check" 失败

三个 worker 几乎同时调用 record_strategy_outcome("add_null_check", ...)
```

**WAL 模式下的文件系统级行为**：

SQLite WAL 是进程内机制，但跨进程靠的是文件系统锁（POSIX advisory locks）。在一个操作系统进程内（如一个 uvicorn worker），WAL 允许一写多读。但三个独立的 uvicorn worker 进程 = 三个操作系统进程，它们通过同一个 `.db` 文件和 `.db-wal` 文件进行协调。

实际行为：
- 同一时刻只有一个进程能持有写锁
- 另外两个进程会收到 SQLITE_BUSY 或阻塞等待（取决于 busy_timeout）
- 如果设置了 busy_timeout，三个写操作会排队串行执行，每个写入 ~5-20ms
- 串行化本身不会丢数据，但会导致写入延迟

**真正致命的是业务层 read-modify-write 竞争**：

v1.0 设计中的 `record_strategy_outcome` 伪代码：

```python
# 伪代码 —— 这不是原子操作！
async def record_strategy_outcome(self, strategy_name, success, repo, issue_type):
    row = await db.execute("SELECT success_count, failure_count FROM strategies WHERE name=?", (strategy_name,))
    if success:
        new_success = row['success_count'] + 1
    else:
        new_failure = row['failure_count'] + 1
    await db.execute("UPDATE strategies SET success_count=?, failure_count=? WHERE name=?", ...)
```

三个 worker 同时执行：

```
W1: SELECT → success=10, failure=3
W2: SELECT → success=10, failure=3  ← 同时读到旧值
W3: SELECT → success=10, failure=3  ← 同时读到旧值
W1: UPDATE success=11, failure=3
W2: UPDATE success=11, failure=4  ← 覆盖了 W1 的 success=11（应该是 12）
W3: UPDATE success=11, failure=3  ← 覆盖了 W1 和 W2（应该是 12 和 4）
```

**实际丢了 2 次更新。** 贝叶斯 confidence 计算结果错误 → Layer 2 的策略推荐系统输出错误的信心值 → PLAN 阶段推荐了错误的策略 → 整个"边用边学"的核心价值被破坏。

**SQLite 写锁只保证单个 SQL 语句的原子性，不保证 SELECT → application logic → UPDATE 的原子性。**

#### 场景 S3：100 个 issue 涌入

Layer 2 的全局 SQLite 是所有请求的写瓶颈：
- 无论 issue 属于哪个 repo，每个 issue 的 REFLECT 和 COMMIT 阶段都向同一个 `reflections.db` 写入
- 100 个 issue × 2 次写入/issue（策略更新 + 失败模式） = 200 次写入
- 200 次写入串行化在单个 SQLite 文件上 → 即使每次 10ms，总共需要 2 秒纯写入时间
- 加上排队延迟，高峰期写入延迟可达数秒甚至数十秒

### 2.4 Layer 3：Meta Memory（Global SQLite）

#### 场景 S2+S3：用量统计的竞争

`usage_stats` 和 `token_efficiency` 的更新同样存在 read-modify-write 竞争：

```
W1: daily_stats["2026-06-08"].issues_processed += 1
W2: daily_stats["2026-06-08"].issues_processed += 1
→ 可能只增加 1 而不是 2
```

Layer 3 的统计数据如果丢失，影响相对较小（用户不会因为统计不准而修错 bug），但作为"自我优化"层的输入，不准确的统计数据会导致错误的资源分配决策（如动态调整 max_retries）。

### 2.5 跨层交互分析

**Layer 0 → Layer 1 的时间窗口问题**：

v1.0 设计中标注：

> Layer 0（内存）和 Layer 1（SQLite）之间存在时间窗口：LOCATE 完成后内存有 file_cache，
> 但 Layer 1 的 file_index 在异步写入完成前可能不一致。

在并发场景下，这个时间窗口的影响被放大：

1. Request A 完成 LOCATE，file_cache 有 `src/auth.py`，异步写 file_index
2. Request B 开始 LOCATE，读 Layer 1 file_index（还没有 `src/auth.py`）
3. Request B 重新从 GitHub 搜索，浪费 API 调用

这不是数据损坏问题，只是效率损失。但在 100 个 issue 涌入时，这种重复搜索会放大 GitHub API 限流压力。

---

## 3. 致命问题清单（按严重程度排序）

### P0 — 直接导致数据损坏或系统不可用

#### P0-1：Layer 2 策略计数的 read-modify-write 竞争（严重程度：致命）

- **症状**：多 worker 下，策略的 success/failure 计数随机丢失，贝叶斯 confidence 计算错误
- **根因**：SELECT + 应用层计算 + UPDATE 不是原子操作，SQLite WAL 不提供跨语句的事务隔离保证
- **影响**：Layer 2 的核心价值（"边用边学"）被破坏。错误的高 confidence 值导致 PLAN 阶段推荐失败的策略
- **修复方向**：必须改为原子 UPDATE（`UPDATE ... SET success_count = success_count + 1`）或引入乐观锁/事务

#### P0-2：Layer 1 file_index 的覆盖写入（严重程度：致命）

- **症状**：两个并发请求对同一 repo 的 file_index 做全量覆盖，后完成的覆盖先完成的
- **根因**：`update_file_index(files: list[RepoFileEntry])` 是全量替换接口，不是增量更新
- **影响**：文件索引丢失条目 → LOCATE 阶段错过相关文件 → 修复质量下降
- **修复方向**：改为增量 upsert（`INSERT OR REPLACE` 单条），或使用 optimistic locking（version 字段）

#### P0-3：fire-and-forget 写入零容错（严重程度：致命）

- **症状**：SQLITE_BUSY 时写操作静默失败，无重试，无日志，数据永久丢失
- **根因**：设计假设"异步 = 可靠"，但实际上 `asyncio.create_task()` 产生的协程如果抛出异常，只会打印到 stderr，主流程完全不知情
- **影响**：所有标记 `(async)` 的写操作在并发压力下可能随机丢失
- **修复方向**：至少要实现带重试的写入（exponential backoff），捕获异常并记录日志

### P1 — 导致性能严重下降或部分功能失效

#### P1-1：单全局 SQLite 的写吞吐天花板（严重程度：高）

- **症状**：Layer 2 和 Layer 3 共享一个全局 SQLite，所有 worker 的所有写入排队
- **数据**：SQLite 在 SSD 上约 50-100 写事务/秒。100 个 issue/小时平均 ~0.03 写/秒，但 burst 场景（10 个 issue 同时完成 REFLECT）会产生 10 个并发写 → 串行化延迟 ~100-200ms
- **影响**：主流程的异步写入在 burst 场景下显著延迟，可能导致 fire-and-forget task 积压
- **何时到天花板**：当并发请求 > 20-30 时，SQLite 的写排队成为显著瓶颈

#### P1-2：多 worker 无进程间协调（严重程度：高）

- **症状**：3 个 uvicorn worker 各自独立运行 LangGraph，各自独立执行 fire-and-forget 写入
- **根因**：无共享的请求队列、无分布式锁、无 leader 选举
- **影响**：两个 worker 可能同时 clone 同一个 repo 到不同的 temp 目录，浪费磁盘空间；两个 worker 可能同时向同一个 issue 发表 comment；两个 worker 可能创建重复的 branch
- **当前代码已有此问题**：`git_clone` 使用 `tempfile.mkdtemp`，每个请求独立 clone，同一 repo 被 clone 多次

#### P1-3：Layer 0 Redis 备份的一致性陷阱（严重程度：中）

- **v1.0 设计标注**：Redis 备份可选（`key=repopilot:wm:{issue_url_hash}`, TTL=1h）
- **风险**：如果启用，两个 worker 处理同一个 issue URL（用户重复提交），Redis 中的 AgentState 会被覆盖，后启动的请求可能读到前一个的半成品状态
- **当前影响**：Redis 尚未实现，风险为零。但如果按设计实现，需要加入 optimistic locking

### P2 — 影响可扩展性或长期可靠性

#### P2-1：SQLite 文件膨胀与 VACUUM 竞争（严重程度：中）

- v1.0 设计的淘汰策略（每写入后检查 > 10MB → 触发压缩）在并发场景下可能被多个请求同时触发
- 两个请求同时执行 VACUUM → 第二个收到 SQLITE_BUSY → VACUUM 失败 → 文件继续膨胀

#### P2-2：Layer 1 per-repo 隔离的假安全感（严重程度：中）

- per-repo SQLite 文件隔离确实解决了不同 repo 之间的写竞争
- 但同一个热门 repo 的并发请求全部排队在同一个 SQLite 文件上
- 如果某公司 80% 的 issue 集中在 2-3 个核心 repo，per-repo 隔离的价值大打折扣

#### P2-3：Layer 3 数据对自我优化的反馈循环风险（严重程度：低）

- 如果 statistics 因竞争而不准确 → 动态调整 max_retries 错误 → 更多 failure → 进一步污染 statistics
- 这是一个正反馈循环，但需要较长时间和较大数据量才会显现

---

## 4. 工业级方案设计

### 4.1 核心设计原则

1. **写入可靠性 > 写入速度**：宁可在写入时阻塞 50ms，也不能静默丢数据
2. **原子更新**：所有计数器类更新必须使用数据库原子操作（`UPDATE ... SET x = x + 1`），禁止应用层 read-modify-write
3. **写入与请求解耦**：持久化写入不能阻塞 HTTP 响应，但必须有 guaranteed delivery
4. **优雅降级**：写入层故障时，主流程仍能完成（修复 bug 是第一优先级）
5. **可观测性**：所有写入操作必须有成功/失败指标，不能静默失败

### 4.2 推荐的写入架构

```
                        ┌─────────────────┐
                        │   FastAPI App   │
                        │   (Worker × N)  │
                        └────────┬────────┘
                                 │ spawn (non-blocking)
                                 ▼
                        ┌─────────────────┐
                        │  Write Buffer   │  ← 进程内 asyncio.Queue
                        │  (in-memory)    │
                        └────────┬────────┘
                                 │ batch drain
                                 ▼
                        ┌─────────────────┐
                        │  Write Worker   │  ← 单线程，顺序消费
                        │  (per process)  │
                        └────────┬────────┘
                                 │ serial writes
                                 ▼
                        ┌─────────────────┐
                        │  Storage Layer  │  ← SQLite / PostgreSQL
                        └─────────────────┘
```

关键决策：

- **每个 FastAPI worker 进程内有一个 Write Worker 协程**，从 asyncio.Queue 消费写入任务
- HTTP 请求 handler 将写入任务 `put_nowait` 到 queue，立即返回（真正的 fire-and-forget + 背压检测）
- Write Worker 单线程顺序写入，避免了 SQLite 在同一进程内的并发写竞争
- 跨进程的竞争由存储层处理（PostgreSQL 的事务机制，或 SQLite 的 WAL + retry）

### 4.3 各层的具体修复

#### Layer 0：Working Memory — 保持现状，加边界保护

**当前设计已足够安全**（进程内存，请求隔离），需要增加的：

1. **禁用 LangGraph checkpointer 的跨请求共享**：如果未来启用 checkpointer 做状态恢复，必须确保每个请求使用唯一的 thread_id
2. **Redis 备份的 optimistic locking**：如果启用 Redis 备份，写回时检查 version 字段（CAS）

#### Layer 1：Execution Memory — 改全量覆盖为增量 upsert

核心修复：

```sql
-- 旧（全量覆盖，不安全）
DELETE FROM file_index WHERE repo_id = ?;
INSERT INTO file_index ... (多次)

-- 新（增量 upsert，安全）
INSERT INTO file_index (repo_id, path, topics, fix_count, ...)
VALUES (?, ?, ?, ?, ...)
ON CONFLICT(repo_id, path) DO UPDATE SET
    topics = excluded.topics,
    fix_count = file_index.fix_count + 1,
    last_modified_issue = excluded.last_modified_issue,
    relevance_persistence = 1.0;
```

- 将全量替换接口改为单条 upsert 接口
- 每次 LOCATE 完成后，只 upsert 新发现的文件，不影响已有条目
- `recent_issues` 使用纯 INSERT（追加），不读不改现有数据
- `test_patterns` 使用 `INSERT OR REPLACE`（整个 record 替换是可接受的，因为低频更新）

#### Layer 2：Reflection Memory — 原子计数器 + 写入队列

策略计数的核心修复：

```sql
-- 改用原子更新，消除 read-modify-write 竞争
UPDATE strategy_catalog
SET success_count = success_count + 1,
    confidence = CAST(success_count + 1 AS REAL) / CAST(success_count + failure_count + 2 AS REAL)
WHERE strategy_name = ?;
```

- 所有计数类更新使用 `UPDATE ... SET col = col + 1`
- 跨进程竞争完全委托给数据库引擎的事务机制
- 如果使用 SQLite：WAL + busy_timeout=5000 + `BEGIN IMMEDIATE` 事务
- 如果使用 PostgreSQL：利用其成熟的 MVCC 和行级锁

#### Layer 3：Meta Memory — 同 Layer 2

所有统计更新改为原子 SQL。`daily_stats` 使用 upsert：

```sql
INSERT INTO daily_stats (date, issues_processed, tokens_consumed, ...)
VALUES (?, 1, ?, ...)
ON CONFLICT(date) DO UPDATE SET
    issues_processed = daily_stats.issues_processed + 1,
    tokens_consumed = daily_stats.tokens_consumed + excluded.tokens_consumed,
    ...
```

---

## 5. 三种规模下的推荐架构方案

### 5.1 轻量级方案：单用户 WSL 环境

**适用场景**：
- 1 个开发者，串行使用（或偶尔 2-3 个 issue 同时处理）
- 单进程 FastAPI（`uvicorn --workers 1`）
- 不关心高可用，允许偶尔重启丢数据

**架构**：

```
┌─────────────────────────────────────┐
│         FastAPI (1 worker)          │
│  ┌───────────────────────────────┐  │
│  │     LangGraph AgentState      │  │
│  │  Layer 0: Working Memory      │  │
│  └───────────────────────────────┘  │
│              │                      │
│              ▼ (async, with retry)  │
│  ┌───────────────────────────────┐  │
│  │     SQLite WAL mode           │  │
│  │  Layer 1: per-repo .db        │  │
│  │  Layer 2: reflections.db      │  │
│  │  Layer 3: meta.db             │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

**与 v1.0 设计的差异**：

| 项目 | v1.0 设计 | 轻量级方案 |
|------|----------|-----------|
| SQLite 连接 | aiosqlite 默认 | WAL + busy_timeout=5000 |
| Layer 1 写入 | 全量覆盖 file_index | 增量 upsert（单条 INSERT OR REPLACE） |
| Layer 2 计数 | SELECT + 计算 + UPDATE | 原子 `UPDATE SET x=x+1` |
| 写入可靠性 | fire-and-forget，无重试 | async task + 3 次 retry + error log |
| 多 worker | 未考虑 | 强制 single worker |
| Layer 0 Redis | 可选 | 删除（单 worker 不需要） |

**成本**：零新增依赖。改动 ~200 行代码（主要是 UPSERT 语句和 retry 逻辑）。

**限制**：
- 不支持并发写入同一个 repo（通过 per-repo asyncio.Lock 排队，延迟 < 100ms 可接受）
- 不支持多 worker
- 进程重启丢失未完成的 fire-and-forget task（但重试 + 日志已保证不静默丢失）

### 5.2 中型方案：5-10 人团队，单机多 Worker

**适用场景**：
- 5-10 个开发者，同一时刻 2-5 个并发请求
- 单台机器，3 个 uvicorn worker
- 同一 repo 偶有并发（2-3 个请求）
- 可接受分钟级的短暂不可用

**架构**：

```
┌──────────────────────────────────────────────────────────────┐
│                       单机 (Linux)                            │
│                                                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│  │ Worker 1 │  │ Worker 2 │  │ Worker 3 │   ← uvicorn       │
│  │ + L0 WM  │  │ + L0 WM  │  │ + L0 WM  │                   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                   │
│       │             │             │                           │
│       │  ┌──────────┴──────────┐  │                           │
│       │  │    Redis (local)    │  │   ← 写队列 + 缓存        │
│       │  │  - Write Queue      │  │                           │
│       │  │  - Per-Repo Locks   │  │                           │
│       │  │  - Hot File Cache   │  │                           │
│       │  └──────────┬──────────┘  │                           │
│       │             │             │                           │
│       ▼             ▼             ▼                           │
│  ┌──────────────────────────────────────┐                     │
│  │       PostgreSQL (local)             │                     │
│  │  Layer 1: repo_memory schema         │   ← 多 worker      │
│  │  Layer 2: reflections schema         │     共享存储       │
│  │  Layer 3: meta_memory schema         │                     │
│  └──────────────────────────────────────┘                     │
│                                                               │
│  ┌──────────────────────────────────────┐                     │
│  │    Background Write Consumer         │   ← 独立进程       │
│  │    (消费 Redis 写队列 → PG)          │                     │
│  └──────────────────────────────────────┘                     │
└──────────────────────────────────────────────────────────────┘
```

**核心变更**：

1. **Layer 1/2/3 从 SQLite 迁移到 PostgreSQL**
   - 原因：多 worker 跨进程共享 SQLite 的最大问题是全局写锁。PG 的行级锁 + MVCC 天然解决了并发写竞争，不需要应用层加锁。
   - 但 **Layer 1 per-repo 隔离的文件级隔离优势丧失**。替代方案：
     - PG 中为每个 repo 建立独立的 schema（`repo_numpy`, `repo_pandas`），获得逻辑隔离
     - 或使用 PG 的 row-level security (RLS) 做软隔离
     - 实际上，团队内部共享一个 PG 实例没有隐私风险，简单的表 + owner/repo 列就足够

2. **引入 Redis 做三件事**
   - **写队列**：HTTP handler 把写入任务 LPUSH 到 Redis List，由独立 consumer 进程 RPOP 消费写入 PG。解耦 HTTP 请求和 DB 写入的生命周期。
   - **per-repo 分布式锁**：`SET repopilot:lock:{owner}/{repo} worker_id NX EX 30`，防止两个 worker 同时向同一 repo 开 branch / 发表 comment（这个问题在当前代码中就存在，不是 memory 引入的）。
   - **热点文件缓存**：`repopilot:file:{owner}/{repo}:{path}` → content，TTL 5 分钟。减少 GitHub API 调用。

3. **写入可靠性保障**
   - Redis List 持久化（AOF + RDB），consumer 进程确认写入 PG 后才 RPOP
   - 如果 consumer 进程挂了，Redis 中的写入任务不会丢失（重启后继续消费）
   - HTTP 请求不再等待写入完成（只等 LPUSH 到 Redis，~1ms），真正 decouple

4. **Layer 0 保持不变**
   - 仍在进程内存中，请求间隔离
   - 不引入 Redis 备份（简化设计，中型方案不需要）

**数据库 Schema 变更（PostgreSQL）**：

```sql
-- Layer 1: 仓库执行记忆（PG 中单表 + per-repo partition）
CREATE TABLE repo_file_index (
    id BIGSERIAL,
    owner TEXT NOT NULL,
    repo TEXT NOT NULL,
    file_path TEXT NOT NULL,
    topics TEXT[],           -- PG array
    fix_count INT DEFAULT 1,
    last_modified_issue TEXT,
    relevance_persistence REAL DEFAULT 1.0,
    last_accessed TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (owner, repo, file_path)
);

CREATE TABLE repo_issue_log (
    id BIGSERIAL,
    owner TEXT NOT NULL,
    repo TEXT NOT NULL,
    issue_number INT NOT NULL,
    issue_type TEXT,
    files_modified TEXT[],
    fix_strategy TEXT,
    success BOOL,
    tokens_consumed INT,
    completed_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (owner, repo, issue_number)
);

-- Layer 2: 全局反思记忆
CREATE TABLE strategy_catalog (
    strategy_id UUID DEFAULT gen_random_uuid(),
    strategy_name TEXT UNIQUE NOT NULL,
    success_count INT DEFAULT 0,
    failure_count INT DEFAULT 0,
    confidence REAL DEFAULT 0.5,
    -- ... 其他字段
    PRIMARY KEY (strategy_id)
);

-- 原子更新（PG 原生支持）
UPDATE strategy_catalog
SET success_count = success_count + 1,
    last_used = NOW()
WHERE strategy_name = $1;
```

**成本**：

- 新增依赖：`redis`, `asyncpg`（或 `psycopg`）, PostgreSQL
- 需要安装和配置 Redis + PostgreSQL（可以用 docker-compose 简化）
- 新增模块：
  - `WriteBuffer`（asyncio.Queue + consumer coroutine，约 100 行）
  - `DistributedRepoLock`（Redis 锁，约 50 行）
  - `HotFileCache`（Redis 缓存层，约 80 行）
- 数据库迁移脚本（约 100 行 SQL）
- 改动量：约 500-800 行新增代码 + 200 行修改

**此方案的天花板**：

- Redis 是单线程但吞吐足够（10K writes/sec），中型方案不会碰到瓶颈
- PostgreSQL 在单机上的写入吞吐约 1000-5000 TPS，足够 5-10 人团队
- 真正的上限在 GitHub API rate limit（5000 req/hour），而不是 RepoPilot 自身

### 5.3 工业级方案：多实例，100+ 并发

**适用场景**：
- 30+ 开发者，100+ 并发请求
- 多台机器部署（3-5 个 FastAPI 实例）
- CI 自动触发 + 手动提交混合
- 需要高可用、可观测性、多租户隔离

**架构**：

```
                          ┌──────────────┐
                          │   Nginx LB   │
                          └──────┬───────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
  ┌───────▼────────┐   ┌────────▼───────┐   ┌─────────▼──────┐
  │  Instance A    │   │  Instance B    │   │  Instance C    │
  │  FastAPI × 4   │   │  FastAPI × 4   │   │  FastAPI × 4   │
  │  Layer 0: WM   │   │  Layer 0: WM   │   │  Layer 0: WM   │
  └───────┬────────┘   └────────┬───────┘   └─────────┬──────┘
          │                     │                      │
          │  ┌──────────────────┴──────────────────┐   │
          │  │        Redis Cluster                │   │
          │  │  - Distributed Locks (Redlock)     │   │
          │  │  - Hot Cache (LRU, ~10GB)          │   │
          │  │  - Rate Limiter                    │   │
          │  └──────────────────┬──────────────────┘   │
          │                     │                      │
          │  ┌──────────────────┴──────────────────┐   │
          │  │     RabbitMQ / NATS / Kafka         │   │
          │  │  - Write Events: repo.file_indexed  │   │
          │  │  - Write Events: strategy.updated   │   │
          │  │  - Write Events: meta.usage_report  │   │
          │  └──────────────────┬──────────────────┘   │
          │                     │                      │
          └─────────────────────┼──────────────────────┘
                                │
          ┌─────────────────────┴──────────────────────┐
          │          PostgreSQL (HA: Patroni)           │
          │  Per-repo partitions + connection pooling   │
          │  Read replicas for Layer 2/3 queries        │
          └─────────────────────┬──────────────────────┘
                                │
          ┌─────────────────────┴──────────────────────┐
          │         Write Consumers (3 instances)       │
          │  消费 MQ 事件 → 合并写入 → PG               │
          │  去重 + 幂等 + 批量 upsert                  │
          └────────────────────────────────────────────┘
```

**相对于中型方案的关键差异**：

1. **消息队列替代 Redis List**
   - Redis List 适合单机 consumer。多实例部署使用 RabbitMQ/NATS/Kafka：
     - 消息持久化（磁盘存储，不丢）
     - Consumer group（多个 consumer 并行消费，自动负载均衡）
     - 消息去重（基于 trace_id）
     - 死信队列（写入 PG 失败的消息进入 DLQ，人工/自动处理）

2. **写入幂等性保证**
   - 每个写入事件携带 `trace_id` + `event_id`
   - Consumer 侧做去重：`INSERT ... ON CONFLICT (trace_id, event_id) DO NOTHING`
   - 支持 at-least-once 语义（允许重复消费，通过去重保证 exactly-once 效果）

3. **Layer 1 合并写入（Coalescing Writes）**
   - 同一个 repo 的高频 file_index 更新在 consumer 侧合并
   - 例如：10 个 issue 对 `numpy/numpy` 的 file_index 更新，consumer 在 1 秒时间窗口内收集，合并为一次批量 upsert
   - 大幅减少 PG 写入次数

4. **多租户隔离**
   - 每个团队的数据库在 PG 中独立的 database（而非 schema）
   - 或使用 Citus 做分布式 PG，按 tenant_id shard
   - 绝对的数据隔离，满足企业安全审计

5. **可观测性**
   - 写队列深度 → Prometheus metric `repopilot_write_queue_depth`
   - 写延迟 → `repopilot_write_latency_seconds` histogram
   - 写失败率 → `repopilot_write_failure_total` counter
   - Layer 2 策略 confidence 漂移检测 → 异常告警
   - GitHub API rate limit 剩余 → Grafana dashboard

6. **Layer 2 语义检索（Phase 4 升级）**
   - 向量数据库（pgvector 扩展，或 Qdrant sidecar）
   - 策略检索不再靠条件匹配，而是 embedding similarity
   - 解决 v1.0 中策略碎片化和跨语言泛化失败的问题

**成本**：

- 基础设施：PostgreSQL HA (Patroni), Redis Cluster, RabbitMQ/NATS, pgvector
- 运维：Kubernetes + Helm charts，或 docker-compose + systemd
- 新增模块：
  - `WriteEventEmitter`（发布事件到 MQ，约 150 行）
  - `WriteConsumer`（消费 + 合并 + 去重 + PG 写入，约 400 行）
  - `RateLimiter`（分布式限流，约 100 行）
  - `MetricsCollector`（Prometheus 指标，约 200 行）
- 改动量：约 1500-2500 行新增代码

---

## 6. 迁移代价评估：当前架构 → 中型方案

### 6.1 可保持不变的模块

| 模块 | 原因 |
|------|------|
| `src/tools.py` | 纯 GitHub API 封装，仅依赖 httpx。无需修改。 |
| `src/llm.py` | LLM 调用封装。无需修改。 |
| `src/schemas.py` | Pydantic 模型。可能需要增加 Memory 相关模型，但现有模型不变。 |
| `src/tracer.py` | JSONL 日志输出。无需修改。 |
| `src/agent.py` | 旧版线性 agent。标记废弃，不再维护。 |
| `src/agent_loop.py` | 旧版 agent loop。标记废弃，不再维护。 |
| `scripts/` | 数据采集脚本。无需修改。 |
| `tests/` | 大部分测试可保持不变。需要新增 Memory 层的单元/集成测试。 |

### 6.2 需要重构的模块

#### `src/new_agent.py` — 中度重构（核心改动）

| 组件 | 改动程度 | 描述 |
|------|---------|------|
| `AgentState` | 重构 | 拆分为嵌套结构（WorkingMemory 内嵌），增加 memory 引用字段 |
| `WorkingMemory` | 新建 | 从 AgentState 提取 conversation/file_cache/attempts 到独立 model |
| `understand_issue` | 微调 | 无 memory 读写，基本不变 |
| `locate_code` | 修改 | 增加 Layer 1 file_index 读取（优先查缓存，减少 GitHub API 调用）；增加写入请求 enqueue |
| `plan_fix` | 修改 | 增加 conversation_context 注入（修复 P0 bug）；增加 Layer 2 策略检索 |
| `execute_fix` | 修改 | 增加 Layer 1 test_patterns 读取；增加 per-repo 分布式锁（防止 branch 冲突） |
| `verify_fix` | 微调 | 逻辑基本不变 |
| `reflect_on_failure` | 修改 | reflection_notes 从 str 改为 list[ReflectionNote]；增加 Layer 2 failure_pattern 匹配 |
| `commit_fix` | 修改 | 增加 Layer 1/2/3 写入请求 enqueue；增加 PR 创建前的分布式锁 |
| `handle_failure` | 修改 | 增加 Layer 1/2/3 写入请求 enqueue |
| `build_agent_graph` | 微调 | 基本不变 |
| `agent_v2` | 修改 | 创建 WriteBuffer 实例，传入 node 函数 |
| `git_clone` | 微调 | 增加本地 clone 缓存（同 repo 复用 clone，减少网络开销） |

**预估总改动量**：~400 行修改 + ~300 行新增（WorkingMemory model, write enqueue, 策略检索）

#### `src/main.py` — 轻度修改

| 改动 | 描述 |
|------|------|
| 启动时初始化连接 | 创建 PG 连接池、Redis 客户端、WriteBuffer |
| 依赖注入 | 将 WriteBuffer/Store 实例传入 agent_v2 |
| Graceful shutdown | 等待 WriteBuffer 排空 + 关闭连接 |

**预估改动量**：~50 行

### 6.3 需要新建的模块

| 模块 | 描述 | 预估行数 |
|------|------|---------|
| `src/memory/__init__.py` | Memory 层 package 入口 | 10 |
| `src/memory/models.py` | WorkingMemory, ReflectionNote, RepoMemory 等 Pydantic model | 150 |
| `src/memory/repo_store.py` | Layer 1: RepoMemoryStore (PG CRUD + 增量 upsert) | 200 |
| `src/memory/reflection_store.py` | Layer 2: ReflectionStore (策略原子更新 + 失败模式) | 180 |
| `src/memory/meta_store.py` | Layer 3: MetaStore (统计原子更新) | 120 |
| `src/memory/write_buffer.py` | WriteBuffer (asyncio.Queue + consumer + retry) | 100 |
| `src/memory/cache.py` | Redis 热点文件缓存 + per-repo 分布式锁 | 120 |
| `migrations/001_init.sql` | PG schema 初始化 | 100 |
| `docker-compose.yml` | 本地开发 Redis + PG | 30 |

**总新增代码量**：约 1000 行

### 6.4 迁移执行计划

**Phase 1（1-2 天）**：先修复 P0 bug，不引入新基础设施

1. `plan_fix` 注入 conversation_context（10 行改动）
2. `reflection_notes`: str → list[ReflectionNote]（30 行改动）
3. AgentState 拆分，WorkingMemory 模型（80 行改动）
4. 所有 Layer 1/2/3 写入改为原子 SQL（即使用 SQLite 也是原子 UPDATE）
5. 写入增加 retry 逻辑（exponential backoff, max 3 retries）
6. 不引入 PG/Redis，继续用 SQLite

**Phase 2（3-5 天）**：引入 PG + Redis，部署中型方案

1. 安装 PostgreSQL + Redis（docker-compose）
2. 运行 migration 创建 schema
3. 实现 `repo_store.py` + `reflection_store.py` + `meta_store.py`
4. 实现 `write_buffer.py` + `cache.py`
5. 修改 `new_agent.py` 中的各节点函数，集成 memory 读写
6. 集成测试：模拟并发写入 PG，验证原子性和数据正确性
7. 部署到团队机器，观察 1 周

**Phase 3（可选，1-2 周）**：语义检索升级

1. 安装 pgvector 扩展
2. 对 strategy 生成 embedding（调用 embedding API）
3. 替换条件匹配为语义检索

---

## 7. 设计决策记录（相对于 v1.0 的变更）

### 决策 1：为什么中型方案必须迁移到 PostgreSQL？

SQLite WAL 在单进程多协程场景表现良好（一写多读）。但在多进程场景：
- 写操作在 OS 级别串行化（POSIX 文件锁） → 3 worker 的写入吞吐不会超过单 worker
- 没有行级锁 → 即使使用原子 UPDATE，也要等写锁释放
- 没有连接池 → 每个 worker 独立打开文件，增加文件系统压力

PostgreSQL 的 MVCC + 行级锁 + 连接池，是中型方案的最低基础设施门槛。代价是增加了运维复杂度（安装、备份、监控），但对于 5-10 人团队，这个代价是可接受的。

### 决策 2：为什么轻量级方案保留 SQLite 而不是直接上 PG？

对于单用户场景：
- SQLite 零运维（`pip install aiosqlite` 即用）
- WAL 模式对单进程串行写入足够
- 没有多 worker 的跨进程竞争问题
- 数据文件可以用 rsync/git 备份

在轻量级方案中引入 PG 是过度设计。但需要在代码中保持 PG 和 SQLite 的抽象接口一致（Repository 模式），以便未来平滑升级。

### 决策 3：为什么要引入消息队列而不是直接用 PG LISTEN/NOTIFY？

中型方案中用 Redis List 做写队列是因为 Redis 已经存在（做缓存和锁），不需要额外基础设施。工业级方案中引入 RabbitMQ/NATS 是为了：
- Consumer group 支持多个 consumer 并行消费（Redis List 是单 consumer）
- 消息持久化保证（Redis RDB/AOF 有数据丢失窗口）
- 死信队列（写入 PG 永久失败的消息不会阻塞队列）

PG 的 LISTEN/NOTIFY 不适合做写队列：没有持久化（通知期间不在线的 consumer 会丢失消息），没有重试机制，没有死信。

### 决策 4：Layer 1 per-repo 隔离在 PG 中如何保持？

PostgreSQL 本身不支持"每个 repo 一个数据库"（那会创建数百个 database，连接管理灾难）。替代方案：
- **Oracle/MySQL style**：单 database，按 `(owner, repo)` 复合主键分区
- **PG Partitioning**：按 `owner_repo_hash` 做 HASH 分区（几十个分区足够）
- **RLS**：如果未来需要严格的多租户隔离，启用 Row Level Security

对于中型方案的 5-10 人团队，简单的复合主键 + 索引即可，不需要分区。

---

## 8. 总结

### v1.0 设计在工业并发场景下的总体评估

v1.0 的设计在**单进程、串行请求**的场景下是可用的。设计者对 SQLite WAL 和 fire-and-forget 的理解基本正确，但低估了多进程场景下的竞争问题。

三个致命缺陷：
1. **Layer 2 策略计数的 read-modify-write 竞争**（跨进程必然丢更新）
2. **Layer 1 file_index 的全量覆盖接口**（同 repo 并发必然丢数据）
3. **fire-and-forget 零容错**（SQLITE_BUSY 时静默丢数据，无重试无日志）

这三个问题不是"在极端并发下才出现"，而是在**中等并发（2-3 个请求同时操作同一 repo，或 3 个 worker 同时运行）就已经存在**。

### 推荐路径

| 阶段 | 方案 | 适用场景 | 核心改动 |
|------|------|---------|---------|
| 立即 | Phase 1 修复 | 当前的 WSL 环境 | 原子 SQL + retry，不改基础设施 |
| 1 周内 | 轻量级方案 | 单用户稳定使用 | + WriteBuffer + asyncio.Lock |
| 1 月内 | 中型方案 | 5-10 人团队部署 | + PG + Redis |
| 长期 | 工业级方案 | 多实例 100+ 并发 | + MQ + HA + 可观测性 |

**核心原则**：先修 bug，再扩展规模。v1.0 的 P0 问题可以在不改基础设施的情况下修复（Phase 1），让系统在单用户场景稳定运行，然后用 1-2 周平滑迁移到中型方案满足团队需求。

---

*本文档与 /docs/MEMORY_DESIGN.md v1.0 配套阅读。v1.0 描述了"要做什么"，v2.0 描述了"怎么做才可靠"。*
