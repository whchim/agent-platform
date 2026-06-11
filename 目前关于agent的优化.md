# Agent 优化记录

---

## 1. 连续工具失败保护（consecutive_failures）

**问题：** 工具连续失败时，LLM 可能在同一错误上死循环，浪费 token 和时间。

**方案：** 在 LangGraph State 中引入 `consecutive_failures` 计数器，轮次级跟踪。

### 改动点

| 文件 | 位置 | 内容 |
|------|------|------|
| `react_agent.py` | `ReActState` | 新增字段 `consecutive_failures: int` |
| `react_agent.py` | `ReActAgent` | 新增常量 `MAX_CONSECUTIVE_FAILURES = 3` |
| `react_agent.py` | `_tool_node` | 维护计数：本轮任一工具成功 → 归零；本轮全部失败 → +1 |
| `react_agent.py` | `_should_continue` | 新增判断：`consecutive_failures >= 3` → 强制 `end` |
| `react_agent.py` | `run()` | `initial_state` 补 `consecutive_failures: 0` |

### 行为

```
第1轮: tools 全部失败 → failures=1 → 继续循环
第2轮: tools 全部失败 → failures=2 → 继续循环
第3轮: tools 全部失败 → failures=3 → _should_continue → END（强制终止）
中途任一轮有成功 → failures 归零
```

---

## 2. 三态熔断器接入（CircuitBreaker）

**问题：** 同一工具宕机时，LLM 会反复重试，加速系统级联故障。

**方案：** 为每个工具挂一个独立的 `CircuitBreaker` 实例，工具执行前先过熔断器。

### 改动点

| 文件 | 位置 | 内容 |
|------|------|------|
| `react_agent.py` | import | `from app.core.circuit_breaker import CircuitBreaker` |
| `react_agent.py` | `__init__` | 为每个工具创建独立 `CircuitBreaker`（`failure_threshold=3, recovery_timeout=30s`），存入 `self._breakers` |
| `react_agent.py` | `_tool_node` | 用 `cb(tool_fn.invoke)(tool_args)` 包裹工具调用 |

### 熔断器状态机

```
CLOSED（正常）── 连续失败 3 次 ──→ OPEN（熔断，拒绝所有调用）
OPEN ── 等待 30s ──→ HALF_OPEN（放行一个试探请求）
HALF_OPEN ── 成功 → CLOSED
HALF_OPEN ── 失败 → OPEN
```

---

## 3. 失败类型区分

**问题：** 所有工具失败（熔断拒绝 / API 异常 / LLM 编造不存在的工具）统一成一条模糊错误，LLM 无法区分原因，无法有效决策。

**方案：** 自定义 `CircuitBreakerOpenError` 异常 + `_tool_node` 分支处理。

### 改动点

| 文件 | 位置 | 内容 |
|------|------|------|
| `circuit_breaker.py` | 新增 | `class CircuitBreakerOpenError(RuntimeError)` |
| `circuit_breaker.py` | `__call__` 的 wrapper | `raise CircuitBreakerOpenError(...)` 替换泛型 `RuntimeError` |
| `react_agent.py` | import | 加 `CircuitBreakerOpenError` |
| `react_agent.py` | `_tool_node` | 三种分支：`CircuitBreakerOpenError` / `Exception` / 未知工具 |

### LLM 收到的反馈

| 失败原因 | ToolMessage 内容 |
|----------|-----------------|
| 熔断器 OPEN 拒绝调用 | `[熔断] calculator 暂时不可用，请稍后重试或使用其他工具` |
| 工具内部抛异常（API 500、参数错误等） | `[工具异常] xxx` |
| LLM 调用了不存在的工具名 | `[未知工具] xxx` |

LLM 看到 `[熔断]` 知道该换工具，看到 `[工具异常]` 可能修正参数重试。

---

## 4. 兜底话术与失败状态

**问题：** 连续失败被强制终止后，用户看到的可能是 `"(Agent 未产出答案)"` 或 LLM 乱生成的无效回复，且 `status` 永远是 `"success"`，前后端无法区分正常/降级。

**方案：** `run()` 结束时检查 `consecutive_failures`，命中阈值则返回兜底话术 + `status="partial"`。

### 改动点

| 文件 | 位置 | 内容 |
|------|------|------|
| `react_agent.py` | `run()` 的 Step 3 | 连续失败达阈值时，`final_answer` 设为兜底话术，不再从 messages 中提取 |
| `react_agent.py` | `run()` 的 Step 4 | `status` 改为 `"partial"`，`metadata` 中加 `tool_failures` 字段 |

### 用户看到的效果

```json
{
  "status": "partial",
  "answer": "抱歉，工具连续调用失败，我暂时无法完成您的请求，请稍后重试或联系人工客服。",
  "metadata": { "tool_failures": 3, ... }
}
```

---

## 三层防护全景

```
     单次工具调用失败
          │
          ▼
   ┌──────────────────┐
   │ CircuitBreaker   │  ← 工具级：同一工具连续失败 3 次 → OPEN，拒绝调用
   │ (per-tool)       │     30s 后半开试探，成功则恢复
   └──────┬───────────┘
          │ 抛 CircuitBreakerOpenError → LLM 看到 [熔断]
          ▼
   ┌──────────────────┐
   │ consecutive_      │  ← 轮次级：本轮全部工具失败 → +1
   │ failures          │     任一成功 → 归零
   └──────┬───────────┘
          │ 达到 3 → _should_continue 返回 "end"
          ▼
   ┌──────────────────┐
   │ 兜底话术          │  ← 用户级：返回 status="partial" + 兜底文案
   │ status="partial" │
   └──────────────────┘
```

三层机制逐级兜底，各司其职、互不冲突。
