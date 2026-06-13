# pi 架构分析

[earendil-works/pi](https://github.com/earendil-works/pi) 是一个生产级的 AI 编程 Agent，TypeScript 实现，monorepo 结构。本文档记录了对其架构的深入分析，作为用 Python 重写 core-agent 的参考。

---

## 项目结构（4 层架构）

| 包 | 路径 | 职责 |
|---|------|------|
| `@earendil-works/pi-ai` | `packages/ai/` | 多 Provider 统一 LLM API（OpenAI, Anthropic, Google, Bedrock 等 18+ 个 Provider） |
| `@earendil-works/pi-agent-core` | `packages/agent/` | Agent 运行时：事件循环、工具调用、状态管理、流式事件 |
| `@earendil-works/pi-coding-agent` | `packages/coding-agent/` | 交互式编程 Agent CLI（产品层） |
| `@earendil-works/pi-tui` | `packages/tui/` | 终端 UI 库（增量渲染、组件系统） |

---

## Layer 1：pi-ai — 多 Provider LLM API

### Provider 注册机制

```typescript
// api-registry.ts
registerApiProvider({ api: "anthropic-messages", stream, streamSimple })

// stream.ts
function streamSimple(model, context) {
  const provider = apiRegistry[model.api]
  return provider.streamSimple(model, context)
}
```

**关键设计：**
- Provider 通过 `registerApiProvider()` 注册，映射 `Api` 字符串到 `StreamFunction`
- `stream()` — 原始流，支持 Provider 特有参数
- `streamSimple()` — 统一接口，Agent 层不感知 Provider 差异
- Provider 实现完全不抛异常 —— 错误编码在事件流中

### EventStream 协议

```typescript
class EventStream<T, R> {
  push(event: T): void           // Producer 推送事件
  [Symbol.asyncIterator]()       // Consumer 消费事件
  result(): Promise<R>           // 最终结果
  end(): void                    // 结束流
}
```

事件类型（13 种）：`start`, `text_delta`, `toolcall_delta`, `toolcall_args_done`, `done`, `error`, `aborted`...

**关键设计：**
- 基于 async queue + callback 的生产者-消费者模式
- 支持 backpressure
- 最终结果独立于事件流单独获取

---

## Layer 2：pi-agent-core — Agent 运行时

### 双层事件循环

```
外层循环 (while true):
  - 检查是否有 follow-up 消息（agent 即将停止时注入）
  - 有则继续

  内层循环 (hasMoreToolCalls || pendingMessages):
    - 轮询 steering 消息并注入到上下文
    - 调用 LLM (streamAssistantResponse)
    - 提取 tool_use 块
    - 执行工具（并行或串行模式）
    - 将工具结果追加到上下文
    - emit turn_end 事件
    - 调用 shouldStopAfterTurn() hook
    - 调用 prepareNextTurn() hook
    - 轮询 steering 消息
```

### AgentMessage 抽象

```typescript
// 自定义消息类型通过 declaration merging 注入
declare module "@earendil-works/pi-agent-core" {
  interface CustomAgentMessages {
    artifact: ArtifactMessage
  }
}
```

Agent 循环内部使用 `AgentMessage[]`，到 LLM 边界才转换为 `Message[]`。

### Hook 系统

| Hook | 触发时机 | 用途 |
|------|---------|------|
| `beforeToolCall` | 工具执行前 | 阻止/修改工具调用 |
| `afterToolCall` | 工具执行后 | 修改工具结果 |
| `shouldStopAfterTurn` | 每轮结束 | 优雅停止 |
| `prepareNextTurn` | 下一轮开始前 | 切换模型/清理上下文 |
| `transformContext` | 上下文发送前 | 修剪消息 |
| `getApiKey` | 需要认证时 | 动态获取 API key |

### 工具执行模式

- **sequential**：逐个 prepare → execute → finalize
- **parallel**：prepare 串行，execute 并发，结果按完成顺序 emit

### 消息队列

- **steering 消息**：下一轮边界注入，用于实时干预
- **follow-up 消息**：Agent 即将停止时注入，用于后台任务

---

## Layer 3：pi-coding-agent — CLI 产品

### 核心组件

| 文件 | 行数 | 职责 |
|------|------|------|
| `main.ts` | ~550 | CLI 入口、模式分发、会话管理 |
| `agent-session.ts` | ~3100 | 中心类：工具注册、会话持久化、压缩、重试 |
| `sdk.ts` | ~200 | 编程 API |
| `settings-manager.ts` | ~300 | 全局/项目级配置 |
| `model-registry.ts` | ~200 | 模型注册和认证 |

### 运行模式

- **interactive**：全功能 TUI（聊天、编辑器、Markdown 渲染）
- **print**：单次执行（`pi -p "prompt"`）
- **rpc**：JSON-RPC（IDE 集成）

### 工具系统

每个工具 = `name` + `description` + `parameters`（TypeBox JSON Schema）+ `execute()` + `label`

支持可插拔操作接口（如 `BashOperations` 可替换为 SSH / Docker 执行）。

---

## 关键模式总结

| 模式 | pi 实现 | 我们的实现 |
|------|---------|-----------|
| **Provider 注册** | `registerApiProvider()` | `providers.py` `PROVIDERS` dict |
| **事件流** | `EventStream<T, R>` async iterable | `Generator[StreamEvent]` |
| **Agent 事件** | `agent_start/turn_start/message_*/tool_*/turn_end/agent_end` | `AgentEvent(text_delta/tool_call/tool_result/done)` |
| **错误处理** | 错误编码在事件流中，不抛异常 | `StreamEvent(type="error")` |
| **Hook 系统** | 6 种 hooks，在循环中注入 | 暂无 |
| **并行执行** | 工具并发执行 | 串行 |
| **消息队列** | steering + follow-up 双队列 | 暂无 |
| **会话持久化** | 自动保存（每个事件后） | 暂无 |

---

## Python 重写的关键挑战

1. **async generator vs sync generator** — TypeScript 的 `for await...of` 对应 Python 的 `async for`。Python 需要 `asyncio` 来实现真正的异步流。
2. **类型系统** — TypeScript 的 discriminated union 在 Python 中用 `Literal` + `dataclass` 模拟。
3. **declaration merging** — TypeScript 特有的特性，Python 中需要用注册表或插件机制替代。
4. **并发工具执行** — Python 的 `asyncio.gather()` 可以替代 TypeScript 的 `Promise.all()`。
