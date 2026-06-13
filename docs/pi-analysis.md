# pi Architecture Analysis

[earendil-works/pi](https://github.com/earendil-works/pi) is a production-grade AI coding agent in TypeScript, organized as a monorepo. This document captures a deep analysis of its architecture as reference for a Python reimplementation of the core-agent layer.

---

## Project Structure (4-layer architecture)

| Package | Path | Purpose |
|---------|------|---------|
| `@earendil-works/pi-ai` | `packages/ai/` | Unified multi-provider LLM API (OpenAI, Anthropic, Google, Bedrock — 18+ providers) |
| `@earendil-works/pi-agent-core` | `packages/agent/` | Agent runtime: event loop, tool calling, state management, streaming events |
| `@earendil-works/pi-coding-agent` | `packages/coding-agent/` | Interactive coding agent CLI (the product layer) |
| `@earendil-works/pi-tui` | `packages/tui/` | Terminal UI library (differential rendering, component system) |

---

## Layer 1: pi-ai — Multi-Provider LLM API

### Provider Registry

```typescript
// api-registry.ts
registerApiProvider({ api: "anthropic-messages", stream, streamSimple })

// stream.ts
function streamSimple(model, context) {
  const provider = apiRegistry[model.api]
  return provider.streamSimple(model, context)
}
```

**Key design decisions:**
- Providers register via `registerApiProvider()`, mapping an `Api` string to a `StreamFunction`
- `stream()` — raw stream, supports provider-specific options
- `streamSimple()` — unified interface, agent layer is provider-agnostic
- Provider implementations never throw — errors are encoded in the event stream

### EventStream Protocol

```typescript
class EventStream<T, R> {
  push(event: T): void           // Producer pushes events
  [Symbol.asyncIterator]()       // Consumer consumes events
  result(): Promise<R>           // Final result
  end(): void                    // End the stream
}
```

13 event types: `start`, `text_delta`, `toolcall_delta`, `toolcall_args_done`, `done`, `error`, `aborted`...

**Key design decisions:**
- Producer-consumer pattern based on async queue + callbacks
- Supports backpressure
- Final result is independent from the event stream

---

## Layer 2: pi-agent-core — Agent Runtime

### Two-level Event Loop

```
Outer loop (while true):
  - Check for follow-up messages (injected when agent would stop)
  - Continue if present

  Inner loop (hasMoreToolCalls || pendingMessages):
    - Poll for steering messages and inject into context
    - Call LLM (streamAssistantResponse)
    - Extract tool_use blocks
    - Execute tools (parallel or sequential mode)
    - Append tool results to context
    - Emit turn_end event
    - Call shouldStopAfterTurn() hook
    - Call prepareNextTurn() hook
    - Poll for steering messages
```

### AgentMessage Abstraction

```typescript
// Custom message types injected via declaration merging
declare module "@earendil-works/pi-agent-core" {
  interface CustomAgentMessages {
    artifact: ArtifactMessage
  }
}
```

The agent loop works with `AgentMessage[]` internally and only converts to LLM-compatible `Message[]` at the API boundary.

### Hook System

| Hook | Trigger | Purpose |
|------|---------|---------|
| `beforeToolCall` | Before tool execution | Block/modify tool calls |
| `afterToolCall` | After tool execution | Modify tool results |
| `shouldStopAfterTurn` | End of each turn | Graceful stop |
| `prepareNextTurn` | Before next turn | Switch model / prune context |
| `transformContext` | Before context is sent | Trim messages |
| `getApiKey` | When auth is needed | Dynamic API key resolution |

### Tool Execution Modes

- **sequential**: prepare → execute → finalize one at a time
- **parallel**: prepare sequentially, execute concurrently, results emitted in completion order

### Message Queues

- **steering messages**: injected at the next turn boundary, for real-time intervention
- **follow-up messages**: injected when the agent is about to stop, for background tasks

---

## Layer 3: pi-coding-agent — CLI Product

### Core Components

| File | Lines | Purpose |
|------|-------|---------|
| `main.ts` | ~550 | CLI entry point, mode dispatch, session management |
| `agent-session.ts` | ~3100 | Central class: tool registration, persistence, compaction, retry |
| `sdk.ts` | ~200 | Programmatic API |
| `settings-manager.ts` | ~300 | Global / project-level config |
| `model-registry.ts` | ~200 | Model registration and auth |

### Run Modes

- **interactive**: full-featured TUI (chat, editor, markdown rendering)
- **print**: one-shot execution (`pi -p "prompt"`)
- **rpc**: JSON-RPC (IDE integration)

### Tool System

Each tool = `name` + `description` + `parameters` (TypeBox JSON Schema) + `execute()` + `label`

Supports pluggable operation interfaces (e.g., `BashOperations` can be swapped for SSH / Docker execution).

---

## Key Pattern Comparison

| Pattern | pi Implementation | Our Implementation |
|---------|------------------|--------------------|
| **Provider registry** | `registerApiProvider()` | `providers.py` `PROVIDERS` dict |
| **Event stream** | `EventStream<T, R>` async iterable | `Generator[StreamEvent]` |
| **Agent events** | `agent_start/turn_start/message_*/tool_*/turn_end/agent_end` | `AgentEvent(text_delta/tool_call/tool_result/done)` |
| **Error handling** | Errors encoded in event stream, never throw | `StreamEvent(type="error")` |
| **Hook system** | 6 hooks, injected in the loop | Not yet |
| **Parallel execution** | Concurrent tool execution | Sequential |
| **Message queues** | steering + follow-up dual queue | Not yet |
| **Session persistence** | Auto-save after every event | Not yet |

---

## Key Challenges for Python Rewrite

1. **async generator vs sync generator** — TypeScript's `for await...of` maps to Python's `async for`. Requires `asyncio` for true async streaming.
2. **Type system** — TypeScript's discriminated unions are simulated with `Literal` + `dataclass` in Python.
3. **Declaration merging** — TypeScript-specific; requires registry or plugin mechanism in Python.
4. **Concurrent tool execution** — Python's `asyncio.gather()` replaces TypeScript's `Promise.all()`.
