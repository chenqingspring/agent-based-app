# Python Rewrite of pi core-agent — Task List

Goal: fully rewrite pi's agent-core layer in Python, focusing on the event loop, hook system, and tool execution engine.

---

## Phase 1: Foundation

- [ ] 1.1 Project init: `pip install anthropic httpx python-dotenv`
- [ ] 1.2 Type definitions: `AgentMessage`, `ToolDefinition`, `ToolCall`, `AgentEvent`, and other core types
- [ ] 1.3 EventStream: `EventStream[T, R]` base class (async queue + callback, based on pi's `event-stream.ts`)
- [ ] 1.4 Provider registry: `register_api_provider()` + `stream_simple()` dispatch

## Phase 2: Agent Core Loop

- [ ] 2.1 `AgentLoopConfig` class (model, tools, hooks, maxTurns...)
- [ ] 2.2 Inner loop: `has_more_tool_calls` + `pending_messages` logic
- [ ] 2.3 Outer loop: follow-up message polling
- [ ] 2.4 Steering message injection (at turn boundary)
- [ ] 2.5 `convert_to_llm()` — AgentMessage → LLM Message conversion

## Phase 3: Hook System

- [ ] 3.1 Hook interface definition (`before_tool_call`, `after_tool_call`, `should_stop_after_turn`, `prepare_next_turn`, `transform_context`, `get_api_key`)
- [ ] 3.2 Hook injection points in the agent loop
- [ ] 3.3 Default hook implementations

## Phase 4: Tool Execution Engine

- [ ] 4.1 `sequential` mode (prepare → execute → finalize one at a time)
- [ ] 4.2 `parallel` mode (prepare serial → execute concurrent → emit results in source order)
- [ ] 4.3 Pluggable operation interfaces (e.g., `BashOperations` swappable for subprocess / SSH / Docker)
- [ ] 4.4 Tool error handling (encode as events, never throw)

## Phase 5: Session Management

- [ ] 5.1 `AgentSession` class (tool registration + persistence + lifecycle)
- [ ] 5.2 Conversation state persistence (auto-save after every event)
- [ ] 5.3 Context compaction — auto-summarize long conversations
- [ ] 5.4 Session resumption (resume / fork / continue)

## Phase 6: Advanced Features

- [ ] 6.1 Auto-retry (transient error → exponential backoff)
- [ ] 6.2 Model switching (swap model in `prepare_next_turn`)
- [ ] 6.3 Abort mechanism (signal propagation)
- [ ] 6.4 Streaming output (`async for event in agent.run_stream()`)

## Phase 7: CLI and Integration

- [ ] 7.1 CLI entry point (async REPL + single prompt)
- [ ] 7.2 Configuration management (global settings + project-level overrides)
- [ ] 7.3 Model registry and API key management

---

## Priority

| Priority | Phase | Reason |
|----------|-------|--------|
| 🔴 P0 | 1, 2 | Core event loop must work first |
| 🟡 P1 | 3, 4 | Hooks + tool engine are pi's core differentiators |
| 🟡 P1 | 6.4 | Streaming is fundamental to UX |
| 🟢 P2 | 5 | Session management needed for production, but can wait |
| 🟢 P3 | 6.1-6.3, 7 | Nice-to-have |

---

## Planned File Structure

```
pi-core/
├── src/
│   ├── __init__.py
│   ├── types.py           # Core type definitions
│   ├── event_stream.py    # EventStream base class
│   ├── api_registry.py    # Provider registration and dispatch
│   ├── agent_loop.py      # Two-level event loop
│   ├── agent.py           # Agent class (state management)
│   ├── hooks.py           # Hook interface and injection points
│   ├── tool_executor.py   # Tool execution engine
│   └── session.py         # AgentSession
├── tests/
├── requirements.txt
└── README.md
```
