# Python 重写 pi core-agent 任务清单

目标：用 Python 完整重写 pi 的 agent-core 层，重点关注事件循环、Hook 系统、工具执行引擎。

---

## Phase 1：基础设施

- [ ] 1.1 项目初始化：`pip install anthropic httpx python-dotenv`
- [ ] 1.2 类型定义：`AgentMessage`, `ToolDefinition`, `ToolCall`, `AgentEvent` 等核心类型
- [ ] 1.3 EventStream 实现：`EventStream[T, R]` 基类（async queue + callback，参考 pi 的 `event-stream.ts`）
- [ ] 1.4 Provider 注册机制：`register_api_provider()` + `stream_simple()` 分发

## Phase 2：Agent 核心循环

- [ ] 2.1 `AgentLoopConfig` 配置类（model, tools, hooks, maxTurns...）
- [ ] 2.2 内层循环：`has_more_tool_calls` + `pending_messages` 判断
- [ ] 2.3 外层循环：follow-up 消息轮询
- [ ] 2.4 Steering 消息注入（在 turn boundary 插入）
- [ ] 2.5 `convert_to_llm()` — AgentMessage → LLM Message 转换

## Phase 3：Hook 系统

- [ ] 3.1 Hook 接口定义（`before_tool_call`, `after_tool_call`, `should_stop_after_turn`, `prepare_next_turn`, `transform_context`, `get_api_key`）
- [ ] 3.2 Hook 在 Agent 循环中的注入点
- [ ] 3.3 默认 Hook 实现

## Phase 4：工具执行引擎

- [ ] 4.1 `sequential` 模式（逐个 prepare → execute → finalize）
- [ ] 4.2 `parallel` 模式（prepare 串行 → execute 并发 → 结果按序 emit）
- [ ] 4.3 可插拔操作接口（如 `BashOperations` 可替换为 subprocess / SSH / Docker）
- [ ] 4.4 工具错误处理（编码为事件，不抛异常）

## Phase 5：会话管理

- [ ] 5.1 `AgentSession` 类（工具注册 + 持久化 + 生命周期）
- [ ] 5.2 对话状态持久化（自动保存每个事件后）
- [ ] 5.3 上下文压缩（compaction）—— 超长对话自动摘要
- [ ] 5.4 会话恢复（resume / fork / continue）

## Phase 6：高级特性

- [ ] 6.1 自动重试（transient error → exponential backoff）
- [ ] 6.2 模型切换（prepare_next_turn 中更换模型）
- [ ] 6.3 Abort 机制（AbortController 信号传播）
- [ ] 6.4 流式输出（`async for event in agent.run_stream()`）

## Phase 7：CLI 和集成

- [ ] 7.1 CLI 入口（async REPL + single prompt）
- [ ] 7.2 配置管理（全局设置 + 项目级覆盖）
- [ ] 7.3 模型注册和 API key 管理

---

## 优先级

| 优先级 | Phase | 原因 |
|--------|-------|------|
| 🔴 P0 | 1, 2 | 核心事件循环必须先跑通 |
| 🟡 P1 | 3, 4 | Hook + 工具引擎是 pi 的核心差异化能力 |
| 🟡 P1 | 6.4 | 流式输出是用户体验的基础 |
| 🟢 P2 | 5 | 会话管理是生产级必需但可以后做 |
| 🟢 P3 | 6.1-6.3, 7 | 锦上添花 |

---

## 预计文件结构

```
pi-core/
├── src/
│   ├── __init__.py
│   ├── types.py           # 核心类型定义
│   ├── event_stream.py    # EventStream 基类
│   ├── api_registry.py    # Provider 注册和分发
│   ├── agent_loop.py      # 双层事件循环
│   ├── agent.py           # Agent 类（状态管理）
│   ├── hooks.py           # Hook 接口和注入点
│   ├── tool_executor.py   # 工具执行引擎
│   └── session.py         # AgentSession
├── tests/
├── requirements.txt
└── README.md
```
