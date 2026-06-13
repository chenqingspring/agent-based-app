# Agent-based App

从零构建的 AI Agent 学习项目 —— 不用任何框架，理解 Agent 的底层原理。

## 快速开始

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 配置 .env（复制 .env.example 并填入 API key）
cp .env.example .env

# 启动（火山引擎 ARK）
python -m src.cli --provider volcengine

# 或使用 Anthropic
python -m src.cli --provider anthropic
```

## 架构

```
User Input → Agent.run_stream() → AgentEvent 流 → CLI 渲染
                ↑        ↓
           LLMClient ── LLM API (Anthropic / 火山引擎)
                           ↓
              返回 text_delta / tool_use 事件
                           ↓
              工具处理器在本地执行
                           ↓
              结果以事件形式回传，继续循环
```

遵循 **ReAct 模式**：Think → Act → Observe → 循环直到 Done。

## 项目结构

```
src/
├── agent.py          # Agent 主循环（事件驱动）
├── llm.py            # LLM API 封装（batch + streaming）
├── cli.py            # 命令行入口
├── conversation.py   # 多轮对话记忆管理
├── providers.py      # Provider 注册（Anthropic / 火山引擎）
└── tools/
    ├── __init__.py   # Tool + ToolRegistry
    ├── file_tools.py # 文件读写、目录列表
    └── web_tools.py  # 网页搜索、URL 抓取
tests/
├── test_agent.py     # Agent 循环测试
├── test_llm.py       # LLM 客户端测试
└── test_tools.py     # 工具系统测试
docs/
├── pi-analysis.md    # pi 项目架构分析
└── todo-pi-reimplementation.md  # Python 重写 pi core-agent 计划
```

## 特性

- **流式输出**：文字逐字出现，工具调用实时显示
- **事件驱动**：Agent 发出事件，CLI 订阅渲染
- **多 Provider**：Anthropic / 火山引擎 ARK / 任意 Anthropic 兼容端点
- **多轮记忆**：跨轮次对话上下文
- **工具系统**：文件读写、Web 搜索、URL 抓取

## 命令

```bash
# 单次提问
python -m src.cli --prompt "List files"

# 选择 Provider
python -m src.cli -p volcengine -m deepseek-v4-flash

# 禁用流式
python -m src.cli --prompt "Hi" --no-stream

# 查看可用 Provider
python -m src.cli --list-providers

# 运行测试
pytest tests/ -v
```

## 参考

本项目从 [earendil-works/pi](https://github.com/earendil-works/pi) 中借鉴了事件驱动架构和 Provider 注册模式。详见 `docs/pi-analysis.md`。
