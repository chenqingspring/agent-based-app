# Agent-based App

A general-purpose AI agent built from scratch in Python — learn how agents work without frameworks.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure .env (copy .env.example and add your API key)
cp .env.example .env

# Run with Volcengine ARK
python -m src.cli --provider volcengine

# Or use Anthropic
python -m src.cli --provider anthropic
```

## Architecture

```
User Input → Agent.run_stream() → AgentEvent stream → CLI renders
                ↑        ↓
           LLMClient ── LLM API (Anthropic / Volcengine ARK)
                            ↓
              returns text_delta / tool_use events
                            ↓
              Tool handlers execute locally
                            ↓
              Results fed back as events, loop continues
```

Follows the **ReAct pattern**: Think → Act → Observe → Loop until Done.

## Project Structure

```
src/
├── agent.py          # Agent main loop (event-driven)
├── llm.py            # LLM API wrapper (batch + streaming)
├── cli.py            # CLI entry point
├── conversation.py   # Multi-turn conversation memory
├── providers.py      # Provider registry (Anthropic / Volcengine)
└── tools/
    ├── __init__.py   # Tool + ToolRegistry
    ├── file_tools.py # File read/write, directory listing
    └── web_tools.py  # Web search, URL fetching
tests/
├── test_agent.py     # Agent loop tests
├── test_llm.py       # LLM client tests
└── test_tools.py     # Tool system tests
docs/
├── pi-analysis.md    # Architecture analysis of pi
└── todo-pi-reimplementation.md  # Python reimplementation roadmap
```

## Features

- **Streaming output**: text appears in real-time, tool calls displayed as they happen
- **Event-driven**: agent emits events, CLI subscribes and renders
- **Multi-provider**: Anthropic / Volcengine ARK / any Anthropic-compatible endpoint
- **Multi-turn memory**: cross-turn conversation context
- **Tool system**: file I/O, web search, URL fetching

## Commands

```bash
# Single prompt
python -m src.cli --prompt "List files"

# Choose provider
python -m src.cli -p volcengine -m deepseek-v4-flash

# Disable streaming
python -m src.cli --prompt "Hi" --no-stream

# List available providers
python -m src.cli --list-providers

# Run tests
pytest tests/ -v
```

## Reference

This project draws from [earendil-works/pi](https://github.com/earendil-works/pi) — event-driven architecture and provider registry patterns. See `docs/pi-analysis.md` for details.
