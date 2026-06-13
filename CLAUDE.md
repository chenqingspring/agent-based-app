# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A general-purpose AI agent built from scratch in Python. Follows the **ReAct pattern**:
LLM reasons → requests tool → agent executes tool → feeds result back → loops until done.

**Goal**: learn how agents work by building one without frameworks.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run with Volcengine (火山引擎 ARK) — default model: deepseek-v4-pro[1m]
python -m src.cli --provider volcengine

# Run with Anthropic
python -m src.cli --provider anthropic

# Single prompt mode (any provider)
python -m src.cli --provider volcengine --prompt "List files in this directory"

# Choose model
python -m src.cli --provider volcengine --model deepseek-v4-flash

# List available providers
python -m src.cli --list-providers

# Run tests
python -m pytest tests/ -v
```

## Architecture

```
User Input → Agent.run() → Agent Loop → final text response
                ↑               ↓
           LLMClient ──── LLM API (Anthropic or Volcengine)
                                ↓
                    returns text or tool_use blocks
                                ↓
                    Tool handlers execute locally
                                ↓
                    Results fed back as conversation
```

LLMClient supports multiple backends via Anthropic Messages API format:
- **Anthropic** (default): direct API, `ANTHROPIC_API_KEY`
- **Volcengine** (火山引擎 ARK): Anthropic-compatible endpoint at `ark.cn-beijing.volces.com/api/coding`, `VOLCENGINE_API_KEY`

Provider config lives in `src/cli.py` (`PROVIDERS` dict). To add a new Anthropic-compatible provider, just add an entry with `base_url`, `api_key_env`, and `default_model`.

**Core files (read in this order to understand the system):**

1. `src/llm.py` — Thin API wrapper. Accepts custom `base_url` for any Anthropic Message API-compatible backend. Returns `LLMResponse` (text + optional `ToolCall` objects).
2. `src/agent.py` — The agent loop. Orchestrates: send to LLM → handle tool_use → execute tools → feed results → loop. Configurable MAX_TURNS (25) to prevent infinite loops.
3. `src/tools/__init__.py` — `Tool` dataclass (name + JSON Schema + handler) and `ToolRegistry` for collecting tools.
4. `src/tools/file_tools.py` — `read_file`, `write_file`, `list_directory` with path safety (restricted to CWD).
5. `src/tools/web_tools.py` — `web_search` (SerpAPI) and `fetch_url`.
6. `src/cli.py` — Entry point. Provider config, CLI args, creates agent with all tools.
7. `src/conversation.py` — Message history manager with auto-trim when exceeding MAX_MESSAGES.

**Key design decisions:**
- Only dependency is `anthropic` SDK + `python-dotenv`. No agent frameworks.
- Volcengine integration is zero-extra-code — their API is Anthropic Messages-compatible, so `anthropic.Anthropic(base_url=...)` just works.
- Tool definitions use Anthropic's native JSON Schema format (maps 1:1, no translation layer).
- File tools restrict access to CWD and subdirectories for safety.

## Environment

- `ANTHROPIC_API_KEY` — Anthropic API key
- `VOLCENGINE_API_KEY` — Volcengine (火山引擎 ARK) API key
- `SERPAPI_API_KEY` — Optional, for web search
