"""
CLI entry point — ties everything together into a working agent.

Usage:
    python -m src.cli                          # Anthropic (default)
    python -m src.cli --provider volcengine    # Volcengine (火山引擎)
    python -m src.cli --provider volcengine --model deepseek-v4-flash
    python -m src.cli --prompt "List files"    # Single prompt mode
    python -m src.cli --prompt "Hi" --no-stream  # Disable streaming
"""

import argparse
import os

from src.agent import Agent
from src.providers import get_provider, list_providers, PROVIDERS
from src.tools.file_tools import create_file_tools
from src.tools.web_tools import create_web_tools

DEFAULT_SYSTEM = """You are a helpful AI assistant with access to tools.

When you need to do something, check if you have a tool for it:
- To read files: use read_file
- To write files: use write_file
- To see what files exist: use list_directory
- To look up current information: use web_search
- To read a web page: use fetch_url

Be direct and concise. If a tool fails, try an alternative approach.
Answer in Chinese unless the user asks otherwise (用中文回答)."""


def create_agent(
    system: str = DEFAULT_SYSTEM,
    enable_file_tools: bool = True,
    enable_web_tools: bool = True,
    model: str | None = None,
    provider: str = "anthropic",
) -> Agent:
    """Create a fully-configured agent with all tools for the given provider."""
    provider_cfg = get_provider(provider)
    api_key = provider_cfg.resolve_api_key()
    model = model or provider_cfg.default_model

    agent = Agent(
        system=system,
        tools=[],
        model=model,
        api_key=api_key,
        base_url=provider_cfg.base_url,
        provider_name=provider,
    )

    if enable_file_tools:
        file_registry = create_file_tools()
        file_registry.register_all(agent)
        print(f"  File tools: {len(file_registry)} loaded")

    if enable_web_tools:
        web_registry = create_web_tools()
        web_registry.register_all(agent)
        print(f"  Web tools: {len(web_registry)} loaded")

    return agent


def main():
    parser = argparse.ArgumentParser(
        description="Agent-based app — an AI agent with tools"
    )
    parser.add_argument(
        "--provider", "-p",
        type=str,
        default="anthropic",
        choices=list(PROVIDERS.keys()),
        help="LLM provider to use",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Model ID (provider default if omitted)",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=DEFAULT_SYSTEM,
        help="Custom system prompt",
    )
    parser.add_argument(
        "--no-file-tools",
        action="store_true",
        help="Disable file system tools",
    )
    parser.add_argument(
        "--no-web-tools",
        action="store_true",
        help="Disable web tools",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Single prompt mode (non-interactive)",
    )
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="List available providers and exit",
    )
    args = parser.parse_args()

    if args.list_providers:
        print("Available providers:")
        for cfg in list_providers():
            has_key = bool(os.environ.get(cfg.api_key_env))
            status = " [key found]" if has_key else " [no key]"
            print(f"  {cfg.id}: {cfg.name} (model: {cfg.default_model}){status}")
        return

    provider_cfg = get_provider(args.provider)
    model = args.model or provider_cfg.default_model

    print(f"Building agent...")
    print(f"  Provider: {provider_cfg.name}")
    print(f"  Base URL: {provider_cfg.base_url or '(SDK default)'}")
    agent = create_agent(
        system=args.system,
        enable_file_tools=not args.no_file_tools,
        enable_web_tools=not args.no_web_tools,
        model=model,
        provider=args.provider,
    )
    print(f"  Model: {model}")
    print(f"  Total tools: {len(agent._tool_handlers)}\n")

    use_stream = not args.no_stream

    if args.prompt:
        if use_stream:
            for event in agent.run_stream(args.prompt):
                _render_event(event)
        else:
            result = agent.run(args.prompt)
            print(result)
    else:
        agent.run_interactive(stream=use_stream)


def _render_event(event):
    """Render a single agent event to stdout."""
    if event.type == "text_delta":
        print(event.text, end="", flush=True)
    elif event.type == "tool_call":
        print(f"\n  ⚙ {event.tool_name}({_fmt_args(event.tool_input)})", flush=True)
    elif event.type == "tool_result":
        summary = event.tool_output[:200].replace("\n", " ")
        status = "✗" if event.is_error else "✓"
        print(f" → {status} {summary}", flush=True)


def _fmt_args(input_dict: dict) -> str:
    """Format tool input args for display."""
    parts = []
    for k, v in input_dict.items():
        s = str(v)
        if len(s) > 50:
            s = s[:47] + "..."
        parts.append(f"{k}={s}")
    return ", ".join(parts)


if __name__ == "__main__":
    main()
