"""
CLI entry point — ties everything together into a working agent.

Usage:
    python -m src.cli                          # Anthropic (default)
    python -m src.cli --provider volcengine    # Volcengine (火山引擎)
    python -m src.cli --provider volcengine --model deepseek-v4-flash
    python -m src.cli --prompt "List files"    # Single prompt mode
"""

import argparse
import os

from src.agent import Agent
from src.tools.file_tools import create_file_tools
from src.tools.web_tools import create_web_tools

# ---------------------------------------------------------------------------
# Provider configurations
# ---------------------------------------------------------------------------

PROVIDERS = {
    "anthropic": {
        "name": "Anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url": None,  # Use SDK default
        "default_model": "claude-sonnet-4-6",
    },
    "volcengine": {
        "name": "Volcengine (火山引擎 ARK)",
        "api_key_env": "VOLCENGINE_API_KEY",
        "base_url": "https://ark.cn-beijing.volces.com/api/coding",
        "default_model": "deepseek-v4-pro[1m]",
    },
}

DEFAULT_SYSTEM = """You are a helpful AI assistant with access to tools.

When you need to do something, check if you have a tool for it:
- To read files: use read_file
- To write files: use write_file
- To see what files exist: use list_directory
- To look up current information: use web_search
- To read a web page: use fetch_url

Be direct and concise. If a tool fails, try an alternative approach.
Answer in Chinese unless the user asks otherwise (用中文回答)."""


def get_provider_config(provider_id: str) -> dict:
    """Get the config dictionary for a provider."""
    if provider_id not in PROVIDERS:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(f"Unknown provider '{provider_id}'. Available: {available}")
    return PROVIDERS[provider_id]


def resolve_api_key(provider_cfg: dict) -> str:
    """Resolve API key from environment or raise a helpful error."""
    env_var = provider_cfg["api_key_env"]
    api_key = os.environ.get(env_var)
    if not api_key:
        raise ValueError(
            f"{env_var} not set. Add it to your .env file:\n"
            f"  {env_var}=your-key-here\n\n"
            f"Provider: {provider_cfg['name']}"
        )
    return api_key


def create_agent(
    system: str = DEFAULT_SYSTEM,
    enable_file_tools: bool = True,
    enable_web_tools: bool = True,
    model: str | None = None,
    provider: str = "anthropic",
) -> Agent:
    """Create a fully-configured agent with all tools for the given provider."""
    provider_cfg = get_provider_config(provider)
    api_key = resolve_api_key(provider_cfg)
    model = model or provider_cfg["default_model"]
    base_url = provider_cfg["base_url"]

    agent = Agent(
        system=system,
        tools=[],
        model=model,
        api_key=api_key,
        base_url=base_url,
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
        for pid, cfg in PROVIDERS.items():
            model = cfg["default_model"]
            env = cfg["api_key_env"]
            has_key = bool(os.environ.get(env))
            status = " [key found]" if has_key else " [no key]"
            print(f"  {pid}: {cfg['name']} (model: {model}){status}")
        return

    provider_cfg = get_provider_config(args.provider)
    model = args.model or provider_cfg["default_model"]

    print(f"Building agent...")
    print(f"  Provider: {provider_cfg['name']}")
    print(f"  Base URL: {provider_cfg['base_url'] or '(SDK default)'}")
    agent = create_agent(
        system=args.system,
        enable_file_tools=not args.no_file_tools,
        enable_web_tools=not args.no_web_tools,
        model=model,
        provider=args.provider,
    )
    print(f"  Model: {model}")
    print(f"  Total tools: {len(agent._tool_handlers)}\n")

    if args.prompt:
        result = agent.run(args.prompt)
        print(result)
    else:
        agent.run_interactive()


if __name__ == "__main__":
    main()
