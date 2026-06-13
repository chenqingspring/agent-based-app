"""
Tool system — defines what the agent can do.

A Tool has two parts:
1. A JSON Schema definition — tells the LLM what the tool does and what inputs it needs
2. A handler function — the actual code to execute when the LLM requests the tool

This separation is the key insight: the LLM never runs code. It only describes what it
wants to do. The agent runtime (our code) is the one that executes it.
"""

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    """
    A tool the agent can use.

    Attributes:
        name: Unique tool name (e.g., "read_file", "web_search")
        description: What the tool does — LLM uses this to decide when to call it
        parameters: JSON Schema for the tool's input
        handler: Function to call when the LLM requests this tool
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]

    def to_definition(self) -> dict[str, Any]:
        """Convert to Anthropic tool definition format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": list(self.parameters.keys()),
            },
        }

    def execute(self, **kwargs) -> str:
        """Run the handler and return the result as a string."""
        return str(self.handler(**kwargs))


class ToolRegistry:
    """
    Collects tools and provides them to the agent.

    Usage:
        registry = ToolRegistry()

        @registry.register("greet", "Greet a user", {"name": {"type": "string", "description": "..."}})
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        agent = Agent(system="...")
        agent.tools = registry.to_definitions()
        for tool in registry:
            agent.register_tool(tool.to_definition(), tool.handler)
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def add(self, tool: Tool) -> "ToolRegistry":
        """Add a tool to the registry."""
        self._tools[tool.name] = tool
        return self

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
    ) -> Callable:
        """
        Decorator to register a function as a tool.

        @registry.register("read_file", "Read a file", {
            "path": {"type": "string", "description": "Path to the file"}
        })
        def read_file(path: str) -> str:
            ...
        """
        def decorator(fn: Callable) -> Callable:
            self._tools[name] = Tool(
                name=name,
                description=description,
                parameters=parameters,
                handler=fn,
            )
            return fn
        return decorator

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def to_definitions(self) -> list[dict[str, Any]]:
        """Convert all tools to Anthropic tool definitions."""
        return [t.to_definition() for t in self._tools.values()]

    def register_all(self, agent: "Agent") -> None:
        """Register all tools with an agent instance."""
        for tool in self._tools.values():
            agent.register_tool(tool.to_definition(), tool.handler)

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self):
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
