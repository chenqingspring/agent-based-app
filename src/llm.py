"""
LLM client — thin wrapper around Anthropic's API.

The key insight: an agent doesn't need a complex client. It needs three things:
1. Send messages (with optional tool definitions)
2. Get back either text or tool_use blocks
3. That's it.
"""

import os
from dataclasses import dataclass, field
from typing import Any

import anthropic
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ToolCall:
    """A parsed tool-use request from the model."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    """What comes back from the model: text, tool calls, or both."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"


class LLMClient:
    """
    Minimal wrapper around the Anthropic Messages API.

    Usage:
        client = LLMClient()
        response = client.send(
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        print(response.text)
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        base_url: str | None = None,
        provider_name: str = "anthropic",
    ):
        self.model = model
        self.provider_name = provider_name
        self.base_url = base_url

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Set it in .env or pass it directly."
            )
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = anthropic.Anthropic(**client_kwargs)

    def send(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """
        Send a conversation to Claude and get back text and/or tool calls.

        Args:
            system: The system prompt.
            messages: Conversation history in Anthropic format.
            tools: Optional list of tool definitions (JSON Schema).
            max_tokens: Maximum response tokens.

        Returns:
            LLMResponse with text content and any tool calls.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        response = self.client.messages.create(**kwargs)

        return self._parse_response(response)

    def _parse_response(self, response) -> LLMResponse:
        """Extract text and tool calls from an API response."""
        result = LLMResponse(stop_reason=response.stop_reason)

        for block in response.content:
            if block.type == "text":
                result.text += block.text
            elif block.type == "tool_use":
                result.tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        input=block.input,
                    )
                )

        return result
