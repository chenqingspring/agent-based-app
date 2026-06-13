"""
LLM client — thin wrapper around Anthropic's API.

The key insight: an agent doesn't need a complex client. It needs three things:
1. Send messages (with optional tool definitions)
2. Get back either text or tool_use blocks
3. That's it.

Supports both batch (send) and streaming (send_stream) modes.
"""

import os
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any, Literal

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


# ---------------------------------------------------------------------------
# Streaming types — inspired by pi's EventStream protocol
# ---------------------------------------------------------------------------

StreamEventType = Literal["text_delta", "tool_use", "done", "error"]


@dataclass
class StreamEvent:
    """A single event from the LLM stream.

    Types:
        text_delta — a chunk of text as the model generates it
        tool_use   — model wants to use a tool (emitted after text streaming)
        done       — streaming complete, with stop_reason
        error      — something went wrong
    """

    type: StreamEventType
    text: str = ""
    tool_call: ToolCall | None = None
    stop_reason: str = ""
    error: str = ""


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

        # Streaming:
        for event in client.send_stream(...):
            if event.type == "text_delta":
                print(event.text, end="", flush=True)
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

    # ------------------------------------------------------------------
    # Batch mode — full response at once
    # ------------------------------------------------------------------

    def send(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a conversation and get the full response."""
        kwargs = self._build_kwargs(system, messages, tools, max_tokens)
        response = self.client.messages.create(**kwargs)
        return self._parse_response(response)

    # ------------------------------------------------------------------
    # Streaming mode — events as they happen
    # ------------------------------------------------------------------

    def send_stream(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> Generator[StreamEvent, None, None]:
        """
        Stream the LLM response as a sequence of events.

        Yields:
            text_delta — text chunks as they arrive
            tool_use   — tool call requests (after text streaming)
            done       — stream complete with stop_reason
            error      — on failure
        """
        kwargs = self._build_kwargs(system, messages, tools, max_tokens)

        try:
            with self.client.messages.stream(**kwargs) as stream:
                # Stream text deltas in real-time
                for text_chunk in stream.text_stream:
                    yield StreamEvent(type="text_delta", text=text_chunk)

                # Get the final accumulated message
                final = stream.get_final_message()

                # Emit any tool calls from the final message
                for block in final.content:
                    if block.type == "tool_use":
                        yield StreamEvent(
                            type="tool_use",
                            tool_call=ToolCall(
                                id=block.id,
                                name=block.name,
                                input=block.input,
                            ),
                        )

                yield StreamEvent(
                    type="done",
                    stop_reason=final.stop_reason,
                )

        except Exception as e:
            yield StreamEvent(type="error", error=str(e))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_kwargs(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        return kwargs

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
