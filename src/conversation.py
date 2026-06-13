"""
Conversation manager — handles message history and context window limits.

Why this matters: every message you send to the LLM costs tokens and fills the context
window. If you don't manage this, your agent will either:
  - Run out of context window (API error)
  - Waste tokens on irrelevant old messages
  - Lose track of what's important

The solution: keep messages but trim when needed, and summarize context on overflow.
"""

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Conversation:
    """
    Manages the message history for an agent conversation.

    Usage:
        conv = Conversation(system="You are a helpful assistant.")
        conv.add_user_message("Hello!")
        conv.add_assistant_message("Hi there!")
        messages = conv.to_api_format()
    """

    system: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)

    # When conversation exceeds this many messages, trim from the beginning
    MAX_MESSAGES = 40

    def add_user_message(self, content: str | list[dict[str, Any]]):
        """Add a user message to the conversation."""
        self.messages.append({"role": "user", "content": content})
        self._trim()

    def add_assistant_message(self, content: str | list[dict[str, Any]]):
        """Add an assistant message to the conversation."""
        self.messages.append({"role": "assistant", "content": content})
        self._trim()

    def _trim(self):
        """Remove oldest messages if we exceed the limit."""
        while len(self.messages) > self.MAX_MESSAGES:
            self.messages.pop(0)

    def to_messages(self) -> list[dict[str, Any]]:
        """Get messages for passing to the agent."""
        return list(self.messages)

    def save(self, filepath: str):
        """Save conversation to a JSON file."""
        with open(filepath, "w") as f:
            json.dump({
                "system": self.system,
                "messages": self.messages,
            }, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "Conversation":
        """Load a conversation from a JSON file."""
        with open(filepath) as f:
            data = json.load(f)
        conv = cls(system=data["system"])
        conv.messages = data["messages"]
        return conv

    def clear(self):
        """Clear all messages but keep the system prompt."""
        self.messages = []

    def __bool__(self) -> bool:
        """Conversation is always truthy — even when empty.

        Without this, Python falls back to __len__, making an empty Conversation
        falsy, which causes "if conversation:" to fail in run()."""
        return True

    def __len__(self):
        return len(self.messages)
