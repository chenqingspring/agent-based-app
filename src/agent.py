"""
The agent loop — the core pattern that makes an "agent" different from a chatbot.

The ReAct loop:
    Think → Act → Observe → Think → Act → Observe → ... → Done

    User: "What files are in /tmp?"
    → Agent sends to LLM
    → LLM: "I need to list files. [tool_use: list_directory /tmp]"
    → Agent executes list_directory("/tmp"), gets result
    → Agent sends result back to LLM
    → LLM: "There are 3 files in /tmp: a.txt, b.txt, c.log"
    → Done. Print response.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

from src.conversation import Conversation
from src.llm import LLMClient, LLMResponse, ToolCall


@dataclass
class ToolResult:
    """The result of executing a tool."""

    tool_call_id: str
    name: str
    output: str
    is_error: bool = False


_UNSET = object()


class Agent:
    """
    A general-purpose agent that can use tools to accomplish tasks.

    Usage:
        agent = Agent(system="You are a helpful assistant.")
        agent.register_tool(...)
        result = agent.run("What's in this project?")
        print(result)
    """

    MAX_TURNS = 25  # Safety limit — prevent infinite loops

    def __init__(
        self,
        system: str,
        tools: list[dict[str, Any]] | None = None,
        model: str = "claude-sonnet-4-6",
        llm_client: Any = _UNSET,
        api_key: str | None = None,
        base_url: str | None = None,
        provider_name: str = "anthropic",
    ):
        self.system = system
        self.tools = tools or []
        self.llm = (
            llm_client if llm_client is not _UNSET
            else LLMClient(
                model=model,
                api_key=api_key,
                base_url=base_url,
                provider_name=provider_name,
            )
        )

        # Map tool names to handler functions
        self._tool_handlers: dict[str, Callable] = {}

    def register_tool(self, definition: dict[str, Any], handler: Callable):
        """Register a tool definition and its handler function."""
        self.tools.append(definition)
        self._tool_handlers[definition["name"]] = handler

    def run(
        self,
        user_message: str,
        conversation: Conversation | None = None,
    ) -> str:
        """
        Run the agent loop on a user message.

        If a Conversation is provided, it maintains multi-turn memory across
        calls — previous messages are included as context and the final
        assistant response is appended back.

        Returns the final text response.
        """
        # Build the initial message list: either from shared conversation or fresh
        if conversation is not None:
            conversation.add_user_message(user_message)
            messages = conversation.to_messages()
        else:
            messages = [{"role": "user", "content": user_message}]

        final_text: list[str] = []

        for turn in range(self.MAX_TURNS):
            response = self.llm.send(
                system=self.system,
                messages=messages,
                tools=self.tools if self.tools else None,
            )

            # Collect any text in this response
            if response.text:
                final_text.append(response.text)

            # Done — return the full response
            if response.stop_reason == "end_turn":
                result = "\n".join(final_text)
                if conversation is not None:
                    conversation.add_assistant_message(result)
                return result

            # Model wants to use tools — execute them and feed results back
            if response.stop_reason == "tool_use":
                assistant_content = self._build_assistant_content(response)
                tool_results = [
                    self._execute_tool(tc) for tc in response.tool_calls
                ]

                messages.append({"role": "assistant", "content": assistant_content})
                messages.append({
                    "role": "user",
                    "content": self._build_tool_result_content(tool_results),
                })
                continue

            # stop_reason == "max_tokens" — model was cut off mid-response.
            # Use _build_assistant_content to correctly include both text and
            # any tool_use blocks (the model may have started a tool call before
            # hitting the token limit).
            assistant_content = self._build_assistant_content(response)
            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})
            messages.append({
                "role": "user",
                "content": (
                    "You were cut off due to output length. "
                    "Please continue from where you stopped."
                ),
            })

        return "\n".join(final_text) or "(Agent reached max turns with no response)"

    def _build_assistant_content(self, response: LLMResponse) -> list[dict[str, Any]]:
        """Build the assistant message content with text and tool_use blocks."""
        content: list[dict[str, Any]] = []
        if response.text:
            content.append({"type": "text", "text": response.text})
        for tc in response.tool_calls:
            content.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.input,
            })
        return content

    def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call and return the result."""
        handler = self._tool_handlers.get(tool_call.name)
        if handler is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                output=f"Error: Unknown tool '{tool_call.name}'",
                is_error=True,
            )

        try:
            output = handler(**tool_call.input)
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                output=str(output),
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                output=f"Error executing {tool_call.name}: {e}",
                is_error=True,
            )

    def _build_tool_result_content(
        self, results: list[ToolResult]
    ) -> list[dict[str, Any]]:
        """Build the user message content with tool results."""
        return [
            {
                "type": "tool_result",
                "tool_use_id": r.tool_call_id,
                "content": r.output,
            }
            for r in results
        ]

    def run_interactive(self):
        """Run the agent in an interactive REPL loop with multi-turn memory."""
        conversation = Conversation(system=self.system)
        print("Agent ready. Type 'quit' or 'exit' to stop, 'clear' to reset.\n")

        while True:
            try:
                user_input = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                break
            if user_input.lower() == "clear":
                conversation.clear()
                print("Conversation cleared.\n")
                continue

            print()
            response = self.run(user_input, conversation=conversation)
            print(response)
            print()
