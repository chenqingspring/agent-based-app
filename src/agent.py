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

from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from src.conversation import Conversation
from src.llm import LLMClient, LLMResponse, StreamEvent, ToolCall


@dataclass
class ToolResult:
    """The result of executing a tool."""

    tool_call_id: str
    name: str
    output: str
    is_error: bool = False


# ---------------------------------------------------------------------------
# Agent events — inspired by pi's agent-loop.ts event protocol
# ---------------------------------------------------------------------------

AgentEventType = Literal["text_delta", "tool_call", "tool_result", "done"]


@dataclass
class AgentEvent:
    """Events emitted by the agent loop during execution.

    Types:
        text_delta  — a text chunk from the LLM (streaming)
        tool_call   — the agent is executing a tool
        tool_result — the result of a tool execution
        done        — agent loop complete, final response ready
    """

    type: AgentEventType
    text: str = ""
    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)
    tool_output: str = ""
    is_error: bool = False
    final_text: str = ""


_UNSET = object()


class Agent:
    """
    A general-purpose agent that can use tools to accomplish tasks.

    Two modes:
    - run(): batch mode, returns final response
    - run_stream(): event-driven, yields events for real-time display

    Usage:
        agent = Agent(system="You are a helpful assistant.")
        agent.register_tool(...)

        # Batch mode
        result = agent.run("What's in this project?")

        # Streaming mode
        for event in agent.run_stream("What's in this project?"):
            if event.type == "text_delta":
                print(event.text, end="", flush=True)
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

    # ------------------------------------------------------------------
    # Batch mode — returns the final response
    # ------------------------------------------------------------------

    def run(
        self,
        user_message: str,
        conversation: Conversation | None = None,
    ) -> str:
        """Run the agent and return the full text response (non-streaming)."""
        parts: list[str] = []
        for event in self.run_stream(user_message, conversation=conversation):
            if event.type == "text_delta":
                parts.append(event.text)
        return event.final_text if event.type == "done" else "".join(parts)

    # ------------------------------------------------------------------
    # Streaming mode — yields events in real-time
    # ------------------------------------------------------------------

    def run_stream(
        self,
        user_message: str,
        conversation: Conversation | None = None,
    ) -> Generator[AgentEvent, None, None]:
        """
        Run the agent loop, yielding events as they happen.

        This is the main entry point for streaming. The CLI subscribes to
        these events and renders them in real-time.
        """
        # Build the initial message list
        if conversation is not None:
            conversation.add_user_message(user_message)
            messages = conversation.to_messages()
        else:
            messages = [{"role": "user", "content": user_message}]

        all_text: list[str] = []

        for turn in range(self.MAX_TURNS):
            # --- Stream LLM response ---
            text_in_turn: list[str] = []
            tool_calls_in_turn: list[ToolCall] = []
            stop_reason = "end_turn"

            for se in self.llm.send_stream(
                system=self.system,
                messages=messages,
                tools=self.tools if self.tools else None,
            ):
                if se.type == "text_delta":
                    text_in_turn.append(se.text)
                    yield AgentEvent(type="text_delta", text=se.text)

                elif se.type == "tool_use":
                    tool_calls_in_turn.append(se.tool_call)

                elif se.type == "error":
                    yield AgentEvent(
                        type="done",
                        final_text=f"Error: {se.error}",
                        is_error=True,
                    )
                    return

                elif se.type == "done":
                    stop_reason = se.stop_reason

            all_text.extend(text_in_turn)

            # --- Handle stop reasons ---

            if stop_reason == "end_turn":
                result = "".join(all_text)
                if conversation is not None:
                    conversation.add_assistant_message(result)
                yield AgentEvent(type="done", final_text=result)
                return

            if stop_reason == "tool_use" and tool_calls_in_turn:
                # Build the assistant message with text + tool_use blocks
                assistant_content = self._build_tool_use_content(
                    "".join(text_in_turn), tool_calls_in_turn
                )
                messages.append({"role": "assistant", "content": assistant_content})

                # Execute each tool and yield events
                tool_results = []
                for tc in tool_calls_in_turn:
                    yield AgentEvent(
                        type="tool_call",
                        tool_name=tc.name,
                        tool_input=tc.input,
                    )
                    result = self._execute_tool(tc)
                    tool_results.append(result)
                    yield AgentEvent(
                        type="tool_result",
                        tool_name=tc.name,
                        tool_output=result.output,
                        is_error=result.is_error,
                    )

                messages.append({
                    "role": "user",
                    "content": self._build_tool_result_content(tool_results),
                })
                # Loop back for the LLM to process tool results
                continue

            if stop_reason == "max_tokens":
                assistant_content = self._build_tool_use_content(
                    "".join(text_in_turn), tool_calls_in_turn
                )
                if assistant_content:
                    messages.append({"role": "assistant", "content": assistant_content})
                messages.append({
                    "role": "user",
                    "content": (
                        "You were cut off due to output length. "
                        "Please continue from where you stopped."
                    ),
                })
                # Loop back for continuation
                continue

        final = "".join(all_text) or "(Agent reached max turns with no response)"
        if conversation is not None:
            conversation.add_assistant_message(final)
        yield AgentEvent(type="done", final_text=final)

    # ------------------------------------------------------------------
    # Interactive REPL
    # ------------------------------------------------------------------

    def run_interactive(self, stream: bool = True):
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
            if stream:
                for event in self.run_stream(user_input, conversation=conversation):
                    if event.type == "text_delta":
                        print(event.text, end="", flush=True)
                    elif event.type == "tool_call":
                        print(f"\n  ⚙ {event.tool_name}({self._fmt_tool_args(event.tool_input)})", flush=True)
                    elif event.type == "tool_result":
                        summary = event.tool_output[:200].replace("\n", " ")
                        status = "✗" if event.is_error else "✓"
                        print(f" → {status} {summary}", flush=True)
                print()
                print()
            else:
                response = self.run(user_input, conversation=conversation)
                print(response)
                print()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_tool_use_content(
        self, text: str, tool_calls: list[ToolCall]
    ) -> list[dict[str, Any]]:
        """Build assistant message content with text and tool_use blocks."""
        content: list[dict[str, Any]] = []
        if text:
            content.append({"type": "text", "text": text})
        for tc in tool_calls:
            content.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.input,
            })
        return content

    def _build_assistant_content(self, response: LLMResponse) -> list[dict[str, Any]]:
        """Build assistant content from a batch LLMResponse (kept for backward compat)."""
        return self._build_tool_use_content(response.text, response.tool_calls)

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

    @staticmethod
    def _fmt_tool_args(input_dict: dict) -> str:
        """Format tool input args for display."""
        parts = []
        for k, v in input_dict.items():
            s = str(v)
            if len(s) > 50:
                s = s[:47] + "..."
            parts.append(f"{k}={s}")
        return ", ".join(parts)
