"""Tests for the agent loop logic (unit tests, no API calls)."""

import pytest
from src.agent import Agent, ToolResult
from src.conversation import Conversation
from src.llm import LLMResponse, ToolCall
from src.tools import ToolRegistry


class TestConversation:
    """Tests for Conversation integration — especially the __bool__ fix."""

    def test_bool_is_always_true(self):
        """Empty conversation must be truthy. Python uses __len__ for bool()
        when __bool__ is not defined, so len=0 would mean bool=False."""
        conv = Conversation()
        assert bool(conv) is True
        conv.add_user_message("Hi")
        assert bool(conv) is True

    def test_multi_turn_memory(self):
        """Messages added to a conversation persist across turns."""
        conv = Conversation(system="Test")
        conv.add_user_message("I'm Bob")
        conv.add_assistant_message("Hi Bob!")
        assert len(conv) == 2
        assert conv.to_messages()[0]["content"] == "I'm Bob"

    def test_clear(self):
        conv = Conversation(system="Test")
        conv.add_user_message("Hello")
        conv.add_assistant_message("Hi")
        conv.clear()
        assert len(conv) == 0


class TestAgentToolExecution:
    """Tests for the agent's internal tool execution logic."""

    def test_execute_tool_success(self):
        agent = Agent(system="Test", llm_client=None)
        agent.register_tool(
            {
                "name": "add",
                "description": "Add two numbers",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                },
            },
            lambda a, b: a + b,
        )

        tool_call = ToolCall(id="tc_1", name="add", input={"a": 1, "b": 2})
        result = agent._execute_tool(tool_call)

        assert not result.is_error
        assert result.tool_call_id == "tc_1"
        assert result.name == "add"
        assert result.output == "3"

    def test_execute_unknown_tool(self):
        agent = Agent(system="Test", llm_client=None)

        tool_call = ToolCall(id="tc_1", name="nonexistent", input={})
        result = agent._execute_tool(tool_call)

        assert result.is_error
        assert "Unknown tool" in result.output

    def test_execute_tool_error(self):
        agent = Agent(system="Test", llm_client=None)
        agent.register_tool(
            {
                "name": "fail",
                "description": "Always fails",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        tool_call = ToolCall(id="tc_1", name="fail", input={})
        result = agent._execute_tool(tool_call)

        assert result.is_error
        assert "Error executing" in result.output

    def test_build_assistant_content(self):
        agent = Agent(system="Test", llm_client=None)
        response = LLMResponse(
            text="Let me help with that.",
            tool_calls=[
                ToolCall(id="tc_1", name="read_file", input={"path": "test.txt"})
            ],
        )

        content = agent._build_assistant_content(response)
        assert len(content) == 2
        assert content[0] == {"type": "text", "text": "Let me help with that."}
        assert content[1]["type"] == "tool_use"
        assert content[1]["name"] == "read_file"

    def test_build_tool_result_content(self):
        agent = Agent(system="Test", llm_client=None)
        results = [
            ToolResult(tool_call_id="tc_1", name="read_file", output="file contents"),
            ToolResult(tool_call_id="tc_2", name="list_directory", output="a.txt\nb.txt"),
        ]

        content = agent._build_tool_result_content(results)
        assert len(content) == 2
        assert content[0]["type"] == "tool_result"
        assert content[0]["tool_use_id"] == "tc_1"
        assert content[0]["content"] == "file contents"


class TestMaxTurns:
    """Test that the agent respects the max turns limit."""

    def test_max_turns_default(self):
        agent = Agent(system="Test", llm_client=None)
        assert agent.MAX_TURNS == 25


class TestAgentEvent:
    """AgentEvent is a dataclass — verify its structure."""

    def test_text_delta_event(self):
        from src.agent import AgentEvent
        event = AgentEvent(type="text_delta", text="Hello")
        assert event.type == "text_delta"
        assert event.text == "Hello"

    def test_tool_call_event(self):
        from src.agent import AgentEvent
        event = AgentEvent(
            type="tool_call",
            tool_name="read_file",
            tool_input={"path": "/tmp"},
        )
        assert event.type == "tool_call"
        assert event.tool_name == "read_file"

    def test_tool_result_event(self):
        from src.agent import AgentEvent
        event = AgentEvent(
            type="tool_result",
            tool_name="read_file",
            tool_output="file contents",
        )
        assert event.type == "tool_result"
        assert event.tool_output == "file contents"

    def test_done_event(self):
        from src.agent import AgentEvent
        event = AgentEvent(type="done", final_text="Complete response")
        assert event.type == "done"
        assert event.final_text == "Complete response"


class TestAgentStreaming:
    """Test that run_stream() yields correct events."""

    def test_run_collects_stream_events(self):
        """run() returns text collected from run_stream() events."""
        from src.agent import Agent, AgentEvent
        from src.llm import StreamEvent, ToolCall

        # Create a mock LLM client that yields stream events
        class MockLLM:
            def send_stream(self, system, messages, tools=None, max_tokens=4096):
                yield StreamEvent(type="text_delta", text="Hello ")
                yield StreamEvent(type="text_delta", text="world!")
                yield StreamEvent(type="done", stop_reason="end_turn")

        agent = Agent(system="Test", llm_client=MockLLM())
        result = agent.run("Hi")
        assert result == "Hello world!"

    def test_run_stream_yields_text_deltas(self):
        """run_stream() forwards text_delta events from the LLM."""
        from src.agent import Agent
        from src.llm import StreamEvent

        class MockLLM:
            def send_stream(self, system, messages, tools=None, max_tokens=4096):
                yield StreamEvent(type="text_delta", text="A")
                yield StreamEvent(type="text_delta", text="B")
                yield StreamEvent(type="done", stop_reason="end_turn")

        agent = Agent(system="Test", llm_client=MockLLM())
        events = list(agent.run_stream("Hi"))
        assert events[0].type == "text_delta"
        assert events[0].text == "A"
        assert events[2].type == "done"
        assert events[2].final_text == "AB"

    def test_run_stream_with_tool_call(self):
        """run_stream() yields tool_call and tool_result events."""
        from src.agent import Agent
        from src.llm import StreamEvent, ToolCall

        class MockLLM:
            def send_stream(self, system, messages, tools=None, max_tokens=4096):
                yield StreamEvent(
                    type="tool_use",
                    tool_call=ToolCall(id="tc_1", name="add", input={"a": 1, "b": 2}),
                )
                yield StreamEvent(type="done", stop_reason="tool_use")

        agent = Agent(system="Test", llm_client=MockLLM())
        agent.register_tool(
            {"name": "add", "description": "Add two numbers",
             "input_schema": {"type": "object", "properties": {
                 "a": {"type": "integer"}, "b": {"type": "integer"}},
                 "required": ["a", "b"]}},
            lambda a, b: a + b,
        )

        events = list(agent.run_stream("Add 1+2"))
        types = [e.type for e in events]
        assert "tool_call" in types
        assert "tool_result" in types

    def test_run_stream_error(self):
        """run_stream() yields done with error on LLM error."""
        from src.agent import Agent
        from src.llm import StreamEvent

        class MockLLM:
            def send_stream(self, system, messages, tools=None, max_tokens=4096):
                yield StreamEvent(type="error", error="API down")

        agent = Agent(system="Test", llm_client=MockLLM())
        events = list(agent.run_stream("Hi"))
        assert events[0].type == "done"
        assert events[0].is_error
        assert "API down" in events[0].final_text
