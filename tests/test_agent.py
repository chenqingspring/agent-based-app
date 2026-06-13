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
