"""Tests for LLM client (unit tests, no API calls)."""

from src.llm import LLMClient, LLMResponse, StreamEvent, ToolCall


class TestStreamEvent:
    """StreamEvent is a dataclass — verify its structure."""

    def test_text_delta_event(self):
        event = StreamEvent(type="text_delta", text="Hello")
        assert event.type == "text_delta"
        assert event.text == "Hello"

    def test_tool_use_event(self):
        tc = ToolCall(id="tc_1", name="read_file", input={"path": "/tmp"})
        event = StreamEvent(type="tool_use", tool_call=tc)
        assert event.type == "tool_use"
        assert event.tool_call.name == "read_file"

    def test_done_event(self):
        event = StreamEvent(type="done", stop_reason="end_turn")
        assert event.type == "done"
        assert event.stop_reason == "end_turn"

    def test_error_event(self):
        event = StreamEvent(type="error", error="Connection refused")
        assert event.type == "error"
        assert "Connection refused" in event.error


class TestLLMResponse:
    """LLMResponse is a dataclass — verify its structure."""

    def test_empty_response(self):
        resp = LLMResponse()
        assert resp.text == ""
        assert resp.tool_calls == []
        assert resp.stop_reason == "end_turn"

    def test_text_response(self):
        resp = LLMResponse(text="Hello, world!", stop_reason="end_turn")
        assert resp.text == "Hello, world!"

    def test_tool_use_response(self):
        resp = LLMResponse(
            tool_calls=[ToolCall(id="tc_1", name="add", input={"a": 1, "b": 2})],
            stop_reason="tool_use",
        )
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "add"
        assert resp.tool_calls[0].input == {"a": 1, "b": 2}
