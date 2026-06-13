"""Tests for the tool system."""

import pytest
from src.tools import Tool, ToolRegistry
from src.tools.file_tools import read_file, write_file, list_directory, create_file_tools
from src.tools.web_tools import create_web_tools, web_search


class TestToolRegistry:
    def test_register_and_get_tool(self):
        registry = ToolRegistry()

        @registry.register("add", "Add two numbers", {
            "a": {"type": "integer", "description": "First number"},
            "b": {"type": "integer", "description": "Second number"},
        })
        def add(a: int, b: int) -> int:
            return a + b

        tool = registry.get("add")
        assert tool is not None
        assert tool.name == "add"
        assert tool.execute(a=1, b=2) == "3"

    def test_to_definitions(self):
        registry = ToolRegistry()

        @registry.register("echo", "Echo back input", {
            "text": {"type": "string", "description": "Text to echo"},
        })
        def echo(text: str) -> str:
            return text

        definitions = registry.to_definitions()
        assert len(definitions) == 1
        assert definitions[0]["name"] == "echo"
        assert definitions[0]["input_schema"]["type"] == "object"

    def test_iteration(self):
        registry = ToolRegistry()

        @registry.register("tool1", "First tool", {})
        def tool1():
            return "1"

        @registry.register("tool2", "Second tool", {})
        def tool2():
            return "2"

        assert len(registry) == 2
        assert "tool1" in registry
        assert "tool2" in registry
        assert "tool3" not in registry


class TestFileTools:
    def test_read_file(self, tmp_path):
        file = tmp_path / "test.txt"
        file.write_text("Hello, world!")

        # We need to mock the safe path since our tool restricts to CWD
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = read_file(str(file))
            assert "Hello, world!" in result
        finally:
            os.chdir(original_cwd)

    def test_read_nonexistent_file(self, tmp_path):
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = read_file("nonexistent.txt")
            assert "Error" in result
        finally:
            os.chdir(original_cwd)

    def test_read_directory_as_file(self, tmp_path):
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = read_file(".")
            assert "Error" in result
            assert "directory" in result
        finally:
            os.chdir(original_cwd)

    def test_write_file(self, tmp_path):
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = write_file("output.txt", "test content")
            assert "Successfully wrote" in result
            assert (tmp_path / "output.txt").read_text() == "test content"
        finally:
            os.chdir(original_cwd)

    def test_list_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        (tmp_path / "subdir").mkdir()

        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = list_directory(".")
            assert "a.txt" in result
            assert "b.txt" in result
            assert "subdir/" in result
        finally:
            os.chdir(original_cwd)

    def test_create_file_tools(self):
        registry = create_file_tools()
        assert len(registry) == 3
        assert "read_file" in registry
        assert "write_file" in registry
        assert "list_directory" in registry


class TestWebTools:
    def test_create_web_tools(self):
        registry = create_web_tools()
        assert len(registry) == 2
        assert "web_search" in registry
        assert "fetch_url" in registry

    def test_web_search_no_api_key(self):
        import os
        # Ensure no API key is set
        old_key = os.environ.pop("SERPAPI_API_KEY", None)
        try:
            result = web_search("test query")
            assert "SERPAPI_API_KEY" in result
        finally:
            if old_key:
                os.environ["SERPAPI_API_KEY"] = old_key
