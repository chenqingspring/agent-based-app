"""
File system tools — let the agent read, write, and explore files.

These are the most fundamental agent tools. An agent that can't interact with the
filesystem is just a chatbot. These tools turn it into something useful.
"""

import os
from pathlib import Path

from src.tools import Tool, ToolRegistry


def _safe_path(path: str) -> Path:
    """Resolve a path and ensure it's within the current working directory."""
    resolved = Path(path).resolve()
    cwd = Path.cwd().resolve()

    # Only allow paths within the current working directory
    try:
        resolved.relative_to(cwd)
    except ValueError:
        raise ValueError(
            f"Access denied: '{path}' is outside the working directory."
        )
    return resolved


def read_file(path: str) -> str:
    """
    Read the contents of a file.

    Args:
        path: Path to the file to read.
    """
    filepath = _safe_path(path)
    if not filepath.exists():
        return f"Error: File not found: {path}"
    if filepath.is_dir():
        return f"Error: '{path}' is a directory, not a file."

    content = filepath.read_text(encoding="utf-8")
    # Truncate very large files
    if len(content) > 50_000:
        content = content[:50_000] + f"\n... (truncated, {len(content)} total chars)"
    return content


def write_file(path: str, content: str) -> str:
    """
    Write content to a file. Creates the file if it doesn't exist.

    Args:
        path: Path to the file to write.
        content: The content to write.
    """
    filepath = _safe_path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(content, encoding="utf-8")
    return f"Successfully wrote {len(content)} bytes to {path}"


def list_directory(path: str = ".") -> str:
    """
    List the contents of a directory.

    Args:
        path: Path to the directory. Defaults to current directory.
    """
    dirpath = _safe_path(path)
    if not dirpath.exists():
        return f"Error: Directory not found: {path}"
    if not dirpath.is_dir():
        return f"Error: '{path}' is not a directory."

    items = []
    try:
        for entry in sorted(dirpath.iterdir()):
            suffix = "/" if entry.is_dir() else ""
            items.append(f"  {entry.name}{suffix}")
    except PermissionError:
        return f"Error: Permission denied: {path}"

    if not items:
        return f"Directory '{path}' is empty."
    return f"Contents of '{path}':\n" + "\n".join(items)


def create_file_tools() -> ToolRegistry:
    """Create a registry with all file system tools."""
    registry = ToolRegistry()

    registry.register(
        name="read_file",
        description="Read the contents of a file. Use this to examine file contents.",
        parameters={
            "path": {
                "type": "string",
                "description": "Path to the file to read.",
            }
        },
    )(read_file)

    registry.register(
        name="write_file",
        description="Write content to a file. Creates parent directories if needed.",
        parameters={
            "path": {
                "type": "string",
                "description": "Path to the file to write.",
            },
            "content": {
                "type": "string",
                "description": "The content to write to the file.",
            },
        },
    )(write_file)

    registry.register(
        name="list_directory",
        description="List the contents of a directory. Use this to explore what files exist.",
        parameters={
            "path": {
                "type": "string",
                "description": "Path to the directory to list. Defaults to '.'",
            }
        },
    )(list_directory)

    return registry
