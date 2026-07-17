from __future__ import annotations

import json
from typing import Any, Callable, Optional


class ToolResult:
    def __init__(self, success: bool, output: str, error: Optional[str] = None) -> None:
        self.success = success
        self.output = output
        self.error = error

    def to_dict(self) -> dict:
        return {"success": self.success, "output": self.output, "error": self.error}

    def __str__(self) -> str:
        return self.output if self.success else f"Error: {self.error}"


ToolFunc = Callable[..., ToolResult]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, dict] = {}

    def register(
        self,
        name: str,
        description: str,
        fn: ToolFunc,
        parameters: dict,
    ) -> None:
        self._tools[name] = {
            "fn": fn,
            "spec": {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            },
        }

    def get_specs(self) -> list[dict]:
        return [t["spec"] for t in self._tools.values()]

    def execute(self, name: str, arguments: dict | str) -> ToolResult:
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(False, "", f"Unknown tool: {name}")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return ToolResult(False, "", f"Invalid JSON arguments: {arguments}")
        try:
            return tool["fn"](**arguments)
        except Exception as e:
            return ToolResult(False, "", f"Tool execution failed: {e}")

    def get_names(self) -> list[str]:
        return list(self._tools.keys())


_default_registry: Optional[ToolRegistry] = None


def get_default_registry() -> ToolRegistry:
    global _default_registry
    if _default_registry is None:
        val = _build_default_registry()
        _default_registry = val
        return val
    return _default_registry


def _build_default_registry() -> ToolRegistry:
    from src.tools.web_search import web_search
    from src.tools.web_fetch import web_fetch
    from src.tools.calculator import calculator
    from src.tools.code_exec import code_execute
    from src.tools.file_io import file_read, file_write, file_list
    from src.tools.rag_query_tool import rag_query, sql_query
    from src.tools.memory_tools import memory_recall, memory_remember, memory_search

    registry = ToolRegistry()

    registry.register(
        name="web_search",
        description="Search the web for current information. Use when you need up-to-date information or facts beyond your training data.",
        fn=web_search,
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
            },
            "required": ["query"],
        },
    )
    registry.register(
        name="web_fetch",
        description="Fetch and read the content of a URL. Use to get detailed information from a specific webpage.",
        fn=web_fetch,
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch"},
            },
            "required": ["url"],
        },
    )
    registry.register(
        name="calculator",
        description="Evaluate a mathematical expression. Supports basic arithmetic, exponentiation, trig, log, and common math functions.",
        fn=calculator,
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "The mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(144)', '3 * 4.5 + 2')"},
            },
            "required": ["expression"],
        },
    )
    registry.register(
        name="code_execute",
        description="Execute Python code in a sandboxed environment. The code runs with safe builtins and no access to os/sys/subprocess. Uses print() for output.",
        fn=code_execute,
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The Python code to execute"},
                "timeout": {"type": "integer", "description": "Execution timeout in seconds", "default": 10},
            },
            "required": ["code"],
        },
    )
    registry.register(
        name="file_read",
        description="Read the contents of a file from the allowed directories (data/, artifacts/, uploads/, ./).",
        fn=file_read,
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read"},
            },
            "required": ["path"],
        },
    )
    registry.register(
        name="file_write",
        description="Write content to a file in the allowed directories. Creates parent directories if needed.",
        fn=file_write,
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to write to"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
    )
    registry.register(
        name="file_list",
        description="List files and directories in the specified directory.",
        fn=file_list,
        parameters={
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Directory to list (default: .)"},
            },
        },
    )
    registry.register(
        name="rag_query",
        description="Search the knowledge base (RAG vector store) for documents relevant to a query.",
        fn=rag_query,
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "top_k": {"type": "integer", "description": "Number of results to return", "default": 3},
            },
            "required": ["query"],
        },
    )
    registry.register(
        name="sql_query",
        description="Execute a SELECT SQL query against the local databases (rag_store, memory, usage, model_registry). Only SELECT is allowed.",
        fn=sql_query,
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The SELECT SQL query"},
                "limit": {"type": "integer", "description": "Max results per database", "default": 20},
            },
            "required": ["query"],
        },
    )
    registry.register(
        name="memory_recall",
        description="Recall a stored memory by key from the persistent memory store.",
        fn=memory_recall,
        parameters={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "The memory key to recall"},
            },
            "required": ["key"],
        },
    )
    registry.register(
        name="memory_remember",
        description="Store a fact in persistent memory for future conversations.",
        fn=memory_remember,
        parameters={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "The memory key"},
                "value": {"type": "string", "description": "The value to remember"},
                "importance": {"type": "number", "description": "Importance score (0.0 to 1.0)", "default": 1.0},
            },
            "required": ["key", "value"],
        },
    )
    registry.register(
        name="memory_search",
        description="Search through stored memories by keyword.",
        fn=memory_search,
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
            },
            "required": ["query"],
        },
    )
    return registry
