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

    registry = ToolRegistry()
    registry.register(
        name="web_search",
        description="Search the web for current information. Use this when you need up-to-date information or facts beyond your training data.",
        fn=web_search,
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
            },
            "required": ["query"],
        },
    )
    registry.register(
        name="web_fetch",
        description="Fetch and read the content of a URL. Use this to get detailed information from a specific webpage.",
        fn=web_fetch,
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch",
                },
            },
            "required": ["url"],
        },
    )
    registry.register(
        name="calculator",
        description="Evaluate a mathematical expression. Supports basic arithmetic, exponentiation, and common math functions.",
        fn=calculator,
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(144)', '3 * 4.5 + 2')",
                },
            },
            "required": ["expression"],
        },
    )
    return registry
