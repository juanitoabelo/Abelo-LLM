from __future__ import annotations

import pytest

from src.tools.registry import ToolRegistry, ToolResult


class TestToolRegistry:
    def test_default_registry_has_tools(self) -> None:
        from src.tools.registry import get_default_registry
        registry = get_default_registry()
        specs = registry.get_specs()
        names = [s["function"]["name"] for s in specs]
        assert "calculator" in names
        assert "web_search" in names
        assert "web_fetch" in names
        assert "code_execute" in names
        assert "file_read" in names
        assert "memory_recall" in names
        assert "rag_query" in names
        assert "sql_query" in names

    def test_register_custom_tool(self) -> None:
        registry = ToolRegistry()
        registry.register("echo", lambda args: ToolResult(success=True, output=str(args)))
        spec = registry.get_specs()
        assert any(s["function"]["name"] == "echo" for s in spec)
        result = registry.execute("echo", {"msg": "hello"})
        assert result.success
        assert "hello" in result.output

    def test_execute_unknown_tool(self) -> None:
        registry = ToolRegistry()
        result = registry.execute("nonexistent", {})
        assert not result.success

    def test_execute_calculator(self) -> None:
        from src.tools.registry import get_default_registry
        registry = get_default_registry()
        result = registry.execute("calculator", {"expression": "2 + 2"})
        assert result.success
        assert "4" in result.output

    def test_get_names(self) -> None:
        registry = ToolRegistry()

        @registry.register("test_tool")
        def _handler(args):
            return ToolResult(success=True, output="ok")

        names = registry.get_names()
        assert "test_tool" in names
