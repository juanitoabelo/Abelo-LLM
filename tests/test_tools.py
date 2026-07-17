from __future__ import annotations

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

    def test_register_custom_tool(self) -> None:
        registry = ToolRegistry()

        def echo_fn(text: str = "") -> ToolResult:
            return ToolResult(success=True, output=text)

        registry.register("echo", "Echoes input", echo_fn, {
            "type": "object",
            "properties": {"text": {"type": "string"}},
        })
        spec = registry.get_specs()
        assert any(s["function"]["name"] == "echo" for s in spec)
        result = registry.execute("echo", {"text": "hello"})
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
        from src.tools.registry import get_default_registry
        registry = get_default_registry()
        names = registry.get_names()
        assert "calculator" in names
