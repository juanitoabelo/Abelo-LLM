"""Memory tools for the agent to store and retrieve persistent information."""

from __future__ import annotations

from pathlib import Path

from src.tools.registry import ToolResult


def _get_memory_store():
    from src.config.settings import get_settings
    from src.memory.store import MemoryStore
    settings = get_settings()
    base_path = Path(settings.data_dir).parent
    return MemoryStore(str(base_path / "memory.db"))


def memory_recall(key: str) -> ToolResult:
    store = _get_memory_store()
    value = store.recall(key)
    if value:
        return ToolResult(True, value)
    return ToolResult(True, f"No memory found for key: {key}")


def memory_remember(key: str, value: str, importance: float = 1.0) -> ToolResult:
    store = _get_memory_store()
    store.remember(key, value, importance=importance)
    return ToolResult(True, f"Remembered: {key} = {value[:100]}" + ("..." if len(value) > 100 else ""))


def memory_search(query: str) -> ToolResult:
    store = _get_memory_store()
    results = store.search(query, limit=10)
    if not results:
        return ToolResult(True, "No matching memories found.")
    output = ""
    for r in results:
        output += f"- {r['key']}: {r['value'][:200]}\n"
    return ToolResult(True, output.strip())
