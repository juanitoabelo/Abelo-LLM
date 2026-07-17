from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.memory.store import MemoryStore


class TestMemoryStore:
    @pytest.fixture
    def store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield MemoryStore(str(Path(tmpdir) / "test_memory.db"))

    def test_remember_and_recall(self, store: MemoryStore) -> None:
        store.remember("user_name", "Alice")
        result = store.recall("user_name")
        assert result == "Alice"

    def test_recall_missing(self, store: MemoryStore) -> None:
        result = store.recall("nonexistent")
        assert result is None

    def test_forget(self, store: MemoryStore) -> None:
        store.remember("temp_key", "temp_value")
        assert store.recall("temp_key") == "temp_value"
        store.forget("temp_key")
        assert store.recall("temp_key") is None

    def test_list_keys(self, store: MemoryStore) -> None:
        store.remember("k1", "v1")
        store.remember("k2", "v2")
        keys = store.list_keys()
        assert "k1" in keys
        assert "k2" in keys

    def test_search(self, store: MemoryStore) -> None:
        store.remember("pet_name", "Rex")
        store.remember("city", "Tokyo")
        results = store.search("Rex")
        assert any("Rex" in str(r) for r in results)

    def test_session_management(self, store: MemoryStore) -> None:
        store.create_session("session_1")
        store.log_message("session_1", "user", "hello")
        store.log_message("session_1", "assistant", "hi there")
        context = store.get_session_context("session_1", limit=10)
        assert len(context) == 2
        assert context[0]["content"] == "hello"
