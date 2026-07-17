from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.knowledge_graph.graph import KnowledgeGraph


class TestKnowledgeGraph:
    @pytest.fixture
    def kg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield KnowledgeGraph(str(Path(tmpdir) / "test_kg.db"))

    def test_add_entity(self, kg: KnowledgeGraph) -> None:
        eid = kg.add_entity("OpenAI", "organization")
        assert eid > 0

    def test_search_entities(self, kg: KnowledgeGraph) -> None:
        kg.add_entity("Google", "organization")
        kg.add_entity("Microsoft", "organization")
        results = kg.search_entities("Google")
        assert len(results) > 0
        names = [r["name"] for r in results]
        assert "Google" in names

    def test_add_relationship(self, kg: KnowledgeGraph) -> None:
        kg.add_entity("Alice", "person")
        kg.add_entity("Bob", "person")
        kg.add_relationship("Alice", "Bob", "knows")
        related = kg.query_related("Alice")
        assert len(related) > 0

    def test_extract_from_text(self, kg: KnowledgeGraph) -> None:
        kg.extract_from_text("Google Inc and Microsoft Corp are working together. John Smith leads the project.")
        results = kg.search_entities("Google")
        assert len(results) > 0

    def test_get_stats(self, kg: KnowledgeGraph) -> None:
        kg.add_entity("Tesla", "organization")
        kg.add_entity("Python", "technology")
        stats = kg.get_stats()
        assert stats["entities"] == 2
        assert "organization" in stats["types"]
        assert "technology" in stats["types"]

    def test_query_related_depth(self, kg: KnowledgeGraph) -> None:
        kg.add_relationship("A", "B", "knows", source_type="person", target_type="person")
        kg.add_relationship("B", "C", "knows", source_type="person", target_type="person")
        related = kg.query_related("A", max_depth=3)
        names = [r["entity"]["name"] for r in related]
        assert "B" in names
        assert "C" in names
