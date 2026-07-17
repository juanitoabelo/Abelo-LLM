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

    def test_add_and_query_entity(self, kg: KnowledgeGraph) -> None:
        kg.add_entity("OpenAI", "organization")
        results = kg.query("OpenAI")
        assert len(results) > 0
        assert any("OpenAI" in str(r) for r in results)

    def test_add_multiple_entities(self, kg: KnowledgeGraph) -> None:
        kg.add_entity("Google", "organization")
        kg.add_entity("Microsoft", "organization")
        entities = kg.get_all_entities()
        names = [e["name"] for e in entities]
        assert "Google" in names
        assert "Microsoft" in names

    def test_add_relationship(self, kg: KnowledgeGraph) -> None:
        kg.add_entity("Alice", "person")
        kg.add_entity("Bob", "person")
        kg.add_relationship("Alice", "Bob", "knows")
        results = kg.query("Alice")
        assert len(results) > 0

    def test_extract_entities(self, kg: KnowledgeGraph) -> None:
        kg.extract_entities("OpenAI and Microsoft are working with Google on AI safety. Apple is also involved.")
        entities = kg.get_all_entities()
        names = [e["name"] for e in entities]
        assert "OpenAI" in names
        assert "Microsoft" in names
        assert "Google" in names
        assert "Apple" in names

    def test_entity_types(self, kg: KnowledgeGraph) -> None:
        kg.add_entity("Tesla", "organization")
        kg.add_entity("Elon Musk", "person")
        kg.add_entity("Python", "technology")
        types = kg.get_entity_types()
        assert "organization" in types
        assert "person" in types
        assert "technology" in types

    def test_traverse(self, kg: KnowledgeGraph) -> None:
        kg.add_entity("A", "person")
        kg.add_entity("B", "person")
        kg.add_entity("C", "person")
        kg.add_relationship("A", "B", "knows")
        kg.add_relationship("B", "C", "knows")
        path = kg.traverse("A", "C", max_depth=5)
        assert path is not None
        assert "A" in path
        assert "C" in path

    def test_traverse_no_path(self, kg: KnowledgeGraph) -> None:
        kg.add_entity("X", "person")
        kg.add_entity("Y", "person")
        path = kg.traverse("X", "Y", max_depth=3)
        assert path is None
