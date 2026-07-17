"""Tests for document ingestion pipeline."""

import pytest
from src.ingestion.pipeline import DocumentParser, ChunkingStrategy, DocumentChunk


@pytest.fixture
def sample_text():
    return "This is a test document.\n\nIt has multiple paragraphs.\n\nAnd some more content." * 10


def test_semantic_chunking(sample_text):
    chunks = ChunkingStrategy.semantic_chunks(sample_text, max_chars=100)
    assert len(chunks) > 0
    for c in chunks:
        assert len(c) <= 100 + 20


def test_recursive_chunking(sample_text):
    chunks = ChunkingStrategy.recursive_character_chunk(sample_text, max_chars=80, overlap=20)
    assert len(chunks) > 0


def test_document_parse(tmp_path):
    file = tmp_path / "test.txt"
    file.write_text("Hello world. " * 10)
    parser = DocumentParser()
    chunks = parser.parse(file)
    assert len(chunks) > 0
    assert all(isinstance(c, DocumentChunk) for c in chunks)


def test_document_parse_batch(tmp_path):
    for name in ["a.txt", "b.txt"]:
        (tmp_path / name).write_text("Content. " * 5)
    parser = DocumentParser()
    chunks = parser.parse_batch([tmp_path / "a.txt", tmp_path / "b.txt"])
    assert len(chunks) == 2


def test_chunk_metadata(tmp_path):
    file = tmp_path / "meta_test.txt"
    file.write_text("Test")
    parser = DocumentParser()
    chunks = parser.parse(file)
    assert chunks[0].metadata.get("format") == "text"
    assert chunks[0].source == str(file)
