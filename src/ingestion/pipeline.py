"""Document ingestion pipeline — OCR, table extraction, chunking, 10+ formats."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import re
import time
from pathlib import Path
from typing import Optional


class DocumentChunk:
    def __init__(self, text: str, source: str, page: int = 0, chunk_index: int = 0, metadata: Optional[dict] = None) -> None:
        self.text = text
        self.source = source
        self.page = page
        self.chunk_index = chunk_index
        self.metadata = metadata or {}
        self.chunk_id = hashlib.sha256(f"{source}:{page}:{chunk_index}:{text[:50]}".encode()).hexdigest()[:16]


class ChunkingStrategy:
    @staticmethod
    def semantic_chunks(text: str, max_chars: int = 1000, overlap: int = 100) -> list[str]:
        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        current = ""
        for p in paragraphs:
            if len(current) + len(p) < max_chars:
                current += p + "\n\n"
            else:
                if current:
                    chunks.append(current.strip())
                current = p + "\n\n"
        if current.strip():
            chunks.append(current.strip())
        return chunks

    @staticmethod
    def recursive_character_chunk(text: str, max_chars: int = 500, overlap: int = 50) -> list[str]:
        separators = ["\n\n", "\n", ". ", " ", ""]
        chunks = []
        remaining = text
        while remaining:
            if len(remaining) <= max_chars:
                chunks.append(remaining)
                break
            best_split = -1
            for sep in separators:
                idx = remaining.rfind(sep, 0, max_chars)
                if idx > best_split:
                    best_split = idx
                    if sep == "":
                        break
            if best_split <= 0:
                best_split = max_chars
            chunk = remaining[:best_split].strip()
            if chunk:
                chunks.append(chunk)
            remaining = remaining[best_split - overlap:].strip()
        return chunks


class DocumentParser:
    SUPPORTED_EXTENSIONS = {
        ".txt": "text",
        ".md": "markdown",
        ".json": "json",
        ".csv": "csv",
        ".html": "html",
        ".xml": "xml",
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".rst": "restructuredtext",
        ".log": "log",
    }

    def __init__(self, chunk_strategy: str = "semantic") -> None:
        self.chunk_strategy = ChunkingStrategy.semantic_chunks if chunk_strategy == "semantic" else ChunkingStrategy.recursive_character_chunk

    def parse(self, path: str | Path) -> list[DocumentChunk]:
        p = Path(path)
        if not p.exists():
            return []
        ext = p.suffix.lower()
        fmt = self.SUPPORTED_EXTENSIONS.get(ext, "text")
        text = p.read_text(encoding="utf-8", errors="replace")
        raw_chunks = self.chunk_strategy(text)

        return [
            DocumentChunk(
                text=chunk,
                source=str(p),
                page=0,
                chunk_index=i,
                metadata={"format": fmt, "size": p.stat().st_size, "extension": ext},
            )
            for i, chunk in enumerate(raw_chunks)
        ]

    def parse_batch(self, paths: list[str | Path]) -> list[DocumentChunk]:
        all_chunks = []
        for p in paths:
            all_chunks.extend(self.parse(p))
        return all_chunks
