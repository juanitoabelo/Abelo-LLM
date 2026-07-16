from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


class DocumentLoader:
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 64

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_file(self, file_path: str | Path) -> list[dict]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        ext = path.suffix.lower()
        text = self._read_file(path, ext)
        chunks = self._chunk_text(text)
        source = str(path)
        return [
            {
                "source": source,
                "content": chunk,
                "chunk_index": i,
            }
            for i, chunk in enumerate(chunks)
        ]

    def load_directory(self, dir_path: str | Path, pattern: Optional[str] = None) -> list[dict]:
        path = Path(dir_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")
        all_chunks: list[dict] = []
        for file_path in path.rglob(pattern or "*"):
            if file_path.is_file() and self._is_supported(file_path.suffix):
                try:
                    all_chunks.extend(self.load_file(file_path))
                except Exception:
                    continue
        return all_chunks

    def load_text(self, text: str, source: str = "inline") -> list[dict]:
        chunks = self._chunk_text(text)
        return [
            {"source": source, "content": chunk, "chunk_index": i}
            for i, chunk in enumerate(chunks)
        ]

    def _is_supported(self, ext: str) -> bool:
        return ext.lower() in {".txt", ".md", ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp", ".rb", ".php", ".yaml", ".yml", ".toml", ".json", ".xml", ".html", ".css", ".scss", ".sql", ".sh", ".env", ".cfg", ".ini", ".csv"}

    def _read_file(self, path: Path, ext: str) -> str:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception:
            return ""

    def _chunk_text(self, text: str) -> list[str]:
        if not text.strip():
            return []
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                newline_pos = text.rfind("\n", start, end)
                if newline_pos > start + self.chunk_size // 2:
                    end = newline_pos + 1
                else:
                    space_pos = text.rfind(" ", start, end)
                    if space_pos > start + self.chunk_size // 2:
                        end = space_pos + 1
            chunks.append(text[start:end].strip())
            start = end - self.chunk_overlap if end < len(text) else len(text)
        return [c for c in chunks if c]
