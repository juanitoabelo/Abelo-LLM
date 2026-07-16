from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional

import numpy as np


class VectorStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path))
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        return self._local.conn

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source, chunk_index)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                doc_id INTEGER NOT NULL,
                vector BLOB NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source)
        """)
        conn.commit()
        conn.close()

    def add_documents(
        self,
        sources: list[str],
        chunk_indices: list[int],
        contents: list[str],
        vectors: list[list[float]],
        metadatas: Optional[list[dict]] = None,
    ) -> list[int]:
        conn = self._get_conn()
        doc_ids: list[int] = []
        conn.execute("BEGIN")
        try:
            for i, (src, ci, content, vec) in enumerate(
                zip(sources, chunk_indices, contents, vectors)
            ):
                meta = json.dumps(metadatas[i] if metadatas else {})
                cursor = conn.execute(
                    """INSERT OR REPLACE INTO documents (source, chunk_index, content, metadata)
                       VALUES (?, ?, ?, ?)""",
                    (src, ci, content, meta),
                )
                doc_id = cursor.lastrowid
                if doc_id is not None:
                    vec_bytes = np.array(vec, dtype=np.float32).tobytes()
                    conn.execute(
                        "INSERT OR REPLACE INTO embeddings (doc_id, vector) VALUES (?, ?)",
                        (doc_id, vec_bytes),
                    )
                    doc_ids.append(doc_id)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return doc_ids

    def similarity_search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        conn = sqlite3.connect(str(self.db_path))
        try:
            rows = conn.execute(
                "SELECT d.id, d.source, d.chunk_index, d.content, d.metadata, e.vector "
                "FROM documents d JOIN embeddings e ON d.id = e.doc_id"
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return []

        query_np = np.array(query_vector, dtype=np.float32)
        results: list[dict] = []
        for row in rows:
            doc_id, source, chunk_idx, content, metadata_json, vec_bytes = row
            vec_np = np.frombuffer(vec_bytes, dtype=np.float32)
            sim = float(np.dot(query_np, vec_np) / (
                np.linalg.norm(query_np) * np.linalg.norm(vec_np) + 1e-10
            ))
            results.append({
                "id": doc_id,
                "source": source,
                "chunk_index": chunk_idx,
                "content": content,
                "metadata": json.loads(metadata_json) if metadata_json else {},
                "similarity": sim,
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def get_sources(self) -> list[str]:
        conn = sqlite3.connect(str(self.db_path))
        try:
            rows = conn.execute(
                "SELECT DISTINCT source FROM documents ORDER BY source"
            ).fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()

    def delete_source(self, source: str) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM documents WHERE source = ?", (source,))
        conn.commit()

    def count_documents(self) -> int:
        conn = sqlite3.connect(str(self.db_path))
        try:
            row = conn.execute("SELECT COUNT(*) FROM documents").fetchone()
            return row[0] if row else 0
        finally:
            conn.close()
