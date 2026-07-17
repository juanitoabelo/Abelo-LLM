from __future__ import annotations

import json
import math
import sqlite3
import threading
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avg_doc_len = 0.0
        self.doc_lengths: list[int] = []
        self.inverted_index: dict[str, list[tuple[int, int]]] = {}
        self.doc_id_map: list[int] = {}
        self._built = False

    def build(self, documents: list[tuple[int, str]]) -> None:
        self.doc_id_map = {}
        term_doc_freq: dict[str, set[int]] = {}
        self.doc_lengths = []

        for idx, (doc_id, content) in enumerate(documents):
            self.doc_id_map[idx] = doc_id
            tokens = content.lower().split()
            self.doc_lengths.append(len(tokens))
            term_counts: dict[str, int] = {}
            for t in tokens:
                term_counts[t] = term_counts.get(t, 0) + 1
            for term, count in term_counts.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append((idx, count))
                if term not in term_doc_freq:
                    term_doc_freq[term] = set()
                term_doc_freq[term].add(idx)

        self.doc_count = len(documents)
        self.avg_doc_len = sum(self.doc_lengths) / max(1, self.doc_count)
        self._built = True

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        if not self._built:
            return []
        query_terms = query.lower().split()
        scores: Counter[int] = Counter()

        for term in query_terms:
            if term not in self.inverted_index:
                continue
            posting = self.inverted_index[term]
            doc_freq = len(posting)
            idf = math.log((self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

            for doc_idx, term_freq in posting:
                doc_len = self.doc_lengths[doc_idx]
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                scores[doc_idx] += idf * numerator / denominator

        top = scores.most_common(top_k)
        return [(self.doc_id_map[doc_idx], score) for doc_idx, score in top]


class CrossEncoderReranker:
    def __init__(self) -> None:
        self._available = False
        self._model = None
        self._tokenizer = None

    async def rerank(self, query: str, results: list[dict], top_k: int = 3) -> list[dict]:
        if not results:
            return results

        try:
            return await self._rerank_with_llm(query, results, top_k)
        except Exception:
            return results[:top_k]

    async def _rerank_with_llm(self, query: str, results: list[dict], top_k: int) -> list[dict]:
        import json as _json
        from urllib.request import Request, urlopen

        pairs = [(query, r["content"][:500]) for r in results]
        prompt = "Rate the relevance of each document to the query from 0.0 to 1.0.\n\n"
        for i, (q, doc) in enumerate(pairs):
            prompt += f"Document {i}: {doc}\n\n"
        prompt += f"Query: {query}\n\nReturn ONLY a JSON array of scores: [score_0, score_1, ...]"

        payload = _json.dumps({
            "model": "llama3.2:1b", "prompt": prompt,
            "stream": False, "options": {"temperature": 0.1},
        }).encode()
        req = Request("http://localhost:11434/api/generate", data=payload, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=30) as resp:
            data = _json.loads(resp.read().decode())
            response = data.get("response", "").strip()

        import re
        match = re.search(r"\[.*?\]", response, re.DOTALL)
        if match:
            try:
                scores = _json.loads(match.group(0))
                for i, score in enumerate(scores):
                    if i < len(results):
                        results[i]["rerank_score"] = float(score)
                results.sort(key=lambda x: x.get("rerank_score", x["similarity"]), reverse=True)
            except Exception:
                pass
        return results[:top_k]


class VectorStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._bm25 = BM25Index()
        self._bm25_built = False
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

    def _rebuild_bm25(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        try:
            rows = conn.execute("SELECT id, content FROM documents").fetchall()
            documents = [(r[0], r[1]) for r in rows]
            if documents:
                self._bm25.build(documents)
                self._bm25_built = True
        finally:
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

        self._rebuild_bm25()
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

    def hybrid_search(
        self,
        query: str,
        query_vector: list[float],
        top_k: int = 5,
        alpha: float = 0.5,
    ) -> list[dict]:
        semantic_results = self.similarity_search(query_vector, top_k=top_k * 2)

        if not self._bm25_built:
            self._rebuild_bm25()
        bm25_results = self._bm25.search(query, top_k=top_k * 2)

        id_map: dict[int, dict] = {}
        for doc in semantic_results:
            doc_id = doc["id"]
            doc["semantic_score"] = doc["similarity"]
            doc["bm25_score"] = 0.0
            id_map[doc_id] = doc

        for doc_id, score in bm25_results:
            if doc_id in id_map:
                id_map[doc_id]["bm25_score"] = score
            else:
                conn = sqlite3.connect(str(self.db_path))
                try:
                    row = conn.execute(
                        "SELECT id, source, chunk_index, content, metadata FROM documents WHERE id = ?",
                        (doc_id,),
                    ).fetchone()
                    if row:
                        id_map[doc_id] = {
                            "id": row[0], "source": row[1], "chunk_index": row[2],
                            "content": row[3], "metadata": json.loads(row[4] or "{}"),
                            "similarity": 0.0, "semantic_score": 0.0, "bm25_score": score,
                        }
                finally:
                    conn.close()

        if not id_map:
            return []

        scores = [d.get("semantic_score", 0.0) for d in id_map.values()]
        bm25_scores = [d.get("bm25_score", 0.0) for d in id_map.values()]
        max_sem = max(scores) if scores else 1.0
        max_bm25 = max(bm25_scores) if bm25_scores else 1.0

        for doc in id_map.values():
            sem_norm = doc.get("semantic_score", 0.0) / max_sem
            bm25_norm = doc.get("bm25_score", 0.0) / max_bm25
            doc["hybrid_score"] = alpha * sem_norm + (1 - alpha) * bm25_norm

        combined = sorted(id_map.values(), key=lambda x: x["hybrid_score"], reverse=True)
        return combined[:top_k]

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
        self._rebuild_bm25()

    def count_documents(self) -> int:
        conn = sqlite3.connect(str(self.db_path))
        try:
            row = conn.execute("SELECT COUNT(*) FROM documents").fetchone()
            return row[0] if row else 0
        finally:
            conn.close()
