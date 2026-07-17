"""Semantic cache — embedding-based similarity caching for LLM responses."""

from __future__ import annotations

import json
import time
import threading
from pathlib import Path
from typing import Optional
import numpy as np


class SemanticCache:
    def __init__(self, threshold: float = 0.92, max_entries: int = 500, persist_path: str = "data/semantic_cache.json") -> None:
        self.threshold = threshold
        self.max_entries = max_entries
        self.persist_path = Path(persist_path)
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[dict] = []
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if self.persist_path.exists():
            try:
                data = json.loads(self.persist_path.read_text())
                self._entries = data.get("entries", [])
            except Exception:
                self._entries = []

    def _save(self) -> None:
        try:
            self.persist_path.write_text(json.dumps({"entries": self._entries[-self.max_entries:]}))
        except Exception:
            pass

    def _normalize(self, vec: list[float]) -> np.ndarray:
        a = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(a)
        return a / norm if norm > 0 else a

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        return float(np.dot(self._normalize(a), self._normalize(b)))

    def get(self, query_embedding: list[float], model: str = "") -> Optional[str]:
        with self._lock:
            for entry in reversed(self._entries):
                if model and entry.get("model") and entry["model"] != model:
                    continue
                sim = self._cosine_similarity(query_embedding, entry["embedding"])
                if sim >= self.threshold:
                    entry["hits"] = entry.get("hits", 0) + 1
                    entry["last_hit"] = time.time()
                    return entry["response"]
        return None

    def set(self, query: str, query_embedding: list[float], response: str, model: str = "", metadata: Optional[dict] = None) -> None:
        with self._lock:
            self._entries.append({
                "query": query,
                "embedding": query_embedding,
                "response": response,
                "model": model,
                "metadata": metadata or {},
                "hits": 0,
                "created": time.time(),
                "last_hit": time.time(),
            })
            while len(self._entries) > self.max_entries:
                self._entries.pop(0)
        self._save()

    def stats(self) -> dict:
        with self._lock:
            total = len(self._entries)
            hits = sum(e.get("hits", 0) for e in self._entries)
            return {"entries": total, "total_hits": hits, "threshold": self.threshold, "max_entries": self.max_entries}

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
        self._save()

    async def get_embedding(self, text: str, ollama_backend) -> list[float]:
        emb = ""
        async for chunk in ollama_backend.generate(prompt=text, model="nomic-embed-text", temperature=0.0, max_tokens=1, raw=True):
            emb += chunk
        try:
            data = json.loads(emb)
            return data.get("embedding", [])
        except Exception:
            return [0.0] * 768

    async def query(self, text: str, ollama_backend, model: str = "") -> Optional[str]:
        emb = await self.get_embedding(text, ollama_backend)
        if not emb or all(v == 0.0 for v in emb):
            return None
        return self.get(emb, model=model)

    async def store(self, query: str, response: str, ollama_backend, model: str = "", metadata: Optional[dict] = None) -> None:
        emb = await self.get_embedding(query, ollama_backend)
        if emb and not all(v == 0.0 for v in emb):
            self.set(query, emb, response, model=model, metadata=metadata)
