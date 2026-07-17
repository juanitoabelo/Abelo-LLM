"""Model registry for tracking available models and their health."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class ModelInfo:
    name: str
    provider: str  # ollama, local, mcp
    size_bytes: int = 0
    family: str = ""
    quantization: str = ""
    context_length: int = 2048
    status: str = "unknown"  # available, unavailable, loading, error
    latency_ms: float = 0.0
    last_checked: float = 0.0
    tags: list[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if d.get("tags") is None:
            d["tags"] = []
        return d


class ModelRegistry:
    def __init__(self, db_path: str | Path = "data/model_registry.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._cache: dict[str, ModelInfo] = {}
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path))
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS models (
                name TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                size_bytes INTEGER DEFAULT 0,
                family TEXT DEFAULT '',
                quantization TEXT DEFAULT '',
                context_length INTEGER DEFAULT 2048,
                status TEXT DEFAULT 'unknown',
                latency_ms REAL DEFAULT 0,
                last_checked REAL DEFAULT 0,
                tags TEXT DEFAULT '[]'
            )
        """)
        conn.commit()
        conn.close()

    def register(self, info: ModelInfo) -> None:
        conn = self._get_conn()
        tags_json = json.dumps(info.tags or [])
        conn.execute(
            """INSERT OR REPLACE INTO models
               (name, provider, size_bytes, family, quantization, context_length, status, latency_ms, last_checked, tags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (info.name, info.provider, info.size_bytes, info.family, info.quantization,
             info.context_length, info.status, info.latency_ms, info.last_checked, tags_json),
        )
        conn.commit()
        self._cache[info.name] = info

    def update_status(self, name: str, status: str, latency_ms: float = 0.0) -> None:
        now = time.time()
        conn = self._get_conn()
        conn.execute(
            "UPDATE models SET status = ?, latency_ms = ?, last_checked = ? WHERE name = ?",
            (status, latency_ms, now, name),
        )
        conn.commit()
        if name in self._cache:
            self._cache[name].status = status
            self._cache[name].latency_ms = latency_ms
            self._cache[name].last_checked = now

    def get_model(self, name: str) -> Optional[ModelInfo]:
        if name in self._cache:
            return self._cache[name]
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM models WHERE name = ?", (name,)).fetchone()
            if row:
                info = ModelInfo(
                    name=row["name"], provider=row["provider"],
                    size_bytes=row["size_bytes"], family=row["family"],
                    quantization=row["quantization"], context_length=row["context_length"],
                    status=row["status"], latency_ms=row["latency_ms"],
                    last_checked=row["last_checked"], tags=json.loads(row["tags"] or "[]"),
                )
                self._cache[name] = info
                return info
        finally:
            conn.close()
        return None

    def list_models(self, provider: Optional[str] = None) -> list[ModelInfo]:
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.row_factory = sqlite3.Row
            if provider:
                rows = conn.execute("SELECT * FROM models WHERE provider = ? ORDER BY name", (provider,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM models ORDER BY provider, name").fetchall()
            result = []
            for row in rows:
                result.append(ModelInfo(
                    name=row["name"], provider=row["provider"],
                    size_bytes=row["size_bytes"], family=row["family"],
                    quantization=row["quantization"], context_length=row["context_length"],
                    status=row["status"], latency_ms=row["latency_ms"],
                    last_checked=row["last_checked"], tags=json.loads(row["tags"] or "[]"),
                ))
            return result
        finally:
            conn.close()

    def get_health_summary(self) -> dict:
        models = self.list_models()
        available = sum(1 for m in models if m.status == "available")
        unavailable = sum(1 for m in models if m.status == "unavailable")
        total = len(models)
        avg_latency = sum(m.latency_ms for m in models if m.status == "available") / max(1, available)
        return {
            "total_models": total,
            "available": available,
            "unavailable": unavailable,
            "avg_latency_ms": round(avg_latency, 1),
            "by_provider": {},
        }

    def sync_from_ollama(self, ollama_models: list[dict]) -> None:
        for m in ollama_models:
            name = m.get("name", "")
            details = m.get("details", {})
            self.register(ModelInfo(
                name=name,
                provider="ollama",
                size_bytes=m.get("size", 0),
                family=details.get("family", ""),
                quantization=details.get("quantization_level", ""),
                context_length=details.get("context_length", 2048),
                status="available",
                tags=["ollama"],
            ))

    def remove(self, name: str) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM models WHERE name = ?", (name,))
        conn.commit()
        self._cache.pop(name, None)
