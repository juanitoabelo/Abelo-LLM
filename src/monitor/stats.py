from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional


class RequestRecord:
    def __init__(
        self,
        endpoint: str,
        model: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        duration_ms: float = 0.0,
        success: bool = True,
        session_id: Optional[str] = None,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.duration_ms = duration_ms
        self.success = success
        self.session_id = session_id
        self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {
            "endpoint": self.endpoint,
            "model": self.model,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "duration_ms": round(self.duration_ms, 1),
            "success": self.success,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
        }


_CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str) -> int:
    return int(len(text) / _CHARS_PER_TOKEN) + 1


class UsageTracker:
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
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint TEXT NOT NULL,
                model TEXT DEFAULT '',
                tokens_in INTEGER DEFAULT 0,
                tokens_out INTEGER DEFAULT 0,
                duration_ms REAL DEFAULT 0,
                success INTEGER DEFAULT 1,
                session_id TEXT DEFAULT '',
                timestamp REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(timestamp)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_requests_endpoint ON requests(endpoint)
        """)
        conn.commit()
        conn.close()

    def log_request(self, record: RequestRecord) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO requests (endpoint, model, tokens_in, tokens_out, duration_ms, success, session_id, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (record.endpoint, record.model, record.tokens_in, record.tokens_out,
             record.duration_ms, 1 if record.success else 0,
             record.session_id or "", record.timestamp),
        )
        conn.commit()

    def get_stats(self, hours: int = 24) -> dict:
        conn = sqlite3.connect(str(self.db_path))
        try:
            cutoff = time.time() - (hours * 3600)
            rows = conn.execute(
                "SELECT endpoint, model, tokens_in, tokens_out, duration_ms, success, session_id, timestamp FROM requests WHERE timestamp > ? ORDER BY timestamp DESC",
                (cutoff,),
            ).fetchall()
        finally:
            conn.close()

        total_tokens_in = sum(r[2] for r in rows)
        total_tokens_out = sum(r[3] for r in rows)
        total_duration = sum(r[4] for r in rows)
        total_requests = len(rows)
        successful = sum(1 for r in rows if r[5])
        failed = total_requests - successful

        endpoint_counts: dict[str, int] = {}
        model_counts: dict[str, int] = {}
        for r in rows:
            endpoint_counts[r[0]] = endpoint_counts.get(r[0], 0) + 1
            if r[1]:
                model_counts[r[1]] = model_counts.get(r[1], 0) + 1

        return {
            "total_requests": total_requests,
            "successful": successful,
            "failed": failed,
            "total_tokens_in": total_tokens_in,
            "total_tokens_out": total_tokens_out,
            "total_tokens": total_tokens_in + total_tokens_out,
            "total_duration_ms": round(total_duration, 1),
            "avg_duration_ms": round(total_duration / total_requests, 1) if total_requests else 0,
            "per_endpoint": endpoint_counts,
            "per_model": model_counts,
            "timeframe_hours": hours,
        }

    def get_recent_requests(self, limit: int = 20) -> list[dict]:
        conn = sqlite3.connect(str(self.db_path))
        try:
            rows = conn.execute(
                "SELECT endpoint, model, tokens_in, tokens_out, duration_ms, success, session_id, timestamp FROM requests ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [
                {
                    "endpoint": r[0], "model": r[1], "tokens_in": r[2],
                    "tokens_out": r[3], "duration_ms": round(r[4], 1),
                    "success": bool(r[5]), "session_id": r[6], "timestamp": r[7],
                }
                for r in rows
            ]
        finally:
            conn.close()
