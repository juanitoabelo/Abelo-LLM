from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional


class MemoryStore:
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
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                namespace TEXT DEFAULT 'default',
                importance REAL DEFAULT 1.0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE(key, namespace)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(key)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                summary TEXT DEFAULT '',
                message_count INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            )
        """)
        conn.commit()
        conn.close()

    def remember(self, key: str, value: str, namespace: str = "default", importance: float = 1.0) -> None:
        now = time.time()
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO memories (key, value, namespace, importance, created_at, updated_at)
               VALUES (?, ?, ?, ?, COALESCE((SELECT created_at FROM memories WHERE key=? AND namespace=?), ?), ?)""",
            (key, value, namespace, importance, key, namespace, now, now),
        )
        conn.commit()

    def recall(self, key: str, namespace: str = "default") -> Optional[str]:
        conn = sqlite3.connect(str(self.db_path))
        try:
            row = conn.execute(
                "SELECT value FROM memories WHERE key = ? AND namespace = ?",
                (key, namespace),
            ).fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def search(self, query: str, namespace: str = "default", limit: int = 10) -> list[dict]:
        conn = sqlite3.connect(str(self.db_path))
        try:
            rows = conn.execute(
                "SELECT key, value, importance, updated_at FROM memories WHERE namespace = ? ORDER BY importance DESC, updated_at DESC LIMIT ?",
                (namespace, limit),
            ).fetchall()
            results = []
            q = query.lower()
            for key, value, importance, updated_at in rows:
                if q in key.lower() or q in value.lower():
                    results.append({
                        "key": key,
                        "value": value,
                        "importance": importance,
                        "updated_at": updated_at,
                    })
            return results
        finally:
            conn.close()

    def forget(self, key: str, namespace: str = "default") -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM memories WHERE key = ? AND namespace = ?", (key, namespace))
        conn.commit()

    def get_all_memories(self, namespace: str = "default") -> list[dict]:
        conn = sqlite3.connect(str(self.db_path))
        try:
            rows = conn.execute(
                "SELECT key, value, importance, created_at, updated_at FROM memories WHERE namespace = ? ORDER BY importance DESC, updated_at DESC",
                (namespace,),
            ).fetchall()
            return [
                {"key": k, "value": v, "importance": imp, "created_at": ca, "updated_at": ua}
                for k, v, imp, ca, ua in rows
            ]
        finally:
            conn.close()

    def create_session(self, session_id: str) -> None:
        now = time.time()
        conn = self._get_conn()
        conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, created_at, updated_at) VALUES (?, ?, ?)",
            (session_id, now, now),
        )
        conn.commit()

    def log_message(self, session_id: str, role: str, content: str) -> None:
        now = time.time()
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO session_messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, content, now),
        )
        conn.execute(
            "UPDATE sessions SET message_count = message_count + 1, updated_at = ? WHERE session_id = ?",
            (now, session_id),
        )
        conn.commit()

    def update_session_summary(self, session_id: str, summary: str) -> None:
        now = time.time()
        conn = self._get_conn()
        conn.execute(
            "UPDATE sessions SET summary = ?, updated_at = ? WHERE session_id = ?",
            (summary, now, session_id),
        )
        conn.commit()

    def get_session_context(self, session_id: str, limit: int = 20) -> list[dict]:
        conn = sqlite3.connect(str(self.db_path))
        try:
            msgs = conn.execute(
                "SELECT role, content FROM session_messages WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
            return [{"role": r, "content": c} for r, c in reversed(msgs)]
        finally:
            conn.close()

    def get_recent_sessions(self, limit: int = 5) -> list[dict]:
        conn = sqlite3.connect(str(self.db_path))
        try:
            rows = conn.execute(
                "SELECT session_id, summary, message_count, created_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [
                {"session_id": s, "summary": summary, "message_count": mc, "created_at": ca, "updated_at": ua}
                for s, summary, mc, ca, ua in rows
            ]
        finally:
            conn.close()
