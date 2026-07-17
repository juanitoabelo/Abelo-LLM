"""Knowledge graph - extract entities and relationships from documents.

During RAG ingestion, entities (people, places, organizations, concepts) and
their relationships are extracted and stored in a graph for semantic retrieval.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional


class KnowledgeGraph:
    def __init__(self, db_path: str | Path = "data/knowledge_graph.db") -> None:
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
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT NOT NULL DEFAULT 'concept',
                source TEXT DEFAULT '',
                description TEXT DEFAULT '',
                first_seen REAL NOT NULL,
                last_seen REAL NOT NULL,
                frequency INTEGER DEFAULT 1,
                UNIQUE(name, type)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_entity_id INTEGER NOT NULL,
                target_entity_id INTEGER NOT NULL,
                relation TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                first_seen REAL NOT NULL,
                FOREIGN KEY (source_entity_id) REFERENCES entities(id) ON DELETE CASCADE,
                FOREIGN KEY (target_entity_id) REFERENCES entities(id) ON DELETE CASCADE,
                UNIQUE(source_entity_id, target_entity_id, relation)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_rels_source ON relationships(source_entity_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_rels_target ON relationships(target_entity_id)
        """)
        conn.commit()
        conn.close()

    def add_entity(self, name: str, entity_type: str = "concept", source: str = "", description: str = "") -> int:
        conn = self._get_conn()
        now = time.time()
        cursor = conn.execute(
            """INSERT INTO entities (name, type, source, description, first_seen, last_seen)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(name, type) DO UPDATE SET
               last_seen = excluded.last_seen,
               frequency = frequency + 1,
               description = CASE WHEN excluded.description != '' THEN excluded.description ELSE description END""",
            (name, entity_type, source, description, now, now),
        )
        conn.commit()
        if cursor.lastrowid:
            return cursor.lastrowid
        row = conn.execute("SELECT id FROM entities WHERE name = ? AND type = ?", (name, entity_type)).fetchone()
        return row[0] if row else -1

    def add_relationship(self, source_name: str, target_name: str, relation: str,
                         source_type: str = "concept", target_type: str = "concept") -> None:
        conn = self._get_conn()
        now = time.time()
        src_id = self.add_entity(source_name, source_type)
        tgt_id = self.add_entity(target_name, target_type)
        conn.execute(
            """INSERT INTO relationships (source_entity_id, target_entity_id, relation, first_seen)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(source_entity_id, target_entity_id, relation) DO UPDATE SET
               weight = weight + 0.5""",
            (src_id, tgt_id, relation, now),
        )
        conn.commit()

    def extract_from_text(self, text: str, source: str = "") -> None:
        import re

        patterns = [
            (r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', 'person'),  # Person Name
            (r'\b[A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|Company|Technologies|Systems|Group)\b', 'organization'),
            (r'\b(?:the )?[A-Z][a-z]+ (?:City|Town|County|State|Country|Island|River|Mountain|Lake|Ocean)\b', 'location'),
            (r'\b\d{4}s\b', 'decade'),
            (r'\b(?:GPT|LLM|AI|ML|API|SDK|REST|SQL|HTML|CSS|JSON|YAML|TTS|STT)\b', 'technology'),
        ]

        entities_found: list[tuple[str, str]] = []
        for pattern, etype in patterns:
            for match in re.finditer(pattern, text):
                name = match.group().strip()
                if len(name) > 3:
                    entities_found.append((name, etype))

        seen = set()
        for name, etype in entities_found:
            key = (name.lower(), etype)
            if key not in seen:
                seen.add(key)
                self.add_entity(name, etype, source)

        related_pairs = []
        found_names = list(set(e[0] for e in entities_found))
        for i in range(len(found_names)):
            for j in range(i + 1, len(found_names)):
                distance = self._cooccurrence_distance(text, found_names[i], found_names[j])
                if distance is not None and distance < 500:
                    related_pairs.append((found_names[i], found_names[j]))

        for src, tgt in related_pairs[:20]:
            self.add_relationship(src, tgt, "related_to")

    def _cooccurrence_distance(self, text: str, name1: str, name2: str) -> Optional[int]:
        import re
        positions = []
        for match in re.finditer(re.escape(name1), text):
            positions.append(("a", match.start()))
        for match in re.finditer(re.escape(name2), text):
            positions.append(("b", match.start()))
        positions.sort(key=lambda x: x[1])
        min_dist = None
        for i in range(len(positions) - 1):
            if positions[i][0] != positions[i + 1][0]:
                dist = positions[i + 1][1] - positions[i][1]
                if min_dist is None or dist < min_dist:
                    min_dist = dist
        return min_dist

    def query_related(self, entity_name: str, max_depth: int = 2) -> list[dict]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            entity = conn.execute(
                "SELECT id, name, type FROM entities WHERE name = ?",
                (entity_name,),
            ).fetchone()
            if not entity:
                return []

            visited = {entity["id"]}
            results = []
            queue = [(entity["id"], 0)]

            while queue:
                current_id, depth = queue.pop(0)
                if depth >= max_depth:
                    continue

                rows = conn.execute(
                    """SELECT e.id, e.name, e.type, r.relation, r.weight
                       FROM relationships r
                       JOIN entities e ON e.id = r.target_entity_id
                       WHERE r.source_entity_id = ?
                       UNION
                       SELECT e.id, e.name, e.type, r.relation, r.weight
                       FROM relationships r
                       JOIN entities e ON e.id = r.source_entity_id
                       WHERE r.target_entity_id = ?""",
                    (current_id, current_id),
                ).fetchall()

                for row in rows:
                    if row["id"] not in visited:
                        visited.add(row["id"])
                        results.append({
                            "entity": {"id": row["id"], "name": row["name"], "type": row["type"]},
                            "relation": row["relation"],
                            "weight": row["weight"],
                            "depth": depth + 1,
                        })
                        queue.append((row["id"], depth + 1))
            return results
        finally:
            conn.close()

    def search_entities(self, query: str, limit: int = 20) -> list[dict]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT id, name, type, source, description, frequency FROM entities WHERE name LIKE ? ORDER BY frequency DESC LIMIT ?",
                (f"%{query}%", limit),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_stats(self) -> dict:
        conn = sqlite3.connect(str(self.db_path))
        try:
            entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            relationships = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
            types = conn.execute("SELECT type, COUNT(*) as cnt FROM entities GROUP BY type ORDER BY cnt DESC").fetchall()
            return {
                "entities": entities,
                "relationships": relationships,
                "types": {r[0]: r[1] for r in types},
            }
        finally:
            conn.close()
