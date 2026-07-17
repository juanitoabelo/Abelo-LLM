"""Conversation branching and prompt templates."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.memory.store import MemoryStore

router = APIRouter(prefix="/api/branch", tags=["branching"])


class BranchDB:
    def __init__(self, db_path: str = "data/branches.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS branches (
                id TEXT PRIMARY KEY,
                parent_branch_id TEXT,
                session_id TEXT NOT NULL,
                label TEXT DEFAULT '',
                created_at REAL NOT NULL,
                message_count INTEGER DEFAULT 0,
                snapshot TEXT DEFAULT '[]'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prompt_templates (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                content TEXT NOT NULL,
                variables TEXT DEFAULT '[]',
                category TEXT DEFAULT 'general',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def create_branch(self, session_id: str, label: str = "", snapshot: list[dict] = None) -> str:
        conn = self._get_conn()
        branch_id = str(uuid.uuid4())[:12]
        now = time.time()
        conn.execute(
            "INSERT INTO branches (id, session_id, label, created_at, message_count, snapshot) VALUES (?, ?, ?, ?, ?, ?)",
            (branch_id, session_id, label, now, len(snapshot or []), json.dumps(snapshot or [])),
        )
        conn.commit()
        conn.close()
        return branch_id

    def list_branches(self, session_id: str) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, parent_branch_id, session_id, label, created_at, message_count FROM branches WHERE session_id = ? ORDER BY created_at DESC",
            (session_id,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_branch(self, branch_id: str) -> Optional[dict]:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM branches WHERE id = ?", (branch_id,)).fetchone()
        conn.close()
        if row:
            d = dict(row)
            d["snapshot"] = json.loads(d["snapshot"])
            return d
        return None

    def save_template(self, name: str, content: str, variables: list[str] = None, category: str = "general") -> str:
        conn = self._get_conn()
        now = time.time()
        tid = str(uuid.uuid4())[:12]
        try:
            conn.execute(
                "INSERT OR REPLACE INTO prompt_templates (id, name, content, variables, category, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (tid, name, content, json.dumps(variables or []), category, now, now),
            )
            conn.commit()
            return tid
        finally:
            conn.close()

    def list_templates(self, category: str = "") -> list[dict]:
        conn = self._get_conn()
        if category:
            rows = conn.execute("SELECT * FROM prompt_templates WHERE category = ? ORDER BY name", (category,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM prompt_templates ORDER BY category, name").fetchall()
        conn.close()
        result = []
        for r in rows:
            d = dict(r)
            d["variables"] = json.loads(d["variables"])
            result.append(d)
        return result

    def get_template(self, name: str) -> Optional[dict]:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM prompt_templates WHERE name = ?", (name,)).fetchone()
        conn.close()
        if row:
            d = dict(row)
            d["variables"] = json.loads(d["variables"])
            return d
        return None


_branch_db: Optional[BranchDB] = None


def get_branch_db() -> BranchDB:
    global _branch_db
    if _branch_db is None:
        _branch_db = BranchDB()
    return _branch_db


class CreateBranchRequest(BaseModel):
    session_id: str
    label: str = ""
    messages: list[dict] = []


class SaveTemplateRequest(BaseModel):
    name: str
    content: str
    variables: list[str] = []
    category: str = "general"


class ApplyTemplateRequest(BaseModel):
    name: str
    variables: dict[str, str] = {}


@router.post("/create")
async def create_branch(request: CreateBranchRequest):
    db = get_branch_db()
    branch_id = db.create_branch(request.session_id, request.label, request.messages)
    return {"branch_id": branch_id, "status": "ok"}


@router.get("/list/{session_id}")
async def list_branches(session_id: str):
    db = get_branch_db()
    branches = db.list_branches(session_id)
    return {"branches": branches}


@router.get("/{branch_id}")
async def get_branch(branch_id: str):
    db = get_branch_db()
    branch = db.get_branch(branch_id)
    if not branch:
        raise HTTPException(status_code=404, detail="Branch not found")
    return {"branch": branch}


@router.post("/templates/save")
async def save_template(request: SaveTemplateRequest):
    db = get_branch_db()
    tid = db.save_template(request.name, request.content, request.variables, request.category)
    return {"id": tid, "status": "ok"}


@router.get("/templates")
async def list_templates(category: str = ""):
    db = get_branch_db()
    templates = db.list_templates(category)
    return {"templates": templates}


@router.post("/templates/apply")
async def apply_template(request: ApplyTemplateRequest):
    db = get_branch_db()
    template = db.get_template(request.name)
    if not template:
        raise HTTPException(status_code=404, detail=f"Template '{request.name}' not found")
    content = template["content"]
    for key, value in request.variables.items():
        content = content.replace(f"{{{key}}}", value)
    return {"content": content, "template": template["name"], "category": template["category"]}
