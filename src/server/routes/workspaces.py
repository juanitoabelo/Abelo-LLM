"""Multi-tenant workspaces with RBAC — team isolation, quotas, roles."""

from __future__ import annotations

import secrets
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

from src.auth.jwt import verify_token

router = APIRouter(prefix="/api/workspaces", tags=["workspaces"])

WS_DB = Path("data/workspaces.db")
WS_DB.parent.mkdir(parents=True, exist_ok=True)


def _init_db():
    conn = sqlite3.connect(str(WS_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS workspaces (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            owner_id INTEGER NOT NULL,
            api_key TEXT UNIQUE,
            max_users INTEGER DEFAULT 10,
            max_tokens_per_day INTEGER DEFAULT 1000000,
            tokens_used_today INTEGER DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS workspace_members (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL DEFAULT 'member' CHECK(role IN ('owner', 'admin', 'member', 'viewer')),
            created_at REAL NOT NULL,
            UNIQUE(workspace_id, user_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS workspace_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            enabled INTEGER DEFAULT 1,
            created_at REAL NOT NULL,
            UNIQUE(workspace_id, model_name)
        )
    """)
    conn.commit()
    conn.close()


_init_db()


def _get_auth_user(authorization: str = Header("")) -> Optional[dict]:
    if not authorization.startswith("Bearer "):
        return None
    return verify_token(authorization[7:])


class WorkspaceCreate(BaseModel):
    name: str
    max_users: int = 10
    max_tokens_per_day: int = 1000000


class MemberAdd(BaseModel):
    user_id: int
    role: str = "member"


class ModelEnable(BaseModel):
    model_name: str
    enabled: bool = True


@router.post("")
async def create_workspace(req: WorkspaceCreate, authorization: str = Header("")):
    user = _get_auth_user(authorization)
    if not user:
        raise HTTPException(401, "Authentication required")
    conn = sqlite3.connect(str(WS_DB))
    try:
        ws_id = str(uuid.uuid4())[:12]
        api_key = f"ws-{secrets.token_hex(16)}"
        now = time.time()
        conn.execute(
            "INSERT INTO workspaces (id, name, owner_id, api_key, max_users, max_tokens_per_day, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (ws_id, req.name, user["sub"], api_key, req.max_users, req.max_tokens_per_day, now, now),
        )
        conn.execute("INSERT INTO workspace_members (workspace_id, user_id, role, created_at) VALUES (?, ?, 'owner', ?)", (ws_id, user["sub"], now))
        conn.commit()
        return {"workspace_id": ws_id, "api_key": api_key, "name": req.name}
    finally:
        conn.close()


@router.get("")
async def list_workspaces(authorization: str = Header("")):
    user = _get_auth_user(authorization)
    if not user:
        raise HTTPException(401, "Authentication required")
    conn = sqlite3.connect(str(WS_DB))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT w.* FROM workspaces w JOIN workspace_members m ON w.id = m.workspace_id WHERE m.user_id = ?",
            (user["sub"],),
        ).fetchall()
        return {"workspaces": [dict(r) for r in rows]}
    finally:
        conn.close()


@router.post("/{workspace_id}/members")
async def add_member(workspace_id: str, req: MemberAdd, authorization: str = Header("")):
    user = _get_auth_user(authorization)
    if not user:
        raise HTTPException(401, "Authentication required")
    conn = sqlite3.connect(str(WS_DB))
    try:
        owner = conn.execute("SELECT owner_id FROM workspaces WHERE id = ?", (workspace_id,)).fetchone()
        if not owner or owner[0] != user["sub"]:
            raise HTTPException(403, "Only workspace owner can add members")
        conn.execute("INSERT OR IGNORE INTO workspace_members (workspace_id, user_id, role, created_at) VALUES (?, ?, ?, ?)", (workspace_id, req.user_id, req.role, time.time()))
        conn.commit()
        return {"status": "ok"}
    finally:
        conn.close()


@router.get("/{workspace_id}/members")
async def list_members(workspace_id: str, authorization: str = Header("")):
    user = _get_auth_user(authorization)
    if not user:
        raise HTTPException(401, "Authentication required")
    conn = sqlite3.connect(str(WS_DB))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT m.user_id, m.role, m.created_at FROM workspace_members m WHERE m.workspace_id = ?",
            (workspace_id,),
        ).fetchall()
        return {"members": [dict(r) for r in rows]}
    finally:
        conn.close()


@router.post("/{workspace_id}/models")
async def toggle_model(workspace_id: str, req: ModelEnable, authorization: str = Header("")):
    user = _get_auth_user(authorization)
    if not user:
        raise HTTPException(401, "Authentication required")
    conn = sqlite3.connect(str(WS_DB))
    try:
        conn.execute(
            "INSERT OR REPLACE INTO workspace_models (workspace_id, model_name, enabled, created_at) VALUES (?, ?, ?, ?)",
            (workspace_id, req.model_name, int(req.enabled), time.time()),
        )
        conn.commit()
        return {"status": "ok"}
    finally:
        conn.close()


@router.get("/{workspace_id}/stats")
async def workspace_stats(workspace_id: str, authorization: str = Header("")):
    user = _get_auth_user(authorization)
    if not user:
        raise HTTPException(401, "Authentication required")
    conn = sqlite3.connect(str(WS_DB))
    conn.row_factory = sqlite3.Row
    try:
        ws = conn.execute("SELECT * FROM workspaces WHERE id = ?", (workspace_id,)).fetchone()
        if not ws:
            raise HTTPException(404, "Workspace not found")
        member_count = conn.execute("SELECT COUNT(*) FROM workspace_members WHERE workspace_id = ?", (workspace_id,)).fetchone()[0]
        d = dict(ws)
        d["member_count"] = member_count
        return d
    finally:
        conn.close()
