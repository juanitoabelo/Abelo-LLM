"""Tests for multi-tenant workspaces."""

import json
import sqlite3
import time
from pathlib import Path

import pytest


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_ws.db")


def test_db_is_initialized():
    """Verify workspaces module imports and sets up DB schema."""
    import src.server.routes.workspaces as ws
    assert ws.WS_DB.exists()
    conn = sqlite3.connect(str(ws.WS_DB))
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = [t[0] for t in tables]
    assert "workspaces" in table_names
    assert "workspace_members" in table_names
    assert "workspace_models" in table_names
    conn.close()


def test_routes_registered():
    import src.server.routes.workspaces as ws
    routes = [r.path for r in ws.router.routes]
    assert "/api/workspaces" in routes
    assert "/api/workspaces/{workspace_id}/members" in routes
    assert "/api/workspaces/{workspace_id}/stats" in routes


def test_manual_db_ops(tmp_path):
    db = str(tmp_path / "manual.db")
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS workspaces (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, owner_id INTEGER NOT NULL,
            api_key TEXT UNIQUE, max_users INTEGER DEFAULT 10,
            max_tokens_per_day INTEGER DEFAULT 1000000, tokens_used_today INTEGER DEFAULT 0,
            created_at REAL NOT NULL, updated_at REAL NOT NULL
        )
    """)
    conn.execute(
        "INSERT INTO workspaces (id, name, owner_id, api_key, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("ws_1", "Test WS", 1, "test-key-123", time.time(), time.time()),
    )
    conn.execute("""
        CREATE TABLE IF NOT EXISTS workspace_members (
            id INTEGER PRIMARY KEY AUTOINCREMENT, workspace_id TEXT NOT NULL,
            user_id INTEGER NOT NULL, role TEXT NOT NULL DEFAULT 'member',
            created_at REAL NOT NULL, UNIQUE(workspace_id, user_id)
        )
    """)
    conn.execute(
        "INSERT INTO workspace_members (workspace_id, user_id, role, created_at) VALUES (?, ?, ?, ?)",
        ("ws_1", 1, "owner", time.time()),
    )
    conn.commit()

    ws = conn.execute("SELECT * FROM workspaces WHERE id = ?", ("ws_1",)).fetchone()
    assert ws is not None
    assert ws[1] == "Test WS"

    members = conn.execute("SELECT * FROM workspace_members WHERE workspace_id = ?", ("ws_1",)).fetchall()
    assert len(members) == 1
    assert members[0][3] == "owner"

    conn.execute("UPDATE workspaces SET tokens_used_today = tokens_used_today + 50 WHERE id = ?", ("ws_1",))
    updated = conn.execute("SELECT tokens_used_today FROM workspaces WHERE id = ?", ("ws_1",)).fetchone()
    assert updated[0] == 50
    conn.close()
