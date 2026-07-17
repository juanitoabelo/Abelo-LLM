"""User feedback — thumbs up/down on responses, stored for RLHF."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/feedback", tags=["feedback"])

FEEDBACK_DB = Path("data/feedback.db")
FEEDBACK_DB.parent.mkdir(parents=True, exist_ok=True)


def _init_db():
    conn = sqlite3.connect(str(FEEDBACK_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            message_index INTEGER NOT NULL,
            user_message TEXT NOT NULL,
            assistant_response TEXT NOT NULL,
            rating INTEGER NOT NULL CHECK(rating IN (1, -1)),
            model TEXT DEFAULT '',
            tags TEXT DEFAULT '[]',
            comment TEXT DEFAULT '',
            created_at REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS preferences (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            chosen TEXT NOT NULL,
            rejected TEXT NOT NULL,
            prompt TEXT NOT NULL,
            model TEXT DEFAULT '',
            created_at REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()


_init_db()


class FeedbackCreate(BaseModel):
    session_id: str
    message_index: int
    user_message: str
    assistant_response: str
    rating: int  # 1 = thumbs up, -1 = thumbs down
    model: str = ""
    tags: list[str] = []
    comment: str = ""


class PreferenceCreate(BaseModel):
    session_id: str
    chosen: str
    rejected: str
    prompt: str
    model: str = ""


@router.post("/rate")
async def submit_feedback(fb: FeedbackCreate):
    conn = sqlite3.connect(str(FEEDBACK_DB))
    try:
        conn.execute(
            "INSERT INTO feedback (id, session_id, message_index, user_message, assistant_response, rating, model, tags, comment, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4())[:12], fb.session_id, fb.message_index, fb.user_message, fb.assistant_response, fb.rating, fb.model, json.dumps(fb.tags), fb.comment, time.time()),
        )
        conn.commit()
        return {"status": "ok"}
    finally:
        conn.close()


@router.post("/preference")
async def submit_preference(pref: PreferenceCreate):
    conn = sqlite3.connect(str(FEEDBACK_DB))
    try:
        conn.execute(
            "INSERT INTO preferences (id, session_id, chosen, rejected, prompt, model, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4())[:12], pref.session_id, pref.chosen, pref.rejected, pref.prompt, pref.model, time.time()),
        )
        conn.commit()
        return {"status": "ok"}
    finally:
        conn.close()


@router.get("/export")
async def export_feedback(limit: int = 1000):
    conn = sqlite3.connect(str(FEEDBACK_DB))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM feedback ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return {"feedback": [dict(r) for r in rows]}
    finally:
        conn.close()


@router.get("/export/preferences")
async def export_preferences(limit: int = 1000):
    conn = sqlite3.connect(str(FEEDBACK_DB))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM preferences ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return {"preferences": [dict(r) for r in rows]}
    finally:
        conn.close()


@router.get("/stats")
async def feedback_stats():
    conn = sqlite3.connect(str(FEEDBACK_DB))
    try:
        total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        up = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating = 1").fetchone()[0]
        down = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating = -1").fetchone()[0]
        return {
            "total": total,
            "thumbs_up": up,
            "thumbs_down": down,
            "ratio": round(up / max(1, total), 3),
        }
    finally:
        conn.close()
