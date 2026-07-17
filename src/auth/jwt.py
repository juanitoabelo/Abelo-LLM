"""JWT-based authentication for multi-user support."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class User:
    id: int
    username: str
    display_name: str
    created_at: float


JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_hex(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRY = 86400 * 7  # 7 days


def _base64url_encode(data: bytes) -> str:
    import base64
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _base64url_decode(s: str) -> bytes:
    import base64
    s = s + "=" * (4 - len(s) % 4)
    return base64.urlsafe_b64decode(s)


def _jwt_header() -> str:
    import base64
    return base64.urlsafe_b64encode(json.dumps({"alg": JWT_ALGORITHM, "typ": "JWT"}).encode()).rstrip(b"=").decode()


def _jwt_payload(user_id: int, username: str) -> str:
    import base64
    payload = json.dumps({
        "sub": user_id,
        "username": username,
        "iat": int(time.time()),
        "exp": int(time.time()) + JWT_EXPIRY,
    })
    return base64.urlsafe_b64encode(payload.encode()).rstrip(b"=").decode()


def _jwt_sign(message: str) -> str:
    import base64
    sig = hmac.new(JWT_SECRET.encode(), message.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(sig).rstrip(b"=").decode()


def create_token(user_id: int, username: str) -> str:
    header = _jwt_header()
    payload = _jwt_payload(user_id, username)
    signature = _jwt_sign(f"{header}.{payload}")
    return f"{header}.{payload}.{signature}"


def verify_token(token: str) -> Optional[dict]:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header, payload, signature = parts
        expected_sig = _jwt_sign(f"{header}.{payload}")
        if not hmac.compare_digest(signature, expected_sig):
            return None
        data = json.loads(_base64url_decode(payload))
        if data.get("exp", 0) < time.time():
            return None
        return data
    except Exception:
        return None


class UserStore:
    def __init__(self, db_path: str | Path = "data/users.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
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
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                display_name TEXT DEFAULT '',
                created_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)
        """)
        conn.commit()
        conn.close()

    def create_user(self, username: str, password: str, display_name: str = "") -> User:
        conn = self._get_conn()
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        now = time.time()
        try:
            cursor = conn.execute(
                "INSERT INTO users (username, password_hash, display_name, created_at) VALUES (?, ?, ?, ?)",
                (username, password_hash, display_name or username, now),
            )
            conn.commit()
            return User(id=cursor.lastrowid, username=username, display_name=display_name or username, created_at=now)
        except sqlite3.IntegrityError:
            raise ValueError(f"User '{username}' already exists")

    def authenticate(self, username: str, password: str) -> Optional[User]:
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.row_factory = sqlite3.Row
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            row = conn.execute(
                "SELECT id, username, display_name, created_at FROM users WHERE username = ? AND password_hash = ?",
                (username, password_hash),
            ).fetchone()
            if row:
                return User(id=row["id"], username=row["username"], display_name=row["display_name"], created_at=row["created_at"])
            return None
        finally:
            conn.close()

    def get_user(self, user_id: int) -> Optional[User]:
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT id, username, display_name, created_at FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
            if row:
                return User(id=row["id"], username=row["username"], display_name=row["display_name"], created_at=row["created_at"])
            return None
        finally:
            conn.close()

    def user_exists(self, username: str) -> bool:
        conn = sqlite3.connect(str(self.db_path))
        try:
            row = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
            return row is not None
        finally:
            conn.close()


def require_auth():
    """FastAPI dependency for JWT auth."""
    from fastapi import Header, HTTPException

    def dependency(authorization: str = Header("")):
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
        token = authorization[7:]
        payload = verify_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return payload

    return dependency


def get_current_user():
    """FastAPI dependency that returns the full User object."""
    from fastapi import Header, HTTPException

    def dependency(authorization: str = Header("")):
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
        token = authorization[7:]
        payload = verify_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        store = UserStore()
        user = store.get_user(payload["sub"])
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user

    return dependency
