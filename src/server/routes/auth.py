"""Authentication routes for multi-user support."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.auth.jwt import UserStore, create_token

router = APIRouter(prefix="/api/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    username: str
    password: str
    display_name: str = ""


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/register")
async def register(request: RegisterRequest):
    if len(request.username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")
    if len(request.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    store = UserStore()
    if store.user_exists(request.username):
        raise HTTPException(status_code=409, detail="Username already taken")

    try:
        user = store.create_user(request.username, request.password, request.display_name)
        token = create_token(user.id, user.username)
        return {
            "status": "ok",
            "user": {"id": user.id, "username": user.username, "display_name": user.display_name},
            "token": token,
        }
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/login")
async def login(request: LoginRequest):
    store = UserStore()
    user = store.authenticate(request.username, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_token(user.id, user.username)
    return {
        "status": "ok",
        "user": {"id": user.id, "username": user.username, "display_name": user.display_name},
        "token": token,
    }


@router.get("/me")
async def get_me(authorization: str = ""):
    from src.auth.jwt import verify_token

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization[7:]
    payload = verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")

    store = UserStore()
    user = store.get_user(payload["sub"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"user": {"id": user.id, "username": user.username, "display_name": user.display_name}}
