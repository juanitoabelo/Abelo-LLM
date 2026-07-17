"""WebSocket real-time collaboration — chat rooms, multi-user sessions."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import defaultdict
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.llm.router import LLMRouter
from src.config.settings import get_settings

router = APIRouter(tags=["websocket"])

rooms: dict[str, dict] = defaultdict(lambda: {
    "clients": {},
    "messages": [],
    "created_at": time.time(),
})


@router.websocket("/ws/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, user: str = "anonymous"):
    await websocket.accept()
    room = rooms[room_id]
    client_id = str(uuid.uuid4())[:8]
    room["clients"][client_id] = websocket
    system_msg = json.dumps({"type": "system", "content": f"{user} joined the room", "user": user, "client_id": client_id, "ts": time.time()})
    for cid, ws in room["clients"].items():
        try:
            await ws.send_text(system_msg)
        except Exception:
            pass
    await websocket.send_text(json.dumps({"type": "welcome", "client_id": client_id, "room": room_id, "user_count": len(room["clients"])}))

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            msg_type = msg.get("type", "message")

            if msg_type == "message":
                content = msg.get("content", "")
                ts = time.time()
                room["messages"].append({"user": user, "content": content, "ts": ts, "client_id": client_id})
                llm = LLMRouter()
                settings = get_settings()
                history = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in room["messages"][-20:]]
                collected = ""
                async for chunk in llm.chat(
                    messages=[{"role": "user", "content": content}],
                    model=settings.default_model,
                    enable_rag=True, enable_tools=True, enable_memory=True,
                    enable_guardrails=True, enable_thinking=True,
                ):
                    collected += chunk
                room["messages"].append({"user": "assistant", "content": collected, "ts": time.time(), "client_id": "assistant"})
                broadcast = json.dumps({"type": "assistant", "content": collected, "user": "assistant", "ts": time.time()})
                for cid, ws in room["clients"].items():
                    try:
                        await ws.send_text(broadcast)
                    except Exception:
                        pass

            elif msg_type == "typing":
                for cid, ws in room["clients"].items():
                    if cid != client_id:
                        try:
                            await ws.send_text(json.dumps({"type": "typing", "user": user, "client_id": client_id}))
                        except Exception:
                            pass

    except WebSocketDisconnect:
        pass
    finally:
        room["clients"].pop(client_id, None)
        leave_msg = json.dumps({"type": "system", "content": f"{user} left the room", "user": user, "ts": time.time()})
        for cid, ws in room["clients"].items():
            try:
                await ws.send_text(leave_msg)
            except Exception:
                pass
        if not room["clients"]:
            del rooms[room_id]
