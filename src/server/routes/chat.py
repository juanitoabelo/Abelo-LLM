from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.llm.router import LLMRouter

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = None
    system: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = True
    images: list[str] = []
    history: list[dict] = []


@router.post("")
async def chat_endpoint(request: ChatRequest):
    llm = LLMRouter()
    backends = await llm.check_backends()
    if not any(backends.values()):
        raise HTTPException(status_code=503, detail="No LLM backend available")

    messages = llm.format_chat_messages(request.system, request.history, request.message)

    if not request.stream:
        response_parts = []
        async for chunk in llm.chat(
            messages=messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            images=request.images or None,
            stream=False,
        ):
            response_parts.append(chunk)
        return {"response": "".join(response_parts)}

    async def event_stream():
        async for chunk in llm.chat(
            messages=messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            images=request.images or None,
            stream=True,
        ):
            yield f"data: {json.dumps({'content': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
