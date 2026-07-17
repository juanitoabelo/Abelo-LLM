"""OpenAI-compatible API endpoint (drop-in for any OpenAI SDK)."""

from __future__ import annotations

import json
import time
import uuid
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.llm.router import LLMRouter
from src.config.settings import get_settings

router = APIRouter(prefix="/v1", tags=["openai"])


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionsRequest(BaseModel):
    model: str = "llama3.2:1b"
    messages: list[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None


@router.get("/models")
async def list_models():
    llm = LLMRouter()
    backends = await llm.check_backends()
    models = []
    if backends.get("ollama"):
        ollama_models = await llm.list_models()
        for m in ollama_models:
            models.append({
                "id": m["name"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": m.get("provider", "ollama"),
            })
    return {
        "object": "list",
        "data": models,
    }


@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionsRequest):
    llm = LLMRouter()
    backends = await llm.check_backends()
    if not any(backends.values()):
        raise HTTPException(503, "No LLM backend available")

    chat_messages = [{"role": m.role, "content": m.content} for m in request.messages]
    chat_messages = llm.format_chat_messages(None, chat_messages[:-1], chat_messages[-1]["content"])

    if request.stream:

        async def event_stream():
            yield f"data: {json.dumps({'id': f'chatcmpl-{uuid.uuid4().hex[:8]}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
            async for chunk in llm.chat(
                messages=chat_messages, model=request.model,
                temperature=request.temperature, max_tokens=request.max_tokens,
                stream=True, enable_rag=False, enable_tools=False, enable_memory=False,
                enable_guardrails=False, enable_thinking=False,
            ):
                content = ""
                try:
                    p = json.loads(chunk)
                    if p.get("type") == "content":
                        content = p.get("content", "")
                except (json.JSONDecodeError, TypeError):
                    content = chunk
                if content:
                    yield f"data: {json.dumps({'id': f'chatcmpl-{uuid.uuid4().hex[:8]}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'delta': {'content': content}, 'finish_reason': None}]})}\n\n"
            yield f"data: {json.dumps({'id': f'chatcmpl-{uuid.uuid4().hex[:8]}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    collected = ""
    async for chunk in llm.chat(
        messages=chat_messages, model=request.model,
        temperature=request.temperature, max_tokens=request.max_tokens,
        stream=False, enable_rag=False, enable_tools=False, enable_memory=False,
        enable_guardrails=False, enable_thinking=False,
    ):
        collected += chunk

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": collected}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
