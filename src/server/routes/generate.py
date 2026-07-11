from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from src.llm.router import LLMRouter
from src.multimodal import (
    ContentPlanner,
    generate_code_artifact,
    generate_image_artifact,
    generate_text_artifact,
    generate_video_artifact,
    generate_audio_artifact,
)

router = APIRouter(prefix="/api/generate", tags=["generate"])


class GenerateTextRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = True


class GenerateArtifactRequest(BaseModel):
    prompt: str
    mode: Optional[str] = None
    scene_count: int = 4
    fps: int = 24


@router.post("/text")
async def generate_text(request: GenerateTextRequest):
    llm = LLMRouter()
    backends = await llm.check_backends()
    if not backends.get("ollama"):
        raise HTTPException(status_code=503, detail="No LLM backend available")

    if not request.stream:
        parts = []
        async for chunk in llm.generate(
            prompt=request.prompt,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=False,
        ):
            parts.append(chunk)
        return {"response": "".join(parts)}

    async def event_stream():
        async for chunk in llm.generate(
            prompt=request.prompt,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=True,
        ):
            yield f"data: {json.dumps({'content': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/artifact")
async def generate_artifact(request: GenerateArtifactRequest):
    planner = ContentPlanner()
    mode = request.mode or planner.classify_request(request.prompt)

    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)

    timestamp = str(hash(request.prompt))[-8:]
    safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in request.prompt[:30]).strip()

    if mode == "image":
        output_path = output_dir / f"{safe_name}_{timestamp}.png"
        await generate_image_artifact(request.prompt, output_path)
        return {"mode": "image", "path": str(output_path), "url": f"/files/{output_path.name}"}

    if mode == "video":
        output_path = output_dir / f"{safe_name}_{timestamp}.mp4"
        await generate_video_artifact(
            request.prompt, output_path,
            scene_count=request.scene_count,
            fps=request.fps,
        )
        return {"mode": "video", "path": str(output_path), "url": f"/files/{output_path.name}"}

    if mode == "code":
        output_path = output_dir / f"{safe_name}_{timestamp}.py"
        await generate_code_artifact(request.prompt, output_path)
        return {"mode": "code", "path": str(output_path), "url": f"/files/{output_path.name}"}

    if mode == "audio":
        output_path = output_dir / f"{safe_name}_{timestamp}.txt"
        await generate_audio_artifact(request.prompt, output_path)
        return {"mode": "audio", "path": str(output_path), "url": f"/files/{output_path.name}"}

    output_path = output_dir / f"{safe_name}_{timestamp}.txt"
    await generate_text_artifact(request.prompt, output_path)
    return {"mode": "text", "path": str(output_path), "url": f"/files/{output_path.name}"}
