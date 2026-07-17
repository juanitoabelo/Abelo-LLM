"""Voice API routes."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from src.voice.interface import SpeechToText, TextToSpeech

router = APIRouter(prefix="/api/voice", tags=["voice"])

AUDIO_DIR = Path("artifacts/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Only audio files are supported")

    path = AUDIO_DIR / f"input_{uuid.uuid4().hex}.wav"
    path.write_bytes(await file.read())

    stt = SpeechToText()
    text = stt.transcribe(path)
    path.unlink(missing_ok=True)

    return {"text": text, "status": "ok"}


class TTSRequest(BaseModel):
    text: str


@router.post("/tts")
async def text_to_speech(request: TTSRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    tts = TextToSpeech()
    data = tts.speak(request.text)
    if data:
        return Response(content=data, media_type="audio/wav")
    return {"status": "placeholder", "text": request.text[:500]}
