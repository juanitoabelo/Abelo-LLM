from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/upload", tags=["upload"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_SIZE = 10 * 1024 * 1024


@router.post("")
async def upload_image(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported type: {file.content_type}. Allowed: jpg, png, webp, gif")

    contents = await file.read()
    if len(contents) > MAX_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Max 10MB")

    ext = file.filename.rsplit(".", 1)[-1] if "." in (file.filename or "") else "png"
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = UPLOAD_DIR / filename
    filepath.write_bytes(contents)

    return JSONResponse({
        "filename": filename,
        "url": f"/uploads/{filename}",
        "size": len(contents),
        "content_type": file.content_type,
    })
