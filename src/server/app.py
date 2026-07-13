from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.config.settings import get_settings
from src.server.routes import chat, generate, models, upload

app = FastAPI(
    title="my_custom_llm",
    version="1.0.0",
    description="Full-featured local LLM tool with multimodal generation",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(generate.router)
app.include_router(models.router)
app.include_router(upload.router)

for _dir, _mount in [("artifacts", "/files"), ("uploads", "/uploads")]:
    try:
        p = __import__("pathlib").Path(_dir)
        p.mkdir(exist_ok=True)
        app.mount(_mount, StaticFiles(directory=str(p)), name=_dir)
    except Exception:
        pass


@app.get("/")
async def root():
    return {
        "name": "my_custom_llm",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/api/chat",
            "generate_text": "/api/generate/text",
            "generate_artifact": "/api/generate/artifact",
            "models": "/api/models",
            "health": "/api/models/health",
        },
    }


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "src.server.app:app",
        host=settings.server_host,
        port=settings.server_port,
        workers=settings.server_workers,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
