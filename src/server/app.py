from __future__ import annotations

import time
from collections import defaultdict

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from src.config.settings import get_settings
from src.server.routes import chat, generate, models, upload, rag, memory, stats, agent
from src.server.routes import auth as auth_routes
from src.server.routes.training import register_training_routes
from src.server.routes.structured import register_structured_routes
from src.server.routes.plugins import register_plugin_routes
from src.server.routes import voice as voice_routes
from src.server.routes import branching as branching_routes
from src.server.routes import openai as openai_routes
from src.server.routes import ws as ws_routes
from src.server.routes import feedback as feedback_routes
from src.server.routes import workspaces as workspaces_routes
from src.server.routes import new_features as new_features_routes

app = FastAPI(
    title="my_custom_llm",
    version="4.0.0",
    description="Full-featured local LLM with RAG, tool calling, memory, agent planning, guardrails, observability, fine-tuning, structured output, plugins, voice, branching, knowledge graph, vision, and model merging",
)


# Rate limiting middleware
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = 30, window_seconds: int = 60) -> None:
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - self.window_seconds
        self._requests[client_ip] = [t for t in self._requests[client_ip] if t > window_start]
        if len(self._requests[client_ip]) >= self.max_requests:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later.", "retry_after": self.window_seconds},
            )
        self._requests[client_ip].append(now)
        return await call_next(request)


settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
app.add_middleware(RateLimitMiddleware, max_requests=settings.rate_limit_max, window_seconds=settings.rate_limit_window)

app.include_router(chat.router)
app.include_router(generate.router)
app.include_router(models.router)
app.include_router(upload.router)
app.include_router(rag.router)
app.include_router(memory.router)
app.include_router(stats.router)
app.include_router(agent.router)
app.include_router(auth_routes.router)
app.include_router(voice_routes.router)
app.include_router(branching_routes.router)
app.include_router(openai_routes.router)
app.include_router(feedback_routes.router)
app.include_router(workspaces_routes.router)
app.include_router(ws_routes.router)
app.include_router(new_features_routes.router)

register_training_routes(app)
register_structured_routes(app)
register_plugin_routes(app)

for _dir, _mount in [("artifacts", "/files"), ("uploads", "/uploads")]:
    try:
        p = __import__("pathlib").Path(_dir)
        p.mkdir(exist_ok=True)
        app.mount(_mount, StaticFiles(directory=str(p)), name=_dir)
    except Exception:
        pass


@app.get("/")
async def root():
    settings = get_settings()
    return {
        "name": "my_custom_llm",
        "version": "4.0.0",
        "endpoints": {
            "chat": "/api/chat",
            "generate_text": "/api/generate/text",
            "generate_artifact": "/api/generate/artifact",
            "models": "/api/models",
            "health": "/api/models/health",
            "rag_status": "/api/rag/status",
            "rag_ingest": "/api/rag/ingest/text",
            "rag_query": "/api/rag/query",
            "memory": "/api/memory",
            "memory_remember": "/api/memory/remember",
            "memory_sessions": "/api/memory/sessions",
            "stats": "/api/stats",
            "stats_requests": "/api/stats/requests",
            "agent_plan": "/api/agent/plan",
            "agent_execute": "/api/agent/plan/{id}/execute",
            "auth_register": "/api/auth/register",
            "auth_login": "/api/auth/login",
            "auth_me": "/api/auth/me",
            "voice_stt": "/api/voice/stt",
            "voice_tts": "/api/voice/tts",
            "branch_create": "/api/branch/create",
            "branch_list": "/api/branch/list/{session_id}",
            "branch_get": "/api/branch/{branch_id}",
            "branch_templates": "/api/branch/templates",
            "branch_templates_save": "/api/branch/templates/save",
            "branch_templates_apply": "/api/branch/templates/apply",
            "openai_models": "/v1/models",
            "openai_chat": "/v1/chat/completions",
            "ws_collab": "/ws/chat/{room_id}",
            "feedback_rating": "/api/feedback/rating",
            "feedback_preference": "/api/feedback/preference",
            "feedback_export": "/api/feedback/export",
            "feedback_stats": "/api/feedback/stats",
            "workspaces": "/api/workspaces",
        },
        "capabilities": {
            "rag": True,
            "hybrid_rag": True,
            "reranking": True,
            "tool_calling": True,
            "persistent_memory": True,
            "context_management": True,
            "agent_planning": True,
            "guardrails": True,
            "observability": True,
            "mcp_support": True,
            "fine_tuning": True,
            "ai_image_gen": True,
            "structured_output": True,
            "plugin_system": True,
            "distillation": True,
            "lora_finetuning": True,
            "eval_harness": True,
            "quantization": True,
            "multi_user_auth": True,
            "thinking_traces": True,
            "training_pipeline": True,
            "voice": settings.enable_voice,
            "vision": settings.enable_vision,
            "knowledge_graph": settings.enable_knowledge_graph,
            "model_merging": True,
            "conversation_branching": True,
            "openai_compatible_api": True,
            "websocket_collaboration": True,
            "user_feedback_loop": True,
            "multi_tenant_workspaces": True,
            "document_pipeline": True,
            "llm_as_judge_eval": True,
            "sub_agent_swarm": True,
            "web_browsing_agent": True,
            "dpo_training": True,
            "model_hub_integration": True,
            "multi_provider_routing": True,
            "telemetry_tracing": True,
            "continuous_batching": True,
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
