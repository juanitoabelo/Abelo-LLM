from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()


class LLMSettings:
    def __init__(self) -> None:
        self.default_model: str = os.getenv("LLM_DEFAULT_MODEL", "llama3.2:1b")
        self.temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        self.max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))
        self.top_k: int = int(os.getenv("LLM_TOP_K", "40"))
        self.top_p: float = float(os.getenv("LLM_TOP_P", "0.95"))

        self.ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_request_timeout: int = int(os.getenv("OLLAMA_TIMEOUT", "120"))

        self.server_host: str = os.getenv("SERVER_HOST", "0.0.0.0")
        self.server_port: int = int(os.getenv("SERVER_PORT", "8000"))
        self.server_workers: int = int(os.getenv("SERVER_WORKERS", "1"))

        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        self.checkpoints_dir: str = os.getenv("CHECKPOINTS_DIR", str(Path(__file__).parent.parent.parent / "checkpoints"))
        self.data_dir: str = os.getenv("DATA_DIR", str(Path(__file__).parent.parent.parent / "data"))

        self.enable_custom_model: bool = os.getenv("ENABLE_CUSTOM_MODEL", "false").lower() == "true"
        self.custom_checkpoint: str = os.getenv("CUSTOM_CHECKPOINT", str(Path(self.checkpoints_dir) / "tiny_transformer_best.pt"))
        self.custom_tokenizer: str = os.getenv("CUSTOM_TOKENIZER", str(Path(self.checkpoints_dir) / "tokenizer.json"))

        self.enable_rag: bool = os.getenv("ENABLE_RAG", "true").lower() == "true"
        self.enable_tools: bool = os.getenv("ENABLE_TOOLS", "true").lower() == "true"
        self.enable_memory: bool = os.getenv("ENABLE_MEMORY", "true").lower() == "true"
        self.enable_guardrails: bool = os.getenv("ENABLE_GUARDRAILS", "true").lower() == "true"
        self.enable_agent: bool = os.getenv("ENABLE_AGENT", "true").lower() == "true"
        self.enable_vision: bool = os.getenv("ENABLE_VISION", "true").lower() == "true"
        self.enable_voice: bool = os.getenv("ENABLE_VOICE", "true").lower() == "true"
        self.enable_knowledge_graph: bool = os.getenv("ENABLE_KNOWLEDGE_GRAPH", "true").lower() == "true"
        self.context_max_tokens: int = int(os.getenv("CONTEXT_MAX_TOKENS", "4096"))

        self.embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        self.rag_top_k: int = int(os.getenv("RAG_TOP_K", "5"))
        self.rag_min_similarity: float = float(os.getenv("RAG_MIN_SIMILARITY", "0.3"))
        self.rag_web_fallback: bool = os.getenv("RAG_WEB_FALLBACK", "true").lower() == "true"
        self.enable_hybrid_rag: bool = os.getenv("ENABLE_HYBRID_RAG", "true").lower() == "true"
        self.enable_reranking: bool = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
        self.enable_thinking: bool = os.getenv("ENABLE_THINKING", "true").lower() == "true"

        self.rate_limit_max: int = int(os.getenv("RATE_LIMIT_MAX", "30"))
        self.rate_limit_window: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

        self.allowed_origins: list[str] = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
        self.allowed_hosts: list[str] = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1,0.0.0.0,*.ngrok-free.app").split(",")

        self.ai_image_gen: bool = os.getenv("AI_IMAGE_GEN", "false").lower() == "true"
        self.mcp_server_url: str = os.getenv("MCP_SERVER_URL", "")

    @property
    def available_remote_models(self) -> list[str]:
        try:
            import httpx
            r = httpx.get(f"{self.ollama_host}/api/tags", timeout=5)
            if r.status_code == 200:
                data = r.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return ["qwen3.5:latest", "gemma4:latest", "llama3.2:1b", "llama3.2:3b", "llama3.2:7b", "mistral:latest", "codellama:latest", "deepseek-coder:latest"]


@lru_cache()
def get_settings() -> LLMSettings:
    return LLMSettings()
