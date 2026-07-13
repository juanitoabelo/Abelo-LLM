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

    @property
    def available_remote_models(self) -> list[str]:
        return ["qwen3.5:latest", "gemma4:latest", "llama3.2:1b", "llama3.2:3b", "llama3.2:7b", "mistral:latest", "codellama:latest", "deepseek-coder:latest"]


@lru_cache()
def get_settings() -> LLMSettings:
    return LLMSettings()
