"""Multi-provider smart routing — route requests to best model by task."""

from __future__ import annotations

import re
import time
from typing import Optional


TASK_PATTERNS: dict[str, list[str]] = {
    "code": [r"\b(code|python|javascript|function|class|def |import |api|endpoint)\b"],
    "math": [r"\b(calculate|solve|equation|math|sum |compute|\d+\s*[\+\-\*\/])\b"],
    "creative": [r"\b(write a poem|story|creative|imagine|fantasy|narrative)\b"],
    "summarize": [r"\b(summarize|tl;dr|brief|condense|key points)\b"],
    "translate": [r"\b(translate|translation|in (spanish|french|german|japanese))\b"],
    "vision": [r"\b(image|photo|picture|visual|describe.*image|what.*see)\b"],
    "simple": [r"\b(hi|hello|hey|thanks|yes|no|okay|sure)\b.{0,20}"],
}


class ProviderRouter:
    def __init__(self, ollama_host: str = "http://localhost:11434") -> None:
        self.ollama_host = ollama_host
        self._latency_cache: dict[str, float] = {}

    def classify_task(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        for task, patterns in TASK_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, prompt_lower):
                    return task
        return "general"

    def recommend_model(self, prompt: str, available_models: list[str]) -> str:
        task = self.classify_task(prompt)
        models_lower = [m.lower() for m in available_models]

        task_model_map = {
            "code": ["deepseek", "codellama", "qwen"],
            "math": ["qwen", "mistral", "llama"],
            "creative": ["llama", "qwen", "mistral"],
            "summarize": ["llama", "qwen", "gemma"],
            "translate": ["qwen", "llama", "mistral"],
            "vision": ["gemma4", "llava"],
            "simple": ["llama3.2:1b"],
            "general": ["qwen", "llama", "gemma", "mistral"],
        }

        preferred = task_model_map.get(task, [])
        for pref in preferred:
            for m in available_models:
                if pref in m.lower():
                    return m

        if available_models:
            return available_models[0]
        return "llama3.2:1b"

    def select_backend(self, prompt: str, model: Optional[str] = None, custom_model_available: bool = False) -> str:
        if model:
            return model
        if custom_model_available:
            task = self.classify_task(prompt)
            if task in ("simple", "summarize"):
                return "custom"
        return "ollama"
