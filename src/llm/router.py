from __future__ import annotations

import asyncio
import json
import uuid
from typing import AsyncGenerator, Optional

from src.config.settings import get_settings
from src.llm.ollama import OllamaBackend
from src.llm.local_model import LocalModelBackend
from src.rag.embedder import OllamaEmbedder
from src.rag.retriever import RAGRetriever
from src.rag.vector_store import VectorStore
from src.context.manager import ContextManager
from src.memory.store import MemoryStore
from src.tools.registry import get_default_registry, ToolRegistry


class LLMRouter:
    WINNER_KEYWORDS = ("winner", "custom", "tinytransformer")

    def __init__(
        self,
        rag_retriever: Optional[RAGRetriever] = None,
        context_manager: Optional[ContextManager] = None,
        memory_store: Optional[MemoryStore] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ) -> None:
        self.settings = get_settings()
        self.ollama = OllamaBackend()
        self.local = LocalModelBackend() if self.settings.enable_custom_model else None
        self._backend_available: Optional[bool] = None

        settings_path = __import__("pathlib").Path(self.settings.data_dir)
        base_path = settings_path.parent

        self.rag_retriever = rag_retriever or RAGRetriever(
            vector_store=VectorStore(str(base_path / "rag_store.db")),
            embedder=OllamaEmbedder(),
        )
        self.context_manager = context_manager or ContextManager()
        self.memory_store = memory_store or MemoryStore(str(base_path / "memory.db"))
        self.tool_registry = tool_registry or get_default_registry()

    def _resolve_backend(self, model: Optional[str] = None) -> OllamaBackend | LocalModelBackend:
        if model and any(kw in model.lower() for kw in self.WINNER_KEYWORDS):
            return self.local or self.ollama
        if model and self.local and self.settings.enable_custom_model:
            return self.ollama
        if self.local and self.settings.enable_custom_model and not model:
            return self.local
        return self.ollama

    async def check_backends(self) -> dict[str, bool]:
        ollama_ok = await self.ollama.is_available()
        local_ok = await self.local.is_available() if self.local else False
        self._backend_available = ollama_ok or local_ok
        return {"ollama": ollama_ok, "local": local_ok}

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stream: bool = True,
    ) -> AsyncGenerator[str, None]:
        backend = self._resolve_backend(model)
        if hasattr(backend, 'generate') and not hasattr(backend, 'chat'):
            async for chunk in backend.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_k=top_k,
                top_p=top_p,
            ):
                yield chunk
        else:
            async for chunk in backend.generate(
                prompt=prompt,
                model=model,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                top_k=top_k,
                top_p=top_p,
                stream=stream,
            ):
                yield chunk

    async def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        images: Optional[list[str]] = None,
        stream: bool = True,
        session_id: Optional[str] = None,
        enable_rag: bool = True,
        enable_tools: bool = True,
        enable_memory: bool = True,
    ) -> AsyncGenerator[str, None]:
        backend = self._resolve_backend(model)
        if backend is not self.ollama:
            user_msg = messages[-1]["content"] if messages else ""
            async for chunk in backend.generate(
                prompt=user_msg,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                yield chunk
            return

        user_query = self._get_last_user_text(messages)

        enriched_messages = list(messages)

        if enable_rag and user_query:
            rag_context = await self.rag_retriever.retrieve_formatted(user_query)
            if rag_context:
                enriched_messages.insert(
                    0,
                    {
                        "role": "system",
                        "content": f"Here is relevant context from your knowledge base to help answer:\n\n{rag_context}",
                    },
                )

        if enable_memory and user_query and session_id:
            memory_hits = self.memory_store.search(user_query, limit=5)
            if memory_hits:
                facts = "\n".join(f"- {m['key']}: {m['value']}" for m in memory_hits)
                enriched_messages.insert(
                    0,
                    {
                        "role": "system",
                        "content": f"Remembered facts about the user:\n{facts}",
                    },
                )
            if session_id:
                session_history = self.memory_store.get_session_context(session_id, limit=6)
                if session_history:
                    summary = self._summarize_session(session_history)
                    if summary:
                        enriched_messages.insert(
                            0,
                            {
                                "role": "system",
                                "content": f"Earlier in this conversation: {summary}",
                            },
                        )

        truncated = self.context_manager.trim_messages(
            enriched_messages,
            extra_text="",
            max_tokens=max_tokens or self.settings.max_tokens,
        )

        if enable_tools:
            tool_specs = self.tool_registry.get_specs()
        else:
            tool_specs = []

        async for item in self._chat_with_tools(
            messages=truncated,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            images=images,
            tool_specs=tool_specs,
            session_id=session_id,
        ):
            yield item

    async def _chat_with_tools(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        images: Optional[list[str]] = None,
        tool_specs: Optional[list[dict]] = None,
        session_id: Optional[str] = None,
        depth: int = 0,
    ) -> AsyncGenerator[str, None]:
        if depth > 5:
            yield "\n\n[Max tool call depth reached]"
            return

        collected = ""
        tool_calls: list[dict] = []

        async for chunk in self.ollama.chat_raw(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            images=images,
            tools=tool_specs,
        ):
            content = chunk.get("content")
            calls = chunk.get("tool_calls")

            if content:
                collected += content
                yield content

            if calls:
                tool_calls.extend(calls)

            if chunk.get("done"):
                break

        if tool_calls:
            max_yield = 2
            for tc in tool_calls[:max_yield]:
                func_name = tc.get("function", {}).get("name", "")
                func_args = tc.get("function", {}).get("arguments", {})

                yield f"\n\n⚡ **Using tool: {func_name}**\n\n"

                result = self.tool_registry.execute(func_name, func_args)

                if result.success:
                    tool_output = result.output[:2000]
                else:
                    tool_output = f"Error: {result.error}"

                messages.append({"role": "assistant", "content": collected} if collected else {
                    "role": "assistant",
                    "content": f"[Calling tool: {func_name}]",
                })
                messages.append({
                    "role": "tool",
                    "content": tool_output,
                    "name": func_name,
                })
                collected = ""

                async for chunk in self._chat_with_tools(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    images=images,
                    tool_specs=tool_specs,
                    session_id=session_id,
                    depth=depth + 1,
                ):
                    yield chunk

    def _get_last_user_text(self, messages: list[dict]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "") or ""
                if "[N image(s) attached]" in content:
                    content = content.split("[N image")[0].strip()
                return content.strip()
        return ""

    def _summarize_session(self, messages: list[dict]) -> str:
        if not messages:
            return ""
        user_topics = set()
        for m in messages:
            if m.get("role") == "user":
                content = m.get("content", "") or ""
                words = content.split()
                for w in words[:10]:
                    clean = w.strip(".,!?;:'\"()[]{}").lower()
                    if len(clean) > 3:
                        user_topics.add(clean)
        topic_list = list(user_topics)[:8]
        if topic_list:
            return f"the user previously asked about: {', '.join(topic_list)}"
        return ""

    def format_chat_messages(
        self,
        system_prompt: Optional[str],
        history: list[dict],
        user_message: str,
    ) -> list[dict]:
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for msg in history:
            messages.append(msg)
        messages.append({"role": "user", "content": user_message})
        return messages

    async def list_models(self) -> list[dict]:
        ollama_models = await self.ollama.list_models()
        models = []
        for m in ollama_models:
            name = m.get("name", "")
            models.append({
                "name": name,
                "size": m.get("size", 0),
                "provider": "ollama",
            })
        if self.local and self.settings.enable_custom_model:
            models.insert(0, {"name": "winner-model (custom)", "size": 0, "provider": "local"})
        return models
