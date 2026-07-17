from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from typing import Any, AsyncGenerator, Optional

from src.config.settings import get_settings
from src.llm.ollama import OllamaBackend
from src.llm.local_model import LocalModelBackend
from src.rag.embedder import OllamaEmbedder
from src.rag.retriever import RAGRetriever
from src.rag.vector_store import VectorStore
from src.context.manager import ContextManager
from src.memory.store import MemoryStore
from src.tools.registry import get_default_registry, ToolRegistry
from src.cache.prompt_cache import get_cache
from src.monitor.stats import UsageTracker, RequestRecord, estimate_tokens


THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


class LLMRouter:
    WINNER_KEYWORDS = ("winner", "custom", "tinytransformer")

    def __init__(
        self,
        rag_retriever=None,
        context_manager=None,
        memory_store=None,
        tool_registry=None,
        agent_planner=None,
        usage_tracker=None,
        content_filter=None,
        pii_filter=None,
        rate_limiter=None,
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
        self.usage_tracker = usage_tracker or UsageTracker(str(base_path / "usage.db"))

        from src.agent.planner import AgentPlanner
        from src.guard.filter import ContentFilter, PIIFilter, RateLimiter
        self.agent_planner = agent_planner or AgentPlanner()
        self.content_filter = content_filter or ContentFilter()
        self.pii_filter = pii_filter or PIIFilter()
        self.rate_limiter = rate_limiter or RateLimiter()

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
                prompt=prompt, temperature=temperature, max_tokens=max_tokens,
                top_k=top_k, top_p=top_p,
            ):
                yield chunk
        else:
            async for chunk in backend.generate(
                prompt=prompt, model=model, system=system,
                temperature=temperature, max_tokens=max_tokens,
                top_k=top_k, top_p=top_p, stream=stream,
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
        enable_guardrails: bool = True,
        enable_thinking: bool = True,
    ) -> AsyncGenerator[str, None]:
        t_start = time.time()
        cache = get_cache()
        user_query = self._get_last_user_text(messages)
        cache_key = (user_query or "", model, temperature)
        if not stream and not enable_rag and not enable_tools and not enable_memory:
            cached = cache.get(*cache_key)
            if cached is not None:
                yield cached
                return

        backend = self._resolve_backend(model)
        if backend is not self.ollama:
            user_msg = messages[-1]["content"] if messages else ""
            async for chunk in backend.generate(
                prompt=user_msg, temperature=temperature, max_tokens=max_tokens,
            ):
                yield chunk
            return

        user_query = self._get_last_user_text(messages)

        if enable_guardrails and user_query:
            blocked = self.content_filter.check_input(user_query)
            if blocked:
                yield json.dumps({"type": "guardrail", "reason": blocked}) + "\n"
                yield f"I can't respond to that request (triggered: {blocked})."
                self._log_request("chat", model, 0, 0, t_start, False, session_id)
                return

            pii_findings = self.pii_filter.scan(user_query)
            if pii_findings:
                pii_types = set(f["type"] for f in pii_findings)
                yield json.dumps({"type": "guardrail", "reason": "pii", "pii_types": list(pii_types)}) + "\n"
                yield f"I notice your message may contain sensitive information ({', '.join(pii_types)}). Please avoid sharing personal data."
                self._log_request("chat", model, 0, 0, t_start, False, session_id)
                return

        enriched_messages = list(messages)

        if enable_rag and user_query:
            rag_context = await self.rag_retriever.retrieve_formatted(user_query)
            if rag_context:
                enriched_messages.insert(0, {"role": "system", "content": f"Here is relevant context from your knowledge base to help answer:\n\n{rag_context}"})

        if enable_memory and user_query and session_id:
            memory_hits = self.memory_store.search(user_query, limit=5)
            if memory_hits:
                facts = "\n".join(f"- {m['key']}: {m['value']}" for m in memory_hits)
                enriched_messages.insert(0, {"role": "system", "content": f"Remembered facts about the user:\n{facts}"})
            session_history = self.memory_store.get_session_context(session_id, limit=6)
            if session_history:
                summary = self._summarize_session(session_history)
                if summary:
                    enriched_messages.insert(0, {"role": "system", "content": f"Earlier in this conversation: {summary}"})

        if enable_thinking and user_query:
            thinking_sys = {
                "role": "system",
                "content": "When you need to reason step-by-step, enclose your reasoning in <think>...</think> tags. The final answer should come after the closing tag."
            }
            enriched_messages.append(thinking_sys)

        truncated = self.context_manager.trim_messages(
            enriched_messages, extra_text="",
            max_tokens=max_tokens or self.settings.max_tokens,
        )

        tool_specs = self.tool_registry.get_specs() if enable_tools else []

        if session_id:
            self.memory_store.create_session(session_id)
            user_content = messages[-1].get("content", "") if messages else ""
            self.memory_store.log_message(session_id, "user", user_content)

        full_response = ""
        async for item in self._chat_with_tools(
            messages=truncated, model=model, temperature=temperature,
            max_tokens=max_tokens, images=images, tool_specs=tool_specs,
            session_id=session_id, enable_thinking=enable_thinking,
        ):
            full_response += item
            yield item

        if enable_guardrails and full_response:
            output_blocked = self.content_filter.check_output(full_response)
            if output_blocked:
                pass
            safe_response = self.pii_filter.sanitize(full_response)
            if safe_response != full_response:
                pass

        if session_id:
            self.memory_store.log_message(session_id, "assistant", full_response)

        if not enable_rag and not enable_tools and not enable_memory:
            try:
                cache.set(*(user_query or "", model, temperature), full_response)
            except Exception:
                pass

        tokens_in = estimate_tokens(user_query)
        tokens_out = estimate_tokens(full_response)
        self._log_request("chat", model or self.settings.default_model, tokens_in, tokens_out, t_start, True, session_id)

    async def _chat_with_tools(
        self, messages: list[dict], model: Optional[str] = None,
        temperature: Optional[float] = None, max_tokens: Optional[int] = None,
        images: Optional[list[str]] = None, tool_specs: Optional[list[dict]] = None,
        session_id: Optional[str] = None, depth: int = 0,
        enable_thinking: bool = True,
    ) -> AsyncGenerator[str, None]:
        if depth > 5:
            yield "\n\n[Max tool call depth reached]"
            return

        collected = ""
        tool_calls: list[dict] = []

        async for chunk in self.ollama.chat_raw(
            messages=messages, model=model, temperature=temperature,
            max_tokens=max_tokens, images=images, tools=tool_specs,
        ):
            content = chunk.get("content")
            calls = chunk.get("tool_calls")

            if content:
                collected += content
                if enable_thinking:
                    think_match = THINK_TAG_RE.search(content)
                    if think_match:
                        thinking_text = think_match.group(1).strip()
                        yield json.dumps({"type": "think", "content": thinking_text}) + "\n"
                    clean_content = THINK_TAG_RE.sub("", content)
                    if clean_content.strip():
                        yield json.dumps({"type": "content", "content": clean_content}) + "\n"
                else:
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

                yield json.dumps({"type": "tool_call", "name": func_name, "args": func_args}) + "\n"
                yield f"\n\n⚡ **Using tool: {func_name}**\n\n"

                result = self.tool_registry.execute(func_name, func_args)

                if result.success:
                    tool_output = result.output[:2000]
                else:
                    tool_output = f"Error: {result.error}"

                yield json.dumps({"type": "tool_result", "name": func_name, "output": tool_output[:500]}) + "\n"

                messages.append({"role": "assistant", "content": collected} if collected else {"role": "assistant", "content": f"[Calling tool: {func_name}]"})
                messages.append({"role": "tool", "content": tool_output, "name": func_name})
                collected = ""

                async for chunk in self._chat_with_tools(
                    messages=messages, model=model, temperature=temperature,
                    max_tokens=max_tokens, images=images, tool_specs=tool_specs,
                    session_id=session_id, depth=depth + 1,
                    enable_thinking=enable_thinking,
                ):
                    yield chunk

        if not tool_calls and enable_thinking and not collected.startswith("{"):
            final_content = THINK_TAG_RE.sub("", collected)
            if final_content.strip():
                yield json.dumps({"type": "done"}) + "\n"

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
                for w in content.split()[:10]:
                    clean = w.strip(".,!?;:'\"()[]{}").lower()
                    if len(clean) > 3:
                        user_topics.add(clean)
        topic_list = list(user_topics)[:8]
        if topic_list:
            return f"the user previously asked about: {', '.join(topic_list)}"
        return ""

    def format_chat_messages(self, system_prompt: Optional[str], history: list[dict], user_message: str) -> list[dict]:
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for msg in history:
            messages.append(msg)
        messages.append({"role": "user", "content": user_message})
        return messages

    def _log_request(self, endpoint: str, model: str, tokens_in: int, tokens_out: int, t_start: float, success: bool, session_id: Optional[str]) -> None:
        try:
            record = RequestRecord(
                endpoint=endpoint, model=model,
                tokens_in=tokens_in, tokens_out=tokens_out,
                duration_ms=(time.time() - t_start) * 1000,
                success=success, session_id=session_id,
            )
            self.usage_tracker.log_request(record)
        except Exception:
            pass

    async def list_models(self) -> list[dict]:
        ollama_models = await self.ollama.list_models()
        models = []
        for m in ollama_models:
            name = m.get("name", "")
            models.append({"name": name, "size": m.get("size", 0), "provider": "ollama"})
        if self.local and self.settings.enable_custom_model:
            models.insert(0, {"name": "winner-model (custom)", "size": 0, "provider": "local"})
        return models
