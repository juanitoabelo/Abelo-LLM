"""Agentic RAG — self-query, self-critique, multi-hop decomposition."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncGenerator, Optional


class AgenticRAG:
    def __init__(self, vector_store, llm_backend, judge=None) -> None:
        self.vector_store = vector_store
        self.llm = llm_backend
        self.judge = judge

    async def decompose(self, question: str) -> list[str]:
        prompt = (
            f"Break this complex question into 2-5 simpler sub-questions that can be answered independently. "
            f"Return a JSON list of strings only.\n\nQuestion: {question}\n\nSub-questions:"
        )
        result = ""
        async for chunk in self.llm.generate(prompt=prompt, temperature=0.1, max_tokens=512):
            result += chunk
        try:
            parsed = json.loads(result)
            if isinstance(parsed, list):
                return parsed[:5]
        except json.JSONDecodeError:
            pass
        return [question]

    async def rewrite_query(self, question: str, conversation_history: Optional[list[dict]] = None) -> str:
        if not conversation_history:
            return question
        context = "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in conversation_history[-3:])
        prompt = (
            f"Given the conversation history, rewrite the latest question to be self-contained for a knowledge base search. "
            f"Respond with only the rewritten query.\n\nHistory:\n{context}\n\nLatest question: {question}\n\nRewritten query:"
        )
        result = ""
        async for chunk in self.llm.generate(prompt=prompt, temperature=0.1, max_tokens=256):
            result += chunk
        return result.strip() or question

    async def critique(self, question: str, context: str, answer: str) -> dict:
        if not self.judge:
            return {"sufficient": True, "gaps": []}
        prompt = (
            f"Given the question, retrieved context, and answer, determine if the context was sufficient. "
            f"Return JSON: {{\"sufficient\": bool, \"gaps\": [\"missing info 1\", ...], \"follow_up\": \"next question to fill gaps\"}}\n\n"
            f"Question: {question}\n\nContext: {context[:2000]}\n\nAnswer: {answer[:1000]}\n\nJudgment:"
        )
        result = ""
        async for chunk in self.llm.generate(prompt=prompt, temperature=0.1, max_tokens=512):
            result += chunk
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"sufficient": True, "gaps": []}

    async def answer(self, question: str, k: int = 5, conversation_history: Optional[list[dict]] = None, max_hops: int = 3) -> dict:
        rewritten = await self.rewrite_query(question, conversation_history)
        sub_questions = await self.decompose(rewritten)

        all_contexts = []
        all_answers = []

        for sq in sub_questions:
            results = self.vector_store.similarity_search(sq, k=k)
            ctx = "\n\n".join(r.page_content for r in results) if hasattr(results[0], 'page_content') else str(results)
            all_contexts.append(ctx)

            prompt = f"Context:\n{ctx}\n\nQuestion: {sq}\n\nAnswer concisely using only the context:"
            answer = ""
            async for chunk in self.llm.generate(prompt=prompt, temperature=0.3, max_tokens=512):
                answer += chunk
            all_answers.append(answer)

        combined = "\n\n".join(f"Q: {sq}\nA: {ans}" for sq, ans in zip(sub_questions, all_answers))
        final_prompt = f"Based on the following sub-answers, provide a comprehensive final answer to the original question.\n\nSub-answers:\n{combined}\n\nOriginal question: {question}\n\nFinal answer:"
        final = ""
        async for chunk in self.llm.generate(prompt=final_prompt, temperature=0.3, max_tokens=1024):
            final += chunk

        critique = await self.critique(question, "\n\n".join(all_contexts), final)

        return {
            "answer": final,
            "sub_questions": sub_questions,
            "sub_answers": all_answers,
            "critique": critique,
            "rewritten_query": rewritten,
            "hops": len(sub_questions),
        }

    async def answer_stream(self, question: str, k: int = 5, conversation_history: Optional[list[dict]] = None) -> AsyncGenerator[str, None]:
        rewritten = await self.rewrite_query(question, conversation_history)
        sub_questions = await self.decompose(rewritten)
        yield json.dumps({"type": "sub_questions", "questions": sub_questions, "rewritten": rewritten}) + "\n"

        all_answers = []
        for sq in sub_questions:
            results = self.vector_store.similarity_search(sq, k=k)
            ctx = "\n\n".join(r.page_content for r in results) if results else ""
            prompt = f"Context:\n{ctx}\n\nQuestion: {sq}\n\nAnswer concisely:"
            answer = ""
            async for chunk in self.llm.generate(prompt=prompt, temperature=0.3, max_tokens=512):
                answer += chunk
            all_answers.append(answer)

        yield json.dumps({"type": "combining", "sub_answers": all_answers}) + "\n"
        combined = "\n\n".join(f"Q: {sq}\nA: {ans}" for sq, ans in zip(sub_questions, all_answers))
        final_prompt = f"Provide a comprehensive final answer.\n\nSub-answers:\n{combined}\n\nQuestion: {question}\n\nFinal:"
        async for chunk in self.llm.generate(prompt=final_prompt, temperature=0.3, max_tokens=1024):
            yield chunk
