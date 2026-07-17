"""Tests for agentic RAG."""

import json
import pytest
from src.rag.agentic import AgenticRAG


class FakeVectorStore:
    def similarity_search(self, query, k=5):
        class FakeDoc:
            def __init__(self):
                self.page_content = f"Content about {query}"
        return [FakeDoc() for _ in range(min(k, 3))]


class FakeLLM:
    async def generate(self, prompt, temperature=0.3, max_tokens=512):
        chunks = []
        if "Break this complex question" in prompt:
            chunks.append('["sub q1", "sub q2"]')
        elif "rewrite" in prompt.lower() and "conversation" in prompt.lower():
            chunks.append("rewritten query here")
        elif "JSON" in prompt and "sufficient" in prompt:
            chunks.append('{"sufficient": true, "gaps": []}')
        else:
            chunks.append("fake answer about " + prompt[-50:])
        for c in chunks:
            yield c


@pytest.fixture
def rag():
    store = FakeVectorStore()
    llm = FakeLLM()
    return AgenticRAG(vector_store=store, llm_backend=llm)


@pytest.mark.asyncio
async def test_decompose(rag):
    subs = await rag.decompose("What is AI and ML?")
    assert len(subs) == 2
    assert "sub q1" in subs


@pytest.mark.asyncio
async def test_rewrite_with_history(rag):
    history = [{"role": "user", "content": "What is Python?"}]
    rewritten = await rag.rewrite_query("What about Java?", history)
    assert "rewritten" in rewritten.lower()


@pytest.mark.asyncio
async def test_rewrite_no_history(rag):
    rewritten = await rag.rewrite_query("What is AI?")
    assert rewritten == "What is AI?"


@pytest.mark.asyncio
async def test_critique(rag):
    result = await rag.critique("test q", "ctx", "answer")
    assert result["sufficient"] is True


@pytest.mark.asyncio
async def test_answer(rag):
    result = await rag.answer("What is AI?", k=2, max_hops=2)
    assert "answer" in result
    assert "sub_questions" in result
    assert "critique" in result
