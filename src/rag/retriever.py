from __future__ import annotations

from typing import Optional

from src.rag.embedder import OllamaEmbedder
from src.rag.loader import DocumentLoader
from src.rag.vector_store import VectorStore


class RAGRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Optional[OllamaEmbedder] = None,
        top_k: int = 5,
        min_similarity: float = 0.3,
        enable_web_fallback: bool = True,
    ) -> None:
        self.store = vector_store
        self.embedder = embedder or OllamaEmbedder()
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.enable_web_fallback = enable_web_fallback

    async def retrieve(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        k = top_k or self.top_k
        query_vector = await self.embedder.embed_query(query)
        if not query_vector:
            return []
        results = self.store.similarity_search(query_vector, top_k=k)
        filtered = [r for r in results if r["similarity"] >= self.min_similarity]

        if not filtered and self.enable_web_fallback:
            web_results = await self._web_fallback(query)
            if web_results:
                return web_results

        return filtered

    async def retrieve_formatted(self, query: str, top_k: Optional[int] = None) -> str:
        results = await self.retrieve(query, top_k)
        if not results:
            return ""
        sections: list[str] = []
        for r in results:
            source = r["source"]
            content = r["content"]
            sim = r["similarity"]
            sections.append(f"[Source: {source} (relevance: {sim:.2f})]\n{content}")
        return "\n\n".join(sections)

    async def _web_fallback(self, query: str) -> list[dict]:
        try:
            from src.tools.web_search import web_search
            from src.tools.web_fetch import web_fetch

            search_result = web_search(query)
            if not search_result.success or not search_result.output:
                return []

            lines = search_result.output.split("\n")
            urls: list[str] = []
            for line in lines:
                line = line.strip()
                if line.startswith(("http://", "https://")):
                    urls.append(line)
                elif "  " in line:
                    parts = line.split("  ")
                    for p in parts:
                        p = p.strip()
                        if p.startswith(("http://", "https://")):
                            urls.append(p)

            results: list[dict] = []
            for url in urls[:2]:
                fetch_result = web_fetch(url)
                if fetch_result.success and fetch_result.output:
                    results.append({
                        "id": -len(results) - 1,
                        "source": url,
                        "chunk_index": 0,
                        "content": fetch_result.output[:1000],
                        "metadata": {"type": "web_fallback"},
                        "similarity": 0.5,
                    })
            return results
        except Exception:
            return []

    async def ingest_file(self, file_path: str) -> dict:
        loader = DocumentLoader()
        chunks = loader.load_file(file_path)
        if not chunks:
            return {"status": "empty", "chunks": 0, "source": file_path}
        texts = [c["content"] for c in chunks]
        vectors = await self.embedder.embed(texts)
        if not vectors or len(vectors) != len(texts):
            return {"status": "embedding_failed", "chunks": len(chunks), "source": file_path}
        self.store.add_documents(
            sources=[c["source"] for c in chunks],
            chunk_indices=[c["chunk_index"] for c in chunks],
            contents=texts,
            vectors=vectors,
        )
        return {"status": "ok", "chunks": len(chunks), "source": file_path}

    async def ingest_directory(self, dir_path: str) -> list[dict]:
        loader = DocumentLoader()
        chunks = loader.load_directory(dir_path)
        if not chunks:
            return []
        texts = [c["content"] for c in chunks]
        vectors = await self.embedder.embed(texts)
        if not vectors or len(vectors) != len(texts):
            return []
        self.store.add_documents(
            sources=[c["source"] for c in chunks],
            chunk_indices=[c["chunk_index"] for c in chunks],
            contents=texts,
            vectors=vectors,
        )
        return [{"status": "ok", "chunks": len(chunks), "source": dir_path}]

    async def ingest_text(self, text: str, source: str = "inline") -> dict:
        loader = DocumentLoader()
        chunks = loader.load_text(text, source=source)
        if not chunks:
            return {"status": "empty", "chunks": 0, "source": source}
        texts = [c["content"] for c in chunks]
        vectors = await self.embedder.embed(texts)
        if not vectors or len(vectors) != len(texts):
            return {"status": "embedding_failed", "chunks": len(chunks), "source": source}
        self.store.add_documents(
            sources=[c["source"] for c in chunks],
            chunk_indices=[c["chunk_index"] for c in chunks],
            contents=texts,
            vectors=vectors,
        )
        return {"status": "ok", "chunks": len(chunks), "source": source}
