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
    ) -> None:
        self.store = vector_store
        self.embedder = embedder or OllamaEmbedder()
        self.top_k = top_k
        self.min_similarity = min_similarity

    async def retrieve(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        k = top_k or self.top_k
        query_vector = await self.embedder.embed_query(query)
        if not query_vector:
            return []
        results = self.store.similarity_search(query_vector, top_k=k)
        return [r for r in results if r["similarity"] >= self.min_similarity]

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
