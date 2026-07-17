"""RAG query tool for the agent to search the knowledge base."""

from __future__ import annotations

from src.tools.registry import ToolResult


def rag_query(query: str, top_k: int = 3) -> ToolResult:
    import asyncio
    from pathlib import Path
    from src.config.settings import get_settings
    from src.rag.vector_store import VectorStore
    from src.rag.embedder import OllamaEmbedder
    from src.rag.retriever import RAGRetriever

    settings = get_settings()
    base_path = Path(settings.data_dir).parent
    store = VectorStore(str(base_path / "rag_store.db"))
    embedder = OllamaEmbedder()
    retriever = RAGRetriever(vector_store=store, embedder=embedder)

    async def _run():
        results = await retriever.retrieve(query, top_k=top_k)
        if not results:
            return "No relevant documents found."
        output = ""
        for i, r in enumerate(results, 1):
            source = r.get("source", "unknown")
            content = r.get("content", "")[:500]
            score = r.get("similarity", 0.0)
            output += f"{i}. [{source}] (score: {score:.2f})\n{content}\n\n"
        return output.strip()

    try:
        result = asyncio.run(_run())
        return ToolResult(True, result)
    except Exception as e:
        return ToolResult(False, "", f"RAG query failed: {e}")


def sql_query(query: str, limit: int = 20) -> ToolResult:
    import sqlite3
    from pathlib import Path

    if not query.strip().upper().startswith("SELECT"):
        return ToolResult(False, "", "Only SELECT queries are allowed")

    allowed_dbs = ["rag_store.db", "memory.db", "usage.db", "model_registry.db"]
    found = []
    for db in allowed_dbs:
        p = Path("data") / db
        if p.exists():
            found.append(str(p.resolve()))
    if not found:
        return ToolResult(False, "", "No databases found")

    results = []
    for db_path in found:
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query)
            rows = cursor.fetchmany(limit)
            if rows:
                cols = [d[0] for d in cursor.description]
                table = " | ".join(cols) + "\n" + "-" * len(" | ".join(cols))
                for row in rows:
                    table += "\n" + " | ".join(str(v)[:40] for v in row)
                results.append(f"Database: {Path(db_path).name}\n{table}")
            conn.close()
        except Exception as e:
            results.append(f"DB: {Path(db_path).name} - Error: {e}")

    return ToolResult(True, "\n\n".join(results) if results else "No results")
