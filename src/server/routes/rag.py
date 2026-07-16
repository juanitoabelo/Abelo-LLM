from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from src.rag.retriever import RAGRetriever
from src.rag.vector_store import VectorStore
from src.rag.embedder import OllamaEmbedder
from src.config.settings import get_settings

router = APIRouter(prefix="/api/rag", tags=["rag"])


def _get_retriever() -> RAGRetriever:
    settings = get_settings()
    base_path = Path(settings.data_dir).parent
    store = VectorStore(str(base_path / "rag_store.db"))
    embedder = OllamaEmbedder()
    return RAGRetriever(vector_store=store, embedder=embedder)


@router.get("/status")
async def rag_status():
    retriever = _get_retriever()
    embedder_ok = await retriever.embedder.is_available()
    doc_count = retriever.store.count_documents()
    sources = retriever.store.get_sources()
    return {
        "embedder_available": embedder_ok,
        "document_count": doc_count,
        "sources": sources,
    }


class IngestTextRequest(BaseModel):
    text: str
    source: str = "inline"


@router.post("/ingest/text")
async def ingest_text(request: IngestTextRequest):
    retriever = _get_retriever()
    result = await retriever.ingest_text(request.text, source=request.source)
    if result["status"] == "embedding_failed":
        raise HTTPException(status_code=500, detail="Embedding failed")
    return result


@router.post("/ingest/file")
async def ingest_file(file: UploadFile):
    suffix = Path(file.filename or "file.txt").suffix or ".txt"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    retriever = _get_retriever()
    try:
        result = await retriever.ingest_file(tmp_path)
    finally:
        os.unlink(tmp_path)

    if result["status"] == "embedding_failed":
        raise HTTPException(status_code=500, detail="Embedding failed")
    return result


@router.post("/ingest/directory")
async def ingest_directory(directory: str):
    path = Path(directory)
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a valid directory: {directory}")
    retriever = _get_retriever()
    results = await retriever.ingest_directory(directory)
    return {"results": results, "total_chunks": sum(r.get("chunks", 0) for r in results)}


@router.delete("/sources/{source:path}")
async def delete_source(source: str):
    retriever = _get_retriever()
    retriever.store.delete_source(source)
    return {"status": "deleted", "source": source}


@router.post("/query")
async def query_rag(query: str, top_k: Optional[int] = None):
    retriever = _get_retriever()
    results = await retriever.retrieve(query, top_k=top_k)
    return {"results": results}
