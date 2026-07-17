"""Routes for semantic cache, agentic RAG, DAG, red teaming, model management."""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.agent.dag import DAGExecutor, DAGNode
from src.config.settings import get_settings
from src.eval.red_team import RedTeamRunner
from src.llm.ollama import OllamaBackend
from src.llm.semantic_cache import SemanticCache
from src.rag.agentic import AgenticRAG
from src.rag.vector_store import VectorStore

router = APIRouter(tags=["new_features"])

_cache = SemanticCache()
_red_team_runner: Optional[RedTeamRunner] = None


class ChatRequest(BaseModel):
    message: str
    model: str = ""
    stream: bool = False


class AgenticRAGRequest(BaseModel):
    question: str
    k: int = 5
    stream: bool = False


class DAGRequest(BaseModel):
    nodes: list[dict]
    merge_prompt: str = ""


class RedTeamRequest(BaseModel):
    run_all: bool = True


class PullModelRequest(BaseModel):
    name: str


class DeleteModelRequest(BaseModel):
    name: str


@router.post("/api/cache/clear")
async def clear_cache():
    _cache.clear()
    return {"status": "ok"}


@router.get("/api/cache/stats")
async def cache_stats():
    return _cache.stats()


@router.post("/api/rag/agentic")
async def agentic_rag(req: AgenticRAGRequest):
    store = VectorStore()
    llm = OllamaBackend()
    rag = AgenticRAG(vector_store=store, llm_backend=llm)
    result = await rag.answer(req.question, k=req.k)
    return result


@router.post("/api/dag/execute")
async def execute_dag(req: DAGRequest):
    llm = OllamaBackend()
    executor = DAGExecutor(llm_backend=llm)
    for n in req.nodes:
        executor.add_node(DAGNode(n["id"], n.get("task", ""), agent_id=n.get("agent_id", "default"), depends_on=n.get("depends_on", []), retry=n.get("retry", 2), timeout=n.get("timeout", 30.0)))
    if req.merge_prompt:
        final = await executor.run_with_merge(req.merge_prompt)
        return {"final": final, "summary": executor.summary()}
    results = []
    async for r in executor.run():
        results.append(r)
    return {"results": results, "summary": executor.summary()}


@router.post("/api/redteam/run")
async def run_redteam(req: RedTeamRequest):
    global _red_team_runner
    llm = OllamaBackend()
    _red_team_runner = RedTeamRunner(llm_backend=llm)
    report = await _red_team_runner.run_all()
    return {"total": report["total"], "refused": report["refused"], "complied": report["complied"], "safety_score": report["safety_score"]}


@router.get("/api/redteam/results")
async def redteam_results():
    import glob
    from pathlib import Path
    files = sorted(Path("data/redteam").glob("redteam_*.json"), reverse=True)
    results = []
    for f in files[:5]:
        try:
            results.append(json.loads(f.read_text()))
        except Exception:
            pass
    return {"reports": results}


@router.post("/api/models/pull")
async def pull_model(req: PullModelRequest):
    import subprocess
    try:
        subprocess.run(["ollama", "pull", req.name], capture_output=True, check=True, timeout=300)
        return {"status": "ok", "model": req.name}
    except subprocess.CalledProcessError as e:
        raise HTTPException(400, f"Failed to pull model: {e.stderr.decode()}")
    except FileNotFoundError:
        raise HTTPException(500, "Ollama CLI not found")


@router.post("/api/models/delete")
async def delete_model(req: DeleteModelRequest):
    import subprocess
    try:
        subprocess.run(["ollama", "rm", req.name], capture_output=True, check=True, timeout=30)
        return {"status": "ok", "model": req.name}
    except subprocess.CalledProcessError as e:
        raise HTTPException(400, f"Failed to delete model: {e.stderr.decode()}")


@router.get("/api/models/local")
async def local_models():
    import subprocess
    import json as _json
    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, check=True, timeout=15)
        lines = r.stdout.decode().strip().split("\n")[1:]
        models = []
        for line in lines:
            parts = line.split()
            if parts:
                models.append({"name": parts[0], "size": parts[2] if len(parts) > 2 else "?", "modified": parts[-2] if len(parts) > 3 else ""})
        return {"models": models}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"models": []}
