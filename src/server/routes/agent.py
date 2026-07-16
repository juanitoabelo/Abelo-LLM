from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.agent.planner import AgentPlanner
from src.tools.registry import get_default_registry

router = APIRouter(prefix="/api/agent", tags=["agent"])


class PlanRequest(BaseModel):
    goal: str
    model: str = ""


@router.post("/plan")
async def create_plan(request: PlanRequest):
    planner = AgentPlanner()
    plan = await planner.decompose(request.goal, model=request.model or None)
    return {"plan": plan.to_dict()}


@router.get("/plan/{plan_id}")
async def get_plan(plan_id: str):
    planner = AgentPlanner()
    plan = planner.get_plan(plan_id)
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")
    return {"plan": plan.to_dict()}


@router.post("/plan/{plan_id}/execute")
async def execute_plan(plan_id: str, model: str = ""):
    planner = AgentPlanner()
    plan = planner.get_plan(plan_id)
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")

    tool_registry = get_default_registry()

    async def event_stream():
        async for chunk in planner.execute_plan(plan_id, tool_registry, model=model or None):
            yield f"data: {__import__('json').dumps({'content': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
