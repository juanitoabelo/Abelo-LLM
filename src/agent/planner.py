from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncGenerator, Optional

from src.config.settings import get_settings


class TaskStep:
    def __init__(
        self,
        id: str,
        description: str,
        tool: Optional[str] = None,
        tool_args: Optional[dict] = None,
        depends_on: Optional[list[str]] = None,
    ) -> None:
        self.id = id
        self.description = description
        self.tool = tool
        self.tool_args = tool_args or {}
        self.depends_on = depends_on or []
        self.result: Optional[str] = None
        self.status: str = "pending"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "tool": self.tool,
            "tool_args": self.tool_args,
            "depends_on": self.depends_on,
            "result": self.result,
            "status": self.status,
        }


class TaskPlan:
    def __init__(self, goal: str, steps: Optional[list[TaskStep]] = None) -> None:
        self.id = str(uuid.uuid4())[:8]
        self.goal = goal
        self.steps = steps or []
        self.created_at = time.time()
        self.status = "created"

    def add_step(self, step: TaskStep) -> None:
        self.steps.append(step)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "status": self.status,
        }


DECOMPOSE_PROMPT = """You are a task planning agent. Break down the following user request into a sequence of concrete steps.

Each step should either:
1. Use a tool (web_search, web_fetch, calculator, rag_query, memory_recall, memory_remember, file_read, file_write)
2. Be a reasoning/analysis step for the LLM to perform

Respond with ONLY a JSON array of step objects:
[
  {
    "id": "step_1",
    "description": "Clear description of what this step does",
    "tool": "tool_name or null if reasoning step",
    "tool_args": {"key": "value"} or {},
    "depends_on": ["step_0"] or []
  }
]

Tools available:
- web_search: {"query": "search query"}
- web_fetch: {"url": "https://..."}
- calculator: {"expression": "2 + 2"}
- rag_query: {"query": "search knowledge base"}
- memory_recall: {"key": "memory key"}
- memory_remember: {"key": "key", "value": "value"}"""


class AgentPlanner:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._ollama = None
        self._plans: dict[str, TaskPlan] = {}

    @property
    def ollama(self):
        if self._ollama is None:
            from src.llm.ollama import OllamaBackend
            self._ollama = OllamaBackend()
        return self._ollama

    async def decompose(self, goal: str, model: Optional[str] = None) -> TaskPlan:
        plan = TaskPlan(goal=goal)
        messages = [
            {"role": "system", "content": DECOMPOSE_PROMPT},
            {"role": "user", "content": f"Decompose this request into steps: {goal}"},
        ]
        collected = ""
        async for chunk in self.ollama.chat_raw(messages=messages, model=model, temperature=0.3):
            if "content" in chunk:
                collected += chunk["content"]
            if chunk.get("done"):
                break

        steps_data = self._parse_steps(collected)
        for s in steps_data:
            plan.add_step(TaskStep(
                id=s.get("id", f"step_{len(plan.steps)}"),
                description=s.get("description", ""),
                tool=s.get("tool"),
                tool_args=s.get("tool_args", {}),
                depends_on=s.get("depends_on", []),
            ))
        plan.status = "ready"
        self._plans[plan.id] = plan
        return plan

    def _parse_steps(self, text: str) -> list[dict]:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        text = text.strip()
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "steps" in data:
                return data["steps"]
        except json.JSONDecodeError:
            pass
        return []

    def get_plan(self, plan_id: str) -> Optional[TaskPlan]:
        return self._plans.get(plan_id)

    async def execute_plan(
        self,
        plan_id: str,
        tool_registry,
        model: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        plan = self._plans.get(plan_id)
        if not plan:
            yield f"Plan {plan_id} not found"
            return

        plan.status = "running"
        yield f"**Plan: {plan.goal}**\n\n"

        completed: dict[str, str] = {}
        for step in plan.steps:
            deps_not_met = [d for d in step.depends_on if d not in completed]
            if deps_not_met:
                yield f"⏳ Step {step.id} waiting for: {', '.join(deps_not_met)}\n\n"
                continue

            step.status = "running"
            yield f"**Step {step.id}:** {step.description}\n\n"

            if step.tool and step.tool in ("web_search", "web_fetch", "calculator", "memory_recall", "memory_remember"):
                yield f"  Using tool: {step.tool}\n\n"
                result = tool_registry.execute(step.tool, step.tool_args)
                step.result = result.output if result.success else f"Error: {result.error}"
                step.status = "done" if result.success else "failed"
                completed[step.id] = step.result
                yield f"  Result: {step.result[:500]}\n\n"
            else:
                prompt = f"""Context so far:
{chr(10).join(f'- Step {k}: {v[:200]}' for k, v in completed.items())}

Current step: {step.description}

Previous results: {json.dumps(completed)}

Provide a thorough response for this step based on the context above."""

                collected = ""
                msgs = [
                    {"role": "system", "content": "You are executing a multi-step plan. Provide clear, thorough responses."},
                    {"role": "user", "content": prompt},
                ]
                async for chunk in self.ollama.chat_raw(messages=msgs, model=model):
                    if "content" in chunk:
                        collected += chunk["content"]
                        yield chunk["content"]
                    if chunk.get("done"):
                        break
                step.result = collected
                step.status = "done"
                completed[step.id] = collected

        plan.status = "completed"
        yield f"\n\n✅ **Plan completed!**"
