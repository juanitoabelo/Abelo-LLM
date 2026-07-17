"""ReAct-based agent planner with self-reflection and tree-of-thought support."""

from __future__ import annotations

import json
import re
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
        reasoning: Optional[str] = None,
    ) -> None:
        self.id = id
        self.description = description
        self.tool = tool
        self.tool_args = tool_args or {}
        self.depends_on = depends_on or []
        self.result: Optional[str] = None
        self.status: str = "pending"
        self.reasoning = reasoning

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "tool": self.tool,
            "tool_args": self.tool_args,
            "depends_on": self.depends_on,
            "result": self.result,
            "status": self.status,
            "reasoning": self.reasoning,
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


REACT_PROMPT = """You are a ReAct (Reasoning + Acting) agent. Break down the user's request into steps.

For each step, you must output:
1. **Thought**: What you need to figure out or do
2. **Action**: The tool to use (or "reason" if no tool needed)
3. **Action Input**: Arguments for the tool

Available tools:
- web_search: {"query": "search query"}
- web_fetch: {"url": "https://..."}
- calculator: {"expression": "2 + 2"}
- code_execute: {"code": "python code"}
- file_read: {"path": "file path"}
- file_write: {"path": "path", "content": "content"}
- file_list: {"directory": "."}
- rag_query: {"query": "search query"}
- sql_query: {"query": "SELECT..."}
- memory_recall: {"key": "memory key"}
- memory_remember: {"key": "key", "value": "value"}
- memory_search: {"query": "search query"}

Respond with a JSON array of step objects:
[
  {
    "id": "step_1",
    "description": "What this step does",
    "tool": "tool_name or null for reasoning",
    "tool_args": {"key": "value"} or {},
    "depends_on": [],
    "reasoning": "Why this step is needed and what you expect to learn"
  }
]"""

REFLECT_PROMPT = """You are reflecting on the result of a previous action.

Context: {context}
Current step: {step_description}
Tool used: {tool_name}
Tool result: {tool_result}

Thought: Was the result sufficient? What should happen next?
If the result is insufficient or incorrect, output a corrected action.
If the goal is met, output: {"status": "complete", "summary": "..."}
If more work is needed, output: {"status": "continue", "next_step": "..."}"""


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
            {"role": "system", "content": REACT_PROMPT},
            {"role": "user", "content": f"Decompose this request into ReAct steps: {goal}"},
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
                reasoning=s.get("reasoning"),
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
            json_match = re.search(r"\[.*\]", text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
        return []

    def get_plan(self, plan_id: str) -> Optional[TaskPlan]:
        return self._plans.get(plan_id)

    async def reflect(self, step: TaskStep, tool_registry, model: Optional[str] = None) -> str:
        context = "\n".join(f"- Step {s.id}: {s.description} -> {s.status}: {(s.result or '')[:200]}" for s in self._plans.get(step.id[:8] + "'s plan", TaskPlan("")).steps if s.status != "pending")
        prompt = REFLECT_PROMPT.format(
            context=context,
            step_description=step.description,
            tool_name=step.tool or "reasoning",
            tool_result=(step.result or "")[:500],
        )
        collected = ""
        msgs = [
            {"role": "system", "content": "You are a self-reflection agent. Analyze results and decide next actions."},
            {"role": "user", "content": prompt},
        ]
        async for chunk in self.ollama.chat_raw(messages=msgs, model=model, temperature=0.3):
            if "content" in chunk:
                collected += chunk["content"]
            if chunk.get("done"):
                break
        return collected.strip()

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
        yield json.dumps({"type": "plan_start", "goal": plan.goal}) + "\n"
        yield f"**Plan: {plan.goal}**\n\n"

        completed: dict[str, str] = {}
        for step in plan.steps:
            deps_not_met = [d for d in step.depends_on if d not in completed]
            if deps_not_met:
                yield f"⏳ Step {step.id} waiting for: {', '.join(deps_not_met)}\n\n"
                continue

            step.status = "running"
            if step.reasoning:
                yield json.dumps({"type": "think", "content": step.reasoning}) + "\n"
            yield f"**Step {step.id}:** {step.description}\n\n"

            if step.tool and step.tool not in ("null", "reason", "None", None):
                yield json.dumps({"type": "tool_call", "name": step.tool, "args": step.tool_args}) + "\n"
                yield f"  Using tool: {step.tool}\n\n"
                try:
                    result = tool_registry.execute(step.tool, step.tool_args)
                    step.result = result.output if result.success else f"Error: {result.error}"
                    step.status = "done" if result.success else "failed"
                    completed[step.id] = step.result
                    yield json.dumps({"type": "tool_result", "name": step.tool, "output": (step.result or "")[:500]}) + "\n"
                    yield f"  Result: {(step.result or '')[:500]}\n\n"
                except Exception as e:
                    step.result = f"Execution error: {e}"
                    step.status = "failed"
                    completed[step.id] = step.result
                    yield f"  Error: {e}\n\n"

                if step.status == "failed" and step.tool:
                    reflection = await self.reflect(step, tool_registry, model)
                    yield json.dumps({"type": "reflect", "content": reflection[:300]}) + "\n"
                    yield f"  Reflection: {reflection[:300]}\n\n"
            else:
                prompt = f"""Context so far:
{chr(10).join(f'- Step {k}: {v[:200]}' for k, v in completed.items())}

Current step: {step.description}
Reasoning: {step.reasoning or 'Analyze and respond based on context.'}

Previous results: {json.dumps(completed)}

Provide a thorough response for this step."""

                collected = ""
                msgs = [
                    {"role": "system", "content": "You are executing a multi-step plan with ReAct reasoning. Provide clear, thorough responses with your reasoning."},
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
        yield json.dumps({"type": "plan_done", "plan_id": plan_id}) + "\n"
        yield f"\n\n✅ **Plan completed!**"
