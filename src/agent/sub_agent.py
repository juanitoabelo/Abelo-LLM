"""Agent with sub-agents — parallel execution, delegated tasks, DAG workflow."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, AsyncGenerator, Optional


class SubAgentTask:
    def __init__(self, agent_id: str, description: str, tool_calls: Optional[list[dict]] = None, depends_on: Optional[list[str]] = None) -> None:
        self.id = str(uuid.uuid4())[:8]
        self.agent_id = agent_id
        self.description = description
        self.tool_calls = tool_calls or []
        self.depends_on = depends_on or []
        self.result: Any = None
        self.status = "pending"
        self.error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "description": self.description,
            "tool_calls": self.tool_calls,
            "depends_on": self.depends_on,
            "status": self.status,
            "error": self.error,
        }


class SubAgent:
    def __init__(self, agent_id: str, role: str, model: str = "llama3.2:1b") -> None:
        self.agent_id = agent_id
        self.role = role
        self.model = model
        self.tasks: list[SubAgentTask] = []

    def add_task(self, description: str, tool_calls: Optional[list[dict]] = None, depends_on: Optional[list[str]] = None) -> SubAgentTask:
        task = SubAgentTask(self.agent_id, description, tool_calls, depends_on)
        self.tasks.append(task)
        return task

    async def execute(self, tool_registry, ollama_backend) -> AsyncGenerator[str, None]:
        yield json.dumps({"type": "agent_start", "agent_id": self.agent_id, "role": self.role}) + "\n"
        completed: dict[str, Any] = {}

        async def _run_task(task: SubAgentTask):
            deps = [d for d in task.depends_on if d not in completed]
            if deps:
                return
            task.status = "running"
            yield json.dumps({"type": "task_start", "task_id": task.id, "agent_id": self.agent_id, "description": task.description}) + "\n"

            for tc in task.tool_calls:
                try:
                    result = tool_registry.execute(tc.get("name", ""), tc.get("arguments", {}))
                    task.result = result.output if result.success else f"Error: {result.error}"
                except Exception as e:
                    task.result = f"Exception: {e}"

            if not task.tool_calls:
                prompt = f"You are a {self.role} agent. Task: {task.description}\nRespond concisely."
                collected = ""
                async for chunk in ollama_backend.generate(prompt=prompt, model=self.model, max_tokens=256):
                    collected += chunk
                task.result = collected

            task.status = "done"
            completed[task.id] = task.result
            yield json.dumps({"type": "task_done", "task_id": task.id, "agent_id": self.agent_id, "result": (task.result or "")[:300]}) + "\n"

        ready = [t for t in self.tasks if not t.depends_on]
        remaining = [t for t in self.tasks if t.depends_on]

        for task in ready:
            async for msg in _run_task(task):
                yield msg

        for task in remaining:
            async for msg in _run_task(task):
                yield msg

        yield json.dumps({"type": "agent_done", "agent_id": self.agent_id, "role": self.role}) + "\n"


class AgentSwarm:
    def __init__(self) -> None:
        self.agents: dict[str, SubAgent] = {}

    def add_agent(self, agent_id: str, role: str, model: str = "llama3.2:1b") -> SubAgent:
        agent = SubAgent(agent_id, role, model)
        self.agents[agent_id] = agent
        return agent

    async def run_all(self, tool_registry, ollama_backend) -> AsyncGenerator[str, None]:
        for agent_id, agent in self.agents.items():
            async for msg in agent.execute(tool_registry, ollama_backend):
                yield msg
