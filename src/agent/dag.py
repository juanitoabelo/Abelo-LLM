"""Multi-agent DAG orchestration — parallel execution, retry, result merging."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, AsyncGenerator, Callable, Optional


class DAGNode:
    def __init__(self, node_id: str, task: str, agent_id: str = "default", depends_on: Optional[list[str]] = None, retry: int = 2, timeout: float = 30.0, max_concurrency: int = 1) -> None:
        self.node_id = node_id
        self.task = task
        self.agent_id = agent_id
        self.depends_on = depends_on or []
        self.retry = retry
        self.timeout = timeout
        self.max_concurrency = max_concurrency
        self.result: Any = None
        self.error: Optional[str] = None
        self.status = "pending"
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None

    def to_dict(self) -> dict:
        return {"id": self.node_id, "task": self.task, "agent_id": self.agent_id, "depends_on": self.depends_on, "status": self.status, "error": self.error}


class DAGExecutor:
    def __init__(self, llm_backend) -> None:
        self.llm = llm_backend
        self.nodes: dict[str, DAGNode] = {}

    def add_node(self, node: DAGNode) -> DAGNode:
        self.nodes[node.node_id] = node
        return node

    def add_edge(self, from_id: str, to_id: str) -> None:
        if to_id in self.nodes:
            self.nodes[to_id].depends_on.append(from_id)

    def topologically_sorted(self) -> list[list[DAGNode]]:
        visited: set[str] = set()
        layers: list[list[DAGNode]] = []

        def _get_ready(assigned: set[str]) -> list[DAGNode]:
            ready = []
            for n in self.nodes.values():
                if n.node_id in assigned or n.node_id in visited:
                    continue
                if all(d in assigned for d in n.depends_on):
                    ready.append(n)
            return ready

        assigned: set[str] = set()
        while len(assigned) < len(self.nodes):
            layer = _get_ready(assigned)
            if not layer:
                break
            layers.append(layer)
            for n in layer:
                visited.add(n.node_id)
                assigned.add(n.node_id)
        return layers

    async def _execute_node(self, node: DAGNode, tool_registry=None) -> None:
        node.status = "running"
        node.started_at = time.time()

        for attempt in range(node.retry + 1):
            try:
                if tool_registry and node.task.startswith("tool:"):
                    tool_name = node.task[5:].split("(")[0].strip()
                    tool_args_raw = node.task[len(f"tool:{tool_name}("):-1] if "(" in node.task else "{}"
                    try:
                        tool_args = json.loads(tool_args_raw) if tool_args_raw else {}
                    except json.JSONDecodeError:
                        tool_args = {}
                    result = tool_registry.execute(tool_name, tool_args)
                    node.result = result.output if result.success else f"error: {result.error}"
                else:
                    prompt = f"Complete this task concisely.\n\nTask: {node.task}\n\nResponse:"
                    collected = ""
                    async for chunk in self.llm.generate(prompt=prompt, temperature=0.3, max_tokens=512):
                        collected += chunk
                    node.result = collected
                node.status = "done"
                node.error = None
                return
            except Exception as e:
                node.error = str(e)
                await asyncio.sleep(0.5 * (attempt + 1))
        node.status = "failed"

    async def run(self, tool_registry=None) -> AsyncGenerator[dict, None]:
        layers = self.topologically_sorted()
        results: dict[str, Any] = {}

        for layer in layers:
            pending = [self._execute_node(n, tool_registry) for n in layer]
            done = await asyncio.gather(*pending, return_exceptions=True)
            for n, d in zip(layer, done):
                if isinstance(d, Exception):
                    n.status = "failed"
                    n.error = str(d)
                results[n.node_id] = n.result
                yield {"node_id": n.node_id, "status": n.status, "result": (n.result or "")[:200], "error": n.error}

        for n in self.nodes.values():
            n.completed_at = time.time()

    async def run_with_merge(self, merge_prompt: str, tool_registry=None) -> str:
        async for _ in self.run(tool_registry):
            pass
        node_results = {n.node_id: n.result for n in self.nodes.values() if n.status == "done"}
        prompt = f"{merge_prompt}\n\nIndividual results:\n{json.dumps(node_results, indent=2)}\n\nSynthesize these into a final answer:"
        final = ""
        async for chunk in self.llm.generate(prompt=prompt, temperature=0.3, max_tokens=1024):
            final += chunk
        return final

    def summary(self) -> dict:
        return {
            "total": len(self.nodes),
            "done": sum(1 for n in self.nodes.values() if n.status == "done"),
            "failed": sum(1 for n in self.nodes.values() if n.status == "failed"),
            "pending": sum(1 for n in self.nodes.values() if n.status == "pending"),
            "duration": max((n.completed_at or 0) for n in self.nodes.values()) - min((n.started_at or float('inf')) for n in self.nodes.values()) if self.nodes else 0,
        }
