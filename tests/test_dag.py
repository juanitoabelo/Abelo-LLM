"""Tests for multi-agent DAG orchestration."""

import pytest
from src.agent.dag import DAGNode, DAGExecutor


class FakeLLM:
    async def generate(self, prompt, temperature=0.3, max_tokens=512):
        yield f"result for {prompt[-30:]}"


@pytest.fixture
def dag():
    return DAGExecutor(llm_backend=FakeLLM())


def test_add_node(dag):
    n = dag.add_node(DAGNode("n1", "task1"))
    assert "n1" in dag.nodes
    assert n.node_id == "n1"


def test_add_edge(dag):
    dag.add_node(DAGNode("a", "task a"))
    dag.add_node(DAGNode("b", "task b"))
    dag.add_edge("a", "b")
    assert "a" in dag.nodes["b"].depends_on


def test_topological_sort_simple(dag):
    dag.add_node(DAGNode("a", "a"))
    dag.add_node(DAGNode("b", "b", depends_on=["a"]))
    dag.add_node(DAGNode("c", "c", depends_on=["b"]))
    layers = dag.topologically_sorted()
    assert len(layers) == 3
    assert layers[0][0].node_id == "a"
    assert layers[1][0].node_id == "b"


def test_topological_sort_parallel(dag):
    dag.add_node(DAGNode("a", "a"))
    dag.add_node(DAGNode("b", "b", depends_on=["a"]))
    dag.add_node(DAGNode("c", "c", depends_on=["a"]))
    layers = dag.topologically_sorted()
    assert layers[0][0].node_id == "a"
    assert len(layers[1]) == 2


@pytest.mark.asyncio
async def test_run(dag):
    dag.add_node(DAGNode("n1", "task1"))
    results = []
    async for r in dag.run():
        results.append(r)
    assert len(results) == 1
    assert results[0]["status"] == "done"
    assert results[0]["node_id"] == "n1"


@pytest.mark.asyncio
async def test_summary(dag):
    dag.add_node(DAGNode("n1", "t1"))
    dag.add_node(DAGNode("n2", "t2", depends_on=["n1"]))
    async for _ in dag.run():
        pass
    s = dag.summary()
    assert s["total"] == 2
    assert s["done"] == 2


def test_node_to_dict():
    n = DAGNode("test", "do thing", agent_id="agent1", retry=3, timeout=15.0)
    d = n.to_dict()
    assert d["id"] == "test"
    assert d["agent_id"] == "agent1"
    assert d["status"] == "pending"
