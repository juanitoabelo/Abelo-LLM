"""Tests for sub-agent system."""

import pytest
from src.agent.sub_agent import SubAgent, SubAgentTask, AgentSwarm


def test_task_creation():
    task = SubAgentTask("agent1", "Test task")
    assert task.agent_id == "agent1"
    assert task.description == "Test task"
    assert task.status == "pending"


def test_task_dependencies():
    task = SubAgentTask("agent1", "Dep task", depends_on=["task_abc"])
    assert "task_abc" in task.depends_on


def test_sub_agent_creation():
    agent = SubAgent("coder_1", "Python Developer")
    assert agent.agent_id == "coder_1"
    assert agent.role == "Python Developer"


def test_sub_agent_add_task():
    agent = SubAgent("test", "Tester")
    task = agent.add_task("Write tests", depends_on=["setup"])
    assert task in agent.tasks
    assert task.depends_on == ["setup"]


def test_agent_swarm():
    swarm = AgentSwarm()
    agent = swarm.add_agent("a1", "Role1")
    assert "a1" in swarm.agents
    assert swarm.agents["a1"] is agent
