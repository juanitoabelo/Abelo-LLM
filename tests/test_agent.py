from __future__ import annotations

import pytest

from src.agent.planner import AgentPlanner, TaskPlan, TaskStep


class TestTaskStep:
    def test_create_step(self) -> None:
        step = TaskStep(id="s1", description="Test step", tool="calculator", tool_args={"expression": "1+1"}, reasoning="Need to compute")
        assert step.id == "s1"
        assert step.tool == "calculator"
        assert step.status == "pending"
        d = step.to_dict()
        assert d["reasoning"] == "Need to compute"

    def test_step_without_tool(self) -> None:
        step = TaskStep(id="s2", description="Reason only")
        assert step.tool is None
        assert step.status == "pending"


class TestTaskPlan:
    def test_create_plan(self) -> None:
        plan = TaskPlan(goal="Solve problem")
        assert plan.goal == "Solve problem"
        assert plan.status == "created"
        assert len(plan.steps) == 0

    def test_add_step(self) -> None:
        plan = TaskPlan(goal="Test")
        step = TaskStep(id="s1", description="Step one")
        plan.add_step(step)
        assert len(plan.steps) == 1

    def test_to_dict(self) -> None:
        plan = TaskPlan(goal="Learn Python")
        plan.add_step(TaskStep(id="s1", description="Research"))
        d = plan.to_dict()
        assert d["goal"] == "Learn Python"
        assert len(d["steps"]) == 1
        assert d["steps"][0]["id"] == "s1"


class TestAgentPlanner:
    def test_parse_steps_json(self) -> None:
        planner = AgentPlanner()
        json_str = '[{"id":"s1","description":"Test","tool":"calculator","tool_args":{"expression":"1+1"},"depends_on":[],"reasoning":"Test"}]'
        steps = planner._parse_steps(json_str)
        assert len(steps) == 1
        assert steps[0]["id"] == "s1"

    def test_parse_steps_fenced_code_block(self) -> None:
        planner = AgentPlanner()
        text = '```json\n[{"id":"s1","description":"Test"}]\n```'
        steps = planner._parse_steps(text)
        assert len(steps) == 1
        assert steps[0]["id"] == "s1"

    def test_parse_steps_with_steps_key(self) -> None:
        planner = AgentPlanner()
        text = '{"steps": [{"id":"s1","description":"Test"}]}'
        steps = planner._parse_steps(text)
        assert len(steps) == 1

    def test_parse_steps_invalid(self) -> None:
        planner = AgentPlanner()
        steps = planner._parse_steps("not json at all")
        assert steps == []

    def test_get_plan_nonexistent(self) -> None:
        planner = AgentPlanner()
        plan = planner.get_plan("nonexistent")
        assert plan is None
