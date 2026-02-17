import pytest

from agent.core.planner import Planner


def test_plan_from_tool_call_builds_single_step_plan():
    plan = Planner._plan_from_tool_call("do math", {"name": "calculator", "arguments": {"expression": "2+2"}})
    assert plan.goal == "do math"
    assert plan.final_output == "step_1_output"
    assert len(plan.steps) == 1
    assert plan.steps[0].required_capability == "calculator"
    assert plan.steps[0].tool_args["expression"] == "2+2"


def test_plan_from_tool_call_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        Planner._plan_from_tool_call("x", {"name": "calculator", "arguments": "expression=2+2"})

