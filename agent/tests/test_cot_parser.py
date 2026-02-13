import pytest
from pydantic import BaseModel

from agent.llm.cot_parser import extract_thinking, extract_json_candidate, parse_structured, CoTParseError
from agent.core.planner import Planner
from agent.core.schema import Plan


class SimpleModel(BaseModel):
    a: int
    b: str


def test_extract_thinking_none():
    assert extract_thinking('{"a":1}') is None


def test_extract_thinking_multiple_blocks():
    text = "<thinking>one</thinking>\n...\n<thinking>two</thinking>"
    assert extract_thinking(text) == "one\n\ntwo"


def test_extract_json_candidate_fenced_json():
    text = "<thinking>x</thinking>\n```json\n{\"a\":1,\"b\":\"y\"}\n```\n"
    assert extract_json_candidate(text) == "{\"a\":1,\"b\":\"y\"}"


def test_extract_json_candidate_balanced_from_text_with_trailing():
    text = "<thinking>x</thinking>\n{\"a\":1,\"b\":\"y\"}\ntrailing"
    assert extract_json_candidate(text) == "{\"a\":1,\"b\":\"y\"}"


def test_extract_json_candidate_no_json_returns_stripped_text():
    assert extract_json_candidate("no json here") == "no json here"


def test_extract_json_candidate_unbalanced_returns_whole_text():
    text = "{\"a\": 1"
    assert extract_json_candidate(text) == text


def test_extract_json_candidate_fenced_any():
    text = "```\n{\"a\":1,\"b\":\"y\"}\n```"
    assert extract_json_candidate(text) == "{\"a\":1,\"b\":\"y\"}"


def test_extract_json_candidate_handles_escaped_quotes():
    text = "{\"a\":\"x\\\\\\\"y\"} trailing"
    assert extract_json_candidate(text) == "{\"a\":\"x\\\\\\\"y\"}"


def test_parse_structured_ok():
    text = "<thinking>x</thinking>\n```json\n{\"a\":1,\"b\":\"y\"}\n```"
    parsed = parse_structured(text, SimpleModel)
    assert parsed.a == 1
    assert parsed.b == "y"


def test_parse_structured_error_contains_preview():
    text = "<thinking>x</thinking>\n```json\n{\"a\":1,\"b\":}\n```"
    with pytest.raises(CoTParseError) as exc:
        parse_structured(text, SimpleModel)
    assert exc.value.preview
    assert exc.value.length == len(text)


def test_parse_structured_fallback_to_json_loads(monkeypatch):
    monkeypatch.setattr(SimpleModel, "model_validate_json", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("nope")))
    text = "{\"a\":1,\"b\":\"y\"}"
    parsed = parse_structured(text, SimpleModel)
    assert parsed.a == 1


def test_planner_parse_json_attaches_thinking_to_metadata():
    planner = Planner()
    content = """
<thinking>
step 1
step 2
</thinking>
```json
{
  "goal": "g",
  "steps": [
    {
      "step_id": "step_1",
      "intent": "do",
      "operation": { "type": "reason", "inputs": {} },
      "output_var": "out",
      "required_capability": "llm_reasoning",
      "certainty": "certain",
      "fallback_strategy": "fail"
    }
  ],
  "final_output": "out",
  "metadata": {}
}
```
"""
    plan = planner._parse_json(content, Plan)
    assert isinstance(plan, Plan)
    assert plan.metadata.get("thinking") and "step 1" in plan.metadata["thinking"]


def test_planner_plan_end_to_end_offline_llm(offline_llm_and_mcp):
    planner = Planner()
    tools = [
        {"name": "write_file", "description": "write file", "args_schema": {"properties": {"path": {}, "content": {}}}},
        {"name": "read_file", "description": "read file", "args_schema": {"properties": {"path": {}}}},
        {"name": "count_words", "description": "count words", "args_schema": {"properties": {"text": {}}}},
    ]
    objective = "Create a file at sandbox/offline.txt with content 'hello world'. Then read it back and count the words in it."
    plan = planner.plan(objective, tools)
    assert isinstance(plan, Plan)
    assert plan.steps and plan.steps[-1].output_var == plan.final_output
    assert plan.metadata.get("thinking")


def test_planner_refine_plan_offline_llm(offline_llm_and_mcp):
    planner = Planner()
    tools = [{"name": "summarize_text", "description": "summarize", "args_schema": {"properties": {"text": {}}}}]
    objective = "Summarize: abc"
    original = planner.plan(objective, tools)
    refined = planner.refine_plan(objective, original, "validation error")
    assert isinstance(refined, Plan)


def test_planner_plan_fallback_on_exception(monkeypatch):
    planner = Planner()
    monkeypatch.setattr(planner, "_generate_skeleton", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    plan = planner.plan("x", [])
    assert isinstance(plan, Plan)
    assert plan.steps and plan.steps[0].step_id == "step_1"


def test_config_expands_env_placeholders(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "abc")
    monkeypatch.delenv("AGENT_CONFIG", raising=False)

    from agent.utils import config as config_mod

    config_mod.Config._instance = None
    config_mod.Config._config = {}

    cfg = config_mod.Config()
    assert cfg.get("mcp.servers.official_github.env.GITHUB_PERSONAL_ACCESS_TOKEN") == "abc"


def test_normalize_github_owner_repo_from_full_name():
    from agent.core.executor import _normalize_github_owner_repo

    out = _normalize_github_owner_repo({"repo": "modelcontextprotocol/servers", "path": "README.md"})
    assert out["owner"] == "modelcontextprotocol"
    assert out["repo"] == "servers"


def test_normalize_github_owner_repo_from_url():
    from agent.core.executor import _normalize_github_owner_repo

    out = _normalize_github_owner_repo({"repo": "https://github.com/modelcontextprotocol/servers", "path": "README.md"})
    assert out["owner"] == "modelcontextprotocol"
    assert out["repo"] == "servers"


def test_normalize_github_owner_repo_from_search_json_string():
    import json
    from agent.core.executor import _normalize_github_owner_repo

    repo_json = json.dumps([{"full_name": "modelcontextprotocol/servers"}], ensure_ascii=False)
    out = _normalize_github_owner_repo({"repo": repo_json, "path": "."})
    assert out["owner"] == "modelcontextprotocol"
    assert out["repo"] == "servers"


def test_planner_fix_final_output_uses_last_step_output_var():
    from agent.core.planner import Planner
    from agent.core.schema import Plan, PlanStep, OperationSpec, OperationType, VariableBinding, VariableRef

    plan = Plan(
        goal="x",
        steps=[
            PlanStep(
                step_id="step_1",
                intent="write",
                operation=OperationSpec(
                    type=OperationType.WRITE,
                    inputs={
                        "file_path": VariableBinding(value_ref=VariableRef(source="literal", name="github_summary.txt")),
                        "content": VariableBinding(value_ref=VariableRef(source="literal", name="hello")),
                    },
                ),
                output_var="write_result",
                required_capability="mcp:official_filesystem:write_file",
                certainty="certain",
                fallback_strategy="fail",
            )
        ],
        final_output="github_summary.txt",
    )

    planner = Planner()
    fixed = planner._fix_final_output(plan)
    assert fixed.final_output == "write_result"
