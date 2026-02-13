import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

from agent.utils.config import config


def pytest_configure(config):
    if os.environ.get("RUN_INTEGRATION_TESTS") != "1":
        return

    opt = getattr(config, "option", None)
    if not opt:
        return

    if hasattr(opt, "cov_fail_under"):
        opt.cov_fail_under = 0
    if hasattr(opt, "no_cov"):
        opt.no_cov = True


@dataclass
class _Message:
    content: str


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Response:
    choices: List[_Choice]


def _lit(value: str) -> Dict[str, Any]:
    return {"value_ref": {"source": "literal", "name": value}}


def _step_out(step_id: str) -> Dict[str, Any]:
    return {"value_ref": {"source": "step_output", "name": step_id}}


def _plan_for_objective(objective: str) -> Dict[str, Any]:
    obj = objective.strip()
    low = obj.lower()

    if "search for a github repository" in low and "readme" in low:
        summary_name = "github_summary.txt"
        if "github_summary.txt" in low:
            summary_name = "github_summary.txt"
        plan = {
            "goal": "Offline GitHub analysis (mock)",
            "steps": [
                {
                    "step_id": "step_1",
                    "intent": "Search repositories",
                    "operation": {"type": "search", "inputs": {"query": _lit("modelcontextprotocol/servers")}},
                    "output_var": "repo_search",
                    "required_capability": "search_repositories",
                    "certainty": "certain",
                    "fallback_strategy": "fail",
                },
                {
                    "step_id": "step_2",
                    "intent": "Read README",
                    "operation": {
                        "type": "read",
                        "inputs": {"repo": _lit("modelcontextprotocol/servers"), "path": _lit("README.md")},
                    },
                    "output_var": "readme_text",
                    "required_capability": "get_file_contents",
                    "certainty": "certain",
                    "fallback_strategy": "fail",
                },
                {
                    "step_id": "step_3",
                    "intent": "Summarize README",
                    "operation": {"type": "transform", "inputs": {"text": _step_out("step_2")}},
                    "output_var": "readme_summary",
                    "required_capability": "summarize_text",
                    "certainty": "certain",
                    "fallback_strategy": "fail",
                },
                {
                    "step_id": "step_4",
                    "intent": "Write summary to sandbox",
                    "operation": {
                        "type": "write",
                        "inputs": {"file_path": _lit(summary_name), "content": _step_out("step_3")},
                    },
                    "output_var": "write_result",
                    "required_capability": "write_file",
                    "certainty": "certain",
                    "fallback_strategy": "fail",
                },
            ],
            "final_output": "write_result",
            "metadata": {"offline_test": True},
        }
        return plan

    m = re.search(r"create a file at\s+(.+?)\s+with content\s+'([^']*)'", obj, flags=re.IGNORECASE)
    if m:
        file_path, content = m.group(1).strip(), m.group(2)
        plan = {
            "goal": "Offline filesystem flow",
            "steps": [
                {
                    "step_id": "step_1",
                    "intent": "Write file",
                    "operation": {"type": "write", "inputs": {"file_path": _lit(file_path), "content": _lit(content)}},
                    "output_var": "write_result",
                    "required_capability": "write_file",
                    "certainty": "certain",
                    "fallback_strategy": "fail",
                },
                {
                    "step_id": "step_2",
                    "intent": "Read file",
                    "operation": {"type": "read", "inputs": {"file_path": _lit(file_path)}},
                    "output_var": "file_content",
                    "required_capability": "read_file",
                    "certainty": "certain",
                    "fallback_strategy": "fail",
                },
                {
                    "step_id": "step_3",
                    "intent": "Count words",
                    "operation": {"type": "transform", "inputs": {"text": _step_out("step_2")}},
                    "output_var": "word_count",
                    "required_capability": "count_words",
                    "certainty": "certain",
                    "fallback_strategy": "fail",
                },
            ],
            "final_output": "word_count",
            "metadata": {"offline_test": True},
        }
        return plan

    m = re.search(r"clone\s+(\S+)\s+to\s+(\S+)", obj, flags=re.IGNORECASE)
    if m:
        repo_url, target_dir = m.group(1), m.group(2)
        read_path = None
        m2 = re.search(r"read the file\s+(\S+)", obj, flags=re.IGNORECASE)
        if m2:
            read_path = m2.group(1)

        steps: List[Dict[str, Any]] = [
            {
                "step_id": "step_1",
                "intent": "Clone repository",
                "operation": {"type": "call_api", "inputs": {"repo_url": _lit(repo_url), "target_dir": _lit(target_dir)}},
                "output_var": "clone_result",
                "required_capability": "git_clone",
                "certainty": "certain",
                "fallback_strategy": "fail",
            }
        ]
        if read_path:
            steps.append(
                {
                    "step_id": "step_2",
                    "intent": "Read file",
                    "operation": {"type": "read", "inputs": {"file_path": _lit(read_path)}},
                    "output_var": "file_content",
                    "required_capability": "read_file",
                    "certainty": "certain",
                    "fallback_strategy": "fail",
                }
            )
        plan = {
            "goal": "Offline MCP + local flow",
            "steps": steps,
            "final_output": steps[-1]["output_var"],
            "metadata": {"offline_test": True},
        }
        return plan

    return {
        "goal": "Fallback offline plan",
        "steps": [
            {
                "step_id": "step_1",
                "intent": "Summarize objective",
                "operation": {"type": "transform", "inputs": {"text": _lit(obj)}},
                "output_var": "summary",
                "required_capability": "summarize_text",
                "certainty": "certain",
                "fallback_strategy": "fail",
            }
        ],
        "final_output": "summary",
        "metadata": {"offline_test": True},
    }


def _skeleton_for_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    steps = []
    for s in plan["steps"]:
        steps.append(
            {
                "step_id": s["step_id"],
                "intent": s["intent"],
                "required_capability": s.get("required_capability", "reasoning"),
                "description": s["intent"],
            }
        )
    return {"goal": plan.get("goal", ""), "steps": steps}


class FakeOpenAI:
    def __init__(self, *args, **kwargs):
        self.chat = self._Chat()

    class _Chat:
        def __init__(self):
            self.completions = FakeOpenAI._Completions()

    class _Completions:
        def create(self, model: str, messages: List[Dict[str, str]], **kwargs):
            system = next((m["content"] for m in messages if m.get("role") == "system"), "")
            user = next((m["content"] for m in messages if m.get("role") == "user"), "")
            objective = user.replace("Objective:", "").strip() if user else ""
            plan = _plan_for_objective(objective)

            if "Break down the user's objective" in system:
                payload = _skeleton_for_plan(plan)
            elif "PlanDebugger" in system or "convert the provided Plan Skeleton" in system:
                payload = plan
            elif "Map semantic arguments to tool parameters" in system:
                payload = {"tool_args": {}, "reasoning": "offline mapping"}
            else:
                payload = plan

            content = f"<thinking>\nOffline deterministic response.\n</thinking>\n```json\n{__import__('json').dumps(payload)}\n```"
            return _Response(choices=[_Choice(message=_Message(content=content))])


@pytest.fixture
def offline_llm_and_mcp(monkeypatch):
    python_exe = os.environ.get("PYTHON", None) or __import__("sys").executable
    mock_git_script = os.path.join(os.getcwd(), "agent", "tests", "mock_mcp_server.py")

    servers = config.get("mcp.servers", {}) or {}
    rewritten = {}
    for name, cfg in servers.items():
        if isinstance(cfg, dict) and str(cfg.get("command", "")).lower().startswith("npx"):
            rewritten[name] = {"command": python_exe, "args": [mock_git_script], "env": os.environ.copy()}
        else:
            rewritten[name] = cfg

    if rewritten:
        config.update("mcp.servers", rewritten)

    import agent.core.planner as planner_mod
    import agent.core.executor as executor_mod
    import agent.core.repair as repair_mod
    import agent.llm.client as llm_client_mod

    monkeypatch.setattr(planner_mod, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(executor_mod, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(repair_mod, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(llm_client_mod, "OpenAI", FakeOpenAI)

    yield
