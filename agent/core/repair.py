from typing import Any, Dict, List, Optional

from agent.core.schema import PlanStep
from agent.llm.client import LLMClient
from agent.utils.logger import logger


class Repairer:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    @staticmethod
    def _safe_str(value: Any, limit: int = 800) -> str:
        try:
            text = str(value)
        except Exception:
            text = "<unprintable>"
        if len(text) > limit:
            return text[:limit] + "..."
        return text

    def _summarize_tools(self, tool_defs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        summarized: List[Dict[str, Any]] = []
        for t in tool_defs or []:
            if not isinstance(t, dict):
                continue
            name = t.get("name")
            args_schema = t.get("args_schema") or {}
            props = args_schema.get("properties") if isinstance(args_schema, dict) else None
            params: List[str] = []
            if isinstance(props, dict):
                params = sorted([str(k) for k in props.keys()])
            summarized.append({"name": name, "params": params})
        return summarized

    def repair_step(
        self,
        objective: str,
        failed_step: PlanStep,
        error: str,
        context_variables: Dict[str, Any],
        tool_defs: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        tools_summary = self._summarize_tools(tool_defs)
        available_tool_names = [t.get("name") for t in tools_summary if t.get("name")]

        context_preview: Dict[str, Any] = {}
        for k, v in (context_variables or {}).items():
            context_preview[str(k)] = self._safe_str(v)
            if len(context_preview) >= 60:
                break

        schema_example = {
            "required_capability": "tool_name",
            "tool_args": {},
        }

        system_prompt = (
            "你是一个工具调用修复器。\n"
            "你会得到一个失败的 plan step、报错信息、当前 context_variables、以及可用工具列表。\n"
            "你的目标是输出一个修复后的工具调用（required_capability 与 tool_args），让该 step 更可能执行成功。\n"
            "约束：\n"
            "- 只输出 JSON，必须严格匹配给定 schema\n"
            "- required_capability 必须从可用工具列表中选择\n"
            "- 如果 tool_args 需要引用 context 变量，使用 \"$变量名\"，且变量名必须存在于 context_variables\n"
            "- 不要输出任何额外字段\n"
        )

        prompt = (
            f"objective:\n{objective}\n\n"
            f"failed_step:\n{failed_step.model_dump()}\n\n"
            f"error:\n{error}\n\n"
            f"available_tools:\n{tools_summary}\n\n"
            f"context_keys:\n{sorted(list((context_variables or {}).keys()))}\n\n"
            f"context_preview:\n{context_preview}\n"
        )

        try:
            patched = self.llm.generate_structured(prompt, schema_example, system_prompt=system_prompt, max_retries=3)
        except Exception as e:
            logger.error(f"Repair generation failed: {e}")
            return None

        if not isinstance(patched, dict):
            return None

        required_capability = patched.get("required_capability")
        tool_args = patched.get("tool_args")
        if not isinstance(required_capability, str) or not required_capability.strip():
            return None
        if required_capability not in available_tool_names:
            return None
        if not isinstance(tool_args, dict):
            return None

        return {"required_capability": required_capability, "tool_args": tool_args}
