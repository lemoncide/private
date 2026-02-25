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

    def _generate_patch(
        self,
        prompt: str,
        tool_defs: List[Dict[str, Any]],
        context_variables: Dict[str, Any],
        system_prompt: str,
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

        full_prompt = (
            f"{prompt}\n\n"
            f"available_tools:\n{tools_summary}\n\n"
            f"context_keys:\n{sorted(list((context_variables or {}).keys()))}\n\n"
            f"context_preview:\n{context_preview}\n"
        )

        try:
            patched = self.llm.generate_structured(full_prompt, schema_example, system_prompt=system_prompt, max_retries=3)
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

    def repair_plan(
        self,
        objective: str,
        failed_step: PlanStep,
        error: str,
        context_variables: Dict[str, Any],
        tool_defs: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        system_prompt = (
            "你是一个计划修复器。\n"
            "你会得到一个失败的 plan step、报错信息、当前 context_variables、以及可用工具列表。\n"
            "你的目标是为该 step 重新选择正确的工具，并给出匹配的 tool_args。\n"
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
            "请为该 step 重新选择合适工具，确保 required_capability 存在于工具列表中。\n"
        )
        return self._generate_patch(prompt, tool_defs=tool_defs, context_variables=context_variables, system_prompt=system_prompt)

    def repair_params(
        self,
        objective: str,
        failed_step: PlanStep,
        error: str,
        context_variables: Dict[str, Any],
        tool_defs: List[Dict[str, Any]],
        args_schema: Any = None,
        actual_args: Any = None,
    ) -> Optional[Dict[str, Any]]:
        system_prompt = (
            "你是一个参数修复器。\n"
            "你会得到一个失败的 plan step、工具参数定义、实际传入参数、报错信息、当前 context_variables、以及可用工具列表。\n"
            "你的目标是修正 tool_args 的参数名与类型，使其符合工具参数定义。\n"
            "约束：\n"
            "- 只输出 JSON，必须严格匹配给定 schema\n"
            "- required_capability 必须从可用工具列表中选择\n"
            "- 如果 tool_args 需要引用 context 变量，使用 \"$变量名\"，且变量名必须存在于 context_variables\n"
            "- 不要输出任何额外字段\n"
        )

        prompt = (
            f"objective:\n{objective}\n\n"
            f"failed_step:\n{failed_step.model_dump()}\n\n"
            f"args_schema:\n{self._safe_str(args_schema, limit=2000)}\n\n"
            f"actual_args:\n{self._safe_str(actual_args, limit=2000)}\n\n"
            f"error:\n{error}\n"
        )
        return self._generate_patch(prompt, tool_defs=tool_defs, context_variables=context_variables, system_prompt=system_prompt)

    def repair_query(
        self,
        objective: str,
        failed_step: PlanStep,
        error: str,
        context_variables: Dict[str, Any],
        tool_defs: List[Dict[str, Any]],
        actual_args: Any = None,
    ) -> Optional[Dict[str, Any]]:
        system_prompt = (
            "你是一个 API 查询修复器。\n"
            "你会得到一个失败的 plan step、报错信息、当前参数、当前 context_variables、以及可用工具列表。\n"
            "你的目标是判断报错是否可通过修正查询参数解决，并给出修正后的 tool_args。\n"
            "约束：\n"
            "- 只输出 JSON，必须严格匹配给定 schema\n"
            "- required_capability 必须从可用工具列表中选择\n"
            "- 如果 tool_args 需要引用 context 变量，使用 \"$变量名\"，且变量名必须存在于 context_variables\n"
            "- 不要输出任何额外字段\n"
        )

        prompt = (
            f"objective:\n{objective}\n\n"
            f"failed_step:\n{failed_step.model_dump()}\n\n"
            f"actual_args:\n{self._safe_str(actual_args, limit=2000)}\n\n"
            f"error:\n{error}\n"
        )
        return self._generate_patch(prompt, tool_defs=tool_defs, context_variables=context_variables, system_prompt=system_prompt)

    def repair_step(
        self,
        objective: str,
        failed_step: PlanStep,
        error: str,
        context_variables: Dict[str, Any],
        tool_defs: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        return self.repair_params(
            objective=objective,
            failed_step=failed_step,
            error=error,
            context_variables=context_variables,
            tool_defs=tool_defs,
        )
