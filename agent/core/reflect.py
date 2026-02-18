from typing import Any, Dict, List, Optional

from agent.core.state import AgentState
from agent.llm.client import LLMClient


class Reflector:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    @staticmethod
    def _extract_final_result(state: AgentState) -> Any:
        if not state.plan:
            return None

        final_var = state.plan.final_output
        result = state.context_variables.get(final_var)
        if result is not None:
            return result

        result = state.context_variables.get(f"step_output.{final_var}")
        if result is not None:
            return result

        if state.past_steps:
            return state.past_steps[-1].result

        return None

    @staticmethod
    def _steps_summary(state: AgentState) -> List[Dict[str, Any]]:
        summary: List[Dict[str, Any]] = []
        for r in state.past_steps or []:
            step_dict = r.model_dump() if hasattr(r, "model_dump") else dict(r)
            meta = step_dict.get("meta") or {}
            result = step_dict.get("result")
            result_preview = None
            if result is not None:
                try:
                    result_preview = str(result)
                except Exception:
                    result_preview = None
            if isinstance(result_preview, str) and len(result_preview) > 1200:
                result_preview = result_preview[:1200] + "..."
            summary.append(
                {
                    "step_id": step_dict.get("step_id"),
                    "status": step_dict.get("status"),
                    "tool": meta.get("tool"),
                    "error_type": step_dict.get("error_type"),
                    "error": step_dict.get("error"),
                    "result_preview": result_preview,
                }
            )
        return summary

    def reflect(self, state: AgentState) -> str:
        objective = state.input or ""
        steps = self._steps_summary(state)
        final_result = self._extract_final_result(state)
        status = state.status
        error = state.error

        system_prompt = (
            "你是一个结果解读器。你只负责把已有执行结果整理成面向用户的回答。\n"
            "约束：\n"
            "- 不要提出新的计划，不要要求调用工具，不要编造不存在的事实\n"
            "- 只基于输入的 objective、steps、final_result、error 进行总结\n"
            "- 用中文输出\n"
        )

        user_prompt = (
            f"objective:\n{objective}\n\n"
            f"status: {status}\n"
            f"error: {error}\n\n"
            f"steps:\n{steps}\n\n"
            f"final_result:\n{final_result}\n"
        )

        text = self.llm.generate(user_prompt, system_prompt=system_prompt) or ""
        if text.startswith("Error generating response:"):
            if status == "completed":
                return f"任务已完成。\n最终结果：{final_result}"
            if status == "failed":
                if error and "outside sandbox" in error.lower():
                    return f"任务失败。\n错误：{error}\n建议：把写入路径改为 sandbox 内相对路径，例如 \"readme_content.md\"。"
                return f"任务失败。\n错误：{error}"
            return f"状态：{status}\n{error or ''}".strip()

        return text.strip() or f"任务状态：{status}"
