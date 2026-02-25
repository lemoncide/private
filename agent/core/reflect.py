from typing import Any, Dict, List, Optional

from agent.core.state import AgentState
from agent.llm.client import LLMClient


class Reflector:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    @staticmethod
    def _safe_preview(value: Any, limit: int = 800) -> Optional[str]:
        if value is None:
            return None
        try:
            text = str(value)
        except Exception:
            return None
        if len(text) > limit:
            return text[:limit] + "..."
        return text

    @staticmethod
    def _prompt_preview(value: Any, limit: int = 12000) -> Optional[str]:
        if value is None:
            return None
        try:
            text = str(value)
        except Exception:
            return None
        if len(text) > limit:
            return text[:limit] + "...(内容已截断)"
        return text

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
                result_preview = result_preview[:1200] + "...(内容已截断)"
            summary.append(
                {
                    "step_id": step_dict.get("step_id"),
                    "status": step_dict.get("status"),
                    "tool": meta.get("tool"),
                    "tool_args": meta.get("args") if meta.get("args") is not None else meta.get("tool_args"),
                    "error_type": step_dict.get("error_type"),
                    "error": step_dict.get("error"),
                    "result_preview": result_preview,
                }
            )
        return summary

    def _format_steps_fallback(self, steps: List[Dict[str, Any]]) -> str:
        if not steps:
            return "无步骤执行记录。"
        lines: List[str] = []
        for i, s in enumerate(steps, start=1):
            step_id = s.get("step_id")
            status = s.get("status")
            tool = s.get("tool")
            tool_args = self._safe_preview(s.get("tool_args"))
            error_type = s.get("error_type")
            error = s.get("error")
            result_preview = s.get("result_preview")
            parts: List[str] = []
            parts.append(f"{i}. step_id={step_id} status={status}")
            if tool:
                parts.append(f"tool={tool}")
            if tool_args:
                parts.append(f"args={tool_args}")
            if error_type:
                parts.append(f"error_type={error_type}")
            if error:
                parts.append(f"error={error}")
            if result_preview:
                parts.append(f"result_preview={result_preview}")
            lines.append(" | ".join(parts))
        return "\n".join(lines)

    @staticmethod
    def _known_suggestions(error_text: str) -> List[str]:
        text = (error_text or "").lower()
        suggestions: List[str] = []
        if "outside sandbox" in text:
            suggestions.append("把写入/读取路径改为 sandbox 内相对路径，例如 \"readme_content.md\"。")
        if "tool " in text and " not found" in text:
            suggestions.append("检查 plan 里的 required_capability 是否与工具列表名称一致。")
        if "mcp server not found" in text:
            suggestions.append("检查 mcp.servers 配置与 server_name 是否匹配，并确认服务器已启动。")
        if "变量 '" in (error_text or "") and "在 context 中不存在" in (error_text or ""):
            suggestions.append("检查 tool_args 中的 \"$变量名\" 是否在 context_variables 里已存在，或先添加生成该变量的步骤。")
        return suggestions

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

        final_result_for_prompt = self._prompt_preview(final_result, limit=12000)
        user_prompt = (
            f"objective:\n{objective}\n\n"
            f"status: {status}\n"
            f"error: {error}\n\n"
            f"steps:\n{steps}\n\n"
            f"final_result:\n{final_result_for_prompt}\n"
        )

        text = self.llm.generate(user_prompt, system_prompt=system_prompt) or ""
        if text.startswith("Error generating response:"):
            base_error = error
            if not base_error and state.past_steps:
                try:
                    base_error = state.past_steps[-1].error
                except Exception:
                    base_error = error
            report_lines: List[str] = []
            if status == "completed":
                report_lines.append("任务已完成。")
                report_lines.append(f"最终结果：{final_result}")
                if steps:
                    report_lines.append("")
                    report_lines.append("执行步骤：")
                    report_lines.append(self._format_steps_fallback(steps))
                return "\n".join(report_lines).strip()
            if status == "failed":
                report_lines.append("任务失败。")
                if base_error:
                    report_lines.append(f"错误：{base_error}")
                if steps:
                    report_lines.append("")
                    report_lines.append("执行步骤：")
                    report_lines.append(self._format_steps_fallback(steps))
                suggestions = self._known_suggestions(base_error or "")
                if suggestions:
                    report_lines.append("")
                    report_lines.append("建议：")
                    for s in suggestions:
                        report_lines.append(f"- {s}")
                return "\n".join(report_lines).strip()
            report_lines.append(f"状态：{status}")
            if base_error:
                report_lines.append(str(base_error))
            if steps:
                report_lines.append("")
                report_lines.append("执行步骤：")
                report_lines.append(self._format_steps_fallback(steps))
            return "\n".join(report_lines).strip()

        return text.strip() or f"任务状态：{status}"
