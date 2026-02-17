from typing import Any, Dict

from openai import OpenAI

from agent.core.schema import ExecutionContext, ExecutionResult, PlanStep
from agent.tools.manager import ToolManager
from agent.utils.logger import logger


class ToolExecutor:
    def __init__(self, tool_manager: ToolManager):
        self.tool_manager = tool_manager

    def _resolve_tool_args(self, step: PlanStep, context: ExecutionContext) -> Dict[str, Any]:
        resolved: Dict[str, Any] = {}
        for key, val in (step.tool_args or {}).items():
            if isinstance(val, str) and val.startswith("$"):
                var_name = val[1:]
                resolved_value = context.variables.get(var_name)
                if resolved_value is None:
                    raise ValueError(f"变量 '{var_name}' 在 context 中不存在，step: {step.step_id}")
                resolved[key] = resolved_value
            else:
                resolved[key] = val
        return resolved

    async def execute_step(self, step: PlanStep, context: ExecutionContext) -> ExecutionResult:
        logger.info(f"Executing Step {step.step_id}: {step.intent}")

        try:
            resolved_args = self._resolve_tool_args(step, context)
            result_data = self.tool_manager.execute_tool(step.required_capability, **resolved_args)
            context.set(step.output_var, result_data)
            return ExecutionResult(
                step_id=step.step_id,
                status="success",
                result=result_data,
                meta={"tool": step.required_capability, "args": resolved_args},
            )
        except Exception as e:
            return ExecutionResult(
                step_id=step.step_id,
                status="failed",
                error=str(e),
                error_type="tool_error",
                meta={"tool": step.required_capability, "tool_args": step.tool_args},
            )
