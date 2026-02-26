import asyncio
import re
from typing import Any, Dict

OpenAI = None

from agent.core.errors import ToolNotFoundError
from agent.core.schema import ExecutionContext, ExecutionResult, PlanStep
from agent.tools.manager import ToolManager
from agent.utils.config import config
from agent.utils.logger import logger


class ToolExecutor:
    def __init__(self, tool_manager: ToolManager):
        self.tool_manager = tool_manager

    def _resolve_tool_args(self, step: PlanStep, context: ExecutionContext) -> Dict[str, Any]:
        var_pattern = re.compile(r"\$([A-Za-z_][A-Za-z0-9_\\.:-]*)")

        def resolve_string(s: str) -> Any:
            full = var_pattern.fullmatch(s)
            if full:
                var_name = full.group(1)
                resolved_value = context.variables.get(var_name)
                if resolved_value is None:
                    raise ValueError(f"变量 '{var_name}' 在 context 中不存在，step: {step.step_id}")
                return resolved_value

            def repl(m: re.Match) -> str:
                var_name = m.group(1)
                resolved_value = context.variables.get(var_name)
                if resolved_value is None:
                    raise ValueError(f"变量 '{var_name}' 在 context 中不存在，step: {step.step_id}")
                try:
                    return str(resolved_value)
                except Exception:
                    return "<unprintable>"

            return var_pattern.sub(repl, s)

        def resolve_value(val: Any) -> Any:
            if isinstance(val, str):
                if "$" in val:
                    return resolve_string(val)
                return val
            if isinstance(val, dict):
                return {k: resolve_value(v) for k, v in val.items()}
            if isinstance(val, list):
                return [resolve_value(v) for v in val]
            return val

        resolved: Dict[str, Any] = {}
        for key, val in (step.tool_args or {}).items():
            resolved[key] = resolve_value(val)
        return resolved

    def _normalize_result_data(self, result_data: Any) -> Any:
        if isinstance(result_data, list):
            text_parts = []
            has_text_block = False
            for item in result_data:
                if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                    has_text_block = True
                    text_parts.append(item["text"])
                else:
                    return result_data
            if has_text_block:
                return "".join(text_parts)
        return result_data

    def _classify_exception(self, exc: Exception, stage: str) -> str:
        if isinstance(exc, ToolNotFoundError) or stage == "tool_lookup":
            return "tool_not_found"
        text = (str(exc) or "").lower()
        schema_markers = [
            "invalid_type",
            "invalid input",
            "expected number",
            "expected string",
            "missing required",
            "field required",
            "unexpected argument",
            "unexpected keyword",
            "got an unexpected",
            "validation error",
            "pydantic",
            "type error",
            "basemodel.__init__",
            "参数",
            "tool_args",
        ]
        if any(m in text for m in schema_markers) or ("变量" in str(exc) and "context" in str(exc)):
            return "schema_error"
        if re.search(r"\b4\d\d\b", text) or "forbidden" in text or "unauthorized" in text or "resource not found" in text:
            return "api_error"
        network_markers = [
            "timeout",
            "timed out",
            "connection refused",
            "connection reset",
            "connecterror",
            "readtimeout",
            "connecttimeout",
            "name or service not known",
            "temporarily unavailable",
            "network is unreachable",
        ]
        if any(m in text for m in network_markers):
            return "network_error"
        return "unknown_error"

    async def execute_step(self, step: PlanStep, context: ExecutionContext) -> ExecutionResult:
        logger.info(f"Executing Step {step.step_id}: {step.intent}")

        tool_obj = None
        resolved_args = None
        try:
            tool_obj = self.tool_manager.get_tool(step.required_capability)
            if tool_obj is None:
                raise ToolNotFoundError(step.required_capability)

            resolved_args = self._resolve_tool_args(step, context)
            max_retries = int(config.get("executor.max_retries", 3) or 3)
            if max_retries < 1:
                max_retries = 1
            retry_delay_seconds = float(config.get("executor.retry_delay_seconds", 0.5) or 0.5)
            if retry_delay_seconds < 0:
                retry_delay_seconds = 0.0

            attempts_used = 0
            for attempt in range(1, max_retries + 1):
                try:
                    result_data = await self.tool_manager.execute_tool(step.required_capability, **resolved_args)
                    attempts_used = attempt
                    break
                except Exception as e:
                    error_type = self._classify_exception(e, stage="tool_run")
                    if error_type == "network_error" and attempt < max_retries and retry_delay_seconds > 0:
                        await asyncio.sleep(retry_delay_seconds)
                        continue
                    raise

            normalized = self._normalize_result_data(result_data)
            context.set(step.output_var, normalized)
            return ExecutionResult(
                step_id=step.step_id,
                status="success",
                result=normalized,
                meta={"tool": step.required_capability, "args": resolved_args, "attempts": attempts_used},
            )
        except Exception as e:
            error_type = self._classify_exception(e, stage="tool_run")
            stage = "tool_run"
            if isinstance(e, ToolNotFoundError):
                stage = "tool_lookup"
            elif "变量 '" in str(e) and "context 中不存在" in str(e):
                stage = "args_resolve"
            args_schema = None
            try:
                args_schema = getattr(tool_obj, "args_schema", None) if tool_obj is not None else None
            except Exception:
                args_schema = None
            return ExecutionResult(
                step_id=step.step_id,
                status="failed",
                error=str(e),
                error_type=error_type,
                meta={
                    "tool": step.required_capability,
                    "tool_args": step.tool_args,
                    "resolved_args": resolved_args,
                    "args_schema": args_schema.schema() if hasattr(args_schema, "schema") else args_schema.model_json_schema() if hasattr(args_schema, "model_json_schema") else args_schema,
                    "stage": stage,
                },
            )
