import logging
from typing import Any, Dict, List, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from agent.core.schema import Plan, PlanStep
from agent.llm.cot_parser import CoTParseError, extract_thinking, parse_structured
from agent.llm.client import LLMClient
from agent.llm.url import normalize_base_url
from agent.utils.config import config

logger = logging.getLogger("Agent")

T = TypeVar("T", bound=BaseModel)


class Planner:
    def __init__(self):
        api_key = config.get("llm.api_key", "lm-studio")
        base_url = normalize_base_url(config.get("llm.api_base", "http://127.0.0.1:1234/v1"))
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = config.get("llm.model", "gpt-3.5-turbo")
        self.llm = LLMClient()

    def _parse_json(self, text: str, model_class: Type[T]) -> T:
        thinking = extract_thinking(text or "")
        if thinking:
            logger.debug(f"[Planner Thinking] len={len(thinking)} preview={thinking[:400]}")

        try:
            parsed = parse_structured(text or "", model_class)
        except CoTParseError as e:
            logger.error(f"Failed to parse structured output: {e} len={e.length} preview={e.preview}")
            raise

        if thinking and hasattr(parsed, "metadata"):
            try:
                metadata = getattr(parsed, "metadata") or {}
                if isinstance(metadata, dict):
                    metadata = dict(metadata)
                    metadata["thinking"] = thinking
                    parsed = parsed.model_copy(update={"metadata": metadata})
            except Exception:
                pass

        return parsed

    def plan(self, objective: str, tools: List[Dict[str, Any]]) -> Plan:
        logger.info(f"Planning for objective: {objective}")
        try:
            tool_plan = self._generate_tool_call_plan(objective, tools)
            if tool_plan:
                return tool_plan
            plan = self._generate_plan(objective, tools)
            return self._fix_final_output(plan)
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return self._create_fallback_plan(objective, tools)

    @staticmethod
    def _plan_from_tool_call(objective: str, tool_call: Dict[str, Any]) -> Plan:
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("arguments")

        if not tool_name or not isinstance(tool_name, str):
            raise ValueError("Tool call missing 'name'")
        if tool_args is None:
            tool_args = {}
        if not isinstance(tool_args, dict):
            raise ValueError("Tool call 'arguments' must be an object")

        step = PlanStep(
            step_id="step_1",
            intent=objective,
            required_capability=tool_name,
            tool_args=tool_args,
            output_var="step_1_output",
            depends_on=[],
            fallback_strategy="fail",
        )
        return Plan(goal=objective, steps=[step], final_output="step_1_output", metadata={"planner_mode": "tool_call"})

    def _generate_tool_call_plan(self, objective: str, tools: List[Dict[str, Any]]) -> Plan | None:
        if not tools:
            return None

        system_prompt = """
你是一个工具路由器。你必须从提供的工具清单中选择一个最合适的工具，并给出该工具调用所需的参数。
要求：
- 只能选择一个工具调用
- 参数必须与该工具的参数 schema 匹配
- 不要输出自然语言解释
""".strip()

        result = self.llm.generate_with_tools(
            objective,
            tool_defs=tools,
            system_prompt=system_prompt,
            tool_choice="auto",
            temperature=0.0,
        )

        if result.get("error"):
            return None

        tool_calls = result.get("tool_calls") or []
        if not tool_calls:
            return None

        return self._plan_from_tool_call(objective, tool_calls[0])

    def _generate_plan(self, objective: str, tools: List[Dict[str, Any]]) -> Plan:
        tool_lines: List[str] = []
        for t in tools:
            name = t.get("name", "")
            args_schema = t.get("args_schema") or {}
            props = args_schema.get("properties") if isinstance(args_schema, dict) else None
            if isinstance(props, dict) and props:
                params = ", ".join(sorted(props.keys()))
                tool_lines.append(f"- {name}: {params}")
            else:
                tool_lines.append(f"- {name}")
        tool_list = "\n".join(tool_lines)

        system_prompt = f"""
你是一个任务规划器。根据用户目标和可用工具列表，生成一个执行计划。

可用工具（每行一个，格式：工具名: 参数列表）：
{tool_list}

输出格式要求：
- required_capability 必须是上方工具列表中的完整工具名，不能自造
- tool_args 中引用上一步结果用 "$上一步的output_var"，引用用户输入用 "$input"
- 其他值直接写字面量
- final_output 必须等于最后一步的 output_var

输出（先写 <thinking> 再写 JSON；JSON 不要使用 markdown code block）：
{{
  "goal": "...",
  "steps": [
    {{
      "step_id": "step_1",
      "intent": "...",
      "required_capability": "完整工具名",
      "tool_args": {{"参数名": "值或$变量"}},
      "output_var": "step_1_output",
      "depends_on": [],
      "fallback_strategy": "fail"
    }}
  ],
  "final_output": "最后一步的output_var",
  "metadata": {{}}
}}
""".strip()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Objective: {objective}"},
            ],
            temperature=0.0,
            timeout=600.0,
        )
        return self._parse_json(response.choices[0].message.content, Plan)

    def _fix_final_output(self, plan: Plan) -> Plan:
        if not plan or not plan.steps:
            return plan

        last_output = plan.steps[-1].output_var
        if plan.final_output != last_output:
            try:
                plan = plan.model_copy(update={"final_output": last_output})
            except Exception:
                plan.final_output = last_output
        return plan

    def _create_fallback_plan(self, objective: str, tools: List[Dict[str, Any]]) -> Plan:
        available_names = [t.get("name") for t in tools if t.get("name")]
        preferred = "llm_reasoning" if "llm_reasoning" in set(available_names) else (available_names[0] if available_names else "llm_reasoning")
        step = PlanStep(
            step_id="step_1",
            intent="Fallback: summarize objective",
            required_capability=preferred,
            tool_args={"task_description": "$input"} if preferred == "llm_reasoning" else {},
            output_var="final_result",
            depends_on=[],
            fallback_strategy="fail",
        )
        return Plan(goal=objective, steps=[step], final_output="final_result", metadata={"fallback": True})

    def refine_plan(self, objective: str, original_plan: Plan, error_msg: str, tools: List[Dict[str, Any]]) -> Plan:
        tool_names = [t.get("name") for t in tools if t.get("name")]
        tool_lines = "\n".join(f"- {name}" for name in tool_names)

        system_prompt = f"""
你是一个 PlanDebugger。上一次生成的计划未通过校验，请修复它。

可用工具（required_capability 必须从这里选择）：
{tool_lines}

修复规则：
- 只输出一个完整的新计划 JSON（先 <thinking> 再 JSON）
- steps 中每个 step 必须包含：step_id, intent, required_capability, tool_args, output_var, depends_on, fallback_strategy
- tool_args 中的 "$变量" 只能引用 "$input" 或者前序 step 的 output_var
- final_output 必须等于最后一步的 output_var
""".strip()

        user_content = f"""
Objective: {objective}

Failed Plan (JSON):
{original_plan.model_dump_json(indent=2)}

Validation Error:
{error_msg}
""".strip()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
            temperature=0.0,
            timeout=600.0,
        )
        plan = self._parse_json(response.choices[0].message.content, Plan)
        return self._fix_final_output(plan)
