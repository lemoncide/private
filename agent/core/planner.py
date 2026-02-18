import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI

from agent.core.schema import Plan, PlanStep
from agent.core.validator import PlanValidator
from agent.llm.client import LLMClient
from agent.llm.url import normalize_base_url
from agent.utils.config import config

logger = logging.getLogger("Agent")


class Planner:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        api_key = config.get("llm.api_key", "lm-studio")
        base_url = normalize_base_url(config.get("llm.api_base", "http://127.0.0.1:1234/v1"))
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = config.get("llm.model", "gpt-3.5-turbo")
        self.llm = llm_client or LLMClient()
        self.validator = PlanValidator()

    def plan(self, objective: str, tools: List[Dict[str, Any]]) -> Plan:
        logger.info(f"Planning for objective: {objective}")
        plan = self._generate_tool_call_plan(objective, tools)
        if plan.final_output != plan.steps[-1].output_var:
            raise ValueError("final_output must equal the last step output_var")
        metadata = dict(plan.metadata or {})
        metadata.setdefault("planner_mode", "groq_json_schema")
        return plan.model_copy(update={"metadata": metadata})

    def create_plan(self, objective: str, tools: List[Dict[str, Any]], max_retries: int = 3) -> Plan:
        available = [t.get("name") for t in (tools or []) if t.get("name")]
        last_error: Optional[str] = None
        for _ in range(max_retries):
            try:
                plan = self.plan(objective, tools)
            except Exception as e:
                last_error = str(e)
                continue

            validation = self.validator.validate(plan, available_tools=available)
            if validation.valid:
                return plan

            last_error = "; ".join(validation.errors)

        raise ValueError(last_error or "Max retries exceeded")

    def _generate_tool_call_plan(self, objective: str, tools: List[Dict[str, Any]]) -> Plan:
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
你是一个任务规划器。根据用户目标和可用工具列表，生成一个可执行的多步骤计划。

可用工具（格式：工具名: 参数列表）：
{tool_list}

规则：
- Plan 里只包含需要调用外部工具的步骤（读文件、搜索、写文件等）
- 纯推理、总结、解释类的步骤不要放进 Plan，这些由系统自动处理
- required_capability 必须是上方工具列表中的完整工具名
- tool_args 中引用上一步结果用 "$上一步的output_var"，引用用户输入用 "$input"
- final_output 必须等于最后一步的 output_var
- 文件路径约束：当调用 read_file / write_file 时，path 必须使用 sandbox 内的相对路径（例如 "github_summary.txt" 或 "outputs/readme.md"），禁止使用绝对路径（例如 "E:\\tmp\\a.txt"）

严格按照以下 JSON 格式输出，不要输出任何额外文字：
{{
  "goal": "用一句话描述整体目标",
  "steps": [
    {{
      "step_id": "step_1",
      "intent": "用一句话描述这一步要做什么",
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
            response_format={"type": "json_object"},
            timeout=600.0,
        )
        content = response.choices[0].message.content
        plan = Plan.model_validate_json(content)
        if not plan.steps:
            raise ValueError("Plan.steps must not be empty")
        return plan
