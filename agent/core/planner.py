import logging
from typing import Any, Dict, List, Optional
import os
import json

from agent.core.errors import PlanToolNotFoundError
from agent.core.schema import Plan, PlanStep
from agent.core.validator import PlanValidator
from agent.llm.client import LLMClient
from agent.llm.url import normalize_base_url
from agent.utils.config import config

logger = logging.getLogger("Agent")


class Planner:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
        self.validator = PlanValidator()

    def plan(self, objective: str, tools: List[Dict[str, Any]], validation_feedback: Optional[str] = None) -> Plan:
        logger.info(f"Planning for objective: {objective}")
        plan = self._generate_tool_call_plan(objective, tools, validation_feedback=validation_feedback)
        if plan.final_output != plan.steps[-1].output_var:
            raise ValueError("final_output must equal the last step output_var")
        metadata = dict(plan.metadata or {})
        metadata.setdefault("planner_mode", "groq_json_schema")
        return plan.model_copy(update={"metadata": metadata})

    def create_plan(self, objective: str, tools: List[Dict[str, Any]], max_retries: int = 3) -> Plan:
        available = [t.get("name") for t in (tools or []) if t.get("name")]
        last_error: Optional[str] = None
        last_missing_tools: List[str] = []
        last_validation_errors: List[str] = []
        for _ in range(max_retries):
            validation_feedback = last_error
            try:
                plan = self.plan(objective, tools, validation_feedback=validation_feedback)
            except Exception as e:
                last_error = str(e)
                continue

            validation = self.validator.validate(plan, available_tools=available)
            if validation.valid:
                return plan

            last_error = "; ".join(validation.errors)
            last_validation_errors = list(validation.errors or [])
            missing = []
            for step in plan.steps:
                if step.required_capability and step.required_capability not in set(available):
                    missing.append(step.required_capability)
            last_missing_tools = sorted(list(set(missing)))

        if last_missing_tools:
            raise PlanToolNotFoundError(
                last_error or "required_capability not in available tools",
                missing_tools=last_missing_tools,
                errors=last_validation_errors,
            )
        raise ValueError(last_error or "Max retries exceeded")

    def _generate_tool_call_plan(
        self, objective: str, tools: List[Dict[str, Any]], validation_feedback: Optional[str] = None
    ) -> Plan:
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

        prefs_text = ""
        prefs_path = config.get("user_prefs.path", "configs/user_prefs.json")
        try:
            path = prefs_path if os.path.isabs(prefs_path) else os.path.join(os.getcwd(), prefs_path)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    prefs = json.load(f)
                raw = json.dumps(prefs, ensure_ascii=False)
                if len(raw) > 2000:
                    raw = raw[:2000] + "...(已截断)"
                prefs_text = f"\n用户偏好与环境：\n{raw}\n"
        except Exception:
            prefs_text = ""

        system_prompt = f"""
你是一个任务规划器。根据用户目标和可用工具列表，生成一个可执行的多步骤计划。

可用工具（格式：工具名: 参数列表）：
{tool_list}

规则：
【核心规则】
1. 每个步骤必须调用一个具体工具，required_capability 必须是上方工具列表中的完整工具名
2. 步骤之间的数据传递：用 "$上一步的output_var" 引用上一步结果，用 "$input" 引用用户原始输入
3. 数据必须显式传递：如果某步骤需要用到前一步的结果，tool_args 里必须用 "$变量名" 明确引用，禁止跳过
4. final_output 必须等于最后一步的 output_var
5. 文件路径只能使用 sandbox 内的相对路径（如 "report.txt"），禁止绝对路径
6. write_file 的 mode 参数只能是 "write" 或 "append"，不能用 "w" 或 "a"

【哪些步骤放进 Plan】
- 需要调用外部工具的步骤必须放进来：读文件、写文件、搜索、获取数据、调用 API 等
- 需要基于真实数据进行分析、总结、生成报告的步骤，必须使用 llm_reasoning 工具，并把数据通过 "$变量名" 传入 task_description
- 不要省略中间步骤，数据获取和数据分析是两个独立步骤

【llm_reasoning 使用规范】
- 当需要分析数据、生成报告、总结内容时，使用 llm_reasoning 工具
- task_description 必须包含实际数据引用，例如：
  "根据以下文件列表分析项目结构：$file_list_output，请生成项目概览报告"
- 禁止在 task_description 里只写任务描述而不引用任何数据变量

【示例计划】
目标：获取 GitHub 仓库根目录文件列表，分析项目结构，生成报告写入文件

正确示例：
{{
  "goal": "分析 GitHub 仓库结构并生成报告",
  "steps": [
    {{
      "step_id": "step_1",
      "intent": "获取仓库根目录文件列表",
      "required_capability": "mcp:official_github:get_file_contents",
      "tool_args": {{"owner": "lemoncide", "repo": "agent", "path": ""}},
      "output_var": "file_list",
      "depends_on": [],
      "fallback_strategy": "fail"
    }},
    {{
      "step_id": "step_2",
      "intent": "基于文件列表分析项目结构并生成报告",
      "required_capability": "llm_reasoning",
      "tool_args": {{
        "task_description": "根据以下仓库文件列表：$file_list，分析该项目的结构、当前实现、未来目标及优缺点，生成一份完整的项目概览报告"
      }},
      "output_var": "report",
      "depends_on": ["step_1"],
      "fallback_strategy": "fail"
    }},
    {{
      "step_id": "step_3",
      "intent": "将报告写入文件",
      "required_capability": "write_file",
      "tool_args": {{"path": "project_overview.txt", "content": "$report", "mode": "write"}},
      "output_var": "write_result",
      "depends_on": ["step_2"],
      "fallback_strategy": "fail"
    }}
  ],
  "final_output": "write_result",
  "metadata": {{}}
}}
{prefs_text}

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

        user_content = f"Objective: {objective}"
        if validation_feedback:
            feedback = str(validation_feedback)
            if len(feedback) > 2000:
                feedback = feedback[:2000] + "...(内容已截断)"
            user_content += (
                "\n\n上一次生成的计划未通过校验，错误如下：\n"
                f"{feedback}\n\n"
                "请修正计划，required_capability 必须严格从可用工具列表中选择，不能编造不存在的工具名。"
            )

        response = self.llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
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
