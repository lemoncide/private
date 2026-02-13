from typing import List, Optional, Dict, Any
from .schema import Plan, PlanStep, ExecutionResult, ExecutionContext, VariableRef, VariableBinding, OperationSpec
from agent.tools.manager import ToolManager
from agent.llm.client import LLMClient
from pydantic import BaseModel, Field
import difflib
import json
import asyncio
from agent.utils.logger import logger
from agent.utils.config import config
from openai import OpenAI
from agent.llm.cot_parser import extract_thinking, parse_structured

class ExecutionRepairer:
    """运行时修复器 - 修复工具调用错误"""
    def __init__(self, tool_manager: ToolManager):
        self.tool_manager = tool_manager
        
        # Initialize LLM for Adaptation
        api_key = config.get("llm.api_key", "lm-studio")
        base_url = config.get("llm.api_base", "http://127.0.0.1:1234/v1")
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model = config.get("llm.model", "gpt-3.5-turbo")
        
        self.capability_map = {
            "search": ["google_search", "bing_search", "example_search"],
            "calculator": ["calculator"],
            "reasoning": ["llm_reasoning"]
        }

    async def repair(self, step: PlanStep, error: Exception, context: ExecutionContext) -> ExecutionResult:
        """
        Mode 2: Ask LLM to fix the arguments based on the error and context.
        Includes retry logic with exponential backoff.
        """
        max_retries = config.get("execution.max_retries", 3)
        base_delay = config.get("execution.retry_delay", 1.0)
        
        last_error = error
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.info(f"Retrying execution (attempt {attempt+1}/{max_retries}) after {delay}s delay...")
                    await asyncio.sleep(delay)

                # 1. Ask LLM for fix (or just retry if it's a transient error)
                if self._is_transient_error(last_error) and attempt > 0:
                    # For transient errors like timeout, just retry without LLM adaptation first
                    tool_name = self._select_tool(step.required_capability, step.operation.type)
                    if tool_name:
                         # Use existing context to resolve arguments again (in case they changed)
                         try:
                             # This is simplified; in a real retry we'd need the original resolved_args
                             # or re-resolve them. Let's assume we need to re-resolve.
                             # But ExecutionRepairer doesn't have _resolve_arguments.
                             # For simplicity, we use the LLM to provide fresh args on every attempt
                             # or use the first adaptation.
                             pass 
                         except:
                             pass

                # 2. LLM Adaptation
                context_summary = list(context.variables.keys())
                current_bindings = {k: str(v.value_ref) for k, v in step.operation.inputs.items()}

                system_prompt = f"""
                You are a runtime repair agent. A step failed to execute.
                Your job is to provide CORRECTED arguments for the tool call.
                
                Step Intent: {step.intent}
                Tool Capability: {step.required_capability}
                Error: {str(last_error)}
                Available Variables: {context_summary}
                Current Semantic Bindings: {current_bindings}
                
                CRITICAL: The error often indicates incorrect arguments.
                Output the actual TOOL arguments.

                RESPONSE FORMAT:
                You MUST provide a <thinking> block BEFORE the JSON.
                ```json
                {{
                  "new_inputs": {{ ... }},
                  "reasoning": "..."
                }}
                ```
                """
                
                class AdaptedBindings(BaseModel):
                    new_inputs: Dict[str, Any] = Field(description="Corrected input values (literals)")
                    reasoning: str = Field(description="Why these values were chosen")

                last_error = None
                resp = None
                for _attempt in range(2):
                    try:
                        resp = self.client.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "system", "content": system_prompt}],
                        )
                        break
                    except Exception as e:
                        last_error = e
                if resp is None:
                    raise last_error
                content = resp.choices[0].message.content
                thinking = extract_thinking(content or "")
                if thinking:
                    logger.debug(f"[ExecutionRepairer Thinking] len={len(thinking)} preview={thinking[:300]}")
                adaptation = parse_structured(content or "", AdaptedBindings)
                logger.info(f"Adaptation (attempt {attempt+1}): {adaptation.reasoning}")
                
                # 3. Execution
                tool_name = self._select_tool(step.required_capability, step.operation.type)
                if tool_name:
                    result_data = self.tool_manager.execute_tool(tool_name, **adaptation.new_inputs)
                    context.set(f"step_output.{step.step_id}", result_data)
                    context.set(step.output_var, result_data)
                    return ExecutionResult(step_id=step.step_id, status="success", result=result_data)
                else:
                    return ExecutionResult(step_id=step.step_id, status="failed", error_type="tool_not_found", error="No tool found")

            except Exception as e:
                logger.warning(f"Repair attempt {attempt+1} failed: {e}")
                last_error = e
                if not self._is_retryable(e):
                    break
        
        return ExecutionResult(step_id=step.step_id, status="failed", error_type="logic_error", error=f"Max retries reached. Last error: {last_error}")

    def _is_transient_error(self, error: Exception) -> bool:
        msg = str(error).lower()
        return "timeout" in msg or "rate limit" in msg or "connection" in msg

    def _is_retryable(self, error: Exception) -> bool:
        # Most errors are retryable with LLM adaptation, except maybe authentication or permanent tool errors
        msg = str(error).lower()
        permanent_errors = ["auth", "permission", "not found", "invalid api key"]
        return not any(p in msg for p in permanent_errors) or self._is_transient_error(error)

    def _select_tool(self, capability: str, op_type: Any) -> Optional[str]:
        # Try to find a tool that matches the capability
        if self.tool_manager.get_tool(capability):
            return capability
            
        candidates = self.capability_map.get(capability, [])
        for c in candidates:
            if self.tool_manager.get_tool(c):
                return c
        return None

class PlanRepairer:
    """计划修复器 - 基于 Trace 定位并修复失败"""

    def __init__(self, llm_client: LLMClient, tool_manager: ToolManager):
        self.llm = llm_client
        self.tools = tool_manager

    async def repair(
        self,
        failed_step: PlanStep,
        execution_result: ExecutionResult,
        context: ExecutionContext,
        remaining_plan: List[PlanStep]
    ) -> Optional[Plan]:
        """尝试修复失败的步骤 (Async)"""

        error_type = execution_result.error_type
        print(f"--- Repairing Step {failed_step.step_id} (Error: {error_type}) ---")

        # 1. 绑定错误 - 变量不存在或路径错误
        if error_type == "binding_error":
            return self._repair_binding(failed_step, execution_result, context)

        # 2. 工具不可用 - 切换到备用工具
        if error_type == "tool_not_found":
            return self._repair_tool_selection(failed_step, remaining_plan, execution_result)

        # 3. Schema 不匹配 - 调整参数映射
        if error_type == "schema_error":
            return await self._repair_schema_mapping(failed_step, execution_result)

        # 4. 逻辑错误 - 需要重新规划
        if error_type == "logic_error":
            return None  # 触发全量 Replan

        return None

    def _repair_binding(
        self,
        step: PlanStep,
        result: ExecutionResult,
        context: ExecutionContext
    ) -> Optional[Plan]:
        """修复变量绑定错误"""
        
        # 尝试从错误信息中提取参数名
        failed_param = self._extract_failed_param(result.error)

        if not failed_param:
            # 如果无法提取具体参数，尝试检查所有 input 的绑定是否在 context 中存在
            for param_name, binding in step.operation.inputs.items():
                val = context.get(binding.value_ref)
                if val is None and binding.value_ref.source != "literal":
                    failed_param = param_name
                    break
        
        if not failed_param:
            return None

        print(f"Reparing binding for parameter: {failed_param}")

        # 尝试寻找替代变量源
        alternative_ref = self._find_alternative_variable(
            failed_param,
            context
        )

        if alternative_ref:
            # 修改步骤的输入绑定
            repaired_step = step.model_copy(deep=True)
            
            # Find which input parameter corresponds to the failed variable name
            target_param = None
            if failed_param in repaired_step.operation.inputs:
                target_param = failed_param
            else:
                # Search by value_ref.name
                for param, binding in repaired_step.operation.inputs.items():
                    if binding.value_ref.name == failed_param:
                        target_param = param
                        break
            
            if target_param:
                 # Explicitly replace the binding object to ensure update
                 original_binding = repaired_step.operation.inputs[target_param]
                 repaired_step.operation.inputs[target_param] = original_binding.model_copy(update={"value_ref": alternative_ref})

            # 返回只包含修复步骤的 Plan
            return Plan(
                goal="Repair binding",
                steps=[repaired_step],
                initial_inputs={},
                final_output=repaired_step.output_var
            )

        return None

    def _repair_tool_selection(self, step: PlanStep, remaining_plan: List[PlanStep], error: ExecutionResult = None) -> Optional[Plan]:
        """
        修复工具选择错误
        """
        required_cap = step.required_capability
        print(f"Searching for alternative tool for capability: {required_cap}")
        
        # 1. List available tools
        all_tools = self.tools.list_tools()
        
        # 2. Fuzzy match tool names or descriptions
        candidate_tools = []
        for tool in all_tools:
            if required_cap.lower() in tool['name'].lower() or required_cap.lower() in tool['description'].lower():
                candidate_tools.append(tool['name'])
        
        best_tool = None
        if candidate_tools:
            best_tool = candidate_tools[0]
            
        if best_tool:
            print(f"Found alternative tool: {best_tool}")
            
            # Update step to use this tool explicitly or update capability
            repaired_step = step.model_copy(deep=True)
            repaired_step.required_capability = best_tool 
            
            # Update metadata
            if not repaired_step.constraints:
                repaired_step.constraints = {}
            repaired_step.constraints["suggested_tool"] = best_tool
            
            return Plan(
                goal="Repair tool selection",
                steps=[repaired_step],
                initial_inputs={},
                final_output=repaired_step.output_var
            )
            
        return None

    async def _repair_schema_mapping(self, step: PlanStep, result: ExecutionResult) -> Optional[Plan]:
        """
        修复 Schema 不匹配错误
        """
        # 1. Check if it's a Plan Definition Error
        if self._is_plan_definition_error(result):
             # Plan definition error: The planner generated invalid parameters
             return await self._fix_plan_parameters(step, result)
        else:
             return None

    def _is_plan_definition_error(self, error: ExecutionResult) -> bool:
        """判断是否是 Plan 定义错误"""
        if not error.error:
            return False
            
        error_msg = error.error.lower()
        plan_error_indicators = [
            "unknown parameter",
            "invalid operation",
            "unsupported capability",
            "missing required input",
            "extra input",
            "unexpected argument"
        ]
        return any(indicator in error_msg for indicator in plan_error_indicators)

    async def _fix_plan_parameters(self, step: PlanStep, error: ExecutionResult) -> Optional[Plan]:
        """修复 Plan 中的参数定义 (Async)"""
        
        class CorrectedInputs(BaseModel):
            inputs: Dict[str, VariableBinding]
            reasoning: str

        prompt = f"""
        The following step failed due to invalid parameter definitions.
        
        Step Intent: {step.intent}
        Operation Type: {step.operation.type}
        Current Inputs: {step.operation.inputs}
        Error: {error.error}
        
        Please correct the 'inputs' mapping. Ensure parameter names match the tool's requirements.
        Use 'literal' source for hardcoded values if needed.
        """
        
        try:
            # Note: self.llm is LLMClient wrapper, might need to use its method or raw client
            # Assuming LLMClient has generate_structured or we use self.llm.client directly if exposed
            # LLMClient usually wraps OpenAI/Instructor. Let's use a generic generate or assume structure support.
            # If LLMClient doesn't support async generate_structured, we might need to use the one in ExecutionRepairer logic
            # For now, let's assume LLMClient has an async generate or we use run_in_executor if it's sync.
            # Checking LLMClient... it seems to be a custom wrapper.
            # Let's try to use the generate_structured if available, or just use the logic from ExecutionRepairer
            # To be safe, let's use the same Instructor setup as ExecutionRepairer for now if self.llm is not sufficient.
            
            # Re-using the instructor setup for PlanRepairer too for consistency
            # (In a real refactor, we should inject a shared AsyncLLMClient)
            
            # Temporary: Create a local instructor client for repair if self.llm is sync
            api_key = config.get("llm.api_key", "lm-studio")
            base_url = config.get("llm.api_base", "http://127.0.0.1:1234/v1")
            client = instructor.patch(OpenAI(
                base_url=base_url,
                api_key=api_key
            ), mode=instructor.Mode.MD_JSON)
            model = config.get("llm.model", "gpt-3.5-turbo")
            
            last_error = None
            result = None
            for _attempt in range(2):
                try:
                    result = client.chat.completions.create(
                        model=model,
                        response_model=CorrectedInputs,
                        messages=[{"role": "system", "content": prompt}],
                    )
                    break
                except Exception as e:
                    last_error = e
            if result is None:
                raise last_error
            
            repaired_step = step.model_copy(deep=True)
            repaired_step.operation.inputs = result.inputs
            
            return Plan(
                goal="Repair schema mapping",
                steps=[repaired_step],
                initial_inputs={},
                final_output=repaired_step.output_var
            )
        except Exception as e:
            logger.error(f"Plan parameter repair failed: {e}")
            return None

    def _extract_failed_param(self, error_msg: Optional[str]) -> Optional[str]:
        """从错误信息中提取失败的参数名"""
        if not error_msg:
            return None
        # 简单的启发式规则
        if "Missing input" in error_msg:
             try:
                 # format: "Missing input: param_name" or "KeyError: 'param_name'"
                 if ": " in error_msg:
                     return error_msg.split(": ")[1].strip().strip("'").strip('"')
             except:
                 pass
        if "KeyError" in error_msg:
             try:
                 import re
                 match = re.search(r"KeyError: '([^']+)'", error_msg)
                 if match:
                     return match.group(1)
             except:
                 pass
        return None

    def _find_alternative_variable(self, param_name: str, context: ExecutionContext) -> Optional[VariableRef]:
        """在 Context 中寻找替代变量"""
        available_vars = list(context.variables.keys())
        
        # 策略 1: 查找同名变量
        if param_name in context.variables:
             return VariableRef(source="context", name=param_name)
        
        # 策略 2: 模糊匹配 (difflib)
        matches = difflib.get_close_matches(param_name, available_vars, n=1, cutoff=0.6)
        if matches:
            print(f"Fuzzy match found: {param_name} -> {matches[0]}")
            return self._create_var_ref(matches[0])

        # 策略 3: 同义词匹配 (Synonyms)
        synonyms = self._load_synonyms()
        if param_name in synonyms:
            for synonym in synonyms[param_name]:
                if synonym in context.variables:
                    print(f"Synonym match found: {param_name} -> {synonym}")
                    return self._create_var_ref(synonym)
                # Check for matches in available_vars using synonyms
                matches = difflib.get_close_matches(synonym, available_vars, n=1, cutoff=0.8)
                if matches:
                    print(f"Synonym fuzzy match found: {param_name} -> {synonym} -> {matches[0]}")
                    return self._create_var_ref(matches[0])
                    
        # 策略 4: Path 错误修复 (e.g. "data.items[0]" -> "data")
        # If param_name looks like a path access that failed
        if "." in param_name or "[" in param_name:
            root_var = param_name.split(".")[0].split("[")[0]
            if root_var in context.variables:
                 print(f"Path root match found: {param_name} -> {root_var}")
                 return self._create_var_ref(root_var)
        
        return None

    def _create_var_ref(self, key: str) -> VariableRef:
        """Helper to create VariableRef from context key"""
        if key.startswith("step_output."):
            parts = key.split(".")
            if len(parts) >= 2:
                return VariableRef(source="step_output", name=parts[1])
        elif key.startswith("input."):
             return VariableRef(source="input", name=key.split(".")[1])
        else:
             return VariableRef(source="context", name=key)

    def _load_synonyms(self) -> Dict[str, List[str]]:
        # TODO: Load from config/synonyms.json
        return {
            "url": ["link", "address", "uri"],
            "text": ["content", "body", "string"],
            "file": ["path", "filename", "filepath"],
            "query": ["keyword", "search_term", "q"]
        }
