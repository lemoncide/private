import re
import json
import logging
import asyncio
from typing import List, Dict, Any, Type, TypeVar
from pydantic import BaseModel, Field
from openai import OpenAI
from agent.core.schema import Plan, PlanStep, VariableRef, VariableBinding, OperationType, OperationSpec
from agent.llm.cot_parser import extract_thinking, parse_structured, CoTParseError
from agent.utils.config import config

logger = logging.getLogger("Agent")

T = TypeVar("T", bound=BaseModel)

# ============= Stage 1 Schema =============
class SkeletonStep(BaseModel):
    step_id: str
    intent: str
    required_capability: str
    description: str

class PlanSkeleton(BaseModel):
    goal: str = "" # Optional or derived
    steps: List[SkeletonStep]

# ============= Planner =============
class Planner:
    def __init__(self):
        # Initialize standard OpenAI client
        api_key = config.get("llm.api_key", "lm-studio")
        base_url = config.get("llm.api_base", "http://127.0.0.1:1234/v1")
        
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model = config.get("llm.model", "gpt-3.5-turbo")

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
        """
        Two-Stage Planning Strategy:
        1. Skeleton Generation: Identify high-level steps and required capabilities.
        2. Detail Generation: Fill in variable bindings, operations, and fallbacks.
        """
        logger.info(f"Planning for objective: {objective}")
        
        try:
            # Stage 1: Generate Skeleton
            skeleton = self._generate_skeleton(objective, tools)
            logger.info(f"Generated skeleton with {len(skeleton.steps)} steps")
            
            # Stage 2: Generate Full Plan
            plan = self._generate_details(objective, tools, skeleton)
            logger.info(f"Generated full plan with {len(plan.steps)} steps")
            
            return plan
            
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            # Fallback to a simple single-step plan if LLM fails completely
            return self._create_fallback_plan(objective)

    def _generate_skeleton(self, objective: str, tools: List[Dict[str, Any]]) -> PlanSkeleton:
        """Stage 1: High-level steps"""
        # Improved tool description injection - COMPACT MODE
        # To avoid context window overflow, we only include name and description for skeleton generation
        tool_descriptions = []
        for t in tools:
            # Truncate description heavily
            desc = t['description'][:60] + "..." if len(t['description']) > 60 else t['description']
            tool_descriptions.append(f"- {t['name']}: {desc}")
        
        tool_desc_str = "\n".join(tool_descriptions)
        
        system_prompt = f"""
        You are an expert planner. Break down the user's objective into high-level steps.
        
        Available Tools:
        {tool_desc_str}
        
        For each step, identify:
        1. A unique step_id (e.g., step_1)
        2. The intent (what to achieve)
        3. The required capability (MUST be an EXACT tool name from Available Tools)
        4. A brief description

        TOOL SELECTION RULES (CRITICAL):
        - required_capability MUST exactly match one of the tool names listed in Available Tools.
        - Do NOT output generic labels like "search", "read", "write", "reasoning".
        - Do NOT invent tools. If no tool fits, choose "llm_reasoning".
        
        RESPONSE FORMAT:
        You MUST provide a <thinking> block BEFORE the JSON.
        Format:
        <thinking>
        1. Identify the minimum necessary steps.
        2. Validate dependencies between steps.
        </thinking>
        ```json
        {{ ... }}
        ```

        CRITICAL: The JSON must match this schema:
        {{
            "steps": [
                {{
                    "step_id": "string",
                    "intent": "string",
                    "required_capability": "string",
                    "description": "string"
                }}
            ]
        }}
        
        Keep the plan concise and logical.
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Objective: {objective}"}
            ],
            temperature=0.0
        )
        
        return self._parse_json(response.choices[0].message.content, PlanSkeleton)

    def _generate_details(self, objective: str, tools: List[Dict[str, Any]], skeleton: PlanSkeleton) -> Plan:
        """Stage 2: Detailed bindings and configurations"""
        # Improved tool description injection - ULTRA COMPACT MODE
        # SIMPLIFIED: Trust the pre-filtered tool list from nodes.py
        # No need for complex logic here, just inject what we have.
            
        tool_descriptions = []
        for t in tools:
            # Inject all tools passed to this function
            desc = f"- {t['name']}"
            if 'args_schema' in t:
                 props = t['args_schema'].get('properties', {})
                 args_list = list(props.keys())
                 desc += f" (Inputs: {', '.join(args_list)})"
            tool_descriptions.append(desc)
            
        tool_desc_str = "\n".join(tool_descriptions)
        skeleton_json = skeleton.model_dump_json(indent=2)
        
        # NOTE: For official MCP tools, path arguments must be ABSOLUTE paths or valid URIs.
        # Ensure the LLM understands this requirement for filesystem operations.
        
        system_prompt = f"""
        You are an expert autonomous agent planner.
        Your goal is to convert the provided Plan Skeleton into a robust, executable Plan.
        
        Available Tools (Refer to these for exact parameter names):
        {tool_desc_str}
        
        Plan Skeleton:
        {skeleton_json}
        
        CRITICAL RULES:
        1. Maintain the steps from the skeleton.
        2. For each step, define the 'operation' with 'type' and 'inputs'.
        2.5. required_capability MUST be an EXACT tool name from Available Tools. Do NOT use generic capabilities.
        2.6. operation.type MUST be one of: compute, transform, extract, search, read, write, call_api, condition, aggregate, reason, generate.
        2.7. certainty MUST be either: certain, uncertain. (Do NOT use high/medium/low.)
        2.8. fallback_strategy MUST be one of: fail, ask_llm, skip. (Do NOT use none.)
        3. Use 'VariableRef' for inputs to trace data flow:
           - source='input' for initial user inputs
           - source='step_output' for results from previous steps (name=step_id)
           - source='literal' for hardcoded values. THIS IS MANDATORY for values explicitly stated in the user's objective (numbers, paths, query strings).
             * Example (Math): "Calculate 50 * 3" -> inputs: {{ "expression": {{ "value_ref": {{"source": "literal", "name": "50 * 3"}} }} }}
             * Example (File): "Read from data/report.txt" -> inputs: {{ "file_path": {{ "value_ref": {{"source": "literal", "name": "data/report.txt"}} }} }}
             * Example (Search): "Search for 'LLM agents'" -> inputs: {{ "query": {{ "value_ref": {{"source": "literal", "name": "LLM agents"}} }} }}
        4. Define 'output_var' for each step.
        5. Mark 'certainty' as 'uncertain' if the step relies on external data that might be messy.
        6. Set 'fallback_strategy' to 'ask_llm' for uncertain steps.
        
        STRICT TYPE CHECKING:
        - Do not pass a string to a tool expecting an integer/float. 
        - If a tool needs a number (e.g. calculator), extract the numeric value.
        - If a tool needs a JSON string, ensure it is properly formatted.
        
        TOOL PARAMETER AWARENESS:
        - For 'calculator', use 'expression' (e.g., "50 * 3"). Do NOT split into 'a' and 'b'.
        - For 'local_web_search', use 'query'.
        - For 'read_file' / 'write_file', use 'file_path' (and 'content' for write).
        - For MCP tools, strictly follow the parameter names listed in "Available Tools".

        TOOL CHOICE QUALITY:
        - Prefer direct, structured tools over indirect detours (e.g., avoid using a general web search when a dedicated tool can query the target system directly).
        - Do not pass large blobs of text into path/file parameters. If you need a path or identifier, add a step to extract/derive it explicitly.
        
        PATH HANDLING:
        - When dealing with files, always use the FULL ABSOLUTE PATH provided in the objective if available.
        - Do not truncate or modify file paths unless explicitly asked.
        
        DEPENDENCY VALIDATION:
        - Ensure every 'step_output' source refers to a step_id that actually exists in previous steps.
        - Do not reference future steps.
        - The 'final_output' field MUST be the exact 'output_var' name of the last step, NOT a description.
        - NEVER set 'final_output' to a filename like 'github_summary.txt'. That filename belongs in a write step input, not final_output.
        - If the last step writes a file, set that step's output_var to something like 'write_result' and set final_output to 'write_result'.

        PLAN OUTPUT MUST MATCH THIS SHAPE (CRITICAL):
        ```json
        {{
          "goal": "string",
          "steps": [
            {{
              "step_id": "step_1",
              "intent": "string",
              "operation": {{
                "type": "search",
                "description": "optional string",
                "inputs": {{
                  "query": {{ "value_ref": {{ "source": "literal", "name": "..." }} }}
                }},
                "config": {{}}
              }},
              "output_var": "string",
              "depends_on": [],
              "required_capability": "EXACT_TOOL_NAME_FROM_AVAILABLE_TOOLS",
              "constraints": {{}},
              "certainty": "certain",
              "fallback_strategy": "fail"
            }}
          ],
          "initial_inputs": {{}},
          "final_output": "string",
          "metadata": {{}}
        }}
        ```
        IMPORTANT: operation.type is a SEMANTIC ENUM (search/read/write/...). required_capability is the CONCRETE TOOL NAME.
        
        LITERAL EXTRACTION RULES (MANDATORY):
        - You MUST extract specific values (numbers, strings, filenames) from the user's objective.
        - Set 'source' to 'literal'.
        - Set 'name' to the exact value (e.g. "50", "iPhone 16", "data.txt").
        
        RESPONSE FORMAT:
        You MUST provide a <thinking> block BEFORE the JSON.
        Format:
        <thinking>
        1. Analyze tool requirements...
        2. Check if resources exist... (e.g., do I need to clone before reading?)
        3. Verify input types...
        </thinking>
        ```json
        {{
          "steps": [...]
        }}
        ```
        """
        
        try:
            last_error = None
            for _attempt in range(3):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Objective: {objective}"},
                        ],
                        timeout=600.0,
                    )
                    plan = self._parse_json(response.choices[0].message.content, Plan)
                    return self._fix_final_output(plan)
                except Exception as e:
                    last_error = e
            raise last_error
            
        except Exception as e:
            logger.error(f"Plan generation failed or timed out: {e}")
            raise e

    def _fix_final_output(self, plan: Plan) -> Plan:
        if not plan or not plan.steps:
            return plan

        available_vars = set((plan.initial_inputs or {}).keys())
        step_outputs = [s.output_var for s in plan.steps if s.output_var]
        if plan.final_output in step_outputs or plan.final_output in available_vars:
            return plan

        candidates = [s for s in plan.steps if s.output_var]
        if not candidates:
            return plan

        last_step = candidates[-1]
        plan.final_output = last_step.output_var
        return plan

    def _create_fallback_plan(self, objective: str) -> Plan:
        return Plan(
            goal=objective,
            steps=[
                PlanStep(
                    step_id="step_1",
                    intent="Execute task via LLM reasoning",
                    operation=OperationSpec(
                        type=OperationType.REASON,
                        description="Reasoning fallback",
                        inputs={
                            "task_description": VariableBinding(
                                value_ref=VariableRef(source="literal", name=objective)
                            )
                        }
                    ),
                    output_var="final_result",
                    required_capability="reasoning",
                    certainty="uncertain",
                    fallback_strategy="ask_llm"
                )
            ],
            final_output="final_result"
        )

    def refine_plan(self, objective: str, original_plan: Plan, error_msg: str) -> Plan:
        """
        Refine the plan based on validation errors.
        """
        logger.info(f"Refining plan due to error: {error_msg}")
        
        system_prompt = (
            "You are an expert PlanDebugger. Your goal is to fix a broken execution plan.\n"
            "The previous plan failed validation. Do NOT just blindly retry or fill in empty strings.\n\n"
            
            "--- ERROR DIAGNOSIS STRATEGY ---\n"
            "1. IF error is 'Field required' or 'Input is empty' or 'Literal input ... is empty':\n"
            "   - CAUSE: You are likely trying to use a resource (file, variable, id) that hasn't been created yet.\n"
            "   - FIX: You MUST insert a NEW STEP before the failing step to create/fetch that resource.\n"
            "   - EXAMPLE: If 'path' is empty for 'read_file', did you forget to 'git_clone' or 'search_files' first?\n"
            "   - EXAMPLE: If 'repo_id' is empty, did you forget to 'search_repositories' first?\n\n"
            
            "2. IF error is 'Invalid format':\n"
            "   - CAUSE: The tool expects a specific string format (e.g., JSON, date).\n"
            "   - FIX: Check the tool definition strictly.\n\n"
            
            "--- TASK ---\n"
            "Review the Objective, the Failed Plan, and the Error.\n"
            "Generate a CORRECTED plan that solves the logical dependency issues.\n"
            "Output a <thinking> block followed by a valid JSON object matching the Plan schema.\n"
            "CRITICAL CONSTRAINTS:\n"
            "- Your JSON MUST include: goal, steps, final_output.\n"
            "- Each step MUST include: step_id, intent, operation{type,inputs}, output_var, required_capability, certainty, fallback_strategy.\n"
            "- operation.type MUST be one of: compute, transform, extract, search, read, write, call_api, condition, aggregate, reason, generate.\n"
            "- required_capability MUST be an EXACT tool name from the provided tool list. Do NOT output generic capabilities.\n"
            "- certainty MUST be either: certain, uncertain.\n"
            "- fallback_strategy MUST be one of: fail, ask_llm, skip.\n"
        )
        
        user_content = f"""
        Objective: {objective}
        
        Failed Plan (JSON):
        {original_plan.model_dump_json(indent=2)}
        
        Validation Error:
        {error_msg}
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        )
        
        plan = self._parse_json(response.choices[0].message.content, Plan)
        return self._fix_final_output(plan)
