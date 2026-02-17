from typing import Dict, Any, List
import asyncio
from agent.core.state import AgentState
from agent.core.planner import Planner
from agent.core.executor import ToolExecutor, ExecutionContext
from agent.core.repair import PlanRepairer
from agent.core.validator import PlanValidator
from agent.core.schema import Plan, PlanStep, ExecutionResult
from agent.tools.manager import ToolManager
from agent.llm.client import LLMClient
from agent.memory.manager import MemoryManager
from agent.utils.logger import logger

class AgentNodes:
    def __init__(self, llm_client: LLMClient, tool_manager: ToolManager, memory_manager: MemoryManager):
        self.llm = llm_client
        self.tools = tool_manager
        self.memory = memory_manager
        
        self.planner = Planner()
        self.executor = ToolExecutor(tool_manager)
        self.repairer = PlanRepairer(llm_client, tool_manager)
        self.validator = PlanValidator()

    def plan_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Planner Node: Generates an execution plan based on the user's objective.
        """
        logger.info("--- Plan Node ---")
        objective = state.input
        
        # 1. Generate Plan
        # We need to pass schema info now. list_tools returns dicts with name/description.
        # We need to modify ToolManager.list_tools to include schema if we want to use it?
        # Actually ToolManager.list_tools already returns dicts. 
        # But we need to ensure it includes args_schema for Planner to use.
        # Let's check ToolManager.list_tools implementation. 
        # It returns {"name": ..., "description": ...}.
        # We should update list_tools to include schema or fetch full tool objects here.
        
        # Better approach: Get full tool objects here and pass to Planner.
        # But Planner expects list of dicts currently.
        # Let's update ToolManager.list_tools to optionally include schema details
        # OR just use tool_manager.tools.values() directly since we are inside the node.
        
        all_tools_raw = self.tools.list_tools(limit=100) # Get all tools first
        
        # --- TOOL FILTERING LOGIC ---
        # Filter tools based on objective to reduce context window
        filtered_tools = []
        obj_lower = objective.lower()
        
        # 1. Always Include Core Tools
        core_tools = ["read_file", "write_file", "local_web_search", "calculator", "llm_reasoning"]
        
        # 2. Keyword Matchers
        # Map keywords in objective to tool prefixes/names
        # REFINED: Remove broad keywords like 'list', 'search' to prevent explosion
        keyword_map = {
            "git": ["git", "github"],
            "repo": ["git", "github"],
            "code": ["search_code", "read"],
            # "file": ["filesystem", "read", "write", "edit"], # Too broad, remove
            "csv": ["clean_data", "read_csv"],
            "data": ["clean_data", "read_csv"],
            # Specific combinations
            "read": ["read"],
            "write": ["write"],
            "save": ["write"]
        }
        
        relevant_prefixes = set()
        for keyword, prefixes in keyword_map.items():
            if keyword in obj_lower:
                relevant_prefixes.update(prefixes)
                
        # 3. Apply Filter
        for t in all_tools_raw:
            t_name = t['name'].lower()
            
            # Check core
            if any(core in t_name for core in core_tools):
                filtered_tools.append(t)
                continue
                
            # Check relevance
            if any(prefix in t_name for prefix in relevant_prefixes):
                filtered_tools.append(t)
                continue
                
        # Fallback: If filter is too aggressive (< 5 tools), add top N generic tools
        if len(filtered_tools) < 5:
             filtered_tools = all_tools_raw[:10]
             
        # Limit total to prevent overflow (Reduced from 20 to 12)
        filtered_tools = filtered_tools[:12]
        
        logger.info(f"Filtered tools from {len(all_tools_raw)} to {len(filtered_tools)}")
        logger.info(f"Active Tools: {[t['name'] for t in filtered_tools]}")
        
        enriched_tools = []
        for t_dict in filtered_tools:
            tool_obj = self.tools.get_tool(t_dict['name'])
            if tool_obj and hasattr(tool_obj, 'args_schema'):
                # Check if args_schema is a Pydantic model (class) or a dict
                if isinstance(tool_obj.args_schema, dict):
                    t_dict['args_schema'] = tool_obj.args_schema
                elif hasattr(tool_obj.args_schema, 'schema'):
                    t_dict['args_schema'] = tool_obj.args_schema.schema()
                else:
                    # Fallback or raw dict
                    t_dict['args_schema'] = tool_obj.args_schema
            enriched_tools.append(t_dict)
            
        tool_defs = enriched_tools
        
        current_plan = None
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                if attempt == 0:
                    current_plan = self.planner.plan(objective, tool_defs)
                else:
                    # Retry with refinement
                    logger.info(f"Retrying plan generation (attempt {attempt+1})")
                    if current_plan and last_error:
                         current_plan = self.planner.refine_plan(objective, current_plan, last_error, tool_defs)
                    else:
                         current_plan = self.planner.plan(objective, tool_defs)
                    
            except Exception as e:
                logger.error(f"Planning failed (attempt {attempt+1}): {e}")
                last_error = str(e)
                if attempt == max_retries - 1:
                    return {
                        "status": "failed",
                        "error": f"Planning error: {str(e)}"
                    }
                continue # Retry

            # 2. Validate Plan
            validation = self.validator.validate(current_plan)
            if validation.valid:
                logger.info(f"Plan generated and validated with {len(current_plan.steps)} steps")
                
                # Initialize context variables with input
                initial_context = {"input": objective}
                
                return {
                    "plan": current_plan,
                    "current_step_index": 0,
                    "status": "executing", # Ready to execute
                    "context_variables": initial_context,
                    "error": None
                }
            else:
                error_msg = f"Plan validation failed: {'; '.join(validation.errors)}"
                logger.warning(error_msg)
                last_error = error_msg
                
                if attempt == max_retries - 1:
                    return {
                        "status": "failed",
                        "error": error_msg
                    }
        
        return {"status": "failed", "error": "Max retries exceeded"}

    async def execute_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Executor Node: Executes the current step in the plan.
        """
        logger.info("--- Execute Node ---")
        
        if not state.plan or not state.plan.steps:
            return {"status": "failed", "error": "No plan available"}
            
        index = state.current_step_index
        if index >= len(state.plan.steps):
            return {"status": "completed"}
            
        current_step: PlanStep = state.plan.steps[index]
        
        # Reconstruct ExecutionContext
        context = ExecutionContext(variables=state.context_variables)
        
        # Execute Step
        try:
            result: ExecutionResult = await self.executor.execute_step(current_step, context)
        except Exception as e:
            # Fallback for unexpected errors not caught in execute_step
            logger.error(f"Unexpected execution error: {e}")
            result = ExecutionResult(step_id=current_step.step_id, status="failed", error=str(e), error_type="system_error")

        # Update History
        past_steps = state.past_steps.copy()
        past_steps.append(result)
        
        # Handle Result
        if result.status == "success":
            # Check if this was the last step
            next_index = index + 1
            status = "running"
            if next_index >= len(state.plan.steps):
                status = "completed"
                
            return {
                "past_steps": past_steps,
                "context_variables": context.variables, # Persist updated context
                "current_step_index": next_index,
                "status": status,
                "error": None
            }
        else:
            # Execution failed
            logger.warning(f"Step {index} failed: {result.error}")
            return {
                "past_steps": past_steps,
                "status": "failed", # Will trigger repair
                "error": result.error
            }

    async def repair_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Repair Node: Attempts to fix a failed step.
        """
        logger.info("--- Repair Node ---")
        
        if not state.past_steps:
             return {"status": "repair_failed", "error": "No history to repair"}
             
        last_result = state.past_steps[-1]
        if last_result.status != "failed":
            # Should not happen
            return {"status": "running"} # Continue?
            
        index = state.current_step_index
        # Use previous index if we incremented it? 
        # Actually execute_node only increments on success.
        # So index points to the failed step.
        
        failed_step = state.plan.steps[index]
        remaining_plan = state.plan.steps[index+1:]
        
        context = ExecutionContext(variables=state.context_variables)
        
        try:
            repaired_plan = await self.repairer.repair(
                failed_step,
                last_result,
                context,
                remaining_plan
            )
            
            if repaired_plan:
                logger.info(f"Repair successful for step {failed_step.step_id}")
                
                # Update the plan with the repaired step(s)
                # repaired_plan contains the fixed step(s).
                # Usually it returns a single fixed step or a sequence.
                # We replace the current step with the new steps.
                
                # Note: Plan object is immutable-ish (Pydantic), need to copy
                current_plan = state.plan.model_copy(deep=True)
                
                # Replace the current failed step with the first step of repaired plan
                # (Simplification: assuming repair returns 1 replacement step for now)
                if len(repaired_plan.steps) > 0:
                    current_plan.steps[index] = repaired_plan.steps[0]
                    # If there are more steps, we might need to insert them? 
                    # For now assume 1-to-1 repair.
                
                return {
                    "plan": current_plan,
                    "status": "executing", # Retry execution
                    "error": None
                }
            else:
                logger.warning(f"Repair failed for step {failed_step.step_id}")
                return {"status": "repair_failed"}
                
        except Exception as e:
            logger.error(f"Repair process crashed: {e}")
            return {"status": "repair_failed", "error": str(e)}

    def reflect_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Reflect Node: Formats the final response.
        """
        logger.info("--- Reflect Node ---")
        
        if state.status == "completed":
            # Extract final result
            final_var = state.plan.final_output
            result = state.context_variables.get(final_var)
            
            # If not found directly, check step outputs
            if result is None:
                result = state.context_variables.get(f"step_output.{final_var}")
            
            # If still None, maybe it was a direct value?
            if result is None:
                 # Try to find the last step output
                 if state.past_steps:
                     result = state.past_steps[-1].result
            
            response = f"Task Completed.\nFinal Result: {result}"
            return {"response": response}
            
        elif state.status == "repair_failed":
            return {"response": f"Task Failed. Error: {state.error}"}
            
        return {}
