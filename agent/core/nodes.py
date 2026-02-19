from typing import Dict, Any, List
from agent.core.state import AgentState
from agent.core.planner import Planner
from agent.core.executor import ToolExecutor, ExecutionContext
from agent.core.repair import Repairer
from agent.core.schema import PlanStep, ExecutionResult
from agent.core.reflect import Reflector
from agent.core.tool_defs import prepare_tool_defs_with_report
from agent.tools.manager import ToolManager
from agent.llm.client import LLMClient
from agent.memory.manager import MemoryManager
from agent.utils.logger import logger

class AgentNodes:
    def __init__(self, llm_client: LLMClient, tool_manager: ToolManager, memory_manager: MemoryManager):
        self.llm = llm_client
        self.tools = tool_manager
        self.memory = memory_manager
        
        self.planner = Planner(llm_client)
        self.executor = ToolExecutor(tool_manager)
        self.repairer = Repairer(llm_client)
        self.reflector = Reflector(llm_client)

    def plan_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Planner Node: Generates an execution plan based on the user's objective.
        """
        logger.info("--- Plan Node ---")
        objective = state.input

        tool_defs, report = prepare_tool_defs_with_report(self.tools, objective)
        logger.info(f"Filtered tools from {report.get('total')} to {report.get('filtered')}")
        logger.info(f"Active Tools: {[t.get('name') for t in tool_defs]}")

        try:
            plan = self.planner.create_plan(objective, tool_defs)
        except Exception as e:
            return {"status": "failed", "error": f"Planning error: {str(e)}"}

        initial_context = {"input": objective}
        return {
            "plan": plan,
            "current_step_index": 0,
            "status": "executing",
            "context_variables": initial_context,
            "error": None,
            "repair_attempts": 0,
        }

    async def execute_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Executor Node: Executes the current step in the plan.
        """
        logger.info("--- Execute Node ---")
        
        if not state.plan or not state.plan.steps:
            return {
                "context_variables": state.context_variables,
                "status": "failed",
                "error": "No plan available",
            }
            
        index = state.current_step_index
        if index >= len(state.plan.steps):
            return {
                "context_variables": state.context_variables,
                "status": "completed",
            }
            
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
                "error": None,
                "repair_attempts": 0,
            }
        else:
            # Execution failed
            logger.warning(f"Step {index} failed: {result.error}")
            return {
                "past_steps": past_steps,
                "context_variables": context.variables,
                "status": "failed",
                "error": result.error,
                "repair_attempts": state.repair_attempts,
            }

    def repair_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info("--- Repair Node ---")

        if not state.plan or not state.plan.steps:
            return {
                "context_variables": state.context_variables,
                "status": "repair_failed",
                "error": "No plan available",
                "repair_attempts": state.repair_attempts + 1,
            }

        index = state.current_step_index
        if index >= len(state.plan.steps):
            return {
                "context_variables": state.context_variables,
                "status": "repair_failed",
                "error": "Invalid step index",
                "repair_attempts": state.repair_attempts + 1,
            }

        failed_step: PlanStep = state.plan.steps[index]
        last_error = state.error
        if not last_error and state.past_steps:
            last_error = state.past_steps[-1].error

        tool_defs, _ = prepare_tool_defs_with_report(self.tools, state.input)
        patched = self.repairer.repair_step(
            objective=state.input,
            failed_step=failed_step,
            error=last_error or "",
            context_variables=state.context_variables,
            tool_defs=tool_defs,
        )

        attempts = state.repair_attempts + 1
        if not patched:
            return {
                "context_variables": state.context_variables,
                "status": "repair_failed",
                "error": "Repair failed",
                "repair_attempts": attempts,
            }

        new_step = failed_step.model_copy(
            update={
                "required_capability": patched.get("required_capability"),
                "tool_args": patched.get("tool_args") or {},
            }
        )
        new_steps = list(state.plan.steps)
        new_steps[index] = new_step
        new_plan = state.plan.model_copy(update={"steps": new_steps})

        return {
            "plan": new_plan,
            "context_variables": state.context_variables,
            "status": "executing",
            "error": None,
            "repair_attempts": attempts,
            "current_step_index": index,
        }

    def reflect_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Reflect Node: Formats the final response.
        """
        logger.info("--- Reflect Node ---")
        return {"response": self.reflector.reflect(state)}
