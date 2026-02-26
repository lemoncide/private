import asyncio
from typing import Dict, Any, List
from agent.core.state import AgentState
from agent.core.errors import PlanToolNotFoundError
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

    async def plan_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Planner Node: Generates an execution plan based on the user's objective.
        """
        logger.info("--- Plan Node ---")
        objective = state.input
        attempt_limits = [12, 25, 500]
        last_error: str | None = None

        plan = None
        loop = asyncio.get_running_loop()
        for i, limit in enumerate(attempt_limits):
            tool_defs, report = prepare_tool_defs_with_report(self.tools, objective, limit=limit)
            logger.info(f"Filtered tools from {report.get('total')} to {report.get('filtered')} (limit={report.get('limit')})")
            logger.info(f"Active Tools: {[t.get('name') for t in tool_defs]}")
            if report.get("retrieved", 0) <= 0 and i < len(attempt_limits) - 1:
                last_error = "Tool retrieval returned empty; expanding tool set"
                continue
            try:
                # self.planner.create_plan is synchronous, run it in an executor
                plan = await loop.run_in_executor(None, self.planner.create_plan, objective, tool_defs)
                last_error = None
                break
            except PlanToolNotFoundError as e:
                last_error = str(e)
                if i < len(attempt_limits) - 1:
                    continue
                break
            except Exception as e:
                last_error = str(e)
                break

        if plan is None:
            return {"status": "failed", "error": f"Planning error: {last_error}"}

        initial_context = {"input": objective}
        return {
            "plan": plan,
            "current_step_index": 0,
            "status": "executing",
            "context_variables": initial_context,
            "error": None,
            "error_type": None,
            "repair_context": {},
            "repair_history": [],
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
                "error_type": None,
                "repair_context": {},
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
                "error_type": getattr(result, "error_type", None),
                "repair_context": result.meta or {},
                "repair_attempts": state.repair_attempts,
            }

    def error_router_node(self, state: AgentState) -> Dict[str, Any]:
        return {}

    def retry_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info("--- Retry Node ---")
        index = state.current_step_index
        step_id = None
        if state.plan and state.plan.steps and index < len(state.plan.steps):
            step_id = state.plan.steps[index].step_id
        history = list(state.repair_history or [])
        history.append(
            {
                "step_id": step_id,
                "error_type": state.error_type,
                "attempted_fix": "retry",
                "result": "scheduled",
            }
        )
        return {
            "repair_history": history,
            "repair_attempts": state.repair_attempts + 1,
            "status": "running",
            "error": None,
        }

    async def repair_plan_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info("--- Repair Plan Node ---")
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

        tool_defs, _ = prepare_tool_defs_with_report(self.tools, state.input, limit=500)
        
        # Run synchronous repairer method in executor
        loop = asyncio.get_running_loop()
        patched = await loop.run_in_executor(
            None, 
            self.repairer.repair_plan,
            state.input,
            failed_step,
            last_error or "",
            state.context_variables,
            tool_defs
        )

        history = list(state.repair_history or [])
        attempts = state.repair_attempts + 1
        if not patched:
            history.append(
                {
                    "step_id": failed_step.step_id,
                    "error_type": state.error_type,
                    "attempted_fix": "repair_plan",
                    "result": "failed",
                }
            )
            return {
                "context_variables": state.context_variables,
                "status": "repair_failed",
                "error": "Repair plan failed",
                "repair_attempts": attempts,
                "repair_history": history,
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
        history.append(
            {
                "step_id": failed_step.step_id,
                "error_type": state.error_type,
                "attempted_fix": "repair_plan",
                "result": "applied",
            }
        )
        return {
            "plan": new_plan,
            "context_variables": state.context_variables,
            "status": "running",
            "error": None,
            "error_type": None,
            "repair_context": {},
            "repair_attempts": attempts,
            "repair_history": history,
            "current_step_index": index,
        }

    async def repair_params_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info("--- Repair Params Node ---")
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

        tool_defs, _ = prepare_tool_defs_with_report(self.tools, state.input, limit=25)
        args_schema = (state.repair_context or {}).get("args_schema")
        actual_args = (state.repair_context or {}).get("resolved_args")
        
        # Run synchronous repairer method in executor
        loop = asyncio.get_running_loop()
        patched = await loop.run_in_executor(
            None,
            self.repairer.repair_params,
            state.input,
            failed_step,
            last_error or "",
            state.context_variables,
            tool_defs,
            args_schema,
            actual_args
        )

        history = list(state.repair_history or [])
        attempts = state.repair_attempts + 1
        if not patched:
            history.append(
                {
                    "step_id": failed_step.step_id,
                    "error_type": state.error_type,
                    "attempted_fix": "repair_params",
                    "result": "failed",
                }
            )
            return {
                "context_variables": state.context_variables,
                "status": "repair_failed",
                "error": "Repair params failed",
                "repair_attempts": attempts,
                "repair_history": history,
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
        history.append(
            {
                "step_id": failed_step.step_id,
                "error_type": state.error_type,
                "attempted_fix": "repair_params",
                "result": "applied",
            }
        )
        return {
            "plan": new_plan,
            "context_variables": state.context_variables,
            "status": "running",
            "error": None,
            "error_type": None,
            "repair_context": {},
            "repair_attempts": attempts,
            "repair_history": history,
            "current_step_index": index,
        }

    async def repair_query_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info("--- Repair Query Node ---")
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

        tool_defs, _ = prepare_tool_defs_with_report(self.tools, state.input, limit=25)
        actual_args = (state.repair_context or {}).get("resolved_args")
        
        # Run synchronous repairer method in executor
        loop = asyncio.get_running_loop()
        patched = await loop.run_in_executor(
            None,
            self.repairer.repair_query,
            state.input,
            failed_step,
            last_error or "",
            state.context_variables,
            tool_defs,
            actual_args
        )

        history = list(state.repair_history or [])
        attempts = state.repair_attempts + 1
        if not patched:
            history.append(
                {
                    "step_id": failed_step.step_id,
                    "error_type": state.error_type,
                    "attempted_fix": "repair_query",
                    "result": "failed",
                }
            )
            return {
                "context_variables": state.context_variables,
                "status": "repair_failed",
                "error": "Repair query failed",
                "repair_attempts": attempts,
                "repair_history": history,
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
        history.append(
            {
                "step_id": failed_step.step_id,
                "error_type": state.error_type,
                "attempted_fix": "repair_query",
                "result": "applied",
            }
        )
        return {
            "plan": new_plan,
            "context_variables": state.context_variables,
            "status": "running",
            "error": None,
            "error_type": None,
            "repair_context": {},
            "repair_attempts": attempts,
            "repair_history": history,
            "current_step_index": index,
        }

    async def reflect_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info("--- Reflect Node ---")
        
        # self.reflector.reflect is synchronous, run it in an executor
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, self.reflector.reflect, state)
        
        try:
            objective = state.input or ""
            status = state.status or ""
            error = state.error or ""
            tools: List[str] = []
            for r in state.past_steps or []:
                meta = getattr(r, "meta", {}) or {}
                name = meta.get("tool")
                if name:
                    tools.append(name)
            tools_line = ", ".join(tools) if tools else ""
            result_preview = None
            final = None
            try:
                from agent.core.reflect import Reflector
                final = Reflector._extract_final_result(state)
                result_preview = Reflector._prompt_preview(final, limit=1000)
            except Exception:
                result_preview = None
            parts: List[str] = []
            parts.append(f"Objective: {objective}")
            parts.append(f"Status: {status}")
            if tools_line:
                parts.append(f"Tools: {tools_line}")
            if error:
                parts.append(f"Error: {error}")
            if result_preview:
                parts.append(f"Result: {result_preview}")
            summary = "\n".join(parts)
            self.memory.add_task_history(
                summary=summary,
                metadata={
                    "status": status,
                    "tools": tools,
                },
            )
        except Exception:
            pass
        return {"response": text}
