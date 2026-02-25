from langgraph.graph import StateGraph, END
from agent.core.state import AgentState
from agent.core.nodes import AgentNodes
from agent.llm.client import LLMClient
from agent.tools.manager import ToolManager
from agent.memory.manager import MemoryManager
from agent.utils.config import config


def build_graph_with_deps():
    llm_client = LLMClient()
    memory_manager = MemoryManager()
    tool_manager = ToolManager(memory_manager)
    nodes = AgentNodes(llm_client, tool_manager, memory_manager)

    workflow = StateGraph(AgentState)
    workflow.add_node("planner", nodes.plan_node)
    workflow.add_node("executor", nodes.execute_node)
    workflow.add_node("error_router", nodes.error_router_node)
    workflow.add_node("repair_plan", nodes.repair_plan_node)
    workflow.add_node("repair_params", nodes.repair_params_node)
    workflow.add_node("repair_query", nodes.repair_query_node)
    workflow.add_node("retry", nodes.retry_node)
    workflow.add_node("reflect", nodes.reflect_node)

    workflow.set_entry_point("planner")

    def check_plan_status(state: AgentState):
        if state.status == "failed":
            return "reflect"
        return "continue"

    workflow.add_conditional_edges(
        "planner",
        check_plan_status,
        {"continue": "executor", "reflect": "reflect"},
    )

    workflow.add_edge("executor", "error_router")

    def route_after_execution(state: AgentState):
        max_repairs = int(config.get("repair.max_attempts", 2) or 2)
        if max_repairs < 0:
            max_repairs = 0

        if state.status == "completed":
            return "reflect"
        if state.status == "running":
            return "executor"
        if state.status != "failed":
            return "reflect"

        step_id = None
        if state.plan and state.plan.steps and state.current_step_index < len(state.plan.steps):
            step_id = state.plan.steps[state.current_step_index].step_id
        repair_count = 0
        for r in state.repair_history or []:
            if r.get("step_id") == step_id:
                repair_count += 1
        if repair_count >= max_repairs:
            return "reflect"

        if state.error_type == "tool_not_found":
            return "repair_plan"
        if state.error_type == "schema_error":
            return "repair_params"
        if state.error_type == "api_error":
            return "repair_query"
        if state.error_type == "network_error":
            return "retry"
        return "reflect"

    workflow.add_conditional_edges(
        "error_router",
        route_after_execution,
        {
            "executor": "executor",
            "repair_plan": "repair_plan",
            "repair_params": "repair_params",
            "repair_query": "repair_query",
            "retry": "retry",
            "reflect": "reflect",
        },
    )

    def check_repair_result(state: AgentState):
        if state.status == "repair_failed":
            return "reflect"
        return "executor"

    workflow.add_conditional_edges(
        "repair_plan",
        check_repair_result,
        {"executor": "executor", "reflect": "reflect"},
    )
    workflow.add_conditional_edges(
        "repair_params",
        check_repair_result,
        {"executor": "executor", "reflect": "reflect"},
    )
    workflow.add_conditional_edges(
        "repair_query",
        check_repair_result,
        {"executor": "executor", "reflect": "reflect"},
    )
    workflow.add_edge("retry", "executor")

    workflow.add_edge("reflect", END)

    app = workflow.compile()
    return app, tool_manager, memory_manager

def build_graph():
    """
    Builds the LangGraph StateGraph for the agent.
    """
    app, _, _ = build_graph_with_deps()
    return app
