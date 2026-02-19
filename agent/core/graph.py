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
    workflow.add_node("repair", nodes.repair_node)
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

    def check_execution_status(state: AgentState):
        max_attempts = int(config.get("repair.max_attempts", 2) or 2)
        if max_attempts < 0:
            max_attempts = 0
        if state.status == "completed":
            return "reflect"
        elif state.status == "running":
            return "loop"
        elif state.status == "failed" and state.repair_attempts < max_attempts:
            return "repair"
        else:
            return "reflect"

    workflow.add_conditional_edges(
        "executor",
        check_execution_status,
        {"reflect": "reflect", "loop": "executor", "repair": "repair"},
    )

    def check_repair_status(state: AgentState):
        if state.status == "repair_failed":
            return "reflect"
        return "continue"

    workflow.add_conditional_edges(
        "repair",
        check_repair_status,
        {"continue": "executor", "reflect": "reflect"},
    )

    workflow.add_edge("reflect", END)

    app = workflow.compile()
    return app, tool_manager, memory_manager

def build_graph():
    """
    Builds the LangGraph StateGraph for the agent.
    """
    app, _, _ = build_graph_with_deps()
    return app
