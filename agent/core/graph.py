from langgraph.graph import StateGraph, END
from agent.core.state import AgentState
from agent.core.nodes import AgentNodes
from agent.llm.client import LLMClient
from agent.tools.manager import ToolManager
from agent.memory.manager import MemoryManager

def build_graph():
    """
    Builds the LangGraph StateGraph for the agent.
    """
    # Initialize Dependencies
    llm_client = LLMClient()
    tool_manager = ToolManager()
    memory_manager = MemoryManager()
    
    nodes = AgentNodes(llm_client, tool_manager, memory_manager)
    
    # Initialize Graph
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("planner", nodes.plan_node)
    workflow.add_node("executor", nodes.execute_node)
    workflow.add_node("repair", nodes.repair_node)
    workflow.add_node("reflect", nodes.reflect_node)
    
    # Add Edges
    
    # 1. Start -> Planner
    workflow.set_entry_point("planner")
    
    # 2. Planner -> Executor (if success) or End (if failed)
    def check_plan_status(state: AgentState):
        if state.status == "failed":
            return "end" # Planning failed
        return "continue"
        
    workflow.add_conditional_edges(
        "planner",
        check_plan_status,
        {
            "continue": "executor",
            "end": END
        }
    )
    
    # 3. Executor -> (Repair, Reflect/End, Loop)
    def check_execution_status(state: AgentState):
        if state.status == "failed":
            return "repair"
        elif state.status == "completed":
            return "reflect"
        elif state.status == "running":
            return "loop"
        else:
            return "reflect" # Default
            
    workflow.add_conditional_edges(
        "executor",
        check_execution_status,
        {
            "repair": "repair",
            "reflect": "reflect",
            "loop": "executor"
        }
    )
    
    # 4. Repair -> (Executor, End)
    def check_repair_status(state: AgentState):
        if state.status == "executing": # Repaired and ready to retry
            return "retry"
        else: # Repair failed
            return "fail"
            
    workflow.add_conditional_edges(
        "repair",
        check_repair_status,
        {
            "retry": "executor",
            "fail": "reflect" # Go to reflect to report failure
        }
    )
    
    # 5. Reflect -> End
    workflow.add_edge("reflect", END)
    
    # Compile
    app = workflow.compile()
    return app
