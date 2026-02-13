import sys
import os
import asyncio

# Ensure the current directory is in the path
sys.path.append(os.getcwd())

from agent.core.nodes import AgentNodes
from agent.llm.client import LLMClient
from agent.tools.manager import ToolManager
from agent.memory.manager import MemoryManager
# Fix: Import ExecutionResult for type hinting or logic if needed, 
# but Executor returns it, so nodes.py imports it.
# We don't necessarily need it here unless we use it.


from agent.tools.base import BaseTool

# Define Mock tools outside to avoid scope issues
class Calculator(BaseTool):
    name: str = "calculator"
    description: str = "Performs basic arithmetic operations. Inputs: expression (str)."
    
    def run(self, **kwargs):
        expr = kwargs.get("expression")
        if not expr:
             # Check for semantic args
             op1 = kwargs.get("operand1")
             op2 = kwargs.get("operand2")
             operator = kwargs.get("op")
             
             if op1 is not None and op2 is not None:
                 if operator == "add": return float(op1) + float(op2)
                 if operator == "mul": return float(op1) * float(op2)
                 return "Unknown op"
                 
             return f"Missing args. Received: {kwargs}"
        return eval(str(expr))

class GoogleSearch(BaseTool):
    name: str = "google_search"
    description: str = "Search the web."
    
    def run(self, **kwargs):
        # Mapping: query -> search_query
        q = kwargs.get("search_query")
        if q:
            return f"Search results for '{q}'"
        return f"Missing 'search_query'. Received: {kwargs}"

async def test_integration():
    print("--- Testing V8 Integration ---")
    
    # 1. Initialize Nodes
    llm = LLMClient()
    
    # Use real ToolManager but register our mocks manually
    tools = ToolManager()
    
    # Clear auto-loaded tools to focus on test
    tools.tools = {}
    tools.register_tool(Calculator())
    tools.register_tool(GoogleSearch())
    
    memory = MemoryManager()
    nodes = AgentNodes(llm, tools, memory)
    
    # 2. Test Plan Node (Semantic Planning)
    print("\n--- Testing Planner ---")
    state = {
        "input": "Calculate 50 * 3",
        "plan": [],
        "current_step_index": 0,
        "past_steps": [],
        "response": None,
        "status": "pending"
    }
    
    # We expect Planner to generate a step with semantic args: a, b, operation (or similar based on prompt)
    state = nodes.plan_node(state)
    
    if state["status"] == "failed":
        print("Planning failed.")
        return

    print(f"Plan: {state['plan']}")
    
    # 3. Test Execute Node (Mapping & Execution)
    print("\n--- Testing Executor ---")
    # Execute the step
    state = await nodes.execute_node_async(state)
    
    print(f"Status: {state['status']}")
    print(f"History: {state.get('past_steps')}")
    print(f"Error: {state.get('error')}")
    
    # 4. Test Reflection (Completion)
    print("\n--- Testing Reflection ---")
    state = nodes.reflect_node(state)
    print(f"Final Response: {state.get('response')}")

if __name__ == "__main__":
    asyncio.run(test_integration())
