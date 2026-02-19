import argparse
import sys
import os
import asyncio
from contextlib import AsyncExitStack

# Ensure the current directory is in the path
sys.path.append(os.getcwd())

from agent.core.graph import build_graph
from agent.utils.logger import logger

async def run_task(task: str):
    from agent.core.graph import build_graph_with_deps
    app, tool_manager, _ = build_graph_with_deps()
    initial_state = {
        "input": task,
        "plan": None,
        "current_step_index": 0,
        "past_steps": [],
        "context_variables": {},
        "response": None,
        "status": "pending",
        "error": None,
        "scratchpad": [],
        "repair_attempts": 0,
    }

    async with AsyncExitStack() as stack:
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            from langchain_mcp_adapters.tools import load_mcp_tools
        except Exception:
            MultiServerMCPClient = None
            load_mcp_tools = None

        connections = tool_manager.get_mcp_connections() if hasattr(tool_manager, "get_mcp_connections") else {}
        if MultiServerMCPClient and load_mcp_tools and connections:
            client = MultiServerMCPClient(connections, tool_name_prefix=True)
            for server_name in connections.keys():
                session = await stack.enter_async_context(client.session(server_name))
                tools = await load_mcp_tools(session, server_name=server_name, tool_name_prefix=True)
                for tool in tools:
                    name = getattr(tool, "name", "") or ""
                    prefix = f"{server_name}_"
                    tool_name = name[len(prefix):] if name.startswith(prefix) else name
                    tool.name = f"mcp:{server_name}:{tool_name}"
                    tool_manager.register_tool(tool)

        final_state = await app.ainvoke(initial_state)
        return final_state


def main():
    parser = argparse.ArgumentParser(description="Agent CLI (LangGraph)")
    parser.add_argument("--task", type=str, help="The task for the agent to perform")
    parser.add_argument("--task-file", type=str, help="Read the task from a text file")
    
    args = parser.parse_args()
    
    task = args.task or "Calculate 10 + 5 and summarize."
    if args.task_file:
        with open(args.task_file, "r", encoding="utf-8") as f:
            task = f.read().strip() or task
    
    print(f"Starting Agent with LangGraph Workflow for task: {task}")

    try:
        final_state = asyncio.run(run_task(task))
        print("\n--- Final Result ---\n")
        print(final_state.get("response", "No response generated."))
        
        # Optional: Print history
        # print("\n--- Execution History ---")
        # for step in final_state.get("past_steps", []):
        #     print(f"Step: {step['step']} -> Result: {step['result']}")
            
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
