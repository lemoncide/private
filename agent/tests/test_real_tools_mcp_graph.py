import asyncio
import os
import sys
import shutil
import pytest

sys.path.append(os.getcwd())

from agent.core.agent import LangGraphAgent
from agent.core.planner import Planner
from agent.tools.manager import ToolManager
from agent.utils.config import config
from agent.utils.logger import logger

def test_mcp_graph_execution():
    if os.environ.get("RUN_INTEGRATION_TESTS") != "1":
        pytest.skip("integration test")

    print("Initializing Agent with MCP support...")
    
    # Ensure Mock Server Config is present
    server_script = os.path.join(os.getcwd(), "agent", "tests", "mock_mcp_server.py")
    python_exe = sys.executable
    
    config.update("mcp.servers", {
        "MockGitServer": {
            "command": python_exe,
            "args": [server_script],
            "env": os.environ.copy() # Pass current env to propagate PYTHONPATH
        }
    })
    
    # Initialize Agent
    agent = LangGraphAgent()
    
    # User Objective: Mixed Task (MCP + Local)
    # 1. Clone a repo (MCP: git_clone)
    # 2. Read the README (Local or MCP: read_file)
    objective = "Clone https://github.com/test/repo to /tmp/test_repo and read the file /tmp/test_repo/README.md"
    
    print(f"\nRunning Agent with objective: {objective}")
    
    try:
        # Run the agent (Async invocation required for LangGraph with async nodes)
        final_state = asyncio.run(agent.ainvoke(objective))
        
        print("\nExecution Completed.")
        print("Final State Keys:", final_state.keys())
        
        # Validation
        trace = final_state.get("trace", {})
        execution_history = trace.get("past_steps", [])
        print(f"Execution History Length: {len(execution_history)}")
        
        mcp_called = False
        read_called = False
        
        for step_result in execution_history:
            # step_result is a dict here because it came from state dump
            print(f"Step {step_result.get('step_id')}: {step_result.get('status')}")
            meta = step_result.get('meta', {})
            if meta:
                tool_name = meta.get("tool", "")
                print(f"  Tool: {tool_name}")
                if "mcp:MockGitServer:git_clone" in tool_name:
                    mcp_called = True
                if "read_file" in tool_name:
                    read_called = True
                    
        if mcp_called and read_called:
            print("\nSUCCESS: Both MCP git_clone and read_file were called.")
        else:
            print(f"\nFAILURE: MCP Called: {mcp_called}, Read Called: {read_called}")
            
    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mcp_graph_execution()
