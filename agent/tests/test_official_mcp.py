import asyncio
import os
import sys
import shutil
import pytest

sys.path.append(os.getcwd())

from agent.core.agent import LangGraphAgent
from agent.utils.config import config
from agent.utils.logger import logger

def test_official_mcp():
    if os.environ.get("RUN_INTEGRATION_TESTS") != "1":
        pytest.skip("integration test")

    print("Initializing Agent with Official MCP support...")
    
    real_fs_script = os.path.join(os.getcwd(), "agent", "servers", "filesystem_server.py")
    python_exe = sys.executable
    config.update("mcp.servers", {
        "official_filesystem": {
            "command": python_exe,
            "args": [real_fs_script],
            "env": os.environ.copy()
        }
    })

    agent = LangGraphAgent()
    
    # Test Scenario:
    # 1. Write a file using the Official Filesystem Server
    # 2. Read the file back
    # 3. Use a local skill (count_words) on the content
    
    target_dir = os.path.join(os.getcwd(), "sandbox")
    test_file = os.path.join(target_dir, "official_test.txt")
    
    # Ensure sandbox exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    # Clean up before run
    if os.path.exists(test_file):
        os.remove(test_file)
        
    content_to_write = "This is a test of the official MCP filesystem server. It works!"
    
    objective = f"Create a file at {test_file} with content '{content_to_write}'. Then read it back and count the words in it."
    
    print(f"\nRunning Agent with objective: {objective}")
    
    try:
        # Run the agent
        final_state = asyncio.run(agent.ainvoke(objective))
        
        print("\nExecution Completed.")
        
        # Validation
        trace = final_state.get("trace", {})
        execution_history = trace.get("past_steps", [])
        
        print(f"Execution History Length: {len(execution_history)}")
        
        write_called = False
        read_called = False
        count_called = False
        
        for step_result in execution_history:
            print(f"Step {step_result.get('step_id')}: {step_result.get('status')}")
            meta = step_result.get('meta', {})
            if meta:
                tool_name = meta.get("tool", "")
                print(f"  Tool: {tool_name}")
                # Check for official server name in tool string
                if "official_filesystem" in tool_name:
                    if "write" in tool_name: write_called = True
                    if "read" in tool_name: read_called = True
                if "count_words" in tool_name:
                    count_called = True
                    
        if write_called and read_called:
            print("\nSUCCESS: Official MCP Filesystem Server was used for both read and write.")
        else:
            print(f"\nFAILURE: Write Called: {write_called}, Read Called: {read_called}")
            
        if count_called:
            print("SUCCESS: Local skill (count_words) was used.")
        else:
            print("FAILURE: Local skill was not used.")
            
        # Verify file actually exists
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                content = f.read()
            print(f"File content verification: {content}")
            if content == content_to_write:
                print("SUCCESS: File content matches.")
            else:
                print("FAILURE: File content mismatch.")
        else:
            print(f"FAILURE: File {test_file} was NOT created.")
            
    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_official_mcp()
