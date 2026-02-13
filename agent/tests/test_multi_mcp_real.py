import asyncio
import os
import sys
import shutil
import pytest

sys.path.append(os.getcwd())

from agent.core.agent import LangGraphAgent
from agent.utils.config import config
from agent.utils.logger import logger

def test_multi_mcp_real():
    if os.environ.get("RUN_INTEGRATION_TESTS") != "1":
        pytest.skip("integration test")

    print("Initializing Agent with Multi-MCP support...")
    
    # Define absolute paths for servers
    mock_git_script = os.path.join(os.getcwd(), "agent", "tests", "mock_mcp_server.py")
    real_fs_script = os.path.join(os.getcwd(), "agent", "servers", "filesystem_server.py")
    python_exe = sys.executable
    
    # Inject config dynamically to ensure paths are correct
    config.update("mcp.servers", {
        "MockGitServer": {
            "command": python_exe,
            "args": [mock_git_script],
            "env": os.environ.copy()
        },
        "RealFileSystem": {
            "command": python_exe,
            "args": [real_fs_script],
            "env": os.environ.copy()
        }
    })
    
    # Initialize Agent
    agent = LangGraphAgent()
    
    # Test Scenario:
    # 1. Clone a repo (using Mock Git Server)
    # 2. Write a NEW file to that repo (using Real FileSystem Server)
    #    - This tests implicit connection to multiple servers
    #    - This tests auto-mapping of 'write_file' arguments (path, content) without explicit config
    
    target_dir = os.path.join(os.getcwd(), "sandbox", "test_repo_multi")
    test_file = os.path.join(target_dir, "new_feature.txt")
    
    # Clean up before run
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
        
    objective = f"Clone https://github.com/test/repo to {target_dir}. Then create a file named {test_file} with content 'Feature X implemented'."
    
    print(f"\nRunning Agent with objective: {objective}")
    
    try:
        # Run the agent
        final_state = asyncio.run(agent.ainvoke(objective))
        
        print("\nExecution Completed.")
        
        # Validation
        trace = final_state.get("trace", {})
        execution_history = trace.get("past_steps", [])
        
        print(f"Execution History Length: {len(execution_history)}")
        
        git_called = False
        fs_called = False
        
        for step_result in execution_history:
            print(f"Step {step_result.get('step_id')}: {step_result.get('status')}")
            meta = step_result.get('meta', {})
            if meta:
                tool_name = meta.get("tool", "")
                print(f"  Tool: {tool_name}")
                if "mcp:MockGitServer:git_clone" in tool_name:
                    git_called = True
                # Note: write_file might be mapped to mcp:RealFileSystem:write_file
                if "mcp:RealFileSystem:write_file" in tool_name:
                    fs_called = True
                    
        if git_called and fs_called:
            print("\nSUCCESS: Both Git (Mock) and Filesystem (Real) servers were used.")
        else:
            print(f"\nFAILURE: Git Called: {git_called}, FS Called: {fs_called}")
            
        # Verify file actually exists (proving RealFileSystem worked)
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                content = f.read()
            print(f"File content verification: {content}")
            if content == 'Feature X implemented':
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
    test_multi_mcp_real()
