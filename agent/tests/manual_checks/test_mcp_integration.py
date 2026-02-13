import sys
import os
import time
import pytest
# Add project root to sys.path
sys.path.append(os.getcwd())

from agent.tools.manager import ToolManager

def test_mcp_integration():
    if os.environ.get("RUN_INTEGRATION_TESTS") != "1":
        pytest.skip("integration test")

    print("Initializing ToolManager...")
    manager = ToolManager()
    
    # Manually connect to the mock server
    # We assume 'python' is in path and we are in project root
    server_script = os.path.join(os.getcwd(), "agent", "tests", "mock_mcp_server.py")
    
    if not os.path.exists(server_script):
        print(f"Error: Server script not found at {server_script}")
        return

    print(f"Connecting to mock server at {server_script}...")
    
    env = os.environ.copy()

    # Use the same python executable as we are running with
    python_exe = sys.executable

    manager.mcp_adapter.connect_server(
        name="MockGitServer",
        command=python_exe,
        args=[server_script],
        env=env
    )
    
    # Reload tools manually because connect_server was called after init
    # We need to manually register the new tools
    print("Listing tools...")
    tools = manager.mcp_adapter.list_tools()
    print(f"Found {len(tools)} tools from MCP adapter.")
    
    for tool in tools:
        if hasattr(tool, 'server_name'):
             tool.name = f"mcp:{tool.server_name}:{tool.name}"
        manager.register_tool(tool)
        
    # Check if tool exists
    tool_name = "mcp:MockGitServer:git_clone"
    print(f"Checking for {tool_name}...")
    tool = manager.get_tool(tool_name)
    
    if not tool:
        print("FAILED: Tool not found.")
        print("Available tools:", list(manager.tools.keys()))
        return
        
    print(f"Tool found: {tool.name}")
    print(f"Description: {tool.description}")
    print(f"Args Schema: {tool.args_schema}")
    
    # Execute tool
    print("Executing tool...")
    # Note: ToolManager.execute_tool unpacks kwargs.
    result = manager.execute_tool(tool_name, repo_url="https://github.com/test/repo", target_dir="/tmp/test")
    print(f"Result: {result}")
    
    if "Cloned https://github.com/test/repo into /tmp/test" in str(result):
        print("SUCCESS: Tool execution verified.")
    else:
        print("FAILED: Tool execution result incorrect.")

if __name__ == "__main__":
    test_mcp_integration()
