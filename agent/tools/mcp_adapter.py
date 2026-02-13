import asyncio
import os
import sys
import traceback
import time
from typing import List, Dict, Any, Optional
from agent.tools.base import BaseTool
from pydantic import PrivateAttr

# Try to import MCP, handle if missing
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError as e:
    print(f"MCP Import Error: {e}")
    MCP_AVAILABLE = False
    # Define dummy types for runtime to avoid NameError in type hints
    class StdioServerParameters: pass
    class ClientSession: pass
    def stdio_client(*args, **kwargs): pass

class MCPTool(BaseTool):
    """
    A tool that wraps an MCP tool.
    It holds a reference to the adapter to execute itself.
    """
    server_name: str
    tool_name: str
    _adapter: Any = PrivateAttr()

    def set_adapter(self, adapter):
        self._adapter = adapter

    def run(self, **kwargs):
        return self._adapter.execute_tool(self.server_name, self.tool_name, kwargs)

class MCPAdapter:
    def __init__(self):
        self.server_configs: Dict[str, StdioServerParameters] = {}
        self.max_retries = 3
        self.retry_delay = 1.0 # seconds
        if not MCP_AVAILABLE:
            print("Warning: MCP package not installed. MCP features will be disabled.")

    def connect_server(self, name: str, command: str, args: List[str], env: Optional[Dict] = None):
        """
        Register a server configuration.
        """
        if not MCP_AVAILABLE:
            return
            
        # --- FIX: Force environment synchronization & path resolution ---
        
        # 1. Force use of current Python interpreter if command is 'python'
        if command in ["python", "python3"]:
            command = sys.executable
            
        # 2. Resolve relative paths in args (e.g., script names) to absolute paths
        # This prevents "File not found" errors when CWD is different
        resolved_args = []
        for arg in args:
            if arg.endswith(".py") and not os.path.isabs(arg):
                # Assume relative to CWD, convert to absolute
                resolved_args.append(os.path.abspath(arg))
            else:
                resolved_args.append(arg)
                
        # 3. Inject current PYTHONPATH to ensure child process finds modules
        full_env = os.environ.copy()
        
        # Add current sys.path to PYTHONPATH
        # This ensures 'mcp' and other deps installed in current env are visible
        current_pythonpath = full_env.get("PYTHONPATH", "")
        additional_paths = [p for p in sys.path if p] # Filter empty strings
        new_pythonpath = os.pathsep.join(additional_paths + [current_pythonpath])
        full_env["PYTHONPATH"] = new_pythonpath
        
        # 4. Force unbuffered output to prevent IPC hangs
        full_env["PYTHONUNBUFFERED"] = "1"

        if env:
            full_env.update(env)
            
        self.server_configs[name] = StdioServerParameters(
            command=command,
            args=resolved_args,
            env=full_env
        )

    def list_tools(self) -> List[BaseTool]:
        """
        Connect to all registered servers and list their tools.
        """
        if not MCP_AVAILABLE:
            return []

        tools = []
        for name, config in self.server_configs.items():
            try:
                # Run async list_tools in a sync way
                server_tools = asyncio.run(self._fetch_tools(name, config))
                tools.extend(server_tools)
            except Exception as e:
                print(f"Error listing tools from {name}: {e}")
                # Don't crash entire listing if one server fails
                traceback.print_exc()
        return tools

    async def _fetch_tools(self, server_name: str, config: StdioServerParameters) -> List[BaseTool]:
        # Implement simple retry logic for connection
        last_error = None
        for attempt in range(self.max_retries):
            try:
                async with stdio_client(config) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.list_tools()
                        
                        mcp_tools = []
                for t in result.tools:
                    # Map MCP tool to BaseTool
                    # We pass the raw inputSchema to args_schema
                    
                    # --- ENHANCEMENT: Inject Hints into Description ---
                    description = t.description or ""
                    if t.name in ["get_file_contents", "list_directory", "read_file"]:
                        description += " (Rule: path must be local. Did you 'git_clone' first?)"
                        
                    tool = MCPTool(
                        name=t.name, 
                        description=description,
                        args_schema=t.inputSchema,
                        server_name=server_name,
                        tool_name=t.name
                    )
                    tool.set_adapter(self)
                    mcp_tools.append(tool)
                return mcp_tools
            except Exception as e:
                last_error = e
                print(f"Connection attempt {attempt+1} failed for {server_name}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt)) # Exponential backoff
        
        raise last_error

    def execute_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool on a specific server.
        Handles running inside an existing event loop or creating a new one.
        """
        config = self.server_configs.get(server_name)
        if not config:
            raise RuntimeError(f"MCP server not found: {server_name}")
            
        try:
            # Check if we are already in an event loop
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # We are in an async context (e.g., LangGraph ainvoke)
                # But execute_tool is sync. We need to run this async logic.
                # However, since we can't await here, and we can't use asyncio.run(),
                # this is tricky. 
                # Ideally execute_tool should be async, or ToolManager should handle it.
                # BUT, BaseTool.run is sync.
                # Solution: Use nest_asyncio OR run in a separate thread.
                # For simplicity/stability, let's use a thread pool executor to run the async function synchronously.
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._run_tool_with_retry(config, tool_name, arguments))
                    return future.result()
            else:
                return asyncio.run(self._run_tool_with_retry(config, tool_name, arguments))

        except Exception as e:
            raise RuntimeError(f"Error executing MCP tool {server_name}:{tool_name}: {e}") from e

    async def _run_tool_with_retry(self, config: StdioServerParameters, tool_name: str, arguments: Dict[str, Any]) -> Any:
        # Retry logic for execution
        last_error = None
        for attempt in range(self.max_retries):
            try:
                async with stdio_client(config) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool(tool_name, arguments)
                        
                        # Format result content
                        output = []
                        if hasattr(result, 'content'):
                            for content in result.content:
                                if content.type == 'text':
                                    output.append(content.text)
                                elif content.type == 'image':
                                    output.append(f"[Image: {content.mimeType}]")
                                elif content.type == 'resource':
                                    output.append(f"[Resource: {content.uri}]")
                        else:
                             # Fallback if structure is different
                             output.append(str(result))
                        
                        return "\n".join(output)
            except Exception as e:
                last_error = e
                print(f"Execution attempt {attempt+1} failed for {tool_name}: {e}")
                try:
                    if getattr(e, "exceptions", None):
                        for idx, sub in enumerate(getattr(e, "exceptions")):
                            print(f"  Sub-exception[{idx}]: {repr(sub)}")
                    else:
                        traceback.print_exc()
                except Exception:
                    traceback.print_exc()
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
        
        raise last_error
