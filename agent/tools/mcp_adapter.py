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
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(**kwargs))
        raise RuntimeError("MCPTool.run() cannot be used inside a running event loop; use await tool.arun(...)")

    async def arun(self, **kwargs):
        return await self._adapter.execute_tool(self.server_name, self.tool_name, kwargs)

class MCPAdapter:
    def __init__(self):
        self.server_configs: Dict[str, StdioServerParameters] = {}
        self.max_retries = 3
        self.retry_delay = 1.0 # seconds
        self._connections: Dict[str, Dict[str, Any]] = {}
        self._connect_locks: Dict[str, asyncio.Lock] = {}
        self._call_locks: Dict[str, asyncio.Lock] = {}
        self._closed = False
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

    async def execute_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        config = self.server_configs.get(server_name)
        if not config:
            raise RuntimeError(f"MCP server not found: {server_name}")
            
        try:
            return await self._run_tool_with_retry(server_name, tool_name, arguments)
        except Exception as e:
            raise RuntimeError(f"Error executing MCP tool {server_name}:{tool_name}: {e}") from e

    async def connect_all(self):
        if not MCP_AVAILABLE:
            return
        tasks = [self._ensure_connected(name) for name in self.server_configs.keys()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def aclose(self):
        self._closed = True
        names = list(self._connections.keys())
        for name in names:
            try:
                await self._close_server(name)
            except Exception:
                pass

    async def _close_server(self, server_name: str):
        conn = self._connections.pop(server_name, None)
        if not conn:
            return
        session_cm = conn.get("session_cm")
        client_cm = conn.get("client_cm")
        if session_cm is not None:
            try:
                await session_cm.__aexit__(None, None, None)
            except Exception:
                pass
        if client_cm is not None:
            try:
                await client_cm.__aexit__(None, None, None)
            except Exception:
                pass

    def _get_connect_lock(self, server_name: str) -> asyncio.Lock:
        lock = self._connect_locks.get(server_name)
        if lock is None:
            lock = asyncio.Lock()
            self._connect_locks[server_name] = lock
        return lock

    def _get_call_lock(self, server_name: str) -> asyncio.Lock:
        lock = self._call_locks.get(server_name)
        if lock is None:
            lock = asyncio.Lock()
            self._call_locks[server_name] = lock
        return lock

    async def _ensure_connected(self, server_name: str):
        if self._closed:
            raise RuntimeError("MCPAdapter is closed")
        if server_name in self._connections:
            return
        config = self.server_configs.get(server_name)
        if not config:
            raise RuntimeError(f"MCP server not found: {server_name}")
        lock = self._get_connect_lock(server_name)
        async with lock:
            if server_name in self._connections:
                return
            client_cm = stdio_client(config)
            read, write = await client_cm.__aenter__()
            session_cm = ClientSession(read, write)
            session = await session_cm.__aenter__()
            await session.initialize()
            self._connections[server_name] = {
                "client_cm": client_cm,
                "session_cm": session_cm,
                "session": session,
            }

    async def _call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        await self._ensure_connected(server_name)
        conn = self._connections.get(server_name)
        if not conn:
            raise RuntimeError(f"MCP server not connected: {server_name}")
        session = conn.get("session")
        if session is None:
            raise RuntimeError(f"MCP session not available: {server_name}")
        call_lock = self._get_call_lock(server_name)
        async with call_lock:
            return await session.call_tool(tool_name, arguments)

    async def _run_tool_with_retry(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Any:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                result = await self._call_tool(server_name, tool_name, arguments)
                output = []
                if hasattr(result, "content"):
                    for content in result.content:
                        if content.type == "text":
                            output.append(content.text)
                        elif content.type == "image":
                            output.append(f"[Image: {content.mimeType}]")
                        elif content.type == "resource":
                            output.append(f"[Resource: {content.uri}]")
                else:
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
                    try:
                        await self._close_server(server_name)
                    except Exception:
                        pass
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
        
        raise last_error
