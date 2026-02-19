from typing import List, Dict, Optional, Any
from langchain_core.tools import BaseTool
from agent.tools.skill_loader import SkillLoader
from agent.utils.config import config
from agent.utils.logger import logger
import os

from agent.memory.manager import MemoryManager

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except Exception:
    MultiServerMCPClient = None

class ToolManager:
    def __init__(self, memory_manager: MemoryManager = None):
        self.tools: Dict[str, BaseTool] = {}
        self.skill_loader = SkillLoader(os.path.join(os.getcwd(), 'skills'))
        self.memory_manager = memory_manager
        self.auto_mappings: Dict[str, Dict[str, str]] = {}
        self._mcp_tools_initialized = False
        self._mcp_client = None
        self._initialize_tools()

    def _initialize_tools(self):
        # 1. Load Skills with Auto-Mapping
        skills, mappings = self.skill_loader.load_skills()
        self.auto_mappings = mappings
        
        for tool in skills:
            self.register_tool(tool)
            
        logger.info(f"Total tools loaded: {len(self.tools)}")
        logger.info(f"Auto-mappings generated for: {list(self.auto_mappings.keys())}")

    async def init_mcp_tools(self):
        if self._mcp_tools_initialized:
            return
        if MultiServerMCPClient is None:
            logger.warning("langchain-mcp-adapters not installed; skipping MCP tool initialization.")
            return

        connections = self.get_mcp_connections()

        if not connections:
            self._mcp_tools_initialized = True
            return

        client = MultiServerMCPClient(connections, tool_name_prefix=True)
        tools = await client.get_tools()
        self._mcp_client = client

        for tool in tools:
            name = getattr(tool, "name", "") or ""
            matched_server = None
            matched_prefix_len = -1
            for server_name in connections.keys():
                prefix = f"{server_name}_"
                if name.startswith(prefix) and len(prefix) > matched_prefix_len:
                    matched_server = server_name
                    matched_prefix_len = len(prefix)
            if matched_server:
                tool_name = name[matched_prefix_len:]
                tool.name = f"mcp:{matched_server}:{tool_name}"
            self.register_tool(tool)

        self._mcp_tools_initialized = True
        logger.info(f"MCP tools loaded: {len(tools)}")

    def get_mcp_connections(self) -> Dict[str, Dict[str, Any]]:
        connections: Dict[str, Dict[str, Any]] = {}
        mcp_servers = config.get("mcp.servers", {}) or {}
        for server_name, cfg in mcp_servers.items():
            if not isinstance(cfg, dict):
                continue
            command = cfg.get("command")
            if not command:
                continue
            args = cfg.get("args", []) or []
            conn: Dict[str, Any] = {"transport": "stdio", "command": command, "args": args}
            env = cfg.get("env") or {}
            if isinstance(env, dict) and env:
                conn["env"] = env
            connections[server_name] = conn
        return connections

    def register_tool(self, tool: BaseTool):
        self.tools[tool.name] = tool
        if self.memory_manager:
            self.memory_manager.index_tool(tool.name, tool.description)
        logger.debug(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self.tools.get(name)

    def list_tools(self, query: str = None, limit: int = 5) -> List[Dict[str, str]]:
        # Modified to support retrieval
        if query and self.memory_manager:
            relevant_names = self.memory_manager.retrieve_tools(query, limit)
            tools_to_return = [self.tools[name] for name in relevant_names if name in self.tools]
            # Fallback if no relevant tools found or memory not ready
            if not tools_to_return:
                 tools_to_return = list(self.tools.values())[:limit]
        else:
            tools_to_return = list(self.tools.values())

        result = []
        for t in tools_to_return:
            item: Dict[str, Any] = {"name": getattr(t, "name", ""), "description": getattr(t, "description", "")}
            args_schema = getattr(t, "args_schema", None)
            if args_schema is not None:
                if isinstance(args_schema, dict):
                    item["args_schema"] = args_schema
                elif hasattr(args_schema, "schema"):
                    item["args_schema"] = args_schema.schema()
                elif hasattr(args_schema, "model_json_schema"):
                    item["args_schema"] = args_schema.model_json_schema()
            result.append(item)
        return result

    async def execute_tool(self, name: str, **kwargs):
        tool = self.get_tool(name)
        if tool:
            logger.debug(f"Executing tool {name} (type: {type(tool)}) with args: {kwargs}")
            try:
                return await tool.ainvoke(kwargs)
            except Exception as e:
                logger.error(f"Error executing tool {name}: {e}")
                logger.error(f"Tool run method: {getattr(tool, 'run', 'missing')}")
                # Don't try fallback with tool.run(kwargs) as that passes a dict as 1st pos arg,
                # which causes "BaseModel.__init__ takes 1 pos arg but 2 given" if 'run' somehow calls super init
                # or if the tool implementation is confused.
                # Just raise the original error to let Executor/Adaptation handle it.
                raise e
        return f"Tool {name} not found."
