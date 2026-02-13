from typing import List, Dict, Optional
from agent.tools.base import BaseTool
from agent.tools.mcp_adapter import MCPAdapter
from agent.tools.skill_loader import SkillLoader
from agent.utils.config import config
from agent.utils.logger import logger
import os

from agent.memory.manager import MemoryManager

class ToolManager:
    def __init__(self, memory_manager: MemoryManager = None):
        self.tools: Dict[str, BaseTool] = {}
        self.mcp_adapter = MCPAdapter()
        self.skill_loader = SkillLoader(os.path.join(os.getcwd(), 'skills'))
        self.memory_manager = memory_manager
        self.auto_mappings: Dict[str, Dict[str, str]] = {}
        self._initialize_tools()

    def _initialize_tools(self):
        # 1. Load MCP Tools
        mcp_servers = config.get("mcp.servers", {})
        for name, cfg in mcp_servers.items():
            if isinstance(cfg, dict):
                 # New config format: { "command": "...", "args": [...] }
                 command = cfg.get("command")
                 args = cfg.get("args", [])
                 env = cfg.get("env", {})
                 if command:
                     self.mcp_adapter.connect_server(name, command, args, env)
            else:
                 # Legacy URL string format - ignoring for Stdio adapter
                 logger.warning(f"Skipping MCP server {name}: URL configuration not supported by Stdio adapter.")
        
        for tool in self.mcp_adapter.list_tools():
            # Apply Namespace for MCP tools
            if hasattr(tool, 'server_name'):
                 original_name = tool.name
                 # Format: mcp:server_name:tool_name
                 namespaced_name = f"mcp:{tool.server_name}:{original_name}"
                 tool.name = namespaced_name
            self.register_tool(tool)

        # 2. Load Skills with Auto-Mapping
        skills, mappings = self.skill_loader.load_skills()
        self.auto_mappings = mappings
        
        for tool in skills:
            self.register_tool(tool)
            
        logger.info(f"Total tools loaded: {len(self.tools)}")
        logger.info(f"Auto-mappings generated for: {list(self.auto_mappings.keys())}")

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

        # Support LangChain Tools
        result = []
        for t in tools_to_return:
            if hasattr(t, "to_dict"):
                result.append(t.to_dict())
            else:
                result.append({
                    "name": t.name,
                    "description": t.description
                })
        return result

    def execute_tool(self, name: str, **kwargs):
        tool = self.get_tool(name)
        if tool:
            logger.debug(f"Executing tool {name} (type: {type(tool)}) with args: {kwargs}")
            try:
                # Always call run with kwargs unpacked.
                # Our BaseTool is Pydantic, so it expects keywords or defined fields.
                # If run() is defined as run(self, **kwargs), it works.
                # If run() is defined as run(self, expression: str), unpacking works.
                return tool.run(**kwargs)
            except Exception as e:
                logger.error(f"Error executing tool {name}: {e}")
                logger.error(f"Tool run method: {getattr(tool, 'run', 'missing')}")
                # Don't try fallback with tool.run(kwargs) as that passes a dict as 1st pos arg,
                # which causes "BaseModel.__init__ takes 1 pos arg but 2 given" if 'run' somehow calls super init
                # or if the tool implementation is confused.
                # Just raise the original error to let Executor/Adaptation handle it.
                raise e
        return f"Tool {name} not found."
