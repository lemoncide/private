import asyncio
from contextlib import AsyncExitStack
from typing import Dict, Any, Optional
from agent.core.graph import build_graph_with_deps
from agent.core.state import AgentState
from agent.utils.logger import logger

class LangGraphAgent:
    """
    Main Agent class powered by LangGraph.
    """
    def __init__(self):
        logger.info("Initializing LangGraph Agent...")
        self.app, self.tool_manager, self.memory_manager = build_graph_with_deps()
        logger.info("Agent Graph compiled successfully.")

    def invoke(self, query: str) -> Dict[str, Any]:
        """
        Run the agent synchronously.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(query))
        return {"response": "System Error: invoke() cannot be used inside a running event loop; use await ainvoke().", "status": "system_error"}

    async def _maybe_connect_all(self):
        return

    async def _maybe_close_all(self):
        return

    async def ainvoke(self, query: str) -> Dict[str, Any]:
        """
        Run the agent asynchronously.
        """
        logger.info(f"Async Invoking Agent with query: {query}")
        initial_state = AgentState(input=query)
        
        async with AsyncExitStack() as stack:
            await self._maybe_connect_all()
            connections = self.tool_manager.get_mcp_connections() if hasattr(self.tool_manager, "get_mcp_connections") else {}
            try:
                from langchain_mcp_adapters.client import MultiServerMCPClient
                from langchain_mcp_adapters.tools import load_mcp_tools
            except Exception:
                MultiServerMCPClient = None
                load_mcp_tools = None

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
                        self.tool_manager.register_tool(tool)

            try:
                result_state = await self.app.ainvoke(initial_state.model_dump())
            
                response = result_state.get("response")
                error = result_state.get("error")
                status = result_state.get("status")
            
                if error:
                    return {"response": f"Error: {error}", "status": status, "trace": result_state}
            
                return {"response": response, "status": status, "trace": result_state}
            except Exception as e:
                logger.error(f"Agent execution failed: {e}")
                return {"response": f"System Error: {str(e)}", "status": "system_error"}
            finally:
                await self._maybe_close_all()
