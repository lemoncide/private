import asyncio
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
        self._mcp_initialized = False
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
        if self.tool_manager:
            await self.tool_manager.aclose()

    async def ainvoke(self, query: str) -> Dict[str, Any]:
        """
        Run the agent asynchronously.
        """
        logger.info(f"Async Invoking Agent with query: {query}")
        initial_state = AgentState(input=query)
        
        # Initialize MCP tools once
        if not self._mcp_initialized:
            await self.tool_manager.init_mcp_tools()
            self._mcp_initialized = True

        try:
            result_state = await self.app.ainvoke(initial_state)
            
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
