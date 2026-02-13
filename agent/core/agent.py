from typing import Dict, Any, Optional
from agent.core.graph import build_graph
from agent.core.state import AgentState
from agent.utils.logger import logger

class LangGraphAgent:
    """
    Main Agent class powered by LangGraph.
    """
    def __init__(self):
        logger.info("Initializing LangGraph Agent...")
        self.app = build_graph()
        logger.info("Agent Graph compiled successfully.")

    def invoke(self, query: str) -> Dict[str, Any]:
        """
        Run the agent synchronously.
        """
        logger.info(f"Invoking Agent with query: {query}")
        
        # Initialize State
        initial_state = AgentState(input=query)
        
        # Execute Graph
        try:
            # LangGraph's invoke expects a dict or state object. 
            # Since we used Pydantic, we can pass the model or dict.
            # Passing dict is safer for compatibility.
            result_state = self.app.invoke(initial_state.model_dump())
            
            # Result is a dict (state snapshot)
            response = result_state.get("response")
            error = result_state.get("error")
            status = result_state.get("status")
            
            if error:
                return {"response": f"Error: {error}", "status": status, "trace": result_state}
            
            return {"response": response, "status": status, "trace": result_state}
            
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return {"response": f"System Error: {str(e)}", "status": "system_error"}

    async def ainvoke(self, query: str) -> Dict[str, Any]:
        """
        Run the agent asynchronously.
        """
        logger.info(f"Async Invoking Agent with query: {query}")
        
        initial_state = AgentState(input=query)
        
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
