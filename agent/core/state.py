from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from agent.core.schema import Plan, ExecutionResult

class AgentState(BaseModel):
    """
    Unified Agent State (Pydantic Model).
    Shared state across the LangGraph execution flow.
    """
    input: str = Field(description="The original user objective or query")
    
    plan: Optional[Plan] = Field(default=None, description="The current execution plan")
    
    current_step_index: int = Field(default=0, description="Index of the current step in the plan")
    
    past_steps: List[ExecutionResult] = Field(default_factory=list, description="History of executed steps and their results")
    
    context_variables: Dict[str, Any] = Field(default_factory=dict, description="Shared execution context variables (replacing raw ExecutionContext)")
    
    response: Optional[str] = Field(default=None, description="Final response to the user")
    
    status: str = Field(default="pending", description="Current agent status: pending, planning, executing, completed, failed, repaired, repair_failed")
    
    error: Optional[str] = Field(default=None, description="Current error message if any")
    
    scratchpad: List[Dict[str, Any]] = Field(default_factory=list, description="Intermediate reasoning or notes")

    class Config:
        arbitrary_types_allowed = True
