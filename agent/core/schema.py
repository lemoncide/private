from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    step_id: str
    intent: str
    required_capability: str
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    output_var: str
    depends_on: List[str] = Field(default_factory=list)
    fallback_strategy: Literal["fail", "ask_llm", "skip"] = "fail"


class Plan(BaseModel):
    goal: str
    steps: List[PlanStep]
    final_output: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    step_id: str
    status: Literal["success", "failed"]
    result: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class ExecutionContext(BaseModel):
    variables: Dict[str, Any] = Field(default_factory=dict)

    def set(self, name: str, value: Any):
        self.variables[name] = value
