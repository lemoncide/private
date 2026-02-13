from typing import List, Dict, Any, Optional, Literal, Union
from pydantic import BaseModel, Field
from enum import Enum

# ============= Data Flow & Variables =============

class VariableRef(BaseModel):
    """Reference to a variable in the execution context"""
    source: Literal["input", "step_output", "context", "literal"]
    name: str
    path: Optional[str] = None # JSON path e.g. "data.items[0]"

    def __str__(self):
        if self.source == "literal":
            return f"'{self.name}'"
        base = f"${self.source}.{self.name}"
        return f"{base}.{self.path}" if self.path else base

class VariableBinding(BaseModel):
    """Binding an input parameter to a variable reference"""
    # Make param_name optional as it's redundant with the key in OperationSpec.inputs
    param_name: Optional[str] = None 
    value_ref: VariableRef
    transform: Optional[str] = None # e.g. "to_float", "extract_first"

# ============= Operation Semantics =============

class OperationType(str, Enum):
    """Semantic Operation Types"""
    # Data Ops
    COMPUTE = "compute"
    TRANSFORM = "transform"
    EXTRACT = "extract"
    
    # Interaction Ops
    SEARCH = "search"
    READ = "read"
    WRITE = "write"
    CALL_API = "call_api"
    
    # Logic Ops
    CONDITION = "condition"
    AGGREGATE = "aggregate"
    
    # LLM Ops
    REASON = "reason"
    GENERATE = "generate"

class OperationSpec(BaseModel):
    """Specification of an operation (The 'What')"""
    type: OperationType
    description: Optional[str] = None
    
    # Inputs with explicit bindings
    inputs: Dict[str, VariableBinding] = Field(
        default_factory=dict,
        description="Map of semantic parameter names to variable bindings"
    )

    
    # Config (non-data parameters)
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration like precision, timeout, etc."
    )

# ============= High-Level Plan =============

class PlanStep(BaseModel):
    """A single step in the high-level plan"""
    step_id: str
    intent: str
    
    operation: OperationSpec
    
    # Outputs
    output_var: Optional[str] = None
    output_type: Optional[str] = None
    
    # Dependencies
    depends_on: List[str] = Field(default_factory=list)
    
    # Constraints
    required_capability: Optional[str] = None
    constraints: Dict[str, Any] = Field(default_factory=dict)
    
    # V8: Certainty & Fallback
    certainty: Literal["certain", "uncertain"] = Field(
        default="certain",
        description="Is this step's execution path deterministic?"
    )
    fallback_strategy: Literal["fail", "ask_llm", "skip"] = Field(
        default="fail",
        description="What to do if execution fails or is uncertain"
    )

class Plan(BaseModel):
    """Full Execution Plan"""
    goal: str
    steps: List[PlanStep]
    initial_inputs: Dict[str, Any] = Field(default_factory=dict)
    final_output: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

# ============= Low-Level Execution =============

class ToolCallSpec(BaseModel):
    """Concrete Tool Call Specification"""
    tool_name: str
    tool_type: Literal["mcp", "local", "llm"]
    arguments: Dict[str, Any]
    timeout: int = 30
    retry_config: Dict[str, int] = Field(
        default_factory=lambda: {"max_retries": 3, "backoff": 2}
    )

class ExecutionResult(BaseModel):
    step_id: str
    status: str # "success", "failed"
    error_type: Optional[str] = None # "binding_error", "tool_not_found", "schema_error", "logic_error"
    result: Any = None
    error: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

class ExecutableAction(BaseModel):
    """Action ready for execution"""
    step_id: str
    original_operation: OperationSpec
    binding_trace: Dict[str, Any] # Trace of parameter resolution
    tool_call: ToolCallSpec
    validation_rules: Dict[str, Any] = Field(default_factory=dict)

# from jsonpath_ng import parse as jsonpath_parse

# ============= Runtime Context =============

class ExecutionContext(BaseModel):
    """Runtime storage for variables"""
    variables: Dict[str, Any] = Field(default_factory=dict)
    
    def set(self, name: str, value: Any):
        self.variables[name] = value
        
    def get(self, ref: VariableRef) -> Any:
        if ref.source == "literal":
            return ref.name
            
        key = f"{ref.source}.{ref.name}"
        value = self.variables.get(key)
        
        if value is None:
            # Try simple variable name if structured key fails (for inputs)
            if ref.source == "input":
                value = self.variables.get(ref.name)
            
            # V8 Fix: If still None, return None (Adaptation will handle it)
            if value is None:
                return None

        if ref.path:
            # Simple path resolution without external lib for now to avoid dependency issues in this env
            try:
                parts = ref.path.split(".")
                curr = value
                for part in parts:
                    if isinstance(curr, dict):
                        curr = curr.get(part)
                    elif isinstance(curr, list) and part.isdigit():
                        curr = curr[int(part)]
                    else:
                        return None
                return curr
            except:
                return None
                
        return value

    def resolve_binding(self, binding: VariableBinding) -> Any:
        value = self.get(binding.value_ref)
        if binding.transform:
            value = self._apply_transform(value, binding.transform)
        return value

    def _apply_transform(self, value: Any, transform: str) -> Any:
        # Simple transform registry
        transforms = {
            "to_string": str,
            "to_int": int,
            "to_float": float,
            "len": len,
            "first": lambda x: x[0] if isinstance(x, (list, tuple)) and x else None
        }
        if transform in transforms:
            try:
                return transforms[transform](value)
            except:
                return value
        return value
