from typing import List, Dict, Any
from agent.core.schema import Plan, PlanStep, VariableRef, VariableBinding, OperationType
from agent.utils.logger import logger

class ValidationResult:
    def __init__(self, valid: bool, errors: List[str]):
        self.valid = valid
        self.errors = errors
        self.warnings = []

class PlanValidator:
    """Validates Plan structure and dependencies"""
    
    def validate(self, plan: Plan) -> ValidationResult:
        errors = []
        
        # 1. Variable Dependencies
        errors.extend(self._check_variable_dependencies(plan))

        # 1.5 Cross-domain binding sanity checks
        errors.extend(self._check_cross_domain_bindings(plan))

        # 1.6 Tool binding sanity checks
        errors.extend(self._check_tool_bindings(plan))
        
        # 2. Dependency Cycles
        errors.extend(self._check_dependency_cycles(plan))
        
        # 3. Duplicate Outputs
        errors.extend(self._check_duplicate_outputs(plan))
        
        # 4. Literal Values
        errors.extend(self._check_literals(plan))
        
        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def _check_tool_bindings(self, plan: Plan) -> List[str]:
        errors: List[str] = []
        generic_caps = {"search", "read", "write", "reasoning", "filesystem", "transform", "compute"}

        for step in plan.steps:
            cap = (step.required_capability or "").strip()
            if not cap:
                errors.append(f"Step {step.step_id}: required_capability is missing")
                continue
            if cap.lower() in generic_caps:
                errors.append(f"Step {step.step_id}: required_capability '{cap}' is too generic; must be a concrete tool name")

        return errors

    def _check_cross_domain_bindings(self, plan: Plan) -> List[str]:
        errors: List[str] = []

        producer_type_by_key: Dict[str, str] = {}
        for step in plan.steps:
            op_type = step.operation.type
            op_name = op_type.value if isinstance(op_type, OperationType) else str(op_type)
            producer_type_by_key[step.step_id] = op_name
            if step.output_var:
                producer_type_by_key[step.output_var] = op_name

        for step in plan.steps:
            step_op = step.operation.type
            step_op_name = step_op.value if isinstance(step_op, OperationType) else str(step_op)
            if step_op_name not in {"read", "write"}:
                continue

            for param_name, binding in step.operation.inputs.items():
                if param_name not in {"path", "file_path"}:
                    continue
                if binding.value_ref.source != "step_output":
                    continue

                ref_name = binding.value_ref.name
                producer_op = producer_type_by_key.get(ref_name)
                if producer_op == "search":
                    errors.append(
                        f"Step {step.step_id}: '{param_name}' is bound to a SEARCH output '{ref_name}', which is unlikely to be a valid path"
                    )

        return errors

    def _check_literals(self, plan: Plan) -> List[str]:
        errors = []
        for step in plan.steps:
            for param_name, binding in step.operation.inputs.items():
                if binding.value_ref.source == "literal":
                    # Check if name (which holds the value) is present
                    if not binding.value_ref.name:
                        errors.append(f"Step {step.step_id}: Literal input '{param_name}' is empty")
        return errors

    def _check_variable_dependencies(self, plan: Plan) -> List[str]:
        errors = []
        available_vars = set(plan.initial_inputs.keys())
        # Also add "context" variables if any are predefined
        
        step_outputs = set()
        step_ids = set()
        
        # Logging for debug
        logger.debug(f"Validating dependencies. Initial inputs: {available_vars}")
        
        for step in plan.steps:
            logger.debug(f"Checking Step {step.step_id}. Known outputs: {step_outputs}. Known IDs: {step_ids}")
            
            # Check inputs
            for param_name, binding in step.operation.inputs.items():
                ref = binding.value_ref
                source = ref.source
                name = ref.name
                
                if source == "literal":
                    continue
                
                if source == "input":
                    if name not in available_vars:
                        errors.append(f"Step {step.step_id}: Input '{param_name}' references unknown initial input '{name}'")
                elif source == "step_output":
                    # V8 Fix: Support referencing by step_id OR output_var
                    # Also normalize for robustness (case-insensitive check for IDs?)
                    found = False
                    
                    if name in step_outputs or name in step_ids:
                        found = True
                    else:
                        # Try case-insensitive match for IDs
                        name_lower = name.lower()
                        if any(s.lower() == name_lower for s in step_ids):
                            found = True
                            
                    if not found:
                         errors.append(f"Step {step.step_id}: Input '{param_name}' references unknown step output '{name}'")
                         logger.warning(f"Validation failed for '{name}'. Current outputs: {step_outputs}, Current IDs: {step_ids}")
                         
                elif source == "context":
                    # Assume context variables are valid for now or check against schema
                    pass

            # Register output
            if step.output_var:
                step_outputs.add(step.output_var)
            step_ids.add(step.step_id)
            
        # Check final output
        # Final output should be one of the produced variables or inputs
        # But sometimes it might be a step_id too? Let's keep it strict to var name for now as per schema
        if plan.final_output not in step_outputs and plan.final_output not in available_vars:
             # Relax check: maybe it's a step_id?
             if plan.final_output not in step_ids:
                 # Try case-insensitive match
                 if not any(s.lower() == plan.final_output.lower() for s in step_ids):
                    errors.append(f"Final output '{plan.final_output}' is not produced by any step or input")
            
        return errors

    def _check_dependency_cycles(self, plan: Plan) -> List[str]:
        # Build graph
        graph = {s.step_id: s.depends_on for s in plan.steps}
        visited = set()
        stack = set()
        
        def has_cycle(node):
            visited.add(node)
            stack.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor): return True
                elif neighbor in stack:
                    return True
            stack.remove(node)
            return False
            
        for step in plan.steps:
            if step.step_id not in visited:
                if has_cycle(step.step_id):
                    return [f"Cycle detected involving step {step.step_id}"]
        return []

    def _check_duplicate_outputs(self, plan: Plan) -> List[str]:
        seen = set()
        errors = []
        for step in plan.steps:
            if step.output_var in seen:
                errors.append(f"Duplicate output variable '{step.output_var}' in step {step.step_id}")
            seen.add(step.output_var)
        return errors
