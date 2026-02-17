from typing import Iterable, List, Optional, Set

from agent.core.schema import Plan


class ValidationResult:
    def __init__(self, valid: bool, errors: List[str]):
        self.valid = valid
        self.errors = errors


class PlanValidator:
    def validate(self, plan: Plan, available_tools: Optional[Iterable[str]] = None) -> ValidationResult:
        tool_set: Set[str] = set(available_tools or [])
        errors: List[str] = []

        defined_vars: Set[str] = {"input"}

        for step in plan.steps:
            if tool_set and step.required_capability not in tool_set:
                errors.append(
                    f"Step {step.step_id}: required_capability '{step.required_capability}' not in available tools"
                )

            for _, val in (step.tool_args or {}).items():
                if isinstance(val, str) and val.startswith("$"):
                    var_name = val[1:]
                    if var_name not in defined_vars:
                        errors.append(f"Step {step.step_id}: tool_args references undefined variable '{var_name}'")

            defined_vars.add(step.output_var)

        return ValidationResult(valid=len(errors) == 0, errors=errors)
