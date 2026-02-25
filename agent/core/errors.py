from typing import List, Optional


class ToolNotFoundError(Exception):
    def __init__(self, tool_name: str):
        super().__init__(f"Tool '{tool_name}' not found")
        self.tool_name = tool_name


class PlanToolNotFoundError(ValueError):
    def __init__(self, message: str, missing_tools: Optional[List[str]] = None, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.missing_tools = missing_tools or []
        self.errors = errors or []

