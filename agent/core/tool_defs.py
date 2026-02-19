from typing import Any, Dict, List, Tuple

from agent.tools.manager import ToolManager


def _prepare_tool_defs(tool_manager: ToolManager, objective: str, limit: int = 12) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    all_tools_raw = tool_manager.list_tools(limit=500)

    def is_mcp_tool(tool: Dict[str, Any]) -> bool:
        name = str(tool.get("name", "") or "")
        return name.lower().startswith("mcp:")

    def is_local_skill_tool(tool: Dict[str, Any]) -> bool:
        name = str(tool.get("name", "") or "")
        return ":" not in name

    always_keep: List[Dict[str, Any]] = []
    for t in all_tools_raw:
        if is_mcp_tool(t) or is_local_skill_tool(t):
            always_keep.append(t)

    retrieved: List[Dict[str, Any]] = []
    if objective:
        retrieved = tool_manager.list_tools(query=objective, limit=limit)

    merged_by_name: Dict[str, Dict[str, Any]] = {}
    for t in always_keep + retrieved:
        name = t.get("name")
        if name and name not in merged_by_name:
            merged_by_name[name] = t

    filtered_tools = list(merged_by_name.values())

    if len(filtered_tools) < 5:
        filtered_tools = all_tools_raw[: max(10, limit)]

    enriched_tools: List[Dict[str, Any]] = []
    for t_dict in filtered_tools:
        tool_name = t_dict.get("name")
        tool_obj = tool_manager.get_tool(tool_name) if tool_name else None
        if tool_obj and hasattr(tool_obj, "args_schema"):
            if isinstance(tool_obj.args_schema, dict):
                t_dict["args_schema"] = tool_obj.args_schema
            elif hasattr(tool_obj.args_schema, "schema"):
                t_dict["args_schema"] = tool_obj.args_schema.schema()
            else:
                t_dict["args_schema"] = tool_obj.args_schema
        enriched_tools.append(t_dict)

    return enriched_tools, {"total": len(all_tools_raw), "filtered": len(filtered_tools), "limit": limit}


def prepare_tool_defs(tool_manager: ToolManager, objective: str, limit: int = 12) -> List[Dict[str, Any]]:
    tool_defs, _ = _prepare_tool_defs(tool_manager, objective, limit=limit)
    return tool_defs


def prepare_tool_defs_with_report(tool_manager: ToolManager, objective: str, limit: int = 12) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    return _prepare_tool_defs(tool_manager, objective, limit=limit)
