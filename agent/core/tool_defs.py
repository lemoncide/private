from typing import Any, Dict, List, Tuple

from agent.tools.manager import ToolManager


def _prepare_tool_defs(tool_manager: ToolManager, objective: str, limit: int = 12) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    all_tools_raw = tool_manager.list_tools(limit=100)
    objective_lower = (objective or "").lower()

    core_tools = ["read_file", "write_file", "local_web_search", "calculator", "llm_reasoning"]
    keyword_map = {
        "git": ["git", "github"],
        "repo": ["git", "github"],
        "code": ["search_code", "read"],
        "csv": ["clean_data", "read_csv"],
        "data": ["clean_data", "read_csv"],
        "read": ["read"],
        "write": ["write"],
        "save": ["write"],
    }

    relevant_prefixes = set()
    for keyword, prefixes in keyword_map.items():
        if keyword in objective_lower:
            relevant_prefixes.update(prefixes)

    filtered_tools: List[Dict[str, Any]] = []
    for t in all_tools_raw:
        t_name = str(t.get("name", "")).lower()
        if any(core in t_name for core in core_tools):
            filtered_tools.append(t)
            continue
        if any(prefix in t_name for prefix in relevant_prefixes):
            filtered_tools.append(t)
            continue

    if len(filtered_tools) < 5:
        filtered_tools = all_tools_raw[:10]

    filtered_tools = filtered_tools[:limit]

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
