from typing import Any, Dict, List, Tuple

from agent.tools.manager import ToolManager

import logging
logger = logging.getLogger("Agent")

def _prepare_tool_defs(tool_manager: ToolManager, objective: str, limit: int = 12) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    all_tools_raw = tool_manager.list_tools(limit=500)
    all_names_in_order = [t.get("name") for t in all_tools_raw if t.get("name")]

    base_tool_candidates = [
        "write_file",
        "read_file",
        "llm_reasoning",
        "mcp:official_filesystem:read_text_file",
    ]
    base_tool_names = [n for n in base_tool_candidates if tool_manager.get_tool(n)]

    remaining = max(0, int(limit) - len(base_tool_names))
    if remaining >= 8:
        retrieval_k = min(10, remaining)
    else:
        retrieval_k = remaining

    retrieved_names: List[str] = []
    if objective and retrieval_k > 0:
        mm = getattr(tool_manager, "memory_manager", None)
        if mm:
            try:
                retrieved_names = mm.retrieve_tools(objective, limit=retrieval_k) or []
            except Exception:
                retrieved_names = []
        if not retrieved_names:
            try:
                retrieved_names = [t.get("name") for t in tool_manager.list_tools(query=objective, limit=retrieval_k) if t.get("name")]
            except Exception:
                retrieved_names = []
        logger.info(f"Semantic retrieved tools: {retrieved_names}")



    selected_names: List[str] = []
    for name in base_tool_names + retrieved_names:
        if name and name not in selected_names and tool_manager.get_tool(name):
            selected_names.append(name)

    if len(selected_names) < int(limit):
        for name in all_names_in_order:
            if len(selected_names) >= int(limit):
                break
            if name and name not in selected_names and tool_manager.get_tool(name):
                selected_names.append(name)

    enriched_tools: List[Dict[str, Any]] = []
    for tool_name in selected_names:
        tool_obj = tool_manager.get_tool(tool_name)
        if not tool_obj:
            continue
        item: Dict[str, Any] = {"name": getattr(tool_obj, "name", ""), "description": getattr(tool_obj, "description", "")}
        args_schema = getattr(tool_obj, "args_schema", None)
        if args_schema is not None:
            if isinstance(args_schema, dict):
                item["args_schema"] = args_schema
            elif hasattr(args_schema, "model_json_schema"):
                item["args_schema"] = args_schema.model_json_schema()
            elif hasattr(args_schema, "schema"):
                item["args_schema"] = args_schema.schema()
        enriched_tools.append(item)

    report: Dict[str, int] = {
        "total": len(all_tools_raw),
        "filtered": len(enriched_tools),
        "limit": int(limit),
        "base": len(base_tool_names),
        "retrieved": len([n for n in retrieved_names if n and n in selected_names]),
    }
    logger.info(f"Final selected tools ({len(enriched_tools)}): {[t.get('name') for t in enriched_tools]}")
    return enriched_tools, report


def prepare_tool_defs(tool_manager: ToolManager, objective: str, limit: int = 12) -> List[Dict[str, Any]]:
    tool_defs, _ = _prepare_tool_defs(tool_manager, objective, limit=limit)
    return tool_defs


def prepare_tool_defs_with_report(tool_manager: ToolManager, objective: str, limit: int = 12) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    return _prepare_tool_defs(tool_manager, objective, limit=limit)
