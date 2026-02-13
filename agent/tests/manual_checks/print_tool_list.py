import argparse
import json
import os
import sys
from typing import Any, Dict

sys.path.append(os.getcwd())

from agent.tools.manager import ToolManager


def _json_schema_for_tool(tool: Any) -> Dict[str, Any]:
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        return {}
    if isinstance(schema, dict):
        return schema
    if hasattr(schema, "model_json_schema"):
        return schema.model_json_schema()
    if hasattr(schema, "schema"):
        return schema.schema()
    return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    project_root = os.getcwd()
    local_packages = os.path.join(project_root, "local_packages")
    info = {
        "cwd": project_root,
        "python": sys.executable,
        "local_packages": {"path": local_packages, "exists": os.path.exists(local_packages)},
    }

    try:
        __import__("mcp")
        info["mcp_importable"] = True
    except Exception as e:
        info["mcp_importable"] = False
        info["mcp_import_error"] = str(e)

    manager = ToolManager()

    tools = []
    for name, tool in manager.tools.items():
        tools.append(
            {
                "name": name,
                "description": getattr(tool, "description", ""),
                "args_schema": _json_schema_for_tool(tool),
            }
        )

    payload = {"env": info, "tool_count": len(tools), "tools": sorted(tools, key=lambda t: t["name"])}
    text = json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(args.out)
    else:
        print(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
