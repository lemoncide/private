import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.llm.client import LLMClient
from agent.tools.manager import ToolManager


def main() -> int:
    if not os.getenv("GROQ_API_KEY"):
        print("Missing env var: GROQ_API_KEY")
        return 2

    llm = LLMClient()
    tool_manager = ToolManager()
    tool_defs = [
        {
            "name": "calculator",
            "description": "Evaluate a math expression.",
            "args_schema": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        }
    ]

    user_prompt = "Compute 2+2 using the calculator tool."
    result = llm.generate_with_tools(
        user_prompt,
        tool_defs=tool_defs,
        tool_choice={"type": "function", "function": {"name": "calculator"}},
        temperature=0.0,
    )

    print("=== tool selection ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("error"):
        return 1
    if not result.get("tool_calls"):
        return 3

    tool_messages = []
    for call in result["tool_calls"]:
        tool_name = call.get("name")
        args = call.get("arguments")
        if not tool_name or not isinstance(args, dict):
            print(f"Invalid tool call: {call}")
            return 4
        output = tool_manager.execute_tool(tool_name, **args)
        tool_messages.append({"tool_call_id": call.get("id"), "name": tool_name, "output": output})

    print("=== tool execution ===")
    print(json.dumps(tool_messages, ensure_ascii=False, indent=2))

    messages = [{"role": "user", "content": user_prompt}]
    messages.append({"role": "assistant", "content": None, "tool_calls": result.get("openai_tool_calls", [])})
    for tm in tool_messages:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tm["tool_call_id"],
                "content": str(tm["output"]),
            }
        )

    completion = llm.client.chat.completions.create(model=llm.model_name, messages=messages, temperature=0.0)
    final = completion.choices[0].message.content
    print("=== final answer ===")
    print(final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
