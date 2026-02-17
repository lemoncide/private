from agent.llm.client import LLMClient


def test_to_openai_tools_converts_tool_defs():
    tool_defs = [
        {
            "name": "calculator",
            "description": "Evaluate an expression",
            "args_schema": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        }
    ]

    tools = LLMClient._to_openai_tools(tool_defs)
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "calculator"
    assert tools[0]["function"]["parameters"]["properties"]["expression"]["type"] == "string"
    assert "$schema" not in tools[0]["function"]["parameters"]
    assert tools[0]["function"]["parameters"]["additionalProperties"] is False


def test_to_openai_tools_keeps_openai_format():
    tool_defs = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            },
        }
    ]
    tools = LLMClient._to_openai_tools(tool_defs)
    assert tools == tool_defs

