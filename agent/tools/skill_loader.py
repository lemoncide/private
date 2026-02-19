import os
import importlib.util
import inspect
from typing import List, Any, Callable, get_type_hints, Dict, Tuple
from langchain_core.tools import BaseTool
from pydantic import create_model, BaseModel, Field
import traceback

class SkillLoader:
    def __init__(self, skills_dir: str):
        self.skills_dir = skills_dir

    def load_skills(self) -> Tuple[List[BaseTool], Dict[str, Dict[str, str]]]:
        tools = []
        auto_mappings = {}
        
        if not os.path.exists(self.skills_dir):
            return tools, auto_mappings

        for filename in os.listdir(self.skills_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                module_name = filename[:-3]
                file_path = os.path.join(self.skills_dir, filename)
                
                try:
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # 1. Check for explicit get_tools
                    if hasattr(module, "get_tools"):
                        module_tools = module.get_tools()
                        for t in module_tools:
                            if isinstance(t, BaseTool):
                                tools.append(t)
                                self._generate_mappings_for_tool(t, auto_mappings)
                    
                    # 2. Auto-scan for functions
                    else:
                        for name, obj in inspect.getmembers(module):
                            if inspect.isfunction(obj) and obj.__module__ == module_name:
                                # Skip private functions
                                if name.startswith("_"):
                                    continue
                                    
                                # Convert function to Tool
                                tool = self._create_tool_from_func(obj)
                                tools.append(tool)
                                self._generate_mappings_for_tool(tool, auto_mappings)
                                
                except Exception as e:
                    print(f"Failed to load skill {module_name}: {e}")
                    traceback.print_exc()
        
        return tools, auto_mappings

    def _generate_mappings_for_tool(self, tool: BaseTool, mappings: Dict[str, Dict[str, str]]):
        """Generate automatic parameter mappings for a tool."""
        # Common semantic aliases
        semantic_aliases = {
            "path": ["file_path", "filename", "filepath"],
            "url": ["link", "address", "uri"],
            "query": ["q", "search_query", "keyword", "term"],
            "text": ["content", "body", "input_text"],
            "image": ["img", "photo", "picture", "image_path"]
        }
        
        tool_mappings = {}
        args_schema = getattr(tool, "args_schema", None)
        schema: Dict[str, Any] = {}
        if args_schema is None:
            return
        if isinstance(args_schema, dict):
            schema = args_schema
        elif hasattr(args_schema, "schema"):
            schema = args_schema.schema()
        elif hasattr(args_schema, "model_json_schema"):
            schema = args_schema.model_json_schema()
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        
        for param_name in properties.keys():
            # Check if param matches any semantic alias group
            for semantic_key, aliases in semantic_aliases.items():
                if param_name in aliases:
                    # Map semantic key -> actual param name
                    # Wait, mapping structure is {tool_name: {semantic_key: tool_param}}
                    # So if tool has param 'file_path', we map 'path' -> 'file_path'
                    tool_mappings[semantic_key] = param_name
                    
        if tool_mappings:
            mappings[tool.name] = tool_mappings

    def _create_tool_from_func(self, func: Callable) -> BaseTool:
        """Dynamically create a Tool class from a function."""
        tool_name = func.__name__
        tool_description = func.__doc__ or "No description available."
        
        # Parse type hints to create args_schema
        try:
            type_hints = get_type_hints(func)
        except Exception:
            type_hints = {}
            
        fields = {}
        for param_name, param in inspect.signature(func).parameters.items():
            if param_name == 'return':
                continue
            
            annotation = type_hints.get(param_name, Any)
            default = param.default if param.default is not inspect.Parameter.empty else ...
            fields[param_name] = (annotation, default)
            
        ArgsSchema = create_model(f"{tool_name}Args", **fields)
        
        # Create dynamic tool class
        # Use a factory function to capture closure variables correctly
        class DynamicTool(BaseTool):
            name: str = tool_name
            description: str = tool_description
            args_schema: type[BaseModel] = ArgsSchema
            
            def _run(self, **kwargs):
                return func(**kwargs)
                
        return DynamicTool()
