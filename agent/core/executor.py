from typing import Dict, Any, Optional
import yaml
import os
from openai import OpenAI
from pydantic import BaseModel, Field
from agent.core.schema import (
    PlanStep, ExecutableAction, ToolCallSpec, 
    ExecutionContext, OperationType, VariableRef, VariableBinding, ExecutionResult
)
from agent.tools.manager import ToolManager
from agent.utils.config import config
from agent.utils.logger import logger
import json
from agent.core.repair import ExecutionRepairer
from agent.llm.cot_parser import extract_thinking, parse_structured, CoTParseError
import re


def _normalize_github_owner_repo(semantic_args: Dict[str, Any]) -> Dict[str, Any]:
    args = dict(semantic_args or {})
    if args.get("owner") and args.get("repo"):
        return args

    repo_value = args.get("repo") or args.get("full_name") or args.get("repo_full_name")
    if isinstance(repo_value, str):
        repo_value = repo_value.strip()

        match = re.search(r"github\.com/([^/]+)/([^/#?]+)", repo_value)
        if match and not (args.get("owner") and args.get("repo")):
            args["owner"] = match.group(1)
            args["repo"] = match.group(2).removesuffix(".git")
            return args

        if repo_value.startswith("{") or repo_value.startswith("["):
            try:
                parsed = json.loads(repo_value)
            except Exception:
                parsed = None
            if parsed is not None:
                repo_value = parsed
        elif "/" in repo_value and not (args.get("owner") and args.get("repo")) and "://" not in repo_value:
            owner, repo = repo_value.split("/", 1)
            if owner and repo:
                args["owner"] = owner
                args["repo"] = repo
                return args

    if isinstance(repo_value, list) and repo_value:
        repo_value = repo_value[0]

    if isinstance(repo_value, dict) and not (args.get("owner") and args.get("repo")):
        full_name = repo_value.get("full_name")
        if isinstance(full_name, str) and "/" in full_name:
            owner, repo = full_name.split("/", 1)
            args["owner"] = args.get("owner") or owner
            args["repo"] = repo
            return args

        owner_login = None
        owner_obj = repo_value.get("owner")
        if isinstance(owner_obj, dict):
            owner_login = owner_obj.get("login") or owner_obj.get("name")
        repo_name = repo_value.get("name") or repo_value.get("repo")
        if owner_login and repo_name:
            args["owner"] = args.get("owner") or owner_login
            args["repo"] = repo_name
            return args

    return args


def _short_repr(value: Any, max_len: int = 800) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 30] + f"...(truncated,len={len(text)})"


def _validate_against_schema(tool_schema: Dict[str, Any], args: Dict[str, Any]) -> Optional[str]:
    if not tool_schema or not isinstance(tool_schema, dict):
        return None
    if not args or not isinstance(args, dict):
        return None

    required = tool_schema.get("required") or []
    if isinstance(required, list) and required:
        missing = [k for k in required if k not in args or args.get(k) is None]
        if missing:
            return f"Missing required argument(s): {missing}"

    additional = tool_schema.get("additionalProperties")
    props = tool_schema.get("properties") or {}
    if additional is False and isinstance(props, dict) and props:
        unexpected = [k for k in args.keys() if k not in props]
        if unexpected:
            return f"Unexpected argument(s) not in schema: {unexpected}"

    return None

class ToolExecutor:
    def __init__(self, tool_manager: ToolManager):
        self.tool_manager = tool_manager
        
        # Load Mappings (Config + Auto)
        self.mappings = self._load_mappings()
        
        # Initialize LLM for Mode 2 Adaptation
        api_key = config.get("llm.api_key", "lm-studio")
        base_url = config.get("llm.api_base", "http://127.0.0.1:1234/v1")
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model = config.get("llm.model", "gpt-3.5-turbo")
        
        # Simple capability mapping (Can be expanded)
        self.capability_map = {
            "search": ["local_web_search"],
            "calculator": ["calculator"],
            "compute": ["calculator"],
            "reasoning": ["llm_reasoning"],
            "read": ["read_file"],
            "write": ["write_file"],
            "filesystem": ["read_file", "write_file"],
            "transform": ["count_words", "summarize_text"]
        }
        
        # Dynamic discovery of capabilities from ToolManager
        self._discover_capabilities()

        # Initialize Execution Repairer
        self.execution_repairer = ExecutionRepairer(tool_manager)

    def _discover_capabilities(self):
        """
        Dynamically update capability_map based on available tools.
        For MCP tools, we infer capabilities from names or explicit registry.
        """
        for tool_name, tool in self.tool_manager.tools.items():
            # Heuristic: Add MCP tools to capabilities
            if tool_name.startswith("mcp:"):
                # Example: mcp:MockGitServer:git_clone -> capability: git_clone
                parts = tool_name.split(":")
                if len(parts) >= 3:
                    simple_name = parts[-1]
                    # Add to direct capability match
                    if simple_name not in self.capability_map:
                         self.capability_map[simple_name] = []
                    if tool_name not in self.capability_map[simple_name]:
                         self.capability_map[simple_name].append(tool_name)
                    
                    # Also map 'read_file' to 'read' capability if applicable
                    if "read" in simple_name:
                         if "read" not in self.capability_map: self.capability_map["read"] = []
                         self.capability_map["read"].append(tool_name)
        
        logger.info(f"Discovered capabilities: {list(self.capability_map.keys())}")

    def _load_mappings(self) -> Dict[str, Any]:
        """Load parameter mappings from config file and auto-generated mappings"""
        combined_mappings = {}
        
        # 1. Load from Config (Priority)
        try:
            mapping_path = os.path.join(os.getcwd(), "agent", "configs", "mappings.yaml")
            if os.path.exists(mapping_path):
                with open(mapping_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    combined_mappings = data.get("operation_to_tool_mappings", {})
        except Exception as e:
            logger.error(f"Failed to load mappings: {e}")
            
        return combined_mappings

    async def execute_step(self, step: PlanStep, context: ExecutionContext) -> ExecutionResult:
        """
        Execute a single plan step.
        Supports Mode 1 (Certain) & Mode 2 (Uncertain/Fallback).
        """
        logger.info(f"Executing Step {step.step_id}: {step.intent}")
        
        try:
            # 1. Parameter Binding (Semantic Args)
            try:
                semantic_args = self._resolve_arguments(step, context)
            except Exception as e:
                return ExecutionResult(step_id=step.step_id, status="failed", error_type="binding_error", error=str(e))

            # 2. Tool Selection
            tool_name = self._select_tool(step.required_capability, step.operation.type)
            
            if not tool_name:
                 return ExecutionResult(step_id=step.step_id, status="failed", error_type="tool_not_found", error=f"No tool found for capability {step.required_capability}")
            
            logger.info(f"Selected tool: {tool_name}")
            
            # 3. Schema Mapping (Semantic -> Tool Args)
            tool = self.tool_manager.get_tool(tool_name)
            # We assume tool has .args_schema or similar, but for now we rely on 4-level mapping
            # Getting tool schema if available (for Level 2 check)
            tool_schema = {}
            if hasattr(tool, "args_schema") and tool.args_schema:
                 # Robust check for schema format
                 if isinstance(tool.args_schema, dict):
                     tool_schema = tool.args_schema
                 elif hasattr(tool.args_schema, "schema"):
                     tool_schema = tool.args_schema.schema()
            
            logger.info(f"Mapping arguments for tool '{tool_name}' (Semantic: {semantic_args})")
            
            final_args = await self._map_to_tool_schema(
                semantic_args, 
                step.operation.type, 
                tool_name, 
                tool_schema
            )
            
            logger.info(f"Mapped arguments: {final_args}")

            schema_issue = _validate_against_schema(tool_schema, final_args)
            if schema_issue:
                logger.error(f"Schema validation failed. Tool: {tool_name}, Error: {schema_issue}, Args: {_short_repr(final_args)}")
                return ExecutionResult(step_id=step.step_id, status="failed", error_type="schema_error", error=f"Tool arguments invalid: {schema_issue}")
            
            # 4. Execution
            try:
                result_data = self.tool_manager.execute_tool(tool_name, **final_args)
            except Exception as e:
                # Validation error or execution error
                logger.error(f"Execution failed. Tool: {tool_name}, Args: {_short_repr(final_args)}, Error: {e}")
                return ExecutionResult(step_id=step.step_id, status="failed", error_type="schema_error", error=f"Tool execution failed: {e}")
            
            # 5. Store Result
            context.set(f"step_output.{step.step_id}", result_data)
            if step.output_var:
                context.set(step.output_var, result_data) # Store as global var
                context.set(f"step_output.{step.output_var}", result_data) # Store as step output alias
            
            # Debug Context
            logger.debug(f"Updated Context Keys: {list(context.variables.keys())}")
            if result_data is None:
                logger.warning(f"Step {step.step_id} returned None result!")
            
            return ExecutionResult(
                step_id=step.step_id, 
                status="success", 
                result=result_data,
                meta={"tool": tool_name, "args": final_args}
            )

        except Exception as e:
            logger.warning(f"Execution failed for step {step.step_id}: {e}")
            
            # Mode 2: Always attempt adaptation on error for robustness
            logger.info(f"Triggering Mode 2 Adaptation for step {step.step_id} due to error")
            return await self.execution_repairer.repair(step, e, context)

    def _resolve_arguments(self, step: PlanStep, context: ExecutionContext) -> Dict[str, Any]:
        resolved_args = {}
        for param_name, binding in step.operation.inputs.items():
            val = context.resolve_binding(binding)
            if val is None:
                logger.warning(f"Binding resolved to None for param '{param_name}' (Ref: {binding.value_ref})")
            resolved_args[param_name] = val
        return resolved_args

    async def _map_to_tool_schema(
        self, 
        semantic_args: Dict[str, Any], 
        operation_type: str, 
        tool_name: str, 
        tool_schema: Dict
    ) -> Dict[str, Any]:
        """Core Mapping Logic - 4 Level Strategy"""
        
        if tool_name == "mcp:official_github:get_file_contents" or (
            tool_name.startswith("mcp:official_github:") and "owner" in (tool_schema.get("properties") or {}) and "repo" in (tool_schema.get("properties") or {})
        ):
            semantic_args = _normalize_github_owner_repo(semantic_args)

        mapped_args = {}
        
        # Level 1: Config Driven (Operation Specific)
        op_mapping = self.mappings.get(operation_type, {}).get(tool_name, {})
        
        # Level 1.5: Auto-Mappings (Tool Specific)
        # Check if tool_manager has auto_mappings (it should)
        auto_mapping = {}
        if hasattr(self.tool_manager, "auto_mappings"):
             auto_mapping = self.tool_manager.auto_mappings.get(tool_name, {})
        
        # Merge mappings (Config overrides Auto)
        # Note: auto_mapping is {semantic: tool_param}, op_mapping is same structure
        effective_mapping = {**auto_mapping, **op_mapping}
        
        if effective_mapping:
            logger.debug(f"Level 1 Mapping applied for {tool_name} (Map: {effective_mapping})")
            # Map semantic keys to tool keys, keep others as is
            for k, v in semantic_args.items():
                tool_key = effective_mapping.get(k, k) # Default to same name if not mapped
                mapped_args[tool_key] = v
            
            # Verify against schema if available (Level 2 check logic reused)
            tool_props = tool_schema.get("properties", {})
            if tool_props:
                # If all mapped keys exist in schema, we are good
                if all(k in tool_props for k in mapped_args):
                    return mapped_args
                # If not, maybe some unmapped keys are still wrong? 
                # Let's fall through to further levels if not perfect match?
                # Actually, if mapping was explicit, we trust it unless it fails execution.
                # But here we merged auto-mapping which might be partial.
                # Let's check if the result is fully compliant.
                missing_keys = [k for k in mapped_args if k not in tool_props]
                if not missing_keys:
                     return mapped_args
                else:
                     logger.debug(f"Mapping result {mapped_args} has keys not in schema: {missing_keys}. Falling through to Level 2/3.")
            else:
                 return mapped_args # No schema to check, assume correct

        # Level 2: Direct Match (if tool schema available)
        tool_props = tool_schema.get("properties", {})
        if tool_props:
            if all(k in tool_props for k in semantic_args):
                logger.debug(f"Level 2 Direct Match for {tool_name}")
                return semantic_args

        # Level 3: Fuzzy Match (Alias)
        # Simple hardcoded alias map for now
        # Also used for auto-mapping heuristics
        alias_map = {
            "query": ["q", "search_query", "keyword", "term"],
            "max_results": ["num", "limit", "count", "top_k"],
            "file_path": ["path", "filename", "uri", "location"],
            "content": ["data", "text", "body"],
            "repo_url": ["url", "repository", "link"],
            "target_dir": ["path", "destination", "dir", "folder"]
        }
        
        mapped_args = semantic_args.copy()
        for k, v in semantic_args.items():
            if tool_props and k not in tool_props:
                # Try aliases
                aliases = alias_map.get(k, [])
                for alias in aliases:
                    if alias in tool_props:
                        mapped_args[alias] = v
                        del mapped_args[k]
                        break
        
        # Check if all keys are now in tool_props
        if tool_props and all(k in tool_props for k in mapped_args):
             logger.debug(f"Level 3 Fuzzy Match for {tool_name}")
             return mapped_args

        # Level 3.5: Heuristic Auto-Discovery (Schema Inspection)
        # If tool_props exists, try to reverse match based on schema names
        if tool_props:
            mapped_args = {}
            for k, v in semantic_args.items():
                # 1. Check exact match
                if k in tool_props:
                    mapped_args[k] = v
                    continue
                    
                # 2. Check alias map
                found_alias = False
                aliases = alias_map.get(k, [])
                for alias in aliases:
                    if alias in tool_props:
                        mapped_args[alias] = v
                        found_alias = True
                        break
                
                if found_alias: continue
                
                # 3. Fallback: Keep original if no match found (might be correct but missing in schema or dynamic)
                mapped_args[k] = v
            
            # Verify validity
            if all(k in tool_props for k in mapped_args):
                 logger.debug(f"Level 3.5 Heuristic Match for {tool_name}")
                 return mapped_args

        # Level 4: LLM Fallback
        logger.debug(f"Mapping levels 1-3 failed/partial for {tool_name}, attempting LLM mapping")
        return await self._llm_assisted_mapping(semantic_args, tool_schema, operation_type)

    async def _llm_assisted_mapping(
        self, 
        semantic_args: Dict, 
        tool_schema: Dict, 
        operation_type: str
    ) -> Dict:
        """Use LLM to map semantic arguments to tool parameters (Level 4)"""
        
        class MappingResult(BaseModel):
            tool_args: Dict[str, Any]
            reasoning: str

        prompt = f"""
        Map semantic arguments to tool parameters.
        
        Semantic args: {json.dumps(semantic_args)}
        Tool schema properties: {json.dumps(tool_schema.get("properties", {}))}
        Operation type: {operation_type}
        
        RESPONSE FORMAT:
        You MUST provide a <thinking> block BEFORE the JSON.
        ```json
        {{
          "tool_args": {{ ... }},
          "reasoning": "..."
        }}
        ```
        """
        
        try:
            last_error = None
            resp = None
            for _attempt in range(2):
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": prompt}
                        ],
                    )
                    break
                except Exception as e:
                    last_error = e
            if resp is None:
                raise last_error
            content = resp.choices[0].message.content
            thinking = extract_thinking(content or "")
            if thinking:
                logger.debug(f"[Executor Mapping Thinking] len={len(thinking)} preview={thinking[:300]}")
            result = parse_structured(content or "", MappingResult)
            return result.tool_args
        except Exception as e:
            logger.error(f"LLM mapping failed: {e}")
            return semantic_args # Fallback to original

    def _select_tool(self, capability: str, op_type: OperationType) -> Optional[str]:
        if not capability:
            return None

        capability_norm = capability.strip()
        if not capability_norm:
            return None

        if self.tool_manager.get_tool(capability_norm):
            return capability

        all_tool_names = list(self.tool_manager.tools.keys())
        cap_lower = capability_norm.lower()

        direct_matches = [
            t for t in all_tool_names
            if t.lower() == cap_lower
            or t.lower().endswith(":" + cap_lower)
            or t.lower().split(":")[-1] == cap_lower
        ]

        if direct_matches:
            op_name = str(op_type).lower().split(".")[-1]
            op_filtered = [t for t in direct_matches if op_name and op_name in t.lower()]
            chosen_list = op_filtered or direct_matches
            return sorted(chosen_list)[0]
            
        # 2. Map lookup
        candidates = self.capability_map.get(capability, [])
        
        # 3. Filter candidates based on operation type (Heuristic)
        # This handles cases like "filesystem" -> ["read_file", "write_file"]
        filtered_candidates = []
        op_name = str(op_type).lower().split(".")[-1] # e.g. "write", "read"
        
        # Priority filter: exact op match
        for c in candidates:
            if op_name in c.lower():
                filtered_candidates.append(c)
        
        # If heuristics found matches, prioritize them
        search_list = filtered_candidates if filtered_candidates else candidates
        
        # If op is WRITE, strictly avoid search tools
        if op_type == OperationType.WRITE:
             search_list = [c for c in search_list if "search" not in c.lower()]
        
        for c in search_list:
            if self.tool_manager.get_tool(c):
                return c
                
        return None
