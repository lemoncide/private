from typing import List, Dict, Optional, Any, Iterable, Tuple
from agent.utils.config import config
from agent.utils.logger import logger
from openai import OpenAI
import json
import re
from agent.llm.url import normalize_base_url

class LLMClient:
    def __init__(self):
        self.api_key = config.get("llm.api_key", "lm-studio")
        self.base_url = normalize_base_url(config.get("llm.api_base", "http://127.0.0.1:1234/v1"))
        self.llm_model_name = config.get("llm.model", "gpt-3.5-turbo")
        
        logger.info(f"Initializing LLM Client with base_url: {self.base_url}")
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        
        # Logic to bypass OpenAI library validation for local servers
        if "localhost" in self.base_url or "127.0.0.1" in self.base_url:
            self.model_name = "gpt-3.5-turbo" # Override to pass client validation
            logger.info(f"Local LLM detected. Using '{self.model_name}' as alias for '{self.llm_model_name}'")
        else:
            self.model_name = self.llm_model_name

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generates a response from the LLM.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            logger.debug(f"LLM Request: {messages}")
            completion = self.chat(messages, temperature=0.7)
            response = completion.choices[0].message.content
            logger.debug(f"LLM Response: {response[:100]}...")
            return response
        except Exception as e:
            logger.error(f"LLM Generation failed: {e}")
            return f"Error generating response: {e}"

    def chat(
        self, 
        messages: List[Dict[str, Any]], 
        temperature: float = 0.7, 
        response_format: Optional[Dict[str, Any]] = None, 
        timeout: Optional[float] = None
    ) -> Any:
        """
        Direct chat completion call using the underlying OpenAI client.
        """
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            response_format=response_format,
            timeout=timeout,
        )

    def generate_with_tools(
        self,
        prompt: str,
        tool_defs: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        tool_choice: Any = "auto",
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        tools = self._to_openai_tools(tool_defs)

        last_error: Optional[Exception] = None
        for _ in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=temperature,
                )
                message = completion.choices[0].message

                tool_calls_payload: List[Dict[str, Any]] = []
                raw_tool_calls = getattr(message, "tool_calls", None) or []
                for tc in raw_tool_calls:
                    func = getattr(tc, "function", None)
                    name = getattr(func, "name", None) if func else None
                    arguments_raw = getattr(func, "arguments", None) if func else None
                    arguments: Any = arguments_raw
                    if isinstance(arguments_raw, str):
                        try:
                            arguments = json.loads(arguments_raw)
                        except Exception:
                            arguments = arguments_raw

                    tool_calls_payload.append(
                        {
                            "id": getattr(tc, "id", None),
                            "name": name,
                            "arguments": arguments,
                            "arguments_raw": arguments_raw,
                        }
                    )

                legacy_function_call = getattr(message, "function_call", None)
                if legacy_function_call and not tool_calls_payload:
                    name = getattr(legacy_function_call, "name", None)
                    arguments_raw = getattr(legacy_function_call, "arguments", None)
                    arguments: Any = arguments_raw
                    if isinstance(arguments_raw, str):
                        try:
                            arguments = json.loads(arguments_raw)
                        except Exception:
                            arguments = arguments_raw
                    tool_calls_payload.append(
                        {"id": None, "name": name, "arguments": arguments, "arguments_raw": arguments_raw}
                    )

                return {
                    "content": getattr(message, "content", None),
                    "tool_calls": tool_calls_payload,
                    "openai_tool_calls": [
                        {
                            "id": tc.get("id"),
                            "type": "function",
                            "function": {"name": tc.get("name"), "arguments": tc.get("arguments_raw") or ""},
                        }
                        for tc in tool_calls_payload
                        if tc.get("name")
                    ],
                    "model": self.model_name,
                }
            except Exception as e:
                last_error = e
                logger.error(f"LLM Tool call failed: {e}")

        return {
            "content": None,
            "tool_calls": [],
            "openai_tool_calls": [],
            "model": self.model_name,
            "error": str(last_error) if last_error else "unknown error",
        }

    @staticmethod
    def _to_openai_tools(tool_defs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        openai_tools: List[Dict[str, Any]] = []

        for t in tool_defs or []:
            if not isinstance(t, dict):
                continue
            if t.get("type") == "function" and isinstance(t.get("function"), dict):
                openai_tools.append(t)
                continue

            name = t.get("name")
            description = t.get("description", "")
            parameters = t.get("args_schema") or {}

            if isinstance(parameters, dict):
                parameters = dict(parameters)
                parameters.pop("$schema", None)
            else:
                parameters = {}

            if not isinstance(parameters.get("type"), str):
                parameters["type"] = "object"
            parameters.setdefault("properties", {})
            parameters.setdefault("additionalProperties", False)

            openai_tools.append(
                {
                    "type": "function",
                    "function": {"name": name, "description": description or "", "parameters": parameters},
                }
            )

        return openai_tools

    def generate_structured(self, prompt: str, example_schema: Dict[str, Any], system_prompt: Optional[str] = None, max_retries: int = 3) -> Dict[str, Any]:
        """
        Generates a structured JSON response.
        """
        schema_str = json.dumps(example_schema, indent=2)
        system_instruction = f"""
You are a precise JSON generator. 
You must output VALID JSON matching exactly this schema:
{schema_str}
Do not output any markdown formatting like ```json or ```. Just the raw JSON object.
"""
        
        final_system_prompt = system_prompt + "\n" + system_instruction if system_prompt else system_instruction
        
        current_prompt = prompt
        
        for attempt in range(max_retries):
            response = self.generate(current_prompt, system_prompt=final_system_prompt)
            
            # Cleaning response
            cleaned_response = response.strip()
            # Remove markdown code blocks if present
            cleaned_response = re.sub(r'^```json\s*', '', cleaned_response)
            cleaned_response = re.sub(r'^```\s*', '', cleaned_response)
            cleaned_response = re.sub(r'\s*```$', '', cleaned_response)
            
            # Try finding JSON object
            match = re.search(r'\{[\s\S]*\}', cleaned_response)
            if match:
                cleaned_response = match.group(0)
            
            try:
                return json.loads(cleaned_response)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON Parse Error (Attempt {attempt+1}): {e}")
                current_prompt = prompt + f"\n\nError: Previous output was not valid JSON. \nOutput: {response}\nError: {str(e)}\nPlease correct it."
                
        raise ValueError("Failed to generate valid JSON after retries.")
