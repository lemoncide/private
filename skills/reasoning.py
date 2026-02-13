from typing import Optional
from agent.llm.client import LLMClient
from agent.utils.config import config

def llm_reasoning(task_description: str) -> str:
    """
    Perform a general reasoning task using an LLM.
    Use this when no other specific tool is suitable.
    """
    client = LLMClient()
    
    prompt = f"""
    You are a helpful assistant.
    Task: {task_description}
    
    Please provide a concise and accurate response.
    """
    
    return client.generate(prompt)
