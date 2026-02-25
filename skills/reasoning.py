from typing import Optional
from agent.llm.client import LLMClient
from agent.utils.config import config

def llm_reasoning(task_description: str) -> str:
    """
    用 LLM 完成通用推理/文字生成任务。

    何时用：没有更具体的工具可用，但需要生成/改写/归纳文本，或进行一般性推理与解释。
    输入：task_description（对任务的清晰描述，包含必要上下文与输出要求）。
    输出：模型生成的文本（str）。
    典型任务：写摘要、写对用户的解释、把结构化结果改写成自然语言、生成模板化文案。
    """
    client = LLMClient()
    
    prompt = f"""
    You are a helpful assistant.
    Task: {task_description}
    
    Please provide a concise and accurate response.
    """
    
    return client.generate(prompt)
