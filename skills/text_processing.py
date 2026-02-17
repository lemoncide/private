from typing import Dict, Any

def count_words(text: str) -> int:
    """Count the number of words in a text string."""
    if not text:
        return 0
    return len(text.split())

def summarize_text(text: str = "", max_length: int = 100) -> str:
    """
    Summarize text by truncating or extracting key sentences.
    (Currently a simple truncation/keyword placeholder, can be enhanced with LLM later)
    """
    if not text:
        return ""
    
    # Simple truncation for now to verify flow
    if len(text) <= max_length:
        return text
    
    return text[:max_length] + "..."
