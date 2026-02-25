from typing import Dict, Any

def count_words(text: str) -> int:
    """
    统计文本中的词数（按空白分词）。

    何时用：需要快速估算文本长度或做简单计数逻辑。
    输入：text（文本字符串）。
    输出：词数（int）。
    典型任务：长度检查、内容规模估算、简单质量控制。
    """
    if not text:
        return 0
    return len(text.split())

def summarize_text(text: str = "", max_length: int = 100) -> str:
    """
    对文本做轻量级摘要（当前为截断式摘要）。

    何时用：需要把超长文本压缩成可放入 prompt 的短文本，或做预览展示。
    输入：text（原始文本）、max_length（最大长度，按字符数）。
    输出：摘要文本（str；可能带 ... 截断标记）。
    典型任务：日志/JSON 预览、长文压缩、反射/报告中的 result_preview 生成。
    """
    if not text:
        return ""
    
    # Simple truncation for now to verify flow
    if len(text) <= max_length:
        return text
    
    return text[:max_length] + "..."
