import requests
from typing import List, Dict

def rag_search(query: str, top_k: int = 5) -> str:
    """
    调用本地 RAG 系统的 /api/query，检索并返回文档片段拼接字符串。
    
    何时用：仅当用户在 query 中明确提到“使用本地知识库”、“本地RAG”、“本地检索”等关键词时才启用。严禁在未明确要求的情况下自动调用。
    参数：query 查询词；top_k 返回片段数量（默认 5）。
    返回：按顺序拼接的文档片段字符串，包含必要来源信息，便于后续 llm_reasoning 使用。
    """
    url = "http://127.0.0.1:8000/api/query"
    try:
        resp = requests.post(url, json={"query": query, "top_k": int(top_k)}, timeout=15)
        resp.raise_for_status()
        data: List[Dict] = resp.json() if resp.content else []
        if not isinstance(data, list) or not data:
            return ""
        parts: List[str] = []
        for i, doc in enumerate(data, 1):
            text = (doc.get("text") or "").strip()
            source = (doc.get("source") or "").strip()
            page = doc.get("page_number")
            prefix = f"[{i}] {source}" if source else f"[{i}]"
            if isinstance(page, int):
                prefix += f" (p{page})"
            if text:
                parts.append(f"{prefix}: {text}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"RAG检索失败：{e}"
