import json
from typing import List, Dict

def local_web_search(query: str) -> str:
    """
    本地模拟“网页搜索”（离线测试用），返回 mock 结果 JSON。

    何时用：需要在无外网/无 API 的情况下演示“搜索→提取→总结”的链路。
    输入：query（查询词）。
    输出：JSON 字符串（str），包含若干条 {title/snippet/url}。
    典型任务：离线 demo、测试 planner 是否会正确调用搜索工具、验证结果整合流程。
    """
    query_lower = query.lower()
    
    # Mock Data Database
    mock_db = {
        "iphone 16": [
            {"title": "iPhone 16 - Apple", "snippet": "The new iPhone 16 features the A18 Bionic chip. Price starts at $799.", "url": "https://apple.com/iphone-16"},
            {"title": "iPhone 16 Rumors", "snippet": "Latest rumors suggest a new design and improved battery life.", "url": "https://techradar.com/iphone-16"}
        ],
        "llm agents": [
            {"title": "What are LLM Agents?", "snippet": "LLM agents are systems that use large language models as reasoning engines.", "url": "https://example.com/llm-agents"},
            {"title": "Building Autonomous Agents", "snippet": "Guide to building agents with LangChain and AutoGen.", "url": "https://github.com/agents"}
        ],
        "weather": [
            {"title": "Weather Forecast", "snippet": "Today is sunny with a high of 25°C.", "url": "https://weather.com"}
        ]
    }
    
    # Search logic
    results = []
    for key, items in mock_db.items():
        if key in query_lower:
            results.extend(items)
            
    if not results:
        # Fallback generic result
        results.append({
            "title": f"Results for {query}",
            "snippet": f"This is a simulated search result for '{query}'. No specific mock data found.",
            "url": "https://example.com/search"
        })
        
    return json.dumps(results, indent=2)
