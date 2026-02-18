{
  "name": "README.md",
  "path": "README.md",
  "sha": "b57c0f5c79af133d0398fdbf6c02c7a22713a10f",
  "size": 3254,
  "url": "https://api.github.com/repos/lemoncide/RAG/contents/README.md?ref=main",
  "html_url": "https://github.com/lemoncide/RAG/blob/main/README.md",
  "git_url": "https://api.github.com/repos/lemoncide/RAG/git/blobs/b57c0f5c79af133d0398fdbf6c02c7a22713a10f",
  "download_url": "https://raw.githubusercontent.com/lemoncide/RAG/main/README.md",
  "type": "file",
  "content": "# 本地 RAG 知识库系统 (Local RAG System)\n\n这是一个本地 RAG (检索增强生成) 系统。它使用 **LlamaIndex** 构建，后端采用 **FastAPI**，前端提供 **Streamlit** 界面，并支持通过 **LM Studio** 连接本地运行的大模型（如 Qwen, Llama 3）。\n\n针对资源受限环境（如 4GB 显存）进行了一些优化，确保在普通配置下也能流畅运行。\n\n## ✨ 主要功能\n\n1.  **混合检索 (Hybrid Search)**\n    *   **向量检索**: 使用 ChromaDB 存储和检索语义向量。\n    *   **关键词检索**: 使用 BM25 算法匹配精确关键词。\n    *   **动态融合**: 根据你的问题长度自动调整权重（短问题看重关键词，长问题看重语义）。\n\n2.  **重排序 (Reranking)**\n    *   使用 `BAAI/bge-reranker` 模型对初步检索到的结果进行二次打分，把最相关的排到最前面。\n\n3.  **精细化切分**\n    *   将文档切分为较小的片段（256 token），既节省显存，又能更精准地定位信息。\n    *   支持 PDF 自动转 Markdown，保留文档标题结构。\n\n4.  **流式输出 (Streaming)**\n    *   支持打字机效果，即使本地模型生成较慢，也能立刻看到反馈。\n\n## 🛠️ 环境准备\n\n*   Python 3.10+\n*   LM Studio (用于运行本地 LLM)\n*   推荐配置：8GB+ 内存，4GB+ 显存\n\n### 安装依赖\n\n```bash\npip install -r requirements.txt\n```\n\n*(如果缺少某些库，请根据报错安装，主要依赖包括：`llama-index`, `chromadb`, `fastapi`, `uvicorn`, `streamlit`, `rank_bm25`, `pymupdf4llm`)*\n\n## 🚀 使用指南\n\n### 1. 准备数据\n\n将你的 PDF 文档放入 `data/embodia/pdf/` 目录中。\n\n### 2. 构建索引\n\n运行以下命令。它会将文档处理成向量和关键词索引，存入本地的 `chroma_db` 文件夹。\n\n```bash\npython scripts/build_index.py\n```\n\n> **注意**: 每次添加新文件后，都需要重新运行这一步。\n\n### 3. 启动本地模型 (LM Studio)\n\n1.  打开 LM Studio。\n2.  加载一个模型（推荐 小型模型例如 **Phi-3-mini** 以获得最佳速度，默认使用 Qwen-4B）。\n3.  点击左侧 **Server** 图标。\n4.  点击 **Start Server**，保持默认端口 `1234`。\n\n### 4. 启动后端服务\n\n启动 FastAPI 后端：\n\n```bash\nuvicorn app.main:app --reload\n```\n- API 文档地址: http://127.0.0.1:8000/docs\n### 5. 启动对话界面\n\n另开一个终端，启动 Streamlit 前端：\n\n```bash\nstreamlit run streamlit_app.py\n```\n\n浏览器将自动打开 http://localhost:8501。\n\n## 🔌 API 接口说明\n\n### 1. 对话接口 (RAG + LLM)\n\n- **Endpoint**: `POST /api/chat`\n- **描述**: 检索相关文档并生成回答。\n- **请求示例**:\n  ```json\n  {\n    \"query\": \"机器人驾驶车辆竞赛是什么？\",\n    \"top_k\": 5\n  }\n  ```\n\n### 2. 纯检索接口 (Retriever Only)\n\n- **Endpoint**: `POST /api/query`\n- **描述**: 仅返回相关的文档片段，不生成回答。适合作为 Agent 的工具调用。\n- **请求示例**:\n  ```json\n  {\n    \"query\": \"机器人竞赛规则\",\n    \"top_k\": 10\n  }\n  ```\n\n如果你想在其他程序中调用：\n\n*   **流式对话**: `POST /api/chat/stream` \n*   **普通对话**: `POST /api/chat`\n*   **纯检索**: `POST /api/query` (只返回文档，不生成回答)",
  "encoding": "base64",
  "_links": {
    "self": "https://api.github.com/repos/lemoncide/RAG/contents/README.md?ref=main",
    "git": "https://api.github.com/repos/lemoncide/RAG/git/blobs/b57c0f5c79af133d0398fdbf6c02c7a22713a10f",
    "html": "https://github.com/lemoncide/RAG/blob/main/README.md"
  }
}