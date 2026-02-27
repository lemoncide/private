# LangGraph MCP Agent

基于 LangGraph 和 Model Context Protocol (MCP) 构建的智能 Agent 框架。本项目旨在通过模块化的工具管理、自动化的计划修复逻辑以及灵活的 MCP 扩展，实现一个能够处理复杂多步任务的自主智能体。

## 🚀 核心特性

- **LangGraph 驱动的工作流**：采用有状态的图结构管理任务生命周期，包含 `Planner`（规划）、`Executor`（执行）、`Repair`（自动修复）和 `Reflect`（总结反思）节点。
- **MCP (Model Context Protocol) 集成**：支持 stdio 传输协议，可动态加载并连接多个 MCP 服务器（如 GitHub, Filesystem, Fetch 等）。
- **自动化故障恢复**：内置 `Repairer` 节点，针对工具缺失、参数错误或 API 调用失败进行自动重试与计划修正。
- **工具语义检索**：集成 ChromaDB，通过 `MemoryManager` 对工具描述进行向量化索引，根据用户目标动态召回最相关的工具。
- **本地 Skill 扩展**：支持通过 Python 脚本快速定义本地工具（Skills），并与 MCP 工具无缝集成。
- **RAG 触发保护**：内置逻辑确保 `rag_search` 仅在用户明确提到“本地知识库”、“RAG”等关键词时启用，避免资源浪费和干扰。

## 🛠️ 技术栈

- **框架**: [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://github.com/langchain-ai/langchain)
- **大模型**: OpenAI 兼容接口（如 Groq, LM Studio）
- **向量数据库**: [ChromaDB](https://github.com/chroma-core/chroma)
- **协议**: [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)
- **后端**: FastAPI (Web API 接口)

## 📁 项目结构

```text
.
├── agent/                  # 核心代码
│   ├── core/               # 图节点定义 (Planner, Executor, Repair, Reflect)
│   ├── llm/                # LLM 客户端与 URL 处理
│   ├── memory/             # 基于 ChromaDB 的记忆与工具检索管理
│   ├── tools/              # 工具管理器与 MCP 适配器
│   └── web/                # FastAPI Web 服务与前端静态资源
├── configs/                # 配置文件 (config.yaml, user_prefs.json)
├── skills/                 # 本地 Python 工具扩展 (RAG, WebSearch 等)
├── sandbox/                # 运行期产物存放目录（报告、日志等）
├── main.py                 # CLI 入口
└── requirements.txt        # 项目依赖
```

## ⚙️ 快速启动

### 1. 环境准备
确保已安装 Python 3.10+ 和 Node.js（用于运行 MCP npx 工具）。

```bash
pip install -r requirements.txt
```

### 2. 配置文件
在 `configs/config.yaml` 中配置您的 LLM API 和 MCP 服务器：

```yaml
llm:
  model: "your-model-name"
  api_key: "${YOUR_API_KEY}"
  api_base: "https://api.your-provider.com/v1"

mcp:
  servers:
    official_github:
      command: "npx.cmd"
      args: ["-y", "@modelcontextprotocol/server-github"]
      env:
        GITHUB_PERSONAL_ACCESS_TOKEN: "${GITHUB_TOKEN}"
```

### 3. 运行 Agent

**命令行模式:**
```bash
python main.py --task "列出仓库 datawhalechina/hello-agents 的 open issue 并总结内容"
```

**Web 模式:**
```bash
python -m agent.web.app
```
访问 `http://127.0.0.1:8001` 即可使用可视化界面。

## 📸 界面演示

以下是 Agent 在不同场景下的运行示意图：

### 1. 知识库检索与任务执行
Agent 能够根据指令自动判断是否启用本地 RAG 组件进行知识检索，并基于检索内容进行推理。
![RAG 场景演示](https://github.com/user-attachments/assets/4dffdfea-44c9-4353-80ed-90e9f2b0a959)

### 2. 多步任务与结果保存
Agent 能够读取仓库 README，进行分析总结，并将改善建议自动保存到本地 Sandbox。
![多步任务演示](https://github.com/user-attachments/assets/e0ea80e1-4c29-42ff-9d21-ff38ee36c711)

### 3. 自动修复机制
当工具调用出现参数错误或 API 异常时，Agent 会自动触发 Repair 流程，修正计划并继续执行。
![自动修复演示](https://github.com/user-attachments/assets/b36041bf-395b-4635-8984-070b36665ab8)

## 🧠 关键逻辑说明

### 自动修复机制
当工具执行报错时，`error_router` 会根据错误类型（如 `tool_not_found`, `schema_error`）将状态导向不同的 `repair` 节点。`Repairer` 会利用当前上下文变量和最新工具列表尝试生成修正后的步骤。

### 工具发现
Agent 不会一次性将所有工具（可能成百上千）推给 LLM。它通过 `MemoryManager` 计算用户 Objective 与工具描述的余弦相似度，仅召回 Top-K 个最相关的工具，有效节省 Token 并提高规划准确度。

### RAG 启用策略
为了防止 Agent 滥用本地检索，系统在代码层级设置了过滤：只有当指令中包含“知识库”、“RAG”或“本地检索”等明确字眼时，`rag_search` 才会出现在工具列表中。

