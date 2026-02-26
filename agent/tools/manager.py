from typing import List, Dict, Optional, Any
from langchain_core.tools import BaseTool
from agent.tools.skill_loader import SkillLoader
from agent.utils.config import config
from agent.utils.logger import logger
import os

from agent.memory.manager import MemoryManager
from agent.core.errors import ToolNotFoundError

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except Exception:
    MultiServerMCPClient = None

class ToolManager:
    def __init__(self, memory_manager: MemoryManager = None):
        self.tools: Dict[str, BaseTool] = {}
        self.skill_loader = SkillLoader(os.path.join(os.getcwd(), 'skills'))
        self.memory_manager = memory_manager
        self.auto_mappings: Dict[str, Dict[str, str]] = {}
        self._mcp_tools_initialized = False
        self._mcp_client = None
        self._initialize_tools()

    @staticmethod
    def _augment_tool_description(name: str, description: str) -> str:
        base = (description or "").strip()
        usage: List[str] = []
        io: List[str] = []
        tips: List[str] = []
        avoid: List[str] = []
        tasks: List[str] = []
        lower = (name or "").lower()

        if lower.startswith("mcp:official_github:"):
            if lower.endswith(":get_file_contents"):
                usage.append("读取仓库文件内容；或把 path 设为目录来列出目录下的条目")
                io.append("输入 owner/repo/path；输出文件内容或目录条目列表")
                tips.append("path 为空字符串或 / 可列根目录；path 为目录时返回目录条目")
                tasks.extend(["读 README", "列根目录文件", "遍历目录结构", "抓取配置文件"])
            elif lower.endswith(":list_issues") or lower.endswith(":search_issues"):
                usage.append("获取/筛选 issues 或 PR 列表，或按条件搜索")
                io.append("输入 owner/repo 与过滤条件；输出条目列表")
                tasks.extend(["汇总 open issues", "查找特定关键字相关问题"])
            elif lower.endswith(":search_code"):
                usage.append("跨仓库/仓库内做代码关键词搜索（不是目录/文件列表工具）")
                io.append("输入 q（GitHub 搜索语法）；输出匹配结果列表")
                avoid.append("不要用它来列目录，列目录请用 get_file_contents 并传目录 path")
                tasks.extend(["定位函数定义/用法", "排查配置项来源", "查找包含特定字符串的文件"])
            elif lower.endswith(":list_pull_requests"):
                usage.append("列出/筛选仓库的 Pull Requests")
                io.append("输入 owner/repo 与过滤条件；输出 PR 列表")
                tasks.extend(["汇总 open PR", "按分支/状态筛选 PR"])
            elif lower.endswith(":get_pull_request_files"):
                usage.append("获取某个 PR 改动的文件列表")
                io.append("输入 owner/repo/pull_number；输出文件改动列表")
                tasks.extend(["审查改动范围", "统计改动文件类型"])
            elif lower.endswith(":get_pull_request_comments"):
                usage.append("获取某个 PR 的评论")
                io.append("输入 owner/repo/pull_number；输出评论列表")
                tasks.extend(["汇总讨论点", "发现阻塞问题"])
            elif lower.endswith(":get_pull_request_reviews"):
                usage.append("获取某个 PR 的 Review 状态与记录")
                io.append("输入 owner/repo/pull_number；输出 Review 列表")
                tasks.extend(["检查审批状态", "统计 Review 意见"])
            elif lower.endswith(":create_or_update_file") or lower.endswith(":push_files"):
                usage.append("在仓库中创建/更新文件或批量推送文件")
                io.append("输入 owner/repo/path/content/message 等；输出结果/提交信息")
                avoid.append("非读取/列目录工具，若要列目录请用 get_file_contents")
                tasks.extend(["提交生成的报告", "更新配置文件", "批量写入产物"])

        if lower.startswith("mcp:official_filesystem:"):
            if lower.endswith(":read_text_file") or lower.endswith(":read_file"):
                usage.append("读取本地文件内容（只限允许目录内）")
                io.append("输入 path；输出文件文本")
                tasks.extend(["读取源码/配置", "读取生成的输出文件"])
            elif lower.endswith(":list_directory"):
                usage.append("列出某个目录下的文件/子目录")
                io.append("输入 path；输出目录条目列表")
                tasks.extend(["找文件位置", "确认目录结构"])
            elif lower.endswith(":write_file"):
                usage.append("在本地文件系统写入或覆盖文件（只限允许目录内）")
                io.append("输入 path/content；输出写入结果")
                tasks.extend(["生成报告", "导出处理结果", "写入配置"])
            elif lower.endswith(":search_files"):
                usage.append("按 glob 模式递归搜索文件路径")
                io.append("输入 path/pattern；输出匹配路径列表")
                tasks.extend(["查找某类文件", "定位配置/脚本位置"])
            elif lower.endswith(":directory_tree"):
                usage.append("以树形结构返回目录的递归视图")
                io.append("输入 path；输出目录树 JSON（包含 name/type/children）")
                tasks.extend(["整仓走查", "生成目录结构报告"])
            elif lower.endswith(":move_file"):
                usage.append("移动或重命名文件/目录")
                io.append("输入 source/destination；输出移动结果")
                tasks.extend(["重命名文件", "整理目录结构"])
            elif lower.endswith(":create_directory"):
                usage.append("创建目录或确保目录存在")
                io.append("输入 path；输出创建结果")
                tasks.extend(["初始化项目结构", "准备输出路径"])
            elif lower.endswith(":edit_file"):
                usage.append("按行编辑文本文件并返回 diff")
                io.append("输入 path/edits/dryRun；输出 git 风格 diff")
                tasks.extend(["批量替换内容", "预览修改差异"])

        sections: List[str] = []
        if usage:
            sections.append("何时用：" + "；".join(usage))
        if io:
            sections.append("输入输出：" + "；".join(io))
        if tips:
            sections.append("用法提示：" + "；".join(tips))
        if avoid:
            sections.append("避免误用：" + "；".join(avoid))
        if tasks:
            sections.append("典型任务：" + "；".join(tasks))

        if not sections:
            return base
        structured = "\n".join(sections).strip()
        if not base:
            return structured
        return (base + "\n\n" + structured).strip()

    def _initialize_tools(self):
        # 1. Load Skills with Auto-Mapping
        skills, mappings = self.skill_loader.load_skills()
        self.auto_mappings = mappings
        
        for tool in skills:
            self.register_tool(tool)
            
        logger.info(f"Total tools loaded: {len(self.tools)}")
        logger.info(f"Auto-mappings generated for: {list(self.auto_mappings.keys())}")

    async def init_mcp_tools(self):
        if self._mcp_tools_initialized:
            return
        if MultiServerMCPClient is None:
            logger.warning("langchain-mcp-adapters not installed; skipping MCP tool initialization.")
            return

        connections = self.get_mcp_connections()

        if not connections:
            self._mcp_tools_initialized = True
            return

        client = MultiServerMCPClient(connections, tool_name_prefix=True)
        tools = await client.get_tools()
        self._mcp_client = client

        for tool in tools:
            name = getattr(tool, "name", "") or ""
            matched_server = None
            matched_prefix_len = -1
            for server_name in connections.keys():
                prefix = f"{server_name}_"
                if name.startswith(prefix) and len(prefix) > matched_prefix_len:
                    matched_server = server_name
                    matched_prefix_len = len(prefix)
            if matched_server:
                tool_name = name[matched_prefix_len:]
                tool.name = f"mcp:{matched_server}:{tool_name}"
            self.register_tool(tool)

        self._mcp_tools_initialized = True
        logger.info(f"MCP tools loaded: {len(tools)}")

    async def aclose(self):
        """Close the MCP client connections."""
        if self._mcp_client and hasattr(self._mcp_client, "aclose"):
            try:
                await self._mcp_client.aclose()
                logger.info("MCP client connections closed.")
            except Exception as e:
                logger.error(f"Error closing MCP client: {e}")
        self._mcp_tools_initialized = False

    def get_mcp_connections(self) -> Dict[str, Dict[str, Any]]:
        connections: Dict[str, Dict[str, Any]] = {}
        mcp_servers = config.get("mcp.servers", {}) or {}
        for server_name, cfg in mcp_servers.items():
            if not isinstance(cfg, dict):
                continue
            command = cfg.get("command")
            if not command:
                continue
            args = cfg.get("args", []) or []
            conn: Dict[str, Any] = {"transport": "stdio", "command": command, "args": args}
            env = cfg.get("env") or {}
            if isinstance(env, dict) and env:
                conn["env"] = env
            connections[server_name] = conn
        return connections

    def register_tool(self, tool: BaseTool):
        self.tools[tool.name] = tool
        if self.memory_manager:
            self.memory_manager.index_tool(tool.name, self._augment_tool_description(tool.name, tool.description))
        logger.debug(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self.tools.get(name)

    def list_tools(self, query: str = None, limit: int = 5) -> List[Dict[str, str]]:
        # Modified to support retrieval
        if query and self.memory_manager:
            relevant_names = self.memory_manager.retrieve_tools(query, limit)
            tools_to_return = [self.tools[name] for name in relevant_names if name in self.tools]
            # Fallback if no relevant tools found or memory not ready
            if not tools_to_return:
                 tools_to_return = list(self.tools.values())[:limit]
        else:
            tools_to_return = list(self.tools.values())

        result = []
        for t in tools_to_return:
            item: Dict[str, Any] = {"name": getattr(t, "name", ""), "description": getattr(t, "description", "")}
            args_schema = getattr(t, "args_schema", None)
            if args_schema is not None:
                if isinstance(args_schema, dict):
                    item["args_schema"] = args_schema
                elif hasattr(args_schema, "schema"):
                    item["args_schema"] = args_schema.schema()
                elif hasattr(args_schema, "model_json_schema"):
                    item["args_schema"] = args_schema.model_json_schema()
            result.append(item)
        return result

    async def execute_tool(self, name: str, **kwargs):
        tool = self.get_tool(name)
        if tool:
            logger.debug(f"Executing tool {name} (type: {type(tool)}) with args: {kwargs}")
            try:
                return await tool.ainvoke(kwargs)
            except Exception as e:
                logger.error(f"Error executing tool {name}: {e}")
                logger.error(f"Tool run method: {getattr(tool, 'run', 'missing')}")
                # Don't try fallback with tool.run(kwargs) as that passes a dict as 1st pos arg,
                # which causes "BaseModel.__init__ takes 1 pos arg but 2 given" if 'run' somehow calls super init
                # or if the tool implementation is confused.
                # Just raise the original error to let Executor/Adaptation handle it.
                raise e
        raise ToolNotFoundError(name)
