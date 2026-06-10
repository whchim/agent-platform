"""MCP 协议客户端 — 通过子进程 + JSON-RPC 连接 MCP Server，供 Agent 调用外部工具

MCP (Model Context Protocol) 原理：
    1. 启动 MCP Server 子进程（Python / Node.js / 任意语言）
    2. 通过 stdin/stdout 管道发送 JSON-RPC 2.0 请求
    3. Server 返回工具列表 / 执行结果
    4. Agent 通过 MCP 客户端的 list_tools + call_tool 接入外部能力

典型使用场景：
    - 连接代码解释器 Server（执行 Python 代码）
    - 连接文件系统 Server（读写本地文件）
    - 连接数据库 Server（执行 SQL 查询）
"""

import asyncio
import json
import subprocess
from typing import Any


class MCPClient:
    """
    MCP 客户端 — 管理子进程生命周期 + JSON-RPC 通信

    使用方式：
        async with MCPClient(command=["python", "mcp_server.py"]) as client:
            tools = await client.list_tools()
            result = await client.call_tool("add", {"a": 1, "b": 2})
    """

    def __init__(self, command: list[str]):
        """
        参数：
            command : 启动 MCP Server 的命令，如 ["python", "server.py"]
        """
        self._command = command
        self._process: subprocess.Popen | None = None
        self._request_id = 0

    # ---- 生命周期（async context manager） ----

    async def __aenter__(self) -> "MCPClient":
        """启动 MCP Server 子进程 + 完成初始化握手"""
        self._process = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # Server 日志走 stderr，不影响 JSON-RPC
            text=True,               # stdin/stdout 自动 str ↔ bytes
        )
        await self.initialize()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """关闭子进程，释放资源"""
        await self.close()

    async def close(self) -> None:
        """终止 MCP Server 子进程"""
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()  # 超时强制杀
            self._process = None

    # ---- JSON-RPC 通信核心 ----

    async def _send_request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        发送 JSON-RPC 2.0 请求并等待响应

        通信流程（全双工管道）：
        1. 构造 {"jsonrpc": "2.0", "method": "...", "params": {...}, "id": N}
        2. 写入子进程 stdin + flush
        3. 阻塞读取子进程 stdout 一行
        4. 解析 JSON 响应

        注意：
        - 每次请求必须带自增 id，Server 按 id 匹配响应
        - stdout.readline() 会阻塞直到 Server 输出完整一行 JSON
        - 如果 Server 同时输出多条消息，需要用更复杂的流解析器（本实现假设一行一响应）
        """
        if not self._process or not self._process.stdin or not self._process.stdout:
            raise RuntimeError("MCP Server 未启动，请使用 async with MCPClient(...)")

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": self._request_id,
        }

        # 发送：JSON 一行写入 stdin
        self._process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self._process.stdin.flush()

        # 接收：stdout 一行 JSON
        line = await asyncio.get_event_loop().run_in_executor(
            None, self._process.stdout.readline
        )
        if not line:
            raise ConnectionError("MCP Server 连接中断（stdout 关闭）")

        response = json.loads(line)

        # JSON-RPC 错误处理
        if "error" in response:
            raise RuntimeError(f"MCP 错误: {response['error']}")
        return response.get("result", {})

    # ---- MCP 协议方法 ----

    async def initialize(self) -> dict[str, Any]:
        """
        MCP 初始化握手 — 协商协议版本 + 获取 Server 能力列表
        MCP 规范要求 connect 后第一条消息必须是 initialize
        """
        return await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
        })

    async def list_tools(self) -> list[dict[str, Any]]:
        """
        获取 MCP Server 提供的工具列表

        返回示例：
        [
            {"name": "add", "description": "两数相加", "inputSchema": {...}},
            {"name": "search", "description": "搜索知识库", "inputSchema": {...}},
        ]
        """
        result = await self._send_request("tools/list")
        return result.get("tools", [])

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """
        调用 MCP Server 的指定工具

        参数：
            name      : 工具名（来自 list_tools 的返回）
            arguments : 工具参数，key-value 格式

        返回：
            工具的原始返回值（取决于 Server 实现）
        """
        result = await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments,
        })
        return result


