"""MCP 工具桥接 — 将 MCP Server 的远程工具包装成 LangChain StructuredTool

解决的核心问题：
    MCP Server 通过 JSON-RPC 暴露工具，但 ReAct Agent 需要的是 LangChain Tool 对象。
    本模块做适配：MCP 工具规格 → LangChain StructuredTool，Agent 调用时自动转发到 MCP Server。

为什么用 StructuredTool 而不是 Tool：
    LangChain 的 Tool（simple.py）是单输入兼容类，只接受一个字符串参数。
    LLM 调用工具时传入的是 dict（如 {"code": "...", "language": "python"}），
    Tool._to_args_and_kwargs 会把 dict 的 values 拍平成位置参数 → 多参数工具直接崩溃，
    单参数工具也会丢失参数名。StructuredTool 则原样保留 dict kwargs，正确透传。

    三层工具体系：
        原生 Tool (calculator/search) → 编译时注册
        MCP 远程 Tool (代码解释器/文件系统) → 运行时发现
        Skill (多步子任务) → 语义封装
"""

import asyncio
import json
from typing import Any

from langchain_core.tools import StructuredTool

from app.tools.mcp_client import MCPClient


def _build_mcp_tool(name: str, description: str, client: MCPClient) -> StructuredTool:
    """将单个 MCP 远程工具包装成 StructuredTool，支持任意 dict 参数"""

    def _sync_call(args_dict: dict[str, Any]) -> str:
        """同步桥：把 MCPClient 的 async call_tool 包成同步调用"""
        result = asyncio.run(client.call_tool(name, args_dict))
        # 统一转字符串，兼容各种返回类型
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False, default=str)
        return str(result)

    return StructuredTool.from_function(
        func=_sync_call,
        name=name,
        description=description,
        infer_schema=False,           # MCP 工具 schema 动态发现，不从函数签名推断
        args_schema={
            "type": "object",
            "properties": {},         # 不固定参数 → 运行时由 LLM 根据 description 决定
            "additionalProperties": True,
        },
    )


def wrap_mcp_tools(client: MCPClient) -> list[StructuredTool]:
    """
    从 MCP Server 拉取工具列表，逐个包装成 LangChain StructuredTool

    参数：
        client : 已完成初始化握手的 MCPClient 实例

    返回：
        LangChain StructuredTool 列表，可直接传入 ReActAgent(tools=...)
    """
    mcp_tools = asyncio.run(client.list_tools())
    wrapped: list[StructuredTool] = []

    for spec in mcp_tools:
        name = spec["name"]
        description = spec.get("description", f"MCP 远程工具: {name}")

        # 如果 Server 提供了 inputSchema，拼进 description 帮助 LLM 理解参数
        schema = spec.get("inputSchema", {})
        if schema.get("properties"):
            props_desc = ", ".join(
                f'{k}: {v.get("description", v.get("type", "any"))}'
                for k, v in schema["properties"].items()
            )
            description += f"\n参数: {props_desc}"

        tool = _build_mcp_tool(name, description, client)
        wrapped.append(tool)

    return wrapped
