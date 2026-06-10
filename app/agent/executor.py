"""Agent 编排器 — 根据 AgentRequest.mode 路由到对应执行器

职责：
- 初始化 RAGAgent / ReActAgent（注入 retriever、memory、tools）
- run(request) 时按 mode 分发到对应 Agent
- 统一返回 AgentResponse
"""

from app.agent.rag_agent import RAGAgent
from app.agent.react_agent import ReActAgent
from app.memory.short_term import ShortTermMemory
from app.models import AgentMode, AgentRequest, AgentResponse
from app.rag.retriever import MilvusRetriever


class AgentExecutor:
    """
    Agent 编排器 — 模式路由 + 依赖组装

    使用方式：
        executor = AgentExecutor(retriever, memory, tools)
        response = await executor.run(request)
    """

    def __init__(
        self,
        retriever: MilvusRetriever,
        memory: ShortTermMemory,
        tools: list | None = None,  # pyright: ignore[reportMissingTypeArgument] — LangChain Tool 列表
    ):
        self._rag_agent: RAGAgent = RAGAgent(retriever=retriever, memory=memory)
        self._react_agent: ReActAgent = ReActAgent(
            tools=tools or [],
            memory=memory,
        )

    async def run(self, request: AgentRequest) -> AgentResponse:
        """
        根据 mode 路由：
        - react → ReActAgent（推理-行动循环 + 工具调用）
        - rag   → RAGAgent（检索 + 拼 Prompt + 生成）
        """
        if request.mode == AgentMode.REACT:
            return await self._react_agent.run(request)
        elif request.mode == AgentMode.RAG:
            return await self._rag_agent.run(request)
        else:
            raise ValueError(f"不支持的 Agent 模式: {request.mode}")



