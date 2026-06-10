"""Pydantic 数据模型 — AgentRequest、AgentResponse、中间步骤、流式事件

使用场景总结：
- POST /agent/run  →  接收 AgentRequest，返回 AgentResponse（或 SSE StreamingResponse）
- ReAct Agent      →  AgentStep 记录思考链：thought → tool_call → tool_result → ...
- Gradio 前端      →  StreamEvent 逐条推送，前端按 event 类型渲染不同组件
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ========== 枚举 ==========

class AgentMode(str, Enum):
    """Agent 运行模式 — executor 据此路由到不同的执行器"""
    REACT = "react"   # LangGraph StateGraph 循环推理 + 工具调用
    RAG = "rag"       # LCEL 管道：检索 → 拼 Prompt → 生成


# ========== 请求模型 ==========

class AgentRequest(BaseModel):
    """
    Agent 请求体 — POST /agent/run 的 JSON body

    示例：
    {
        "mode": "react",
        "query": "计算 123 * 456",
        "session_id": "abc-123",
        "user_id": "user-001",
        "config": {"tools": ["calculator", "search"]}
    }
    """
    mode: AgentMode = Field(description="Agent 运行模式：react 或 rag")
    query: str = Field(description="用户输入的问题或指令")
    session_id: str = Field(description="会话 ID，关联历史上下文（Redis key）")
    user_id: str | None = Field(default=None, description="用户标识，预留用于多租户")
    config: dict[str, Any] | None = Field(
        default=None, description="模式相关配置（RAG: top_k / ReAct: tools 列表等）"
    )


# ========== 响应模型 ==========

class AgentStep(BaseModel):
    """
    Agent 中间步骤 — 记录 ReAct 或 RAG 管道的每一步输出

    step_type 取值：
    - thought      : LLM 的思考文字
    - tool_call    : Agent 决定调用哪个工具 + 参数
    - tool_result  : 工具返回的观察结果
    - retrieval    : RAG 检索到的文档片段
    """
    step_type: str = Field(description="步骤类型：thought / tool_call / tool_result / retrieval")
    content: str = Field(description="步骤内容文本（思考原文 / 工具调用 JSON / 检索片段）")
    metadata: dict[str, Any] | None = Field(default=None, description="附加元信息（如 tool_name / source）")


class AgentResponse(BaseModel):
    """
    Agent 响应体 — 非流式模式的完整返回

    示例：
    {
        "status": "success",
        "answer": "123 * 456 = 56088",
        "steps": [{"step_type": "thought", "content": "我需要调用计算器..."}, ...],
        "error": null,
        "metadata": {"elapsed": 2.3, "tokens": 156}
    }
    """
    status: str = Field(description="执行状态：success 或 failed")
    answer: str | None = Field(default=None, description="最终回答文本，失败时为 None")
    steps: list[AgentStep] = Field(default_factory=list, description="中间步骤链（思考 → 工具调用 → 观察 → ...）")
    error: str | None = Field(default=None, description="错误信息，成功时为 None")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元信息（耗时秒数 / token 消耗 / 模型名等）")


# ========== 流式事件模型 ==========

class StreamEvent(BaseModel):
    """
    SSE 流式事件 — 每个事件代表一次增量推送，前端按 event 类型渲染

    event 类型对应前端行为：
    - text_delta      → 追加到答案正文区（逐字效果）
    - thought_delta   → 追加到思考过程区（折叠面板）
    - tool_call       → 展示工具调用卡片（名称 + 参数）
    - tool_result     → 展示工具返回结果
    - retrieval       → 展示检索到的文档引用
    - done            → 关闭 SSE 连接，标记流结束
    - error           → 展示错误信息
    """
    event: str = Field(description="事件类型: text_delta / thought_delta / tool_call / tool_result / retrieval / done / error")
    content: str = Field(default="", description="本次推送的增量文本内容")
    metadata: dict[str, Any] = Field(default_factory=dict, description="附加信息（如 tool_name / sources / elapsed）")


