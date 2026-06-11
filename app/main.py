"""FastAPI 主入口 — 服务启动、路由注册、生命周期管理（Milvus / Redis / PostgreSQL 连接初始化）

启动流程：
1. lifespan 阶段：依次初始化 PG 连接池 → Milvus gRPC 连接 → Redis 异步客户端
2. 挂载到 app.state：路由通过 request.app.state 获取这些资源
3. shutdown 时逆序释放：Redis → Milvus → PG（后初始化的先关闭）

路由：
- GET  /health      : 健康检查（K8s / Docker 探针用）
- POST /agent/run   : Agent 执行入口（支持 ?stream=true 走 SSE 流式）
"""

import warnings
from contextlib import asynccontextmanager

import tempfile
from pathlib import Path

# 抑制 PyMilvus ORM API 的 DeprecationWarning（不影响功能）
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pymilvus")

from fastapi import FastAPI, File, Request, UploadFile
from starlette.responses import StreamingResponse

import asyncpg                       # PostgreSQL 纯异步驱动
import redis.asyncio as aioredis     # Redis 异步客户端
from pymilvus import connections     # Milvus 连接管理（全局连接池模式）

from conf import settings
from app.agent.executor import AgentExecutor
from app.memory.short_term import ShortTermMemory
from app.models import AgentRequest, StreamEvent
from app.rag.chunker import chunk_text
from app.rag.document_loader import load_document
from app.rag.embedder import get_embedder
from app.rag.retriever import MilvusRetriever
from app.tools.search import search_knowledge




# ========== 生命周期 ==========

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期上下文管理器
    - yield 之前 = 服务启动时执行（初始化连接）
    - yield 之后 = 服务关闭时执行（释放资源）
    注意：yield 前后的变量作用域是同一个，关闭阶段直接复用引用
    """

    # ---- 阶段 1：启动 — 初始化连接 ----
    # TODO: PostgreSQL 连接池已初始化但尚未用于任何数据持久化。
    # 后续计划：添加 SQLAlchemy ORM 模型（用户表 / Agent 运行日志 / 工具调用统计等），
    # 届时通过 request.app.state.pg_pool 执行 SQL 查询。
    # 当前保留连接池是为了避免后续重复修改生命周期管理代码。
    pg_pool = await asyncpg.create_pool(
        dsn=settings.pg_dsn,
        min_size=2,         # 启动时最少预热 2 个连接
        max_size=10,        # 最多 10 个，超出排队
    )

    # Milvus：gRPC 连接（同步 API，不需要 await）
    # 注意：这是 pymilvus v2 的旧式 ORM API，后续应迁移到 MilvusClient
    connections.connect(
        alias="default",
        host=settings.milvus_host,
        port=settings.milvus_port,
    )

    # Redis：异步客户端，decode_responses=True 自动 bytes → str
    redis_client = aioredis.from_url(
        settings.redis_url,
        decode_responses=True,
    )

    # ---- 组装业务层（依赖注入） ----
    embedder = get_embedder()
    retriever = MilvusRetriever(embedder)
    memory = ShortTermMemory(redis_client)

    # 长时记忆：基于 Milvus 向量存储，语义检索历史知识（可选，失败不阻塞主流程）
    from app.memory.long_term import LongTermMemory
    long_term_memory = LongTermMemory(embedder)

    # 把 search_knowledge 包装成 LangChain Tool
    # 注意：@tool 装饰器生成的是 StructuredTool（非旧版 Tool），会根据函数签名自动
    # 推断 args_schema，因此 LLM 传入 dict 参数（如 {"query": "..."}）能正确路由
    from langchain_core.tools import tool

    @tool
    def search_tool(query: str) -> str:
        """搜索知识库，获取相关文档。传入自然语言查询，返回相关片段和来源。"""
        docs = search_knowledge(query, retriever=retriever)
        if not docs:
            return "未找到相关文档。"
        return "\n\n---\n\n".join(
            f"[来源: {d['source']}] {d['content']}" for d in docs
        )

    executor = AgentExecutor(
        retriever=retriever,
        memory=memory,
        tools=[search_tool],  # ReAct Agent 可以调用 RAG 搜索
        long_term_memory=long_term_memory,
    )

    # ---- 阶段 2：运行 — 挂载到 app.state ----
    # 路由里通过 request.app.state.pg_pool / request.app.state.redis 访问
    # Milvus 不挂载 — pymilvus 内部是全局连接池，直接用 connections.connect("default") 引用
    app.state.pg_pool = pg_pool
    app.state.redis = redis_client
    app.state.retriever = retriever
    app.state.memory = memory
    app.state.long_term_memory = long_term_memory
    app.state.executor = executor

    yield   # ← 服务在此运行

    # ---- 阶段 3：关闭 — 清理 + 释放 ----
    # 先清理 app.state 引用，避免后续代码意外访问已关闭的连接
    del app.state.pg_pool
    del app.state.redis
    del app.state.long_term_memory

    # 逆序释放：先关最后初始化的 Redis，最后关先初始化的 PG
    await redis_client.aclose()
    connections.disconnect(alias="default")
    await pg_pool.close()


# ========== FastAPI 实例 ==========

app = FastAPI(lifespan=lifespan)


# ========== 流式生成器 ==========

async def _stream_agent_response(executor: AgentExecutor, payload: AgentRequest):
    """
    SSE 流式生成器 — 调用 executor 后，把 AgentResponse.steps 转成 SSE 事件流

    流程：
    1. await executor.run(payload) → AgentResponse（含 steps）
    2. 把每个 AgentStep 打包成 StreamEvent
    3. 推送 text_delta（最终答案）+ done 事件
    """
    response = await executor.run(payload)

    # 逐个推送中间步骤
    for step in response.steps:
        ev = StreamEvent(
            event=f"{step.step_type}_delta",
            content=step.content,
            metadata=step.metadata or {},
        )
        yield f"data: {ev.model_dump_json()}\n\n"

    # 推送最终答案
    yield f"data: {StreamEvent(event='text_delta', content=response.answer or '').model_dump_json()}\n\n"
    yield f"data: {StreamEvent(event='done', content='').model_dump_json()}\n\n"


# ========== 路由 ==========

@app.get("/health")
async def health():
    """
    健康检查端点 — 返回 {"status": "ok"}
    用途：Docker healthcheck / K8s liveness probe / 确认服务存活
    """
    return {"status": "ok"}


@app.post("/agent/run")
async def agent_run(payload: AgentRequest, request: Request, stream: bool = False):
    """
    Agent 运行入口 — 根据 mode 字段路由到 ReAct / RAG 执行器

    参数：
    - payload : JSON body，AgentRequest 模型自动校验
    - request  : FastAPI Request 对象，用于访问 request.app.state 中的 executor
    - stream   : 查询参数 ?stream=true 走 SSE 流式，默认 false 走非流式
    """
    executor: AgentExecutor = request.app.state.executor

    if stream:
        return StreamingResponse(
            _stream_agent_response(executor, payload),
            media_type="text/event-stream",
        )
    else:
        return await executor.run(payload)


# ========== 文档上传 ==========

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...), request: Request = None):  # pyright: ignore[reportArgumentType]
    """
    上传文档并入库到 RAG 知识库

    支持格式：.txt / .md / .docx / .pdf / .pptx
    流程：接收文件 → 解析文本 → 分块 → 向量化 → 存入 Milvus
    """
    if not file.filename:
        return {"status": "error", "message": "未提供文件名"}

    # 保存到临时文件（load_document 需要文件路径）
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 1. 解析文档 → 纯文本
        text = load_document(tmp_path)

        # 2. 文本分块
        chunks = chunk_text(text)

        # 3. 入库到 Milvus（retriever 已在 lifespan 中挂载到 app.state）
        retriever: MilvusRetriever = request.app.state.retriever
        ids = retriever.add_documents(chunks, source=file.filename)

        return {
            "status": "ok",
            "filename": file.filename,
            "chunks": len(chunks),
            "ids": ids,
        }
    finally:
        # 清理临时文件
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/history/{session_id}")
async def get_history(session_id: str, request: Request):
    """获取指定会话的对话历史，用于刷新后恢复聊天界面"""
    memory: ShortTermMemory = request.app.state.memory
    history = await memory.load(session_id)
    return {"history": history}
