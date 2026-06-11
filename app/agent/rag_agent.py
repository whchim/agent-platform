"""RAG Agent — LCEL 管道：检索 + 拼 Prompt + LLM 生成

管道流程：
    用户问题 → 加载历史 → 检索知识库 → 拼 Prompt → LLM 推理 → 存记忆 → 返回答案
"""

import datetime

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.core.tracer import Tracer
from app.memory.long_term import LongTermMemory
from app.memory.short_term import ShortTermMemory
from app.models import AgentRequest, AgentResponse, AgentStep
from app.rag.retriever import MilvusRetriever
from app.rag.reranker import rerank
from conf import settings


# RAG Prompt 模板 — 要求 LLM 基于检索结果回答，不知道就说不知道
RAG_SYSTEM_PROMPT = """你是一个知识库助手。请严格基于以下检索到的文档片段回答问题。
如果文档中找不到相关信息，直接说"我没有找到相关信息"，不要编造。

{context}

当前日期: {date}
"""


class RAGAgent:
    """RAG Agent — 检索增强生成的执行器"""

    def __init__(
        self,
        retriever: MilvusRetriever,
        memory: ShortTermMemory,
        long_term_memory: LongTermMemory | None = None,  # 可选：长时向量记忆
    ):
        # LLM：DeepSeek，OpenAI 兼容接口
        self._llm: ChatOpenAI = ChatOpenAI(
            model=settings.deepseek_model,        # deepseek-v4-pro
            api_key=settings.deepseek_api_key,    # pyright: ignore[reportArgumentType]
            base_url=settings.deepseek_base_url,
            temperature=0.3,  # RAG 场景低温度，减少幻觉
        )
        self._retriever: MilvusRetriever = retriever
        self._memory: ShortTermMemory = memory
        self._long_term_memory: LongTermMemory | None = long_term_memory
        self._parser: StrOutputParser = StrOutputParser()  # LLM 输出 → 纯文本字符串

    async def run(self, request: AgentRequest) -> AgentResponse:
        """
        RAG Agent 主流程：
        1. 加载历史上下文
        2. 检索知识库 → 重排
        3. 拼 Prompt（历史 + 检索片段 + 用户问题）
        4. LLM 推理
        5. 记录中间步骤 + 存记忆
        """
        tracer = Tracer(session_id=request.session_id)
        root_span = tracer.start("rag_agent_run", mode="rag", query=request.query[:50])
        steps: list[AgentStep] = []

        try:
            # ---- Step 1：加载会话历史 ----
            history = await self._memory.load(request.session_id)
            history_text = "\n".join(
                f"{h['role']}: {h['content']}" for h in history[-6:]
            )

            # ---- Step 2：检索知识库 + 重排 ----
            retrieval_span = tracer.start("retrieval", query=request.query[:30])
            raw_docs = self._retriever.retrieve(
                request.query,
                top_k=settings.retrieval_top_k or 5,
            )
            docs = rerank(raw_docs, min_score=0.3, max_docs=3)
            retrieval_span.end(ok=True, doc_count=len(raw_docs), reranked=len(docs))

            context = "\n\n---\n\n".join(
                f"[来源: {d['source']}]\n{d['content']}" for d in docs
            )

            # ---- 额外：搜索长时记忆（可选，失败不阻塞主流程） ----
            if self._long_term_memory is not None:
                try:
                    ltm_results = self._long_term_memory.search(
                        request.query, session_id=request.session_id, top_k=3
                    )
                    if ltm_results:
                        ltm_text = "\n\n---\n\n".join(
                            f"[长期记忆] {r['content']}" for r in ltm_results
                        )
                        context = f"{context}\n\n---\n\n{ltm_text}" if context else ltm_text
                except Exception:
                    pass

            steps.append(AgentStep(
                step_type="retrieval",
                content=f"检索到 {len(docs)} 条相关片段",
                metadata={"sources": [d["source"] for d in docs]},
            ))

            # ---- Step 3 + 4：拼 Prompt + LLM 推理 ----
            prompt = ChatPromptTemplate.from_messages([
                ("system", RAG_SYSTEM_PROMPT),
                ("human", "对话历史:\n{history}\n\n用户问题: {question}"),
            ])
            chain = prompt | self._llm | self._parser

            llm_span = tracer.start("llm_call", model=settings.deepseek_model)
            answer = await chain.ainvoke({
                "context": context or "（暂无相关文档）",
                "date": datetime.date.today().isoformat(),
                "history": history_text or "（无历史）",
                "question": request.query,
            })
            llm_span.end(ok=True, answer_len=len(answer))

            # ---- Step 5：存记忆 + 返回 ----
            response = AgentResponse(
                status="success",
                answer=answer,
                steps=steps,
            )
            await self._memory.add(request.session_id, request, response)

            # ---- 存入长时记忆（可选：只存储有实质内容的回答） ----
            if self._long_term_memory is not None and answer and len(answer) > 50:
                try:
                    await self._long_term_memory.add(
                        request.session_id,
                        f"Q: {request.query}\nA: {answer[:1000]}",
                    )
                except Exception:
                    pass

            root_span.end(ok=True)
            return response

        except Exception as e:
            root_span.end(ok=False, error=str(e)[:200])
            raise



