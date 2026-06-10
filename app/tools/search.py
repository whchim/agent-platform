"""知识库搜索 — 调用 RAG 管道（retriever + reranker），供 Agent 工具调用"""

from typing import Any

from app.rag.retriever import MilvusRetriever
from app.rag.reranker import rerank


def search_knowledge(
    query: str,
    retriever: MilvusRetriever,
    top_k: int = 5,
    min_score: float = 0.3,
    max_docs: int = 3,
) -> list[dict[str, Any]]:
    """
    搜索知识库 — 检索 + 重排

    参数：
        query     : 搜索查询（Agent 把用户意图转成关键词）
        retriever : MilvusRetriever 实例（外部注入）
        top_k     : 向量检索的候选数
        min_score : 重排时的最小相似度
        max_docs  : 最终返回的最大文档数

    返回：
        [{"content": "...", "source": "...", "score": 0.9}, ...]
    """
    # Step 1：粗排 — Milvus ANN 检索
    docs = retriever.retrieve(query, top_k=top_k)

    # Step 2：精排 — 分数过滤 + 去重 + 截断
    return rerank(docs, min_score=min_score, max_docs=max_docs)



