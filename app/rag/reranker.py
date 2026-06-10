"""重排序器 — 对检索结果进行二次排序/过滤，提升 RAG 答案质量

当前实现：分数阈值过滤 + 去重 + 降序排列
后续可扩展：接入 Cross-Encoder 模型做语义重排（如 bge-reranker）
"""

from typing import Any


def rerank(
    docs: list[dict[str, Any]],
    min_score: float = 0.0,
    max_docs: int = 3,
) -> list[dict[str, Any]]:
    """
    对检索结果进行重排

    参数：
        docs      : retriever.retrieve() 的原始结果
        min_score : 最小相似度阈值，低于此值的文档被丢弃
        max_docs  : 最终返回的最大文档数

    返回：
        重排后的文档列表（按 score 降序）
    """
    # 1. 按分数降序排序
    sorted_docs = sorted(docs, key=lambda d: d.get("score", 0), reverse=True)

    # 2. 分数阈值过滤：低于 min_score 的丢弃
    filtered = [d for d in sorted_docs if d.get("score", 0) >= min_score]

    # 3. 去重：相同内容只保留分数最高的
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for d in filtered:
        content = d.get("content", "")
        if content not in seen:
            seen.add(content)
            unique.append(d)

    # 4. 截断到 max_docs
    return unique[:max_docs]



