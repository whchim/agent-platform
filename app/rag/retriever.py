"""Milvus 向量检索器 — 文档入库 + 语义检索，RAG 管道的检索引擎

与 long_term.py 的区别：
- long_term 存的是"个人记忆"（带 session_id）
- retriever 存的是"文档片段"（带 source 来源），供 RAG 知识库用
"""

from typing import Any

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    utility,
)

# text-embedding-v4 输出维度
EMBEDDING_DIM = 1024
# 默认 Collection 名
DEFAULT_COLLECTION = "rag_documents"


class MilvusRetriever:
    """Milvus 向量检索器 — embed + insert + ANN 搜索"""

    def __init__(self, embedder: Any, collection_name: str = DEFAULT_COLLECTION):
        self._embedder: Any = embedder
        self._collection_name: str = collection_name
        self._collection: Collection | None = None

    def _ensure_collection(self) -> Collection:
        """确保 Collection 存在并加载"""
        if self._collection is not None:
            return self._collection

        if utility.has_collection(self._collection_name):  # pyright: ignore[reportUnnecessaryComparison]
            self._collection = Collection(self._collection_name)
            self._collection.load()
            return self._collection

        # 首次创建 — schema 包含 content + source + embedding
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),  # 来源标识
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        ]
        schema = CollectionSchema(fields=fields, description="RAG 文档向量存储")
        self._collection = Collection(self._collection_name, schema=schema)

        index_params = {"metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
        self._collection.create_index(  # pyright: ignore[reportUnusedCallResult,reportUnusedCoroutine]
            field_name="embedding", index_params=index_params
        )
        self._collection.load()
        return self._collection

    def add_documents(self, texts: list[str], source: str = "unknown") -> list[int]:
        """
        批量入库文档片段
        流程：batch embed → Milvus insert
        返回：插入的主键 ID 列表
        """
        col = self._ensure_collection()
        vectors: list[list[float]] = self._embedder.embed_documents(texts)

        import time
        payload: list[list[Any]] = [
            texts,                           # content
            [source] * len(texts),           # source: 所有 chunk 同源
            vectors,                         # embedding
        ]
        result = col.insert(payload)
        return list(result.primary_keys)  # pyright: ignore[reportUnknownMemberType]

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """
        语义检索最相关的文档片段
        返回：[{"content": "...", "source": "...", "score": 0.95}, ...]
        """
        col = self._ensure_collection()
        query_vector: list[float] = self._embedder.embed_query(query)

        search_params = {"metric_type": "IP", "params": {"nprobe": 16}}
        results = col.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["content", "source"],
        )

        hits = results[0]  # pyright: ignore[reportIndexIssue]
        return [
            {
                "content": hit.entity.get("content"),  # pyright: ignore[reportUnknownMemberType]
                "source": hit.entity.get("source"),     # pyright: ignore[reportUnknownMemberType]
                "score": hit.score,                     # pyright: ignore[reportUnknownMemberType]
            }
            for hit in hits  # pyright: ignore[reportUnknownVariableType]
        ]


