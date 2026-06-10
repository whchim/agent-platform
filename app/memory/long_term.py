"""Milvus 长时记忆 — 向量存储 + 语义检索，实现长期记忆的存取

设计思路：
- 用 Milvus 存文本向量 + 元信息（session_id / 内容 / 时间戳）
- embedder 从外部注入（工厂函数 get_embedder 产出），方便切换嵌入模型
- 文本先 embed 成向量再存入 Milvus，查询时同样 embed → ANN 搜索
- text-embedding-v4 向量维度 = 1024，写死为常量方便维护
"""

import time
from typing import Any

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    utility,
)

# text-embedding-v4 的输出维度，建 Collection 时用
EMBEDDING_DIM = 1024
# Milvus Collection 名称
COLLECTION_NAME = "long_term_memory"

class LongTermMemory:
    """长期记忆 — 基于 Milvus 向量存储，支持语义添加与检索"""

    def __init__(self, embedder: Any, collection_name: str = COLLECTION_NAME):
        """
        参数：
            embedder       : LangChain Embeddings 实例（由 get_embedder() 产出）
            collection_name: Milvus Collection 名，默认 "long_term_memory"
        """
        self._embedder: Any = embedder
        self._collection_name: str = collection_name
        # 首次使用时懒创建 Collection（schema + index）
        self._collection: Collection | None = None

    def _ensure_collection(self) -> Collection:
        """确保 Collection 已存在并加载到内存，不存在则创建"""
        # 已加载则直接返回，避免重复初始化
        if self._collection is not None:
            return self._collection

        # 检查 Collection 是否已存在（服务重启后不丢失）
        if utility.has_collection(self._collection_name):  # pyright: ignore[reportUnnecessaryComparison]
            self._collection = Collection(self._collection_name)
            self._collection.load()  # 加载到内存，否则无法搜索
            return self._collection

        # 首次创建：定义 schema 字段
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="created_at", dtype=DataType.INT64),  # Unix 时间戳（秒）
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        ]
        schema = CollectionSchema(
            fields=fields,
            description="Agent 长期记忆 — 向量存储 + 语义检索",
        )

        # 创建 Collection
        self._collection = Collection(self._collection_name, schema=schema)

        # 为向量字段建索引 — IVF_FLAT：适用于开发/小规模数据
        index_params = {
            "metric_type": "IP",    # 内积相似度（DashScope embedding 归一化后等价于余弦）
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        self._collection.create_index(
            field_name="embedding",
            index_params=index_params,
        )  # pyright: ignore[reportUnusedCallResult,reportUnusedCoroutine]
        self._collection.load()
        return self._collection


    async def add(self, session_id: str, content: str, metadata: dict[str, Any] | None = None) -> int:
        """
        存入一条长期记忆
        流程：文本 → embed → 向量 + 元信息 → Milvus insert

        返回：Milvus 自动生成的主键 ID
        """
        col = self._ensure_collection()
        # embed_documents 返回 [[...]] 二维列表，取第一个即当前文本的向量
        vector: list[float] = self._embedder.embed_documents([content])[0]

        payload: list[list[Any]] = [
            [session_id],                    # VARCHAR
            [content],                       # VARCHAR
            [int(time.time())],              # INT64 时间戳
            [vector],                        # FLOAT_VECTOR
        ]
        # 忽略未使用的 metadata（预留扩展用）
        _ = metadata
        result = col.insert(payload)
        # primary_keys 可以是单个 int 或列表，取首元素
        return result.primary_keys[0]  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]


    def search(
        self, query: str, session_id: str | None = None, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """
        语义检索与 query 最相似的长期记忆

        流程：query → embed → Milvus ANN 搜索 → 返回最相似的 top_k 条

        返回格式：[{"content": "...", "score": 0.92, "session_id": "...", "created_at": ...}, ...]
        """
        col = self._ensure_collection()
        # 将查询文本转为向量（embed_query 返回一维 list[float]）
        query_vector: list[float] = self._embedder.embed_query(query)

        # 搜索参数：IVF_FLAT 需要 nprobe 控制搜索范围
        search_params = {"metric_type": "IP", "params": {"nprobe": 16}}

        # 可选：按 session_id 过滤，只搜指定会话的记忆
        expr = f'session_id == "{session_id}"' if session_id else None

        results = col.search(
            data=[query_vector],          # 查询向量，需套一层列表
            anns_field="embedding",       # 在哪个向量字段上做 ANN
            param=search_params,
            limit=top_k,
            expr=expr,                    # 标量过滤表达式
            output_fields=["session_id", "content", "created_at"],  # 返回的非向量字段
        )

        # Milvus 返回 SearchResult 对象，遍历命中列表
        hits = results[0]  # pyright: ignore[reportIndexIssue] — pymilvus 类型存根不完整
        return [
            {
                "id": hit.id,
                "content": hit.entity.get("content"),
                "score": hit.score,                          # 相似度得分
                "session_id": hit.entity.get("session_id"),
                "created_at": hit.entity.get("created_at"),
            }
            for hit in hits  # pyright: ignore[reportUnknownVariableType]
        ]


    def clear(self, session_id: str | None = None) -> None:
        """
        清空长期记忆
        - 传 session_id：删除指定会话的所有记忆
        - 不传：删除全部（drop Collection，下次访问自动重建）
        """
        if session_id:
            # 按 session_id 过滤删除
            col = self._ensure_collection()
            expr = f'session_id == "{session_id}"'
            _ = col.delete(expr=expr)
        else:
            # 删除整个 Collection
            if utility.has_collection(self._collection_name):  # pyright: ignore[reportUnnecessaryComparison]
                _ = utility.drop_collection(self._collection_name)  # pyright: ignore[reportUnknownVariableType]
            self._collection = None  # 下次访问 lazy 重建



