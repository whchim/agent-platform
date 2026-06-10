"""嵌入模型工厂 — 支持 DashScope / OpenAI 兼容 / Sentence-Transformers 动态切换

设计思路：
- 工厂模式：调用方只传 provider 名称，不关心底层实现
- 所有实现返回 LangChain 的 Embeddings 接口，确保 embed_text / embed_documents 方法统一
- provider 默认从 settings.EMBEDDING_PROVIDER 读取，改 .env 即可切换，无需改代码
"""

from typing import Literal

from conf import settings

from langchain_community.embeddings import DashScopeEmbeddings

# ========== 工厂入口 ==========

def get_embedder(
    provider: Literal["dashscope", "openai_compatible", "sentence_transformers"] | None = None,
):
    """
    嵌入模型工厂 — 根据 provider 名称返回对应的 LangChain Embeddings 实例

    参数：
        provider: 嵌入提供商标识，可选，默认从 settings.embedding_provider 读取

    返回：
        LangChain Embeddings 实例（全部实现了 embed_documents / embed_query 方法）
    """
    # 未传 provider 时从全局配置读取
    provider = provider or settings.embedding_provider

    if provider == "dashscope":
        # 阿里云 DashScope — 在线 API，模型 text-embedding-v4
        # 注意：langchain_community 内部已封装 HTTP 调用，无需手写 SDK
        return DashScopeEmbeddings(
            model=settings.embedding_model,            # text-embedding-v4
            dashscope_api_key=settings.dashscope_api_key,
        )

    elif provider == "openai_compatible":
        # OpenAI 兼容接口 — 支持 DeepSeek Embedding 等
        # 复用 ChatOpenAI 同款 base_url / api_key 模式，连接任意兼容服务
        raise NotImplementedError("OpenAI-compatible embedder not implemented yet")

    elif provider == "sentence_transformers":
        # 本地模型 — 调用 sentence-transformers 库，免网络
        # 适合离线开发 / 快速验证，缺点：吃内存、吃 CPU
        raise NotImplementedError("Sentence-transformers embedder not implemented yet")

    raise ValueError(f"不支持的嵌入类型: {provider}")  # pyright: ignore[reportUnreachable]

