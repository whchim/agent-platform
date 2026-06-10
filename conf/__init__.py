"""应用配置 — Pydantic Settings 从环境变量加载（LLM / Redis / Milvus / PostgreSQL）

设计思路：
- 使用 pydantic-settings 的 BaseSettings，自动从 .env 文件读取环境变量
- 字段名自动映射：大写 + 下划线 → 小写 + 下划线（REDIS_URL → redis_url）
- 类型自动转换：.env 中全是字符串，BaseSettings 按字段类型注解自动转成 int / bool / float
- 模块级单例 settings，全局 import 使用，避免多次读取 .env

使用方式：
    from conf import settings
    print(settings.redis_url)       # redis://localhost:6379/0
    print(settings.milvus_port)     # 19530 (int, 不是 "19530")
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    全局配置类 — 所有字段自动从 .env 读取，运行时不可变

    分组说明：
    - 基础设施：Redis / PostgreSQL / Milvus 连接参数
    - LLM：DeepSeek API 配置（OpenAI 兼容接口）
    - Embedding：嵌入模型提供商 + 模型名（支持动态切换）
    - 业务参数：检索 top-k / 重排开关 / 熔断阈值
    - 服务：host / port
    """
    model_config = SettingsConfigDict(  # pyright: ignore[reportUnannotatedClassAttribute]
        env_file=".env",               # 指定 .env 文件路径（相对项目根目录）
        env_file_encoding="utf-8",     # 中文注释不会乱码
    )

    # ---- Redis ----
    redis_url: str                     # Redis 连接串, 例: redis://localhost:6379/0

    # ---- PostgreSQL ----
    pg_dsn: str                        # PostgreSQL DSN, 例: postgresql://user:pass@localhost:5432/db

    # ---- Milvus ----
    milvus_host: str                   # Milvus 主机地址, 例: localhost
    milvus_port: int                   # Milvus gRPC 端口, 默认 19530

    # ---- LLM — DeepSeek（OpenAI 兼容） ----
    deepseek_api_key: str              # DeepSeek API Key
    deepseek_base_url: str             # DeepSeek API 地址, 例: https://api.deepseek.com/v1
    deepseek_model: str                # 模型名, 例: deepseek-chat

    # ---- Embedding — 阿里云 DashScope（主用，支持动态切换） ----
    # provider 可选值说明：
    #   dashscope              → 阿里云 text-embedding-v4（在线 API）
    #   openai_compatible      → OpenAI 兼容嵌入（如 DeepSeek embedding）
    #   sentence_transformers  → 本地模型（离线，吃内存）
    embedding_provider: Literal[
        "dashscope", "openai_compatible", "sentence_transformers"
    ]
    embedding_model: str = "text-embedding-v4"   # 嵌入模型名，默认阿里云 text-embedding-v4
    dashscope_api_key: str                       # 阿里云 DashScope API Key

    # ---- Retrieval — 检索参数 ----
    retrieval_top_k: int             # 检索返回的 Top-K 文档数, 例: 5
    rerank_enabled: bool             # 是否启用重排序（ContextualCompressionRetriever）

    # ---- Circuit Breaker — 熔断器（后续用于工具调用保护） ----
    cb_failure_threshold: int        # 连续失败 N 次后熔断, 例: 5
    cb_recovery_timeout: float       # 熔断后等待 N 秒进入半开状态, 例: 30.0

    # ---- Tracing — 可插拔链路追踪 ----
    # backend 可选值：
    #   console → 控制台 print 输出
    #   otel    → OpenTelemetry gRPC 导出（Jaeger / Grafana Tempo）
    tracer_backend: str = "console"
    otel_endpoint: str = "http://localhost:4317"   # OTel Collector / Jaeger gRPC 地址

    # ---- Server ----
    host: str                        # 服务监听地址, 例: 0.0.0.0
    port: int                        # 服务监听端口, 例: 8000


# 模块级单例 — 整个项目唯一入口
# pyright 误报：BaseSettings 在 __init__ 中自动从 .env 填充，不需要传参
settings = Settings()  # pyright: ignore[reportCallIssue]
