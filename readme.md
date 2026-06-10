# Agent Platform

基于 **Python + FastAPI + LangChain + LangGraph + Milvus + Redis + PostgreSQL** 技术栈的智能 Agent 服务平台，提供 ReAct 推理、RAG 检索增强生成、多轮对话记忆等能力。

## 技术栈

| 组件 | 用途 |
|------|------|
| **Python 3.12+** | 开发语言 |
| **FastAPI** | Web 框架，提供 REST API |
| **LangChain** | LLM 集成、工具调用、RAG 管道 |
| **LangGraph** | ReAct Agent 状态图编排 |
| **DeepSeek** | LLM 推理（OpenAI 兼容 API） |
| **阿里云 DashScope** | 嵌入模型（text-embedding-v4），支持动态切换 |
| **Milvus** | 向量数据库，存储文档向量与长时记忆 |
| **Redis** | 短时对话记忆缓存 |
| **PostgreSQL (asyncpg)** | 持久化存储（会话、文档元数据） |
| **Gradio** | Demo 展示界面，可视化交互体验 |

## 项目结构

```
agent-platform/
├── app/                          # 应用主包
│   ├── main.py                   # FastAPI 入口（服务启动、路由注册、生命周期管理）
│   ├── gradio_app.py             # Gradio 演示界面（独立运行的 Web UI）
│   ├── agent/                    # Agent 模块（LangGraph 编排）
│   │   ├── executor.py           # Agent 编排器（auto / react / rag 模式路由）
│   │   ├── rag_agent.py          # RAG Agent（LCEL 管道 + Milvus）
│   │   └── react_agent.py        # ReAct Agent（LangGraph StateGraph 循环）
│   ├── core/                     # 核心基础设施
│   │   ├── circuit_breaker.py    # 三态熔断器（CLOSED/OPEN/HALF_OPEN）
│   │   └── tracer.py             # 链路追踪（LangSmith / OpenTelemetry）
│   ├── memory/                   # 记忆管理
│   │   ├── short_term.py         # Redis 短时滑窗记忆
│   │   └── long_term.py          # Milvus 长时向量记忆
│   ├── models/                   # 数据模型
│   │   └── __init__.py           # Pydantic 请求/响应 + PostgreSQL ORM
│   ├── rag/                      # 检索增强生成
│   │   ├── chunker.py            # 文本分块（RecursiveCharacterTextSplitter）
│   │   ├── embedder.py           # 嵌入模型工厂（dashscope / openai_compatible / sentence_transformers）
│   │   ├── reranker.py           # 重排序（ContextualCompressionRetriever）
│   │   └── retriever.py          # Milvus 向量检索
│   └── tools/                    # LangChain 工具
│       ├── calculator.py         # 安全计算器（AST 白名单）
│       ├── mcp_client.py         # MCP 协议客户端
│       └── search.py             # 知识库搜索
├── conf/                         # 配置
│   └── __init__.py               # Pydantic Settings（.env → Settings 单例）✅
├── data/                         # 本地数据目录
├── .env.example                  # 环境变量模板
├── .env                          # 环境变量（本地，不入库）
└── requirements.txt              # Python 依赖
```

## 快速开始

### 环境要求

- Python 3.12+
- Redis 服务
- Milvus 向量数据库
- PostgreSQL 数据库

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置

1. 复制环境变量模板：
   ```bash
   cp .env.example .env
   ```
2. 按需修改 `.env` 中的配置（LLM API Key、数据库连接串等）

### 启动服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 启动 Demo 界面

```bash
python -m app.gradio_app
# 或指定后端地址与端口
python app/gradio_app.py --port 7860 --api-url http://localhost:8000
```

### API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/agent/run` | 运行 Agent |

## 许可证

MIT
