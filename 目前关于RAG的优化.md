# RAG 优化记录

---

## 1. 中文感知的递归文本分块（chunker）

**问题：** 朴素按字符数切分会把完整句子拦腰截断，导致检索到的 chunk 语义残缺。

**方案：** 基于 LangChain `RecursiveCharacterTextSplitter`，中文句号 `。` 作为高优先级分隔符。

### 改动点

| 文件 | 位置 | 内容 |
|------|------|------|
| `rag/chunker.py` | 全文 | `chunk_text()` 函数，`chunk_size=500, chunk_overlap=50`，分隔符优先级 `["\n\n", "\n", "。", ".", " ", ""]` |

### 设计原理

```
分隔符优先级（从高到低）：
  "\n\n"  →  段落边界，最理想的切分点
  "\n"    →  换行
  "。"    →  中文句号（核心：优先在句子边界切分）
  "."     →  英文句号
  " "     →  空格
  ""      →  逐字符（最终兜底）

chunk_overlap=50：相邻块重叠 50 字符，防止关键信息恰好落在边界被截断
```

---

## 2. 多格式文档加载器（document_loader）

**问题：** 知识库文档来源多样（.txt / .md / .docx / .pdf / .pptx），没有统一入口。

**方案：** `load_document()` 根据后缀自动路由到对应解析器。

### 改动点

| 文件 | 位置 | 内容 |
|------|------|------|
| `rag/document_loader.py` | 全文 | 统一入口 `load_document()` + 4 个格式解析器 |

### 支持格式

| 格式 | 解析库 | 策略 |
|------|--------|------|
| .txt / .md | 内置 `Path.read_text()` | 直接读取 |
| .docx | `python-docx` | 逐段提取 `paragraph.text` |
| .pdf | `pdfplumber` | 逐页 `extract_text()` |
| .pptx | `python-pptx` | 逐页逐形状 `text_frame.text` |

### 设计细节

- 抑制 `pdfplumber` 的 FontBBox 无害警告，避免日志噪音
- 新增格式只需加一个 `elif` 分支 + 一个 `_load_xxx()` 函数，无需改其他代码

---

## 3. 可插拔嵌入模型工厂（embedder）

**问题：** 嵌入模型是 RAG 质量的核心变量，不同场景需要不同方案（在线 API / 本地模型 / OpenAI 兼容），硬编码会导致切换成本极高。

**方案：** 工厂模式，`get_embedder(provider)` 根据 provider 名称返回 LangChain `Embeddings` 接口实例。

### 改动点

| 文件 | 位置 | 内容 |
|------|------|------|
| `rag/embedder.py` | 全文 | `get_embedder()` 工厂函数 |

### 三种 provider

| provider | 实现 | 状态 |
|----------|------|------|
| `dashscope` | 阿里云 `text-embedding-v4`（1024 维，在线 API） | **已实现** |
| `openai_compatible` | DeepSeek Embedding 等兼容接口 | 预留接口 |
| `sentence_transformers` | 本地 `bge-large-zh` 等（离线，吃内存） | 预留接口 |

### 设计细节

- 所有实现返回 LangChain `Embeddings` 接口（`embed_documents` / `embed_query`），调用方无感知
- `provider` 默认从 `settings.embedding_provider` 读取，改 `.env` 即可切换，不改代码
- Milvus 的 `EMBEDDING_DIM = 1024` 写死为常量，切换模型时需同步修改

---

## 4. Milvus 向量检索器（retriever）

**问题：** 文档入库和语义检索是 RAG 管道的核心，需要高效的 ANN 搜索 + 结构化元信息存储。

**方案：** 基于 pymilvus，首次使用时自动创建 Collection + Schema + 索引，后续复用。

### 改动点

| 文件 | 位置 | 内容 |
|------|------|------|
| `rag/retriever.py` | 全文 | `MilvusRetriever` 类 |

### Schema 设计

| 字段 | 类型 | 用途 |
|------|------|------|
| `id` | INT64, 主键, auto_id | Milvus 自动生成 |
| `content` | VARCHAR(4096) | 文档片段文本 |
| `source` | VARCHAR(512) | 来源标识（文件名 / URL） |
| `embedding` | FLOAT_VECTOR(1024) | text-embedding-v4 向量 |

### 索引策略

```
metric_type: IP（内积，DashScope embedding 归一化后等价于余弦相似度）
index_type:  IVF_FLAT（适合中小规模，< 100 万向量）
nlist:       128（聚类中心数）
nprobe:      16（搜索时扫描的聚类数）
```

### 设计细节

- `_ensure_collection()` 懒初始化：首个操作时才建 Collection，避免启动时阻塞
- 服务重启后检测 Collection 已存在 → 直接 `load()` 不重建，数据不丢失
- `add_documents()` 支持批量入库，一次 embed + insert 完成

---

## 5. 重排序器（reranker）

**问题：** 向量检索只做粗排，召回的相关文档可能包含低分噪音或重复内容，直接塞给 LLM 会降低答案质量。

**方案：** 检索后加一层后处理：分数过滤 + 去重 + 截断。

### 改动点

| 文件 | 位置 | 内容 |
|------|------|------|
| `rag/reranker.py` | 全文 | `rerank()` 函数 |

### 处理流程

```
原始 docs（top_k=5）
  → 按 score 降序排列
  → 过滤 score < min_score(0.3) 的文档
  → 相同 content 去重（保留分数最高的）
  → 截断到 max_docs(3)
  → 最终 docs
```

### 设计细节

- `min_score=0.3`：低于此分数的文档与问题几乎无关，过滤后减少 LLM 噪音
- `max_docs=3`：限制上下文长度，避免 prompt 过长 + 降低 token 消耗
- 去重用 `set` 基于 content 字符串，O(n) 时间复杂度
- 预留注释标注后续可接入 Cross-Encoder 模型（如 bge-reranker）做真正语义重排

---

## 6. LCEL 管道 + 幻觉抑制（rag_agent）

**问题：** LLM 在检索不到相关信息时容易编造答案（幻觉），且 RAG 管道的每个环节需要可追踪。

**方案：** 用 LangChain LCEL（`prompt | LLM | parser`）构建管道，System Prompt 明确要求"不知道就说不知道"。

### 改动点

| 文件 | 位置 | 内容 |
|------|------|------|
| `agent/rag_agent.py` | 全文 | `RAGAgent` 类 + `RAG_SYSTEM_PROMPT` |

### 管道流程

```
用户问题
  → 加载 Redis 历史（最近 6 轮）
  → Milvus 检索（top_k=5） → rerank（min_score=0.3, max_docs=3）
  → 拼 Prompt（系统指令 + 检索片段 + 历史 + 日期 + 问题）
  → LLM 推理（temperature=0.3，低幻觉）
  → StrOutputParser 输出纯文本
  → 存 Redis 记忆 → 返回
```

### Prompt 设计

```python
RAG_SYSTEM_PROMPT = """
你是一个知识库助手。请严格基于以下检索到的文档片段回答问题。
如果文档中找不到相关信息，直接说"我没有找到相关信息"，不要编造。

{context}           ← 检索到的文档片段，含 [来源: xxx]

当前日期: {date}    ← 注入日期，帮助 LLM 判断时效性
"""
```

### 设计细节

- `temperature=0.3`：比 ReAct(0) 稍高，平衡忠实度与自然度
- `context` 为空时显示 `"（暂无相关文档）"`，给 LLM 明确信号
- 支持多轮历史：取最近 6 轮 Redis 对话注入 Prompt
- 全流程 Span 追踪（`retrieval` → `llm_call`），可观测每个环节耗时

---

## 7. Milvus 长时记忆（long_term）

**问题：** Redis 短时记忆受限于内存大小 + 无语义检索能力，无法实现"记得三个月前用户说过的偏好"。

**方案：** Milvus 向量存储 + 语义检索，文本 embed 后存入，查询时 ANN 搜索语义最接近的历史。

### 改动点

| 文件 | 位置 | 内容 |
|------|------|------|
| `memory/long_term.py` | 全文 | `LongTermMemory` 类 |

### Schema 设计

| 字段 | 类型 | 用途 |
|------|------|------|
| `id` | INT64, primary, auto_id | 自动主键 |
| `session_id` | VARCHAR(128) | 会话隔离 |
| `content` | VARCHAR(4096) | 记忆文本 |
| `created_at` | INT64 | Unix 时间戳，支持时间过滤 |
| `embedding` | FLOAT_VECTOR(1024) | 语义向量 |

### 与短时记忆（Redis）的对比

| | 短时记忆 (Redis) | 长时记忆 (Milvus) |
|---|---|---|
| 存储形式 | JSON 字符串列表 | 向量 + 元信息 |
| 检索方式 | 精确时间序（最近 N 轮） | 语义相似度搜索 |
| 容量 | 受内存限制 | 受磁盘限制（百万级） |
| 过期策略 | 滑动窗口截断 | 手动删除 |
| 用途 | 当前对话上下文 | 跨会话用户偏好 |

---

## 8. Prompt 防幻觉策略

**问题（已内置，无需代码改动）：** LLM 在知识库无匹配时会编造答案。

**策略：**

| 手段 | 实现 |
|------|------|
| 明确指令 | System Prompt: `"如果找不到相关信息，直接说'我没有找到相关信息'"` |
| 检索为空时 | context 填充 `"（暂无相关文档）"`，给 LLM 显式信号 |
| 低温度 | `temperature=0.3`，降低随机性，减少编造倾向 |
| 来源标注 | 每个片段附 `[来源: xxx]`，LLM 回答时自然会引述出处 |
| 分数过滤 | rerank `min_score=0.3` 过滤低相关文档，避免噪音诱导幻觉 |
