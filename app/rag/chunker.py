"""文本分块 — 基于 LangChain RecursiveCharacterTextSplitter，支持中文"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """
    将长文本递归分割为固定大小的文本块

    递归分割原理：
    依次用 ["\n\n", "\n", "。", ".", " ", ""] 作为分隔符尝试切割，
    优先在段落/句子边界切开，保证每个 chunk 的语义完整性

    参数：
        text          : 待分割的原始文本
        chunk_size    : 每块最大字符数（默认 500）
        chunk_overlap : 相邻块重叠字符数（默认 50），防止关键信息被切断

    返回：
        list[str] — 文本块列表
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],  # 中文句号优先
    )
    return splitter.split_text(text)

