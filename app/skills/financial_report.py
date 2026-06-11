"""财报分析 Skill — 示例：多步子任务的封装

流程：
    1. 读取财报文件（.pdf / .docx / .txt）
    2. 提取关键财务指标（营收、净利润、增长率）
    3. 生成结构化分析报告

这个 Skill 演示了 Skill 的核心价值：
    - 主 Agent 只需要说 "analyze_financial_report(file_path='Q3财报.pdf')"
    - Skill 内部跑了 3 步，Agent 不关心内部细节
    - 返回的是一个完整的分析报告字符串

如果在生产环境，步骤 2 可以换成 LLM 调用做真正的 NLU 提取。
"""

from app.skills import Skill
from app.rag.document_loader import load_document


class FinancialReportSkill(Skill):
    """财报分析 Skill — 读取文件 → 提取指标 → 生成报告"""

    name = "analyze_financial_report"
    description = (
        "分析企业财报文件，提取关键财务指标并生成分析报告。"
        "参数: file_path (str) — 财报文件路径，支持 .pdf / .docx / .txt / .md"
    )

    async def run(self, file_path: str, **_) -> str:
        """
        执行财报分析子任务

        参数：
            file_path : 财报文件路径

        返回：
            结构化的分析报告文本
        """
        # ---- Step 1：读取文件 ----
        try:
            text = load_document(file_path)
        except Exception as e:
            return f"[财报分析失败] 无法读取文件 {file_path}: {e}"

        if not text.strip():
            return f"[财报分析失败] 文件 {file_path} 内容为空"

        # ---- Step 2：提取关键财务指标 ----
        # NOTE: 生产环境此处应接 LLM 做 NLU 提取，当前用关键词规则模拟
        metrics = _extract_financial_metrics(text)

        # ---- Step 3：生成分析报告 ----
        report = _generate_report(file_path, metrics)
        return report


# ========== 辅助函数 ==========

def _extract_financial_metrics(text: str) -> dict:
    """
    从财报文本中提取关键指标

    NOTE: 当前为关键词规则引擎演示。
    生产环境应替换为 LLM 调用：
        llm.invoke(f"从以下财报中提取营收、净利润、增长率: {text[:3000]}")
    """
    # 简单的关键词扫描（演示用，实际应上 LLM 或正则）
    text_lower = text.lower()
    lines = text.split("\n")

    metrics = {
        "revenue": "未找到",
        "net_profit": "未找到",
        "growth_rate": "未找到",
        "key_points": [],
    }

    # 扫描包含"营收""收入""净利润""增长"的行
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if any(kw in stripped for kw in ["营收", "收入", "revenue"]):
            if metrics["revenue"] == "未找到":
                metrics["revenue"] = stripped[:200]
        if any(kw in stripped for kw in ["净利润", "净利", "net profit", "net income"]):
            if metrics["net_profit"] == "未找到":
                metrics["net_profit"] = stripped[:200]
        if any(kw in stripped for kw in ["增长", "同比", "growth"]):
            if metrics["growth_rate"] == "未找到":
                metrics["growth_rate"] = stripped[:200]
        # 收集关键要点
        if any(kw in stripped for kw in ["风险", "重大", "变动", "重要"]):
            metrics["key_points"].append(stripped[:200])

    return metrics


def _generate_report(file_path: str, metrics: dict) -> str:
    """根据提取的指标生成结构化报告"""
    key_points_text = "\n".join(f"  - {p}" for p in metrics["key_points"]) if metrics["key_points"] else "  无"

    return f"""【财报分析报告】

文件: {file_path}

一、关键财务指标
  - 营收: {metrics['revenue']}
  - 净利润: {metrics['net_profit']}
  - 增长率: {metrics['growth_rate']}

二、重要事项
{key_points_text}

---
注：当前为规则引擎提取的初步结果，如需精确分析建议结合 LLM 做深度语义提取。"""
