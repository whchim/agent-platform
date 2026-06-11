"""Skill 模块 — 将多步子任务封装为对外暴露的单一 Tool

设计理念：
    Tool  = 单次调用、无状态、原子操作（如 calculator、search）
    Skill = 多步编排、有内部流程、返回子任务完成报告

Skill 对外是 Tool 接口，对内可以是：
    - 一个 LLM 链（prompt | LLM | parser）
    - 一个 LangGraph 子图
    - 一段规则引擎 + 人工脚本

主 Agent 做高层决策（"需要分析财报"），Skill 做领域执行（读文件→提取指标→计算→生成摘要）。

为什么用 StructuredTool 而不是 Tool：
    Tool 是单输入兼容类，只接受一个字符串参数。Skill.run(**kwargs) 接受多个关键字参数，
    StructuredTool 能把 LLM 传来的 dict 参数原样透传为 kwargs，Tool 会把 dict 拍平导致
    参数丢失。

使用方式：
    # 1. 定义 Skill
    class MySkill(Skill):
        name = "my_skill"
        description = "做某件复杂事情"

        async def run(self, **kwargs) -> str:
            ...

    # 2. 注入 Agent
    agent = ReActAgent(tools=[MySkill(llm=llm).to_tool(), calculator, ...])
"""

import asyncio

from langchain_core.tools import StructuredTool


class Skill:
    """Skill 基类 — 子类只需实现 run()，父类负责 to_tool() 包装"""

    name: str = ""
    description: str = ""

    def to_tool(self) -> StructuredTool:
        """
        把 Skill 包装成 LangChain StructuredTool，ReAct Agent 可直接调用

        StructuredTool 保证 LLM 传来的 dict 参数原样透传为 run(**kwargs)，
        不会像 Tool 那样把 dict 拍平成单个字符串导致参数名丢失。
        """
        if not self.name or not self.description:
            raise ValueError("Skill 子类必须定义 name 和 description")

        return StructuredTool.from_function(
            func=lambda **kwargs: asyncio.run(self.run(**kwargs)),
            name=self.name,
            description=self.description,
            infer_schema=False,           # Skill 参数不固定，不从函数签名推断
            args_schema={
                "type": "object",
                "properties": {},         # 具体参数由子类 run() 的 description 描述
                "additionalProperties": True,
            },
        )

    async def run(self, **kwargs: str) -> str:
        """子类必须实现：执行多步子任务并返回最终结果"""
        raise NotImplementedError("子类必须实现 run() 方法")
