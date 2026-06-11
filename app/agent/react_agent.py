"""ReAct Agent — 基于 LangGraph StateGraph 实现思考-行动-观察循环

循环流程：
    Thought（LLM 推理下一步做什么）
       ↓
    Action（决定调用哪个工具 + 参数）
       ↓
    Observation（工具返回结果）
       ↓
    循环...直到 Final Answer

LangGraph 角色：
    - StateGraph: 定义状态机 + 节点 + 条件边
    - agent_node: LLM 推理 → 输出 tool_call 或 final answer
    - tool_node:  执行工具 → 返回 observation
"""

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.core.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from app.core.tracer import Tracer
from app.memory.long_term import LongTermMemory
from app.memory.short_term import ShortTermMemory
from app.models import AgentRequest, AgentResponse, AgentStep
from app.tools.mcp_client import MCPClient
from app.tools.mcp_tools import wrap_mcp_tools
from conf import settings


# ========== State ==========

class ReActState(TypedDict):
    """LangGraph 状态 — 所有节点共享的上下文"""
    messages: Annotated[list, operator.add]  # pyright: ignore[reportMissingTypeArgument] — LangGraph 内部类型
    thoughts: list[str]                       # 思考过程记录
    steps: list[AgentStep]                    # 中间步骤（返回给前端）
    iteration_count: int                      # 循环次数（防止死循环）
    consecutive_failures: int                 # 连续工具失败次数（成功时归零）


# ========== Agent ==========

class ReActAgent:
    """ReAct Agent — LangGraph 状态图驱动的推理-行动循环"""

    MAX_ITERATIONS = 10            # 最大循环次数，超出强制终止
    MAX_CONSECUTIVE_FAILURES = 3  # 连续工具失败次数，超出强制结束

    def __init__(
        self,
        tools: list,             # pyright: ignore[reportMissingTypeArgument] — LangChain Tool 列表
        memory: ShortTermMemory,
        system_prompt: str | None = None,
        mcp_client: MCPClient | None = None,   # MCP 远程工具客户端（可选）
        skills: list | None = None,            # Skill 实例列表（可选）
        long_term_memory: LongTermMemory | None = None,  # 可选：长时向量记忆
    ):
        self._llm: ChatOpenAI = ChatOpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,  # pyright: ignore[reportArgumentType]
            base_url=settings.deepseek_base_url,
            temperature=0,  # ReAct 推理需要确定性
        )

        # ---- 三层工具合并 ----
        # 第 1 层：原生 Tool（calculator / search）
        all_tools: list = list(tools)

        # 第 2 层：Skill（多步子任务封装）
        if skills:
            for skill in skills:
                all_tools.append(skill.to_tool())

        # 第 3 层：MCP 远程工具（运行时从 MCP Server 动态发现）
        if mcp_client is not None:
            mcp_tools = wrap_mcp_tools(mcp_client)
            all_tools.extend(mcp_tools)

        # 把工具绑定到 LLM — LLM 会自动生成 tool_call
        self._llm_with_tools = self._llm.bind_tools(all_tools)
        self._tools = {t.name: t for t in all_tools}  # 工具名 → 工具对象映射
        # 每个工具一个独立熔断器（连续失败 3 次触发，30s 后半开试探）
        self._breakers: dict[str, CircuitBreaker] = {
            t.name: CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)
            for t in all_tools
        }
        self._memory = memory
        self._long_term_memory: LongTermMemory | None = long_term_memory
        self._system_prompt = system_prompt or "你是一个智能助手，可以使用工具解决复杂问题。"

        # 构建图（编译一次，复用多次）
        self._graph: CompiledStateGraph = self._build_graph()  # pyright: ignore[reportMissingTypeArgument]

    # ========== LangGraph 节点 ==========

    def _agent_node(self, state: ReActState) -> dict:  # pyright: ignore[reportMissingTypeArgument]
        """
        Agent 节点 — LLM 推理
        输入当前状态 → LLM → 输出 AIMessage（可能含 tool_calls 或最终答案）
        """
        messages = state["messages"]
        iteration = state["iteration_count"] + 1

        response: AIMessage = self._llm_with_tools.invoke(messages)

        # 记录思考过程（content 可能是 str | list，强制转 str）
        thought = str(response.content or "")
        if response.tool_calls:
            thought += f"\n[打算调用工具: {[tc['name'] for tc in response.tool_calls]}]"

        return {
            "messages": [response],                          # 追加 LLM 回复到历史
            "thoughts": [thought],                           # 追加本轮思考
            "iteration_count": iteration,
            "steps": [
                AgentStep(step_type="thought", content=thought)
            ],
        }

    def _tool_node(self, state: ReActState) -> dict:  # pyright: ignore[reportMissingTypeArgument]
        """
        工具节点 — 执行 LLM 要求的工具调用
        从最后一条 AIMessage 中提取 tool_calls → 逐个执行 → 返回 ToolMessage
        同时维护 consecutive_failures：本轮全部失败则 +1，任一成功则归零
        """
        last_message = state["messages"][-1]
        tool_calls = last_message.tool_calls

        failures = state.get("consecutive_failures", 0)
        any_success = False

        results: list[ToolMessage] = []
        tool_name = result_text = ""  # 初始化，避免 type checker 的 possibly-unbound 误报
        for tc in tool_calls:  # pyright: ignore[reportUnknownVariableType]
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_fn = self._tools.get(tool_name)

            if tool_fn:
                cb = self._breakers.get(tool_name)
                try:
                    # 熔断器包裹：OPEN 状态抛 CircuitBreakerOpenError，阻断调用
                    if cb:
                        output = cb(tool_fn.invoke)(tool_args)
                    else:
                        output = tool_fn.invoke(tool_args)
                    result_text = str(output)
                    any_success = True
                except CircuitBreakerOpenError:
                    result_text = f"[熔断] {tool_name} 暂时不可用，请稍后重试或使用其他工具"
                except Exception as e:
                    result_text = f"[工具异常] {e}"
            else:
                result_text = f"[未知工具] {tool_name}"

            results.append(ToolMessage(
                content=result_text,
                tool_call_id=tc["id"],
            ))

        # 本轮任一工具成功 → 清零；本轮全部失败 → 累加
        consecutive_failures = 0 if any_success else failures + 1

        return {
            "messages": results,
            "consecutive_failures": consecutive_failures,
            "steps": [
                AgentStep(
                    step_type="tool_result",
                    content=f"{tool_name}: {result_text[:500]}",  # 截断长结果
                    metadata={"tool": tool_name},
                )
            ],
        }

    def _should_continue(self, state: ReActState) -> Literal["tools", "end"]:
        """
        条件边 — 判断下一步
        - LLM 有 tool_calls → 继续执行工具
        - 无 tool_calls → 结束（最终答案）
        - 超过最大迭代 → 强制结束
        - 连续工具失败超阈值 → 强制结束（防止 LLM 在同一错误上死循环）
        """
        if state["iteration_count"] >= self.MAX_ITERATIONS:
            return "end"

        if state.get("consecutive_failures", 0) >= self.MAX_CONSECUTIVE_FAILURES:
            return "end"

        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"

    # ========== 图构建 ==========

    def _build_graph(self) -> CompiledStateGraph:  # pyright: ignore[reportMissingTypeArgument]
        """
        构建 LangGraph 状态图：
            START → agent_node → 条件判断 → tools(循环) 或 END
        """
        builder = StateGraph(ReActState)

        # 注册节点
        builder.add_node("agent", self._agent_node)
        builder.add_node("tools", self._tool_node)

        # 入口：从 agent 开始
        builder.set_entry_point("agent")

        # 条件边：agent → tools 或 END
        builder.add_conditional_edges("agent", self._should_continue, {
            "tools": "tools",
            "end": END,
        })

        # tools → agent（工具结果喂回 LLM）
        builder.add_edge("tools", "agent")

        return builder.compile()

    # ========== 公共接口 ==========

    async def run(self, request: AgentRequest) -> AgentResponse:
        tracer = Tracer(session_id=request.session_id)
        root_span = tracer.start("react_agent_run", mode="react", query=request.query[:50])

        try:
            # ---- Step 0：搜索长时记忆（可选，失败不阻塞主流程） ----
            ltm_context = ""
            if self._long_term_memory is not None:
                try:
                    ltm_results = self._long_term_memory.search(
                        request.query, session_id=request.session_id, top_k=3
                    )
                    if ltm_results:
                        ltm_context = "【长期记忆】以下是与当前问题相关的历史信息:\n" + "\n".join(
                            f"- {r['content']}" for r in ltm_results
                        )
                except Exception:
                    pass

            # ---- Step 1：加载历史 + 拼初始消息 ----
            history = await self._memory.load(request.session_id)
            messages: list = [SystemMessage(content=self._system_prompt)]  # pyright: ignore[reportMissingTypeArgument]

            # 如果搜索到长时记忆，拼入系统消息
            if ltm_context:
                messages = [SystemMessage(content=f"{self._system_prompt}\n\n{ltm_context}")]

            for h in history[-6:]:
                if h["role"] == "user":
                    messages.append(HumanMessage(content=h["content"]))
                else:
                    messages.append(AIMessage(content=h["content"]))

            messages.append(HumanMessage(content=request.query))

            # ---- Step 2：运行状态图 ----
            initial_state: ReActState = {
                "messages": messages,
                "thoughts": [],
                "steps": [],
                "iteration_count": 0,
                "consecutive_failures": 0,
            }

            graph_span = tracer.start("langgraph_invoke")
            final_state = self._graph.invoke(initial_state)
            graph_span.end(ok=True, iterations=final_state["iteration_count"])

            # ---- Step 3：提取最终答案 ----
            tool_failures = final_state.get("consecutive_failures", 0)
            # 默认兜底文案：当图提前终止（超最大迭代/连续失败）且无有效 AIMessage 时使用
            final_answer = "(Agent 未产出答案)"

            if tool_failures >= self.MAX_CONSECUTIVE_FAILURES:
                # 连续失败兜底：不把 LLM 胡乱生成的回复甩给用户
                final_answer = "抱歉，工具连续调用失败，我暂时无法完成您的请求，请稍后重试或联系人工客服。"
            else:
                for msg in reversed(final_state["messages"]):  # pyright: ignore[reportUnknownArgumentType,reportAny]
                    if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                        final_answer = str(msg.content)
                        break

            all_steps: list[AgentStep] = final_state.get("steps", [])  # pyright: ignore[reportUnknownMemberType]

            # ---- Step 4：存记忆 + 返回 ----
            response = AgentResponse(
                status="partial" if tool_failures >= self.MAX_CONSECUTIVE_FAILURES else "success",
                answer=final_answer,
                steps=all_steps,
                metadata={
                    "iterations": final_state["iteration_count"],
                    "model": settings.deepseek_model,
                    "tool_failures": tool_failures,
                },
            )
            await self._memory.add(request.session_id, request, response)

            # ---- 存入长时记忆（可选：只存储有实质内容的回答） ----
            if self._long_term_memory is not None and final_answer and len(final_answer) > 50:
                try:
                    await self._long_term_memory.add(
                        request.session_id,
                        f"Q: {request.query}\nA: {final_answer[:1000]}",
                    )
                except Exception:
                    pass

            root_span.end(ok=True, iterations=final_state["iteration_count"])
            return response

        except Exception as e:
            root_span.end(ok=False, error=str(e)[:200])
            raise


