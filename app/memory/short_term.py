"""Redis 短时记忆 — 滑窗缓存最近 N 轮对话

设计思路：
- 基于 Redis 列表（List），RPUSH 追加，LTRIM 截断
- 每轮对话存两条消息：user + assistant，格式 {"role": "...", "content": "..."}
- 加载后直接拼进 LLM messages 数组，无需转换
- Redis 实例从外部注入（app.state.redis），方便测试 mock
"""

import json

import redis.asyncio as aioredis

from app.models import AgentRequest, AgentResponse


class ShortTermMemory:
    """对话短期记忆 — 基于 Redis 列表实现滑动窗口，自动截断旧消息"""

    def __init__(self, redis_client: aioredis.Redis, max_turns: int = 10):
        self._redis: aioredis.Redis = redis_client   # 异步 Redis 客户端（外部注入）
        self._max_turns: int = max_turns              # 保留最近 N 轮，防止 prompt 过长

    def _key(self, session_id: str) -> str:
        """生成 Redis key：session:{session_id}:history"""
        return f"session:{session_id}:history"

    # ---- 增删查 ----

    async def add(self, session_id: str, request: AgentRequest, response: AgentResponse) -> None:
        """
        存储一轮对话（用户问题 + 助手回答）
        1. 把 user 和 assistant 两条消息 JSON 序列化后 RPUSH 到列表尾部
        2. LTRIM 截断只保留最后 max_turns * 2 条（每轮 2 条），旧消息自动过期
        """
        key = self._key(session_id)
        # ensure_ascii=False：中文不转义为 \uXXXX，Redis 里直接可读
        pair = [
            json.dumps({"role": "user", "content": request.query}, ensure_ascii=False),
            json.dumps({"role": "assistant", "content": response.answer or ""}, ensure_ascii=False),
        ]
        _ = await self._redis.rpush(key, *pair)                            # 追加到列表尾部
        _ = await self._redis.ltrim(key, -self._max_turns * 2, -1)        # 保留最后 max_turns*2 条

    async def load(self, session_id: str) -> list[dict[str, str]]:
        """
        加载会话历史，返回 List[Dict]，可直接拼入 LLM messages 参数
        返回值示例：
        [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮你的？"},
        ]
        """
        key = self._key(session_id)
        raw = await self._redis.lrange(key, 0, -1)   # 取列表全部元素
        return [json.loads(item) for item in raw]     # JSON 反序列化

    async def clear(self, session_id: str) -> None:
        """清空指定会话的全部历史记录"""
        _ = await self._redis.delete(self._key(session_id))



