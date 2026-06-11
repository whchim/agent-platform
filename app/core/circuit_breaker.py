"""三态熔断器 — 防止级联故障，保护外部工具调用

状态机：
    CLOSED（正常）─ 连续失败 N 次 ─→ OPEN（熔断）
    OPEN（拒绝）── 等待 T 秒 ──→ HALF_OPEN（试探）
    HALF_OPEN ── 成功 ──→ CLOSED
    HALF_OPEN ── 失败 ──→ OPEN

使用场景：
    工具调用（calculator / search）失败时触发熔断，
    避免 LLM 反复调用同一个已宕机的工具，浪费 token 和时间。
"""

import time
from collections.abc import Callable
from enum import Enum
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


class CircuitBreakerOpenError(RuntimeError):
    """熔断器打开时抛出，与工具自身异常区分"""


class State(str, Enum):
    CLOSED = "closed"        # 正常通行
    OPEN = "open"            # 拒绝所有请求
    HALF_OPEN = "half_open"  # 允许一个试探请求


class CircuitBreaker:
    """
    三态熔断器

    使用方式：
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=30)
        async with cb:
            result = tool.invoke(...)  # 自动受熔断保护
        if cb.state != State.CLOSED:
            ...  # 熔断中，降级处理
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        """
        参数：
            failure_threshold : 连续失败多少次后触发熔断
            recovery_timeout  : 熔断后多少秒进入半开试探
        """
        self._failure_threshold: int = failure_threshold
        self._recovery_timeout: float = recovery_timeout
        self._state: State = State.CLOSED
        self._failure_count: int = 0   # 连续失败计数
        self._last_failure_time: float = 0.0

    # ---- 状态查询 ----

    @property
    def state(self) -> State:
        """当前状态（自动检查是否该从 OPEN → HALF_OPEN）"""
        self._maybe_transition()
        return self._state

    @property
    def is_open(self) -> bool:
        return self.state == State.OPEN

    # ---- 状态转换 ----

    def _maybe_transition(self) -> None:
        """检查是否到了从 OPEN 进入 HALF_OPEN 的时间"""
        if self._state == State.OPEN:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self._recovery_timeout:
                self._state = State.HALF_OPEN

    def record_success(self) -> None:
        """调用成功 — 重置失败计数，回到 CLOSED"""
        self._state = State.CLOSED
        self._failure_count = 0

    def record_failure(self) -> None:
        """
        调用失败 — 递增失败计数
        达到阈值时触发熔断（CLOSED → OPEN）
        HALF_OPEN 时失败直接回 OPEN
        """
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._failure_count >= self._failure_threshold or self._state == State.HALF_OPEN:
            self._state = State.OPEN

    def __call__(self, func: Callable[P, R]) -> Callable[P, R]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if self.state == State.OPEN:
                raise CircuitBreakerOpenError("熔断器已打开，拒绝调用")
            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception:
                self.record_failure()
                raise
        return wrapper

