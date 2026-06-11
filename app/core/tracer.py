"""链路追踪 — 可插拔后端：console（默认） / OpenTelemetry（Jaeger / Grafana Tempo）

追踪点：
- run_start / run_end : Agent 整体运行耗时
- tool_call          : 工具调用（名称 + 参数 + 结果 + 耗时）
- retrieval          : 检索事件（query + 命中数 + 得分）
- llm_call           : LLM 调用（模型名 + token 预览）

使用方式：
    tracer = Tracer(session_id="abc")
    span = tracer.start("tool_call", tool_name="calculator")
    try:
        result = tool.invoke(...)
        span.end(ok=True, result=str(result))
    except Exception as e:
        span.end(ok=False, error=str(e))
"""

import threading
import time
import uuid
from typing import Any

from conf import settings


class _ConsoleSpan:
    """控制台模式 Span — 纯 print 输出"""

    def __init__(self, name: str, session_id: str, **attrs: Any):
        self.span_name: str = name
        self.session_id: str = session_id
        self.attrs: dict[str, Any] = attrs
        self.span_start: float = time.time()
        self.status: str = "running"

    def end(self, ok: bool = True, **result: Any) -> None:
        elapsed = round(time.time() - self.span_start, 3)
        self.status = "ok" if ok else "error"
        self.attrs.update({"elapsed": elapsed, **result})
        print(f"[TRACE] {self.span_name} | {self.status} | {elapsed}s | sid={self.session_id} | {self.attrs}")


class _OtelSpan:
    """OpenTelemetry 模式 Span — gRPC 导出到 Jaeger / Tempo"""

    _tracer = None          # 惰性初始化（全局唯一 Tracer 实例）
    _tracer_lock = threading.Lock()  # 双重检查锁，防止并发初始化

    @classmethod
    def _get_tracer(cls):
        # 快速路径：已初始化则直接返回，无锁开销
        if cls._tracer is not None:
            return cls._tracer

        with cls._tracer_lock:
            # 双重检查：拿到锁后再次确认，避免竞争窗口内的重复初始化
            if cls._tracer is not None:
                return cls._tracer

            from opentelemetry import trace as otel_trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import SERVICE_NAME, Resource

            resource = Resource(attributes={SERVICE_NAME: "agent-platform"})
            exporter = OTLPSpanExporter(endpoint=settings.otel_endpoint, insecure=True)
            provider = TracerProvider(resource=resource)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            otel_trace.set_tracer_provider(provider)

            cls._tracer = otel_trace.get_tracer(__name__)
            return cls._tracer

    def __init__(self, name: str, session_id: str, **attrs: Any):
        self.span_name: str = name
        self.span_start: float = time.time()
        self._session_id = session_id
        self._otel_span = self._get_tracer().start_span(name, attributes={
            "session_id": session_id,
            **{str(k): str(v) for k, v in attrs.items()},
        })

    def end(self, ok: bool = True, **result: Any) -> None:
        from opentelemetry.trace import Status, StatusCode

        elapsed = round(time.time() - self.span_start, 3)
        self._otel_span.set_attributes({
            "elapsed": elapsed,
            **{str(k): str(v)[:200] for k, v in result.items()},
        })
        status = Status(StatusCode.OK if ok else StatusCode.ERROR)
        self._otel_span.set_status(status)
        self._otel_span.end()


class Tracer:
    """可插拔链路追踪器 — 根据 settings.tracer_backend 选择导出方式"""

    def __init__(self, session_id: str | None = None):
        self.session_id: str = session_id or uuid.uuid4().hex[:8]
        self._spans: list[dict[str, Any]] = []
        self._start_time: float = time.time()

        # 根据配置选择 Span 实现
        if settings.tracer_backend == "otel":
            self._span_cls = _OtelSpan
        else:
            self._span_cls = _ConsoleSpan

    def start(self, name: str, **attrs: Any):
        """开始一个追踪 span"""
        span = self._span_cls(name=name, session_id=self.session_id, **attrs)
        self._spans.append({
            "name": span.span_name,
            "start": span.span_start,
        })
        return span

    def summary(self) -> dict[str, Any]:
        """返回当前 session 的追踪摘要"""
        total_elapsed = round(time.time() - self._start_time, 3)
        return {
            "session_id": self.session_id,
            "total_elapsed": total_elapsed,
            "span_count": len(self._spans),
        }
