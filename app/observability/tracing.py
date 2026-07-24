"""OpenTelemetry 分布式追踪配置"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor


def init_tracing(service_name: str = "ai_manga_agent", otlp_endpoint: str = "http://localhost:4317"):
    """初始化 OpenTelemetry 追踪。

    在 FastAPI app startup 中调用。
    """
    provider = TracerProvider()
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True))
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)


def instrument_fastapi(app):
    """对 FastAPI 应用添加 OpenTelemetry instrumentation。"""
    FastAPIInstrumentor.instrument_app(app)


def get_tracer(name: str = "ai_manga_agent"):
    return trace.get_tracer(name)
