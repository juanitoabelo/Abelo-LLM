"""OpenTelemetry tracing — Langfuse/SigNoz integration for observability."""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Optional


try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False


class Tracer:
    def __init__(self, service_name: str = "my_custom_llm") -> None:
        self.service_name = service_name
        self._tracer = None
        self._setup()

    def _setup(self) -> None:
        otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
        if not otel_endpoint or not _OTEL_AVAILABLE:
            return
        provider = TracerProvider()
        exporter = OTLPSpanExporter(endpoint=f"{otel_endpoint}/v1/traces")
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        self._tracer = trace.get_tracer(self.service_name)

    def is_active(self) -> bool:
        return self._tracer is not None

    def start_span(self, name: str, attributes: Optional[dict] = None):
        if self._tracer:
            return self._tracer.start_as_current_span(name, attributes=attributes)
        return _NoopSpan()


class _NoopSpan:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set_attribute(self, key, value):
        pass

    def add_event(self, name, attributes=None):
        pass


class EventLogger:
    """Structured event logger for traces when OTEL is not available."""

    def __init__(self, log_path: str = "data/traces.jsonl") -> None:
        self.log_path = log_path
        if log_path:
            p = __import__("pathlib").Path(log_path)
            p.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, properties: Optional[dict] = None) -> None:
        if not self.log_path:
            return
        record = {
            "event": event,
            "trace_id": str(uuid.uuid4())[:8],
            "timestamp": time.time(),
            "properties": properties or {},
        }
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

    def log_request(self, method: str, path: str, status: int, duration_ms: float, properties: Optional[dict] = None) -> None:
        self.log("http_request", {
            "method": method,
            "path": path,
            "status": status,
            "duration_ms": round(duration_ms, 1),
            **(properties or {}),
        })

    def log_llm_call(self, model: str, prompt_tokens: int, completion_tokens: int, duration_ms: float) -> None:
        self.log("llm_call", {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "duration_ms": round(duration_ms, 1),
        })


_tracer: Optional[Tracer] = None
_event_logger: Optional[EventLogger] = None


def get_tracer() -> Tracer:
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer


def get_event_logger() -> EventLogger:
    global _event_logger
    if _event_logger is None:
        _event_logger = EventLogger()
    return _event_logger
