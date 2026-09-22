"""
Export benchmark traces to a self-hosted Phoenix instance as OpenInference spans.

This module is what makes `culprit` a Phoenix tool rather than a standalone
script. Every `SpanRec` becomes a real OTel span carrying OpenInference
semantic conventions, so the traces are indistinguishable -- to Phoenix, and
to the localizers reading them back -- from spans emitted by a genuinely
instrumented application.

Ground truth (which span was faulted) is exported too, under the `culprit.*`
metadata keys, because it is genuinely useful to see the label in the Phoenix
UI while debugging the harness. `phoenix_io.load_trace` strips those keys
before any localizer sees a span; the stripping is tested, because a leak
there would silently turn every accuracy number in the report into a 1.0.
"""

from __future__ import annotations

import json
from typing import Any

from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.semconv.trace import SpanAttributes

from .agent import Trace

# Ground-truth keys. Anything under this prefix is benchmark bookkeeping and
# must never reach a localizer.
GT_PREFIX = "culprit.groundtruth."

_PROVIDERS: dict[str, TracerProvider] = {}


def get_tracer(
    project_name: str = "culprit-bench",
    endpoint: str = "http://localhost:6006/v1/traces",
) -> trace_api.Tracer:
    """One tracer per Phoenix project, cached. Phoenix keys projects off the resource."""
    if project_name not in _PROVIDERS:
        provider = TracerProvider(
            resource=Resource.create({"openinference.project.name": project_name})
        )
        provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        _PROVIDERS[project_name] = provider
    return _PROVIDERS[project_name].get_tracer("culprit")


def _dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def export_trace(
    trace: Trace,
    project_name: str = "culprit-bench",
    extra_metadata: dict[str, Any] | None = None,
    **kw: Any,
) -> None:
    """Emit one benchmark trace to Phoenix as a parent span with tool/LLM children."""
    tracer = get_tracer(project_name, **kw)
    root = trace.spans[0]

    with tracer.start_as_current_span("agent.run") as parent:
        parent.set_attribute(
            SpanAttributes.OPENINFERENCE_SPAN_KIND, "AGENT"
        )
        parent.set_attribute(SpanAttributes.INPUT_VALUE, trace.task.question)
        parent.set_attribute(SpanAttributes.OUTPUT_VALUE, str(trace.final_answer))
        parent.set_attribute(
            SpanAttributes.METADATA,
            _dumps(
                {
                    "task_id": trace.task.task_id,
                    "task_kind": trace.task.kind,
                    "culprit.local_trace_id": trace.trace_id,
                    "culprit.local_span_id": root.span_id,
                    f"{GT_PREFIX}gold": trace.task.gold,
                    f"{GT_PREFIX}success": trace.success,
                    f"{GT_PREFIX}culprit_span": trace.culprit_span_id,
                    f"{GT_PREFIX}fault": trace.fault_name,
                    **(extra_metadata or {}),
                }
            ),
        )
        parent.set_status(trace_api.StatusCode.OK if trace.success else trace_api.StatusCode.ERROR)

        for rec in trace.spans[1:]:
            with tracer.start_as_current_span(rec.name) as child:
                kind = "TOOL" if rec.span_kind == "TOOL" else "LLM"
                child.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, kind)
                child.set_attribute(SpanAttributes.INPUT_VALUE, _dumps(rec.input))
                child.set_attribute(SpanAttributes.OUTPUT_VALUE, _dumps(rec.output))
                if rec.span_kind == "TOOL":
                    child.set_attribute(SpanAttributes.TOOL_NAME, rec.input.get("tool", ""))
                    child.set_attribute(
                        SpanAttributes.TOOL_PARAMETERS, _dumps(rec.input.get("args", {}))
                    )
                child.set_attribute(
                    SpanAttributes.METADATA,
                    _dumps(
                        {
                            "culprit.local_span_id": rec.span_id,
                            "culprit.index": rec.index,
                            f"{GT_PREFIX}faulted": rec.faulted,
                            f"{GT_PREFIX}fault_name": rec.fault_name,
                        }
                    ),
                )
                if rec.error:
                    child.set_status(trace_api.StatusCode.ERROR, rec.error)
                    child.record_exception(RuntimeError(rec.error))
                else:
                    child.set_status(trace_api.StatusCode.OK)


def flush(project_name: str = "culprit-bench") -> None:
    prov = _PROVIDERS.get(project_name)
    if prov is not None:
        prov.force_flush()
