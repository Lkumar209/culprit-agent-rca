"""
Read traces out of a self-hosted Phoenix instance, and write verdicts back in.

This is the seam that lets the same localizer code run against the benchmark
corpus and against a user's real Phoenix project. Localizers consume `Trace`
objects; this module manufactures them from Phoenix spans.

`load_traces` strips every `culprit.groundtruth.*` key before handing spans
on. That strip is not a formality -- the benchmark writes the label into span
metadata so it is visible in the Phoenix UI, and a localizer that could read
it would score a perfect 1.0 while doing nothing. `tests/test_no_leakage.py`
exists solely to keep that honest.

Verdicts go back as Phoenix span annotations (`annotator_kind="CODE"`), which
is how they surface in the UI next to the span they accuse.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from phoenix.client import Client

from .agent import SpanRec, Trace
from .tools import Task
from .tracing import GT_PREFIX


def _is_null(v: Any) -> bool:
    """Root spans come back with a pandas NA parent, not None -- check properly."""
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _strip_gt(meta: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in meta.items() if not k.startswith(GT_PREFIX)}


def _parse_meta(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


def _loads(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            out = json.loads(raw)
            return out if isinstance(out, dict) else {"value": out}
        except json.JSONDecodeError:
            return {"value": raw}
    return {}


def load_traces(
    project_name: str = "culprit-bench",
    base_url: str = "http://localhost:6006",
    keep_ground_truth: bool = False,
    limit: int | None = None,
) -> list[Trace]:
    """
    Pull spans from a Phoenix project and reassemble them into `Trace` objects.

    `keep_ground_truth` is for the benchmark harness only -- it repopulates the
    labels needed to *score* a localizer. Localizers themselves always receive
    the stripped view.
    """
    client = Client(base_url=base_url)
    df = client.spans.get_spans_dataframe(project_identifier=project_name)
    if df.empty:
        return []

    # Phoenix returns the span id as the index *and*, depending on version, as a
    # column. Only promote the index when it is not already duplicated.
    df = df.reset_index(drop=bool(df.index.name and df.index.name in df.columns))
    span_id_col = "context.span_id" if "context.span_id" in df.columns else "span_id"
    trace_id_col = "context.trace_id" if "context.trace_id" in df.columns else "trace_id"

    by_trace: dict[str, list[dict[str, Any]]] = {}
    for row in df.to_dict("records"):
        by_trace.setdefault(row[trace_id_col], []).append(row)

    traces: list[Trace] = []
    for tid, rows in by_trace.items():
        rows.sort(key=lambda r: r["start_time"])
        root = next((r for r in rows if _is_null(r.get("parent_id")) or r.get("parent_id") == ""), None)
        if root is None:
            continue
        root_meta = _parse_meta(root.get("attributes.metadata"))

        spans: list[SpanRec] = []
        culprit = None
        for idx, r in enumerate(rows):
            meta = _parse_meta(r.get("attributes.metadata"))
            kind = r.get("attributes.openinference.span.kind") or r.get("span_kind") or "LLM"
            rec = SpanRec(
                span_id=str(r[span_id_col]),
                parent_id=(None if _is_null(r.get("parent_id")) else str(r["parent_id"])),
                name=str(r["name"]),
                span_kind=str(kind),
                index=int(meta.get("culprit.index", idx)),
                input=_loads(r.get("attributes.input.value")),
                output=_loads(r.get("attributes.output.value")),
                error=(str(r.get("status_message")) or None) if r.get("status_code") == "ERROR" else None,
            )
            if keep_ground_truth:
                rec.faulted = bool(meta.get(f"{GT_PREFIX}faulted", False))
                rec.fault_name = meta.get(f"{GT_PREFIX}fault_name")
                # Ground truth is keyed on the *local* span id at export time;
                # map it onto the Phoenix span id we just read back.
                if meta.get("culprit.local_span_id") == root_meta.get(f"{GT_PREFIX}culprit_span"):
                    culprit = rec.span_id
            spans.append(rec)

        task = Task(
            task_id=str(root_meta.get("task_id", tid)),
            kind=str(root_meta.get("task_kind", "unknown")),
            question=str(root.get("attributes.input.value") or ""),
            gold=str(root_meta.get(f"{GT_PREFIX}gold", "")) if keep_ground_truth else "",
            plan=(),
        )
        traces.append(
            Trace(
                trace_id=str(tid),
                task=task,
                spans=spans,
                final_answer=str(root.get("attributes.output.value") or ""),
                success=bool(root_meta.get(f"{GT_PREFIX}success", False)) if keep_ground_truth
                else (root.get("status_code") == "OK"),
                culprit_span_id=culprit,
                fault_name=root_meta.get(f"{GT_PREFIX}fault") if keep_ground_truth else None,
                meta=_strip_gt(root_meta),
            )
        )
        if limit and len(traces) >= limit:
            break
    return traces


def write_verdict(
    span_id: str,
    score: float,
    label: str,
    explanation: str,
    base_url: str = "http://localhost:6006",
    annotation_name: str = "culprit",
    metadata: dict[str, Any] | None = None,
) -> None:
    """Attach a blame verdict to a span, visible in the Phoenix UI."""
    Client(base_url=base_url).spans.add_span_annotation(
        span_id=span_id,
        annotation_name=annotation_name,
        annotator_kind="CODE",
        label=label,
        score=float(score),
        explanation=explanation,
        metadata=metadata or {},
        sync=True,
    )
