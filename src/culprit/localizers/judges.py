"""
LLM-judge localizers: the baseline everyone reaches for first.

Two shapes, matching how people actually use a model on a broken trace:

* `TraceJudge` -- paste the whole trace, ask which span caused the failure.
  One call, sees everything, and has to reason about propagation across steps.
* `SpanJudge` -- score each span independently for "is this step wrong?", then
  take the most suspicious. N calls, each with a narrow view.

The split matters because of what the two can see. A silent fault produces a
span that is locally unremarkable -- well-formed data, plausible values, no
error -- so `SpanJudge` has, in principle, nothing to find. `TraceJudge` at
least has the arithmetic of the whole run available. Whether it exploits that
is the empirical question.

Both render spans through the same `render_trace`, so the judges differ in
strategy and not in what they were shown. Both are also usable as scorers for
`GuidedReplay`, which is the point: a judge that ranks the culprit second is
useless alone and excellent as a way to order replay probes.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..agent import Trace
from .base import Context, Cost, Verdict, timer

SYSTEM = (
    "You are debugging a failed run of a tool-using AI agent. You will be shown the "
    "agent's trace: each step is a span with an id, a name, its input and its output. "
    "Exactly one span is the ROOT CAUSE of the failure. Note that the root cause is "
    "often NOT the span where the problem becomes visible -- a step that returns "
    "plausible but incorrect data will look completely normal, and the damage only "
    "shows up later. Blame the earliest span that is actually wrong, not the span "
    "that reports the bad final answer."
)

TRACE_PROMPT = """The agent was asked:
{question}

It answered: {answer}
This answer is wrong.

Trace ({n} spans):
{spans}

Which span is the root cause? Reply with JSON only:
{{"span_id": "<id>", "confidence": <0-1>, "why": "<one sentence>"}}"""

SPAN_PROMPT = """The agent was asked:
{question}

It answered: {answer}
This answer is wrong.

Consider ONLY this one span, in the context of the steps before it:
{context}

>>> SPAN UNDER REVIEW <<<
{span}

Is THIS span's output incorrect -- not merely downstream of something wrong,
but itself wrong? Reply with JSON only:
{{"wrong": <true|false>, "confidence": <0-1>, "why": "<one sentence>"}}"""


def _truncate(s: str, n: int = 600) -> str:
    return s if len(s) <= n else s[:n] + f"... [{len(s) - n} more chars]"


def render_span(s: Any) -> str:
    err = f" ERROR={s.error}" if s.error else ""
    return (
        f"[{s.span_id}] {s.name} ({s.span_kind}){err}\n"
        f"    input : {_truncate(json.dumps(s.input, default=str))}\n"
        f"    output: {_truncate(json.dumps(s.output, default=str))}"
    )


def render_trace(trace: Trace, spans: list[Any] | None = None) -> str:
    return "\n".join(render_span(s) for s in (spans if spans is not None else trace.candidates))


def _parse(text: str) -> dict[str, Any]:
    """Models wrap JSON in prose and fences often enough to be worth handling."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


class TraceJudge:
    name = "llm_trace_judge"

    def __init__(self, name: str | None = None) -> None:
        if name:
            self.name = name

    def localize(self, trace: Trace, ctx: Context) -> Verdict:
        with timer() as t:
            cands = [s for s in trace.candidates if s.name != "agent.answer"]
            valid = {s.span_id for s in cands}
            prompt = TRACE_PROMPT.format(
                question=trace.task.question, answer=trace.final_answer,
                n=len(cands), spans=render_trace(trace, cands),
            )
            reply = ctx.llm.complete(prompt, system=SYSTEM)
            out = _parse(reply.text)
            pick = out.get("span_id") if out.get("span_id") in valid else None
            rest = [s.span_id for s in cands if s.span_id != pick]
            ranking = ([pick] if pick else []) + rest
        return Verdict(
            span_id=pick or (rest[0] if rest else None),
            score=float(out.get("confidence", 0.5) or 0.5),
            ranking=ranking,
            explanation=str(out.get("why", ""))[:300] or "judge returned no parseable verdict",
            cost=Cost(llm_calls=1, usd=reply.usd, seconds=t.elapsed),
        )


class SpanJudge:
    name = "llm_span_judge"

    def __init__(self, context_spans: int = 3, name: str | None = None) -> None:
        self.context_spans = context_spans
        if name:
            self.name = name

    def localize(self, trace: Trace, ctx: Context) -> Verdict:
        with timer() as t:
            cands = [s for s in trace.candidates if s.name != "agent.answer"]
            scored: list[tuple[float, str]] = []
            usd = 0.0
            for i, s in enumerate(cands):
                ctx_spans = cands[max(0, i - self.context_spans) : i]
                prompt = SPAN_PROMPT.format(
                    question=trace.task.question, answer=trace.final_answer,
                    context=render_trace(trace, ctx_spans) or "(this is the first step)",
                    span=render_span(s),
                )
                reply = ctx.llm.complete(prompt, system=SYSTEM)
                usd += reply.usd
                out = _parse(reply.text)
                conf = float(out.get("confidence", 0.0) or 0.0)
                scored.append((conf if out.get("wrong") else -conf, s.span_id))
            # Ties break toward the earlier span: root causes sit upstream.
            order = sorted(range(len(scored)), key=lambda i: (-scored[i][0], i))
            ranking = [scored[i][1] for i in order]
        return Verdict(
            span_id=ranking[0] if ranking else None,
            score=max((s for s, _ in scored), default=0.0),
            ranking=ranking,
            explanation=f"per-span judging over {len(cands)} spans",
            cost=Cost(llm_calls=len(cands), usd=usd, seconds=t.elapsed),
        )


def judge_scorer(judge: Any) -> Any:
    """Adapt a judge into a `GuidedReplay` scorer."""

    def scorer(trace: Trace, ctx: Context) -> tuple[list[str], Cost]:
        v = judge.localize(trace, ctx)
        return v.ranking, v.cost

    return scorer
