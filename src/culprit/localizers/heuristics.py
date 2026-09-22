"""
Cheap baselines. No model calls, no replays.

These exist to establish the floor, and one of them -- `FirstError` -- is what
a working engineer actually does: scroll the trace, find the red span, blame
it. Measuring how far that gets you on silent faults is half the point of the
benchmark.
"""

from __future__ import annotations

import random

from ..agent import Trace
from .base import Context, Cost, Localizer, Verdict, timer


class RandomLocalizer:
    name = "random"

    def localize(self, trace: Trace, ctx: Context) -> Verdict:
        with timer() as t:
            cands = [s.span_id for s in trace.candidates]
            rng = random.Random(f"{ctx.seed}:{trace.trace_id}")
            rng.shuffle(cands)
        return Verdict(
            span_id=cands[0] if cands else None,
            score=1.0 / max(1, len(cands)),
            ranking=cands,
            explanation="uniform random over candidate spans",
            cost=Cost(seconds=t.elapsed),
        )


class LastSpanLocalizer:
    name = "last_span"

    def localize(self, trace: Trace, ctx: Context) -> Verdict:
        with timer() as t:
            # Latest-first. The final answer span is excluded: blaming the span
            # that merely *reports* the wrong answer is never actionable.
            cands = [s for s in trace.candidates if s.name != "agent.answer"]
            ranking = [s.span_id for s in reversed(cands)]
        return Verdict(
            span_id=ranking[0] if ranking else None,
            score=0.5,
            ranking=ranking,
            explanation="blame the last span before the answer",
            cost=Cost(seconds=t.elapsed),
        )


class FirstErrorLocalizer:
    name = "first_error"

    def localize(self, trace: Trace, ctx: Context) -> Verdict:
        with timer() as t:
            cands = [s for s in trace.candidates if s.name != "agent.answer"]
            errs = [s for s in cands if s.error]
            if errs:
                rest = [s for s in cands if not s.error]
                ranking = [s.span_id for s in errs] + [s.span_id for s in reversed(rest)]
                why = f"first span carrying an error status ({errs[0].name})"
            else:
                # Nothing is red. This is the silent-fault case, and the
                # heuristic has nothing left to say -- it falls back to recency.
                ranking = [s.span_id for s in reversed(cands)]
                why = "no error span in trace; fell back to last-span"
        return Verdict(
            span_id=ranking[0] if ranking else None,
            score=0.9 if errs else 0.3,
            ranking=ranking,
            explanation=why,
            cost=Cost(seconds=t.elapsed),
        )


class EarliestToolLocalizer:
    name = "earliest_tool"

    def localize(self, trace: Trace, ctx: Context) -> Verdict:
        with timer() as t:
            tools = [s for s in trace.candidates if s.span_kind == "TOOL"]
            ranking = [s.span_id for s in tools] + [
                s.span_id for s in trace.candidates if s.span_kind != "TOOL"
            ]
        return Verdict(
            span_id=ranking[0] if ranking else None,
            score=0.3,
            ranking=ranking,
            explanation="blame the earliest tool call (upstream-first prior)",
            cost=Cost(seconds=t.elapsed),
        )


HEURISTICS: list[Localizer] = [
    RandomLocalizer(), LastSpanLocalizer(), FirstErrorLocalizer(), EarliestToolLocalizer()
]
