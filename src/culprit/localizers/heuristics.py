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


class OutputAnomalyLocalizer:
    """
    Blame the tool span whose output looks statistically odd for its own tool.

    This exists to make the baseline honest. `first_error` scoring 0.000 on
    silent faults is true but easy to dismiss: of course a heuristic that looks
    for error statuses finds nothing when there is no error. The stronger claim
    -- that *inspection itself* cannot see a silent fault -- needs a heuristic
    that actually tries.

    So this one does what a competent engineer would build without an LLM: for
    each tool, compare every call's output against the other calls of that same
    tool in the same trace, and flag the one that deviates. It is deliberately
    domain-agnostic. It knows nothing about invoices, pagination or the fault
    taxonomy; it only knows that a number far from its siblings, or an empty
    collection where siblings returned rows, is worth a second look.

    If this also scores near zero on silent faults, the problem is not that the
    baselines were weak. It is that the information is not in the span.
    """

    name = "output_anomaly"

    def _numeric_leaves(self, obj: Any, prefix: str = "") -> dict[str, float]:
        out: dict[str, float] = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                out.update(self._numeric_leaves(v, f"{prefix}.{k}"))
        elif isinstance(obj, list):
            out[f"{prefix}#len"] = float(len(obj))
            for v in obj:
                out.update(self._numeric_leaves(v, f"{prefix}[]"))
        elif isinstance(obj, bool):
            out[prefix] = float(obj)
        elif isinstance(obj, (int, float)):
            out[prefix] = float(obj)
        return out

    def localize(self, trace: Trace, ctx: Context) -> Verdict:
        with timer() as t:
            cands = [s for s in trace.candidates if s.name != "agent.answer"]
            tools = [s for s in cands if s.span_kind == "TOOL"]

            # Group each tool's calls so a span is judged against its own kind.
            by_tool: dict[str, list[Any]] = {}
            for s in tools:
                by_tool.setdefault(s.name, []).append(s)

            scores: dict[str, float] = {}
            for name, group in by_tool.items():
                leaves = [self._numeric_leaves(s.output) for s in group]
                keys = set().union(*leaves) if leaves else set()
                for s, mine in zip(group, leaves):
                    worst = 0.0
                    for k in keys:
                        others = [d[k] for d in leaves if k in d and d is not mine]
                        if k not in mine or len(others) < 2:
                            continue
                        mu = sum(others) / len(others)
                        var = sum((v - mu) ** 2 for v in others) / len(others)
                        sd = var ** 0.5
                        if sd > 1e-9:
                            worst = max(worst, abs(mine[k] - mu) / sd)
                        elif abs(mine[k] - mu) > 1e-9:
                            worst = max(worst, 3.0)  # siblings agreed, this one did not
                    # An empty collection where siblings returned rows is the
                    # single most actionable shape of "this looks wrong".
                    for k, v in mine.items():
                        if k.endswith("#len") and v == 0:
                            sib = [d[k] for d in leaves if k in d and d is not mine]
                            if sib and max(sib) > 0:
                                worst = max(worst, 4.0)
                    scores[s.span_id] = worst

            ranked = sorted(cands, key=lambda s: (-scores.get(s.span_id, -1.0), s.index))
            ranking = [s.span_id for s in ranked]
            top = scores.get(ranking[0], 0.0) if ranking else 0.0
        return Verdict(
            span_id=ranking[0] if ranking else None,
            score=min(1.0, top / 4.0),
            ranking=ranking,
            explanation=(
                f"most anomalous tool output vs sibling calls (z={top:.1f})"
                if top else "no tool output deviated from its siblings"
            ),
            cost=Cost(seconds=t.elapsed),
        )


HEURISTICS.append(OutputAnomalyLocalizer())
