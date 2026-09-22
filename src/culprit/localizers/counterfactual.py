"""
Counterfactual localizers: methods that intervene rather than inspect.

Three strategies over the same causal primitive, trading replays for accuracy:

* `ExhaustiveReplay` -- probe spans in trace order, stop at the first one whose
  repair moves the outcome. Earliest-first is not an arbitrary tie-break: when
  a fault propagates, repairing any downstream span that carries the corruption
  also changes the answer, so several spans are causally implicated and only
  the earliest is the *root* cause. Cost is linear in the culprit's depth.

* `BisectReplay` -- binary search on the monotone prefix probe, O(log n)
  replays, plus one probe to decide whether the guilty step's decision or its
  tool result was at fault.

* `GuidedReplay` -- let a cheap scorer propose an order, then spend replays
  verifying its suggestions in that order. Inherits the scorer's ranking and
  the replay's precision: a wrong guess costs one extra replay, never a wrong
  answer. This is the method that should dominate if cheap suspicion signals
  carry any information at all.
"""

from __future__ import annotations

from typing import Any, Callable

from ..agent import Trace
from ..replay import ReplayEngine, parse_steps
from .base import Context, Cost, Verdict, timer


def _engine(ctx: Context, trace: Trace) -> ReplayEngine:
    return ReplayEngine(
        world=ctx.world,
        env_injector=ctx.env_injector.clone_armed() if ctx.env_injector is not None else None,
        repair_success_prob=ctx.repair_success_prob,
        seed=hash((ctx.seed, trace.trace_id)) % (2**31),
    )


def _fires(result: Any, ctx: Context) -> bool:
    """
    Did the intervention move the outcome, under the configured signal?

    The gold-based signal requires the answer to have *moved* as well as to be
    correct. Without the `changed` conjunct, a run that already succeeded
    satisfies "correct after repair" at the very first probe, and the method
    confidently names span 1 as the culprit of a failure that never happened.
    On genuinely failed traces the conjunct is free -- the original answer is
    wrong, so reaching gold always implies a change.
    """
    if ctx.signal == "fixed":
        return result.fixed and result.changed
    return result.changed


class ExhaustiveReplay:
    name = "cf_exhaustive"

    def __init__(self, max_probes: int | None = None, n_samples: int = 1) -> None:
        self.max_probes = max_probes
        self.n_samples = n_samples
        if n_samples > 1:
            self.name = f"cf_exhaustive_x{n_samples}"

    def localize(self, trace: Trace, ctx: Context) -> Verdict:
        with timer() as t:
            eng = _engine(ctx, trace)
            cands = [s for s in trace.candidates if s.name != "agent.answer"]
            if self.max_probes:
                cands = cands[: self.max_probes]
            hits, misses, probes, tools = [], [], 0, 0
            for s in cands:
                # With a fallible repair oracle a single probe can miss by bad
                # luck, and a miss here is unrecoverable: the scan moves on and
                # blames something downstream. Sampling each span more than
                # once trades replays for a lower false-negative rate.
                fired = False
                for _ in range(self.n_samples):
                    r = eng.intervene(trace, s.span_id, gold=ctx.gold)
                    probes += 1
                    tools += r.n_tool_calls
                    if _fires(r, ctx):
                        fired = True
                        break
                (hits if fired else misses).append(s.span_id)
                if hits:
                    break  # earliest hit is the root cause; stop paying
            ranking = hits + [s.span_id for s in cands if s.span_id not in hits]
        return Verdict(
            span_id=ranking[0] if ranking else None,
            score=1.0 if hits else 0.0,
            ranking=ranking,
            explanation=(
                f"repairing this span changed the outcome; {probes} spans probed"
                if hits else f"no span repair moved the outcome in {probes} probes"
            ),
            cost=Cost(replays=probes, tool_calls=tools, seconds=t.elapsed),
        )


class BisectReplay:
    name = "cf_bisect"

    def localize(self, trace: Trace, ctx: Context) -> Verdict:
        with timer() as t:
            eng = _engine(ctx, trace)
            steps = parse_steps(trace)
            n = len(steps)
            probes = tools = 0

            # Binary search for the smallest k with Q(k) true.
            lo, hi, found = 0, n - 1, None
            while lo <= hi:
                mid = (lo + hi) // 2
                r = eng.intervene_prefix(trace, mid, gold=ctx.gold)
                probes += 1
                tools += r.n_tool_calls
                if _fires(r, ctx):
                    found, hi = mid, mid - 1
                else:
                    lo = mid + 1

            if found is None:
                ranking = [s.span_id for s in trace.candidates if s.name != "agent.answer"]
                return Verdict(
                    span_id=ranking[0] if ranking else None, score=0.0, ranking=ranking,
                    explanation=f"no prefix shield fixed the run ({probes} probes)",
                    cost=Cost(replays=probes, tool_calls=tools, seconds=t.elapsed),
                )

            # One more probe to attribute the guilty step: shield only its
            # decision. If that alone suffices, the agent chose wrongly;
            # otherwise the tool handed it something wrong.
            step = steps[found]
            r = eng.intervene_prefix(trace, found, gold=ctx.gold, protect=("decision",))
            probes += 1
            tools += r.n_tool_calls
            blame_decision = _fires(r, ctx)

            primary = step.decision_span if blame_decision else (step.tool_span or step.decision_span)
            other = (step.tool_span or step.decision_span) if blame_decision else step.decision_span
            rest = [
                s.span_id for s in trace.candidates
                if s.span_id not in (primary.span_id, other.span_id) and s.name != "agent.answer"
            ]
            ranking = [primary.span_id, other.span_id] + rest
        return Verdict(
            span_id=primary.span_id,
            score=1.0,
            ranking=ranking,
            explanation=(
                f"prefix bisection isolated step {found}; "
                f"{'the decision' if blame_decision else 'the tool result'} was at fault "
                f"({probes} probes)"
            ),
            cost=Cost(replays=probes, tool_calls=tools, seconds=t.elapsed),
        )


class GuidedReplay:
    """
    Verify a cheap scorer's ranking with replays, cheapest suspicion first.

    `scorer(trace, ctx) -> (ranked_span_ids, cost)`. Any signal works: an LLM
    judge, a heuristic, a learned model. The replay is what keeps it honest --
    the scorer proposes, the counterfactual disposes.
    """

    def __init__(
        self,
        scorer: Callable[[Trace, Context], tuple[list[str], Cost]],
        name: str = "cf_guided",
        max_probes: int = 6,
    ) -> None:
        self.scorer = scorer
        self.name = name
        self.max_probes = max_probes

    def localize(self, trace: Trace, ctx: Context) -> Verdict:
        with timer() as t:
            order, scorer_cost = self.scorer(trace, ctx)
            eng = _engine(ctx, trace)
            valid = {s.span_id for s in trace.candidates if s.name != "agent.answer"}
            order = [sid for sid in order if sid in valid]

            probes = tools = 0
            hit = None
            for sid in order[: self.max_probes]:
                r = eng.intervene(trace, sid, gold=ctx.gold)
                probes += 1
                tools += r.n_tool_calls
                if _fires(r, ctx):
                    hit = sid
                    break

            if hit is not None:
                ranking = [hit] + [s for s in order if s != hit]
                why = f"scorer rank {order.index(hit) + 1} confirmed by replay ({probes} probes)"
                score = 1.0
            else:
                # Nothing verified. Fall back to the scorer's own ordering and
                # say so -- an unverified guess should not be reported as fact.
                ranking = order
                why = f"no candidate confirmed in {probes} probes; scorer order unverified"
                score = 0.25
        return Verdict(
            span_id=ranking[0] if ranking else None,
            score=score,
            ranking=ranking,
            explanation=why,
            cost=Cost(
                llm_calls=scorer_cost.llm_calls, replays=probes, tool_calls=tools,
                usd=scorer_cost.usd, seconds=t.elapsed,
            ),
        )
