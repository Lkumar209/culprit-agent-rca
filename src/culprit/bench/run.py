"""
The experiment runner.

Everything that produces a number in the report goes through here, so that the
number and the configuration that produced it stay attached to each other.
Every run writes a JSON record containing the full config, the per-case rows,
and the aggregate tables. Re-running with the same seed reproduces it exactly.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from ..localizers.base import Context, Cost, Localizer, Verdict
from ..localizers.counterfactual import BisectReplay, ExhaustiveReplay, GuidedReplay
from ..localizers.heuristics import (
    EarliestToolLocalizer, FirstErrorLocalizer, LastSpanLocalizer, RandomLocalizer,
)
from .corpus import Case, build_corpus, summarize
from .metrics import Row, aggregate, score, table, wilson

RESULTS_DIR = Path(__file__).resolve().parents[3] / "experiments" / "results"


def default_methods() -> list[Localizer]:
    return [
        RandomLocalizer(),
        LastSpanLocalizer(),
        FirstErrorLocalizer(),
        EarliestToolLocalizer(),
        ExhaustiveReplay(),
        BisectReplay(),
    ]


def run_methods(
    world: Any,
    cases: list[Case],
    methods: list[Localizer],
    repair_success_prob: float = 1.0,
    signal: str = "fixed",
    seed: int = 0,
    llm: Any = None,
    statuses: tuple[str, ...] = ("failed",),
    progress: bool = True,
) -> list[Row]:
    """Run every method over every case and score the verdicts."""
    rows: list[Row] = []
    subset = [c for c in cases if c.status in statuses]
    for i, case in enumerate(subset):
        if progress and i % 25 == 0:
            print(f"  case {i + 1}/{len(subset)} ...", flush=True)
        injector = case.injector(world)
        ctx = Context(
            world=world,
            env_injector=injector,
            repair_success_prob=repair_success_prob,
            llm=llm,
            seed=seed,
            signal=signal,
            gold=case.trace.task.gold,
        )
        pos = {
            s.span_id: i
            for i, s in enumerate(
                s for s in case.trace.candidates if s.name != "agent.answer"
            )
        }
        for m in methods:
            v = m.localize(case.trace, ctx)
            hit, rr = score(v, case.trace.culprit_span_id)
            offset = (
                pos[v.span_id] - pos[case.trace.culprit_span_id]
                if v.span_id in pos and case.trace.culprit_span_id in pos
                else None
            )
            rows.append(
                Row(
                    case_id=case.case_id, method=m.name, fault=case.fault,
                    visibility=case.visibility, task_kind=case.task_kind,
                    status=case.status, n_spans=len(case.trace.spans),
                    culprit=case.trace.culprit_span_id, predicted=v.span_id,
                    hit=hit, rr=rr, replays=v.cost.replays, llm_calls=v.cost.llm_calls,
                    usd=v.cost.usd, seconds=v.cost.seconds,
                    abstained=v.score == 0.0, offset=offset,
                )
            )
    return rows


def save(name: str, config: dict[str, Any], rows: list[Row], extra: dict[str, Any] | None = None) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(
        json.dumps(
            {
                "name": name,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "config": config,
                "rows": [asdict(r) for r in rows],
                "extra": extra or {},
            },
            indent=2,
            default=str,
        )
    )
    return path


def headline(rows: list[Row]) -> str:
    """The table that answers 'which method should I use'."""
    agg = aggregate(rows, by=("method",))
    for rec in agg:
        rs = [r for r in rows if r.method == rec["method"]]
        lo, hi = wilson(sum(r.hit for r in rs), len(rs))
        rec["ci95"] = f"[{lo:.2f},{hi:.2f}]"
    return table(agg, ["method", "n", "top1", "ci95", "mrr", "replays", "llm_calls", "ms"],
                 title="Top-1 localization accuracy on failed traces")


def by_visibility(rows: list[Row]) -> str:
    agg = aggregate(rows, by=("method", "visibility"))
    return table(agg, ["method", "visibility", "n", "top1", "mrr", "replays"],
                 title="Accuracy split by fault visibility (silent = no error span in the trace)")


def by_fault(rows: list[Row]) -> str:
    agg = aggregate(rows, by=("method", "fault"))
    return table(agg, ["method", "fault", "n", "top1", "mrr"], title="Accuracy by fault type")


def blame_direction(rows: list[Row]) -> str:
    """
    When a method is wrong, *which way* is it wrong?

    This is the diagnostic that distinguishes two very different failure modes.
    A method that blames spans downstream of the true cause is being fooled by
    propagation -- it has found where the damage became visible rather than
    where it started. A method that scatters in both directions is simply
    guessing. The distinction matters because the first is fixable by giving
    the method a causal signal, and the second is not.
    """
    from collections import defaultdict
    from statistics import mean

    groups: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        if r.offset is not None and not r.hit:
            groups[r.method].append(r)

    recs = []
    for method, rs in sorted(groups.items()):
        downstream = sum(r.offset > 0 for r in rs)
        upstream = sum(r.offset < 0 for r in rs)
        recs.append({
            "method": method,
            "n_wrong": len(rs),
            "blamed_downstream": downstream / len(rs) if rs else 0.0,
            "blamed_upstream": upstream / len(rs) if rs else 0.0,
            "mean_offset": mean(r.offset for r in rs) if rs else 0.0,
        })
    return table(
        recs, ["method", "n_wrong", "blamed_downstream", "blamed_upstream", "mean_offset"],
        title="Direction of error on misses (positive offset = blamed the symptom, not the cause)",
    )
