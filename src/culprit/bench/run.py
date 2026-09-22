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
        for m in methods:
            v = m.localize(case.trace, ctx)
            hit, rr = score(v, case.trace.culprit_span_id)
            rows.append(
                Row(
                    case_id=case.case_id, method=m.name, fault=case.fault,
                    visibility=case.visibility, task_kind=case.task_kind,
                    status=case.status, n_spans=len(case.trace.spans),
                    culprit=case.trace.culprit_span_id, predicted=v.span_id,
                    hit=hit, rr=rr, replays=v.cost.replays, llm_calls=v.cost.llm_calls,
                    usd=v.cost.usd, seconds=v.cost.seconds,
                    abstained=v.score == 0.0,
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
