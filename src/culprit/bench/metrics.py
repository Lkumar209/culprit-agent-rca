"""Scoring for a localization run."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from ..localizers.base import Verdict


@dataclass
class Row:
    case_id: str
    method: str
    fault: str | None
    visibility: str
    task_kind: str
    status: str
    n_spans: int
    culprit: str | None
    predicted: str | None
    hit: bool
    rr: float                   # reciprocal rank, 0 when the culprit is unranked
    replays: int
    llm_calls: int
    usd: float
    seconds: float
    abstained: bool


def score(verdict: Verdict, culprit: str | None) -> tuple[bool, float]:
    hit = culprit is not None and verdict.span_id == culprit
    rr = 0.0
    if culprit is not None and culprit in verdict.ranking:
        rr = 1.0 / (verdict.ranking.index(culprit) + 1)
    return hit, rr


def aggregate(rows: list[Row], by: tuple[str, ...] = ("method",)) -> list[dict[str, Any]]:
    groups: dict[tuple, list[Row]] = defaultdict(list)
    for r in rows:
        groups[tuple(getattr(r, k) for k in by)].append(r)

    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        rec = dict(zip(by, key))
        rec.update(
            {
                "n": len(rs),
                "top1": mean(r.hit for r in rs),
                "mrr": mean(r.rr for r in rs),
                "replays": mean(r.replays for r in rs),
                "llm_calls": mean(r.llm_calls for r in rs),
                "usd": sum(r.usd for r in rs),
                "ms": mean(r.seconds for r in rs) * 1000,
            }
        )
        out.append(rec)
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- honest error bars for a proportion at these n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def table(rows: list[dict[str, Any]], cols: list[str], title: str = "") -> str:
    """Fixed-width table. Results get read in a terminal more than anywhere else."""
    if not rows:
        return f"{title}\n(no rows)\n"
    widths = {c: max(len(c), max(len(_fmt(r.get(c, ""))) for r in rows)) for c in cols}
    head = "  ".join(c.ljust(widths[c]) for c in cols)
    sep = "  ".join("-" * widths[c] for c in cols)
    body = "\n".join("  ".join(_fmt(r.get(c, "")).ljust(widths[c]) for c in cols) for r in rows)
    out = f"{title}\n" if title else ""
    return f"{out}{head}\n{sep}\n{body}\n"


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}"
    return str(v)
