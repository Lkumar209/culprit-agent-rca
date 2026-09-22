"""
The localizer interface every method in this project implements.

A localizer sees a failed trace and nothing else -- no gold answer, no fault
label, no hint about which span was injected. It returns a ranked list of
suspects plus the cost it incurred getting there. Cost is part of the return
value rather than an afterthought because the central claim of this work is
about the accuracy/cost frontier, and a method that is right but needs twenty
replays per trace is a different product from one that is right after two.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..agent import Trace


@dataclass
class Cost:
    llm_calls: int = 0
    replays: int = 0
    tool_calls: int = 0
    seconds: float = 0.0
    usd: float = 0.0

    def __add__(self, other: "Cost") -> "Cost":
        return Cost(
            self.llm_calls + other.llm_calls,
            self.replays + other.replays,
            self.tool_calls + other.tool_calls,
            self.seconds + other.seconds,
            self.usd + other.usd,
        )


@dataclass
class Verdict:
    span_id: str | None
    score: float
    ranking: list[str] = field(default_factory=list)
    explanation: str = ""
    cost: Cost = field(default_factory=Cost)


@dataclass
class Context:
    """Everything a localizer may need, and nothing it may not."""

    world: Any = None
    env_injector: Any = None          # armed environment twin, for replay methods
    repair_success_prob: float = 1.0
    llm: Any = None                   # LLMClient, when a judge needs one
    seed: int = 0
    # Outcome signal: "fixed" needs the gold answer (benchmark ceiling);
    # "changed" needs nothing (what a real user can compute).
    signal: str = "fixed"
    gold: str | None = None


class Localizer(Protocol):
    name: str

    def localize(self, trace: Trace, ctx: Context) -> Verdict: ...


class timer:
    """
    Wall-clock for a localize() call, folded into its Cost.

    `elapsed` is a live property rather than something stamped on exit,
    because several localizers build their Verdict (and therefore read the
    clock) on an early-return path still inside the `with` block.
    """

    def __enter__(self) -> "timer":
        self.t0 = time.perf_counter()
        self.t1: float | None = None
        return self

    def __exit__(self, *exc: object) -> None:
        self.t1 = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return (self.t1 if self.t1 is not None else time.perf_counter()) - self.t0
