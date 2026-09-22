"""
Build the benchmark corpus: traces with a known culprit span.

A `Case` stores the *recipe* for its fault (name, target step, seed) rather
than a live injector object, so the corpus is reproducible from a seed and a
config and the replay environment can be reconstructed identically on demand.

The corpus deliberately keeps three populations:

* `failed`   -- the fault fired and the run produced the wrong answer. These
                carry the localization labels and are the main evaluation set.
* `recovered`-- the fault fired and the agent got the right answer anyway.
                Used as a negative set: a localizer asked to explain a healthy
                run should decline to blame anything.
* `honest`   -- no fault at all. The strictest negative set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator

from ..agent import Trace, run_agent
from ..faults import FAULT_CATALOG, Injector, make_injector
from ..tools import Task, build_tasks
from ..world import World, build_world


@dataclass
class Case:
    case_id: str
    task_id: str
    task_kind: str
    fault: str | None
    visibility: str            # silent | overt | none
    target_step: int
    seed: int
    trace: Trace
    status: str                # failed | recovered | honest

    def injector(self, world: World, policy: Any = None) -> Injector | None:
        """
        Reconstruct the armed environment twin for replay.

        `policy` must be the one that produced the trace. Re-arming with a
        different policy would fire the injector on a different call
        signature than the original run hit, so the replay environment would
        be broken in the wrong place.
        """
        if self.fault is None:
            return None
        inj = make_injector(self.fault, world, self.target_step, self.seed)
        # Re-run to re-arm it on the same call signature the original hit.
        run_agent(world, _task_of(world, self.task_id), policy=policy,
                  injector=inj, run_id=self.case_id)
        return inj


_TASK_CACHE: dict[int, dict[str, Task]] = {}


def _task_of(world: World, task_id: str, task_seed: int = 11) -> Task:
    key = id(world)
    if key not in _TASK_CACHE:
        _TASK_CACHE[key] = {t.task_id: t for t in build_tasks(world, seed=task_seed)}
    return _TASK_CACHE[key][task_id]


def build_corpus(
    world_seed: int = 7,
    task_seed: int = 11,
    n_per_kind: int = 10,
    target_steps: tuple[int, ...] = (0, 1, 2, 3),
    faults: tuple[str, ...] | None = None,
    policy: Any = None,
    max_failed: int | None = None,
) -> tuple[World, list[Case]]:
    """Sweep tasks x faults x injection points and label every resulting trace."""
    world = build_world(seed=world_seed)
    tasks = build_tasks(world, seed=task_seed, n_per_kind=n_per_kind)
    _TASK_CACHE[id(world)] = {t.task_id: t for t in tasks}
    faults = faults or tuple(FAULT_CATALOG)
    # Interleave the task kinds. With `max_failed` set the sweep stops
    # early, and in task order that meant the whole corpus came from the
    # first kind alone -- 40 failures all of one shape, with a recovery rate
    # that said more about that shape than about the agent.
    by_kind: dict[str, list[Task]] = {}
    for t in tasks:
        by_kind.setdefault(t.kind, []).append(t)
    interleaved: list[Task] = []
    for i in range(max((len(v) for v in by_kind.values()), default=0)):
        for k in sorted(by_kind):
            if i < len(by_kind[k]):
                interleaved.append(by_kind[k][i])
    tasks = interleaved
    cases: list[Case] = []

    for task in tasks:
        honest = run_agent(world, task, policy=policy, run_id=f"{task.task_id}|honest")
        cases.append(
            Case(
                case_id=f"{task.task_id}|honest", task_id=task.task_id, task_kind=task.kind,
                fault=None, visibility="none", target_step=-1, seed=0,
                trace=honest, status="honest",
            )
        )
        for fault in faults:
            visibility = FAULT_CATALOG[fault][1]
            for step in target_steps:
                inj = make_injector(fault, world, target_step=step, seed=step)
                tr = run_agent(world, task, policy=policy, injector=inj,
                               run_id=f"{task.task_id}|{fault}|s{step}")
                if not inj.fired:
                    continue
                cases.append(
                    Case(
                        case_id=f"{task.task_id}|{fault}|s{step}",
                        task_id=task.task_id, task_kind=task.kind, fault=fault,
                        visibility=visibility, target_step=step, seed=step, trace=tr,
                        status="recovered" if tr.success else "failed",
                    )
                )
                if max_failed and sum(c.status == "failed" for c in cases) >= max_failed:
                    return world, cases
    return world, cases


def summarize(cases: list[Case]) -> dict[str, Any]:
    from collections import Counter

    by_status = Counter(c.status for c in cases)
    failed = [c for c in cases if c.status == "failed"]
    return {
        "total": len(cases),
        "by_status": dict(by_status),
        "failed_by_visibility": dict(Counter(c.visibility for c in failed)),
        "failed_by_fault": dict(Counter(c.fault for c in failed)),
        "failed_by_task_kind": dict(Counter(c.task_kind for c in failed)),
        "median_spans": sorted(len(c.trace.spans) for c in failed)[len(failed) // 2] if failed else 0,
    }


def stratified(cases: list[Case], n: int, seed: int = 0) -> list[Case]:
    """
    A subsample balanced across fault types.

    Uniform sampling would let the most frequently-firing faults dominate the
    estimate, which matters here because the faults differ in exactly the
    property under study -- whether they are visible in the span they corrupt.
    """
    import random
    from collections import defaultdict

    buckets: dict[str | None, list[Case]] = defaultdict(list)
    for c in cases:
        buckets[c.fault].append(c)
    rng = random.Random(seed)
    per = max(1, n // max(1, len(buckets)))
    out: list[Case] = []
    for f in sorted(buckets, key=str):
        pool = sorted(buckets[f], key=lambda c: c.case_id)
        rng.shuffle(pool)
        out += pool[:per]
    return out[:n]
