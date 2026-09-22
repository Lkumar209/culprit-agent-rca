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

    def injector(self, world: World) -> Injector | None:
        """Reconstruct the armed environment twin for replay."""
        if self.fault is None:
            return None
        inj = make_injector(self.fault, world, self.target_step, self.seed)
        # Re-run to re-arm it on the same call signature the original hit.
        run_agent(world, _task_of(world, self.task_id), injector=inj)
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
) -> tuple[World, list[Case]]:
    """Sweep tasks x faults x injection points and label every resulting trace."""
    world = build_world(seed=world_seed)
    tasks = build_tasks(world, seed=task_seed, n_per_kind=n_per_kind)
    _TASK_CACHE[id(world)] = {t.task_id: t for t in tasks}
    faults = faults or tuple(FAULT_CATALOG)
    cases: list[Case] = []

    for task in tasks:
        honest = run_agent(world, task)
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
                tr = run_agent(world, task, injector=inj)
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
