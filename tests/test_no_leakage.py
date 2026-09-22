"""
Ground truth must never reach a localizer.

The benchmark stores the fault label in span metadata so it is visible in the
Phoenix UI. A localizer that could read it would score a perfect 1.0 while
doing no work, and nothing else in the pipeline would complain. These tests
are the only thing standing between that mistake and a results table full of
ones.
"""

import json

from culprit.agent import run_agent
from culprit.faults import make_injector
from culprit.localizers.judges import render_trace
from culprit.tracing import GT_PREFIX


def _faulted(world, tasks):
    task = tasks[0]
    inj = make_injector("truncated_page", world, target_step=1, seed=3)
    tr = run_agent(world, task, injector=inj)
    assert inj.fired and not tr.success
    return tr


def test_redacted_span_drops_labels(world, tasks):
    tr = _faulted(world, tasks)
    culprit = next(s for s in tr.spans if s.faulted)
    d = culprit.redacted()
    assert "faulted" not in d and "fault_name" not in d


def test_judge_prompt_contains_no_ground_truth(world, tasks):
    """What a judge is shown must not mention the fault, the label, or gold."""
    tr = _faulted(world, tasks)
    rendered = render_trace(tr)
    lowered = rendered.lower()
    for needle in (GT_PREFIX, "faulted", "fault_name", "truncated_page", "culprit"):
        assert needle.lower() not in lowered, f"{needle!r} leaked into the judge prompt"
    assert tr.task.gold not in rendered, "gold answer leaked into the judge prompt"


def test_trace_serialization_keeps_labels_out_of_span_payloads(world, tasks):
    """The span input/output a localizer reads must be label-free."""
    tr = _faulted(world, tasks)
    for s in tr.spans:
        blob = json.dumps({"input": s.input, "output": s.output}, default=str)
        assert "fault" not in blob.lower()
        assert "culprit" not in blob.lower()
