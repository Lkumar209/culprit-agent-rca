"""
Tests for the claims the benchmark's numbers rest on.

These are not unit tests of convenience. Each one guards a property whose
quiet failure would leave the experiments running happily and reporting
meaningless results.
"""

import json

import pytest

from culprit.agent import run_agent
from culprit.faults import FAULT_CATALOG, make_injector
from culprit.replay import ReplayEngine, parse_steps


def test_honest_runs_reach_gold(world, tasks):
    """
    Without a fault the agent must be correct on every task.

    If the agent failed on its own, failures in the corpus would not be
    attributable to the injected fault and every label would be suspect.
    """
    wrong = [t.task_id for t in tasks if not run_agent(world, t).success]
    assert wrong == [], f"agent fails unfaulted tasks: {wrong}"


def test_agent_reads_observations_not_the_plan(world, tasks):
    """
    Corrupting a tool result must change what the agent does next.

    A policy that followed the reference plan regardless of observations would
    sail through every injected fault, and the benchmark would measure nothing.
    """
    task = next(t for t in tasks if t.kind == "quarter_spend")
    inj = make_injector("wrong_entity", world, target_step=0, seed=0)
    faulted = run_agent(world, task, injector=inj)
    honest = run_agent(world, task)
    assert inj.fired
    faulted_args = [s.input.get("args") for s in faulted.spans if s.span_kind == "TOOL"]
    honest_args = [s.input.get("args") for s in honest.spans if s.span_kind == "TOOL"]
    assert faulted_args != honest_args, "fault did not propagate into later tool calls"


@pytest.mark.parametrize("fault", sorted(FAULT_CATALOG))
def test_repairing_the_culprit_fixes_the_run(world, tasks, fault):
    """
    The causal oracle. With a perfect repair, intervening on the labelled
    culprit must fix the run -- otherwise the label is not the cause and
    top-1 accuracy is measuring agreement with a mislabel.
    """
    checked = 0
    for task in tasks:
        for step in (0, 1, 2):
            inj = make_injector(fault, world, target_step=step, seed=step)
            tr = run_agent(world, task, injector=inj)
            if not inj.fired or tr.success:
                continue
            eng = ReplayEngine(world, env_injector=inj.clone_armed(), repair_success_prob=1.0)
            r = eng.intervene(tr, tr.culprit_span_id, gold=tr.task.gold)
            assert r.fixed, f"{fault} @{step} on {task.task_id}: repair did not fix the run"
            checked += 1
    assert checked > 0, f"{fault} never produced a failed trace to check"


@pytest.mark.parametrize("fault", sorted(FAULT_CATALOG))
def test_prefix_probe_is_monotone(world, tasks, fault):
    """
    Q(k) = "shielding steps 0..k fixes the run" must be a step function.

    Bisection is only correct if Q is monotone; if it were not, the binary
    search would be searching a surface with no guarantee it can find.
    """
    for task in tasks[:2]:
        for step in (0, 1):
            inj = make_injector(fault, world, target_step=step, seed=step)
            tr = run_agent(world, task, injector=inj)
            if not inj.fired or tr.success:
                continue
            eng = ReplayEngine(world, env_injector=inj.clone_armed(), repair_success_prob=1.0)
            q = [int(eng.intervene_prefix(tr, k, gold=tr.task.gold).fixed)
                 for k in range(-1, len(parse_steps(tr)))]
            assert all(a <= b for a, b in zip(q, q[1:])), f"{fault}: Q(k) not monotone: {q}"


@pytest.mark.parametrize("fault", sorted(FAULT_CATALOG))
def test_corruption_is_deterministic(world, tasks, fault):
    """
    The same call, corrupted twice, must yield the same thing.

    Counterfactual replay re-runs the agent in the same broken environment, so
    a fault that corrupts differently each time makes the environment itself
    nondeterministic: the final answer moves for reasons unrelated to the
    intervention, and the label-free "did the answer change" signal fires on
    whichever span was probed first. A stateful RNG in `stale_amounts` did
    exactly this and accounted for every miss that signal recorded.
    """
    inj = make_injector(fault, world, target_step=0, seed=0)
    payload_obs = {
        "tool": "list_invoices",
        "args": {"vendor_id": "V1000", "quarter": "Q1", "page": 1},
        "ok": True,
        "result": {
            "invoices": [
                {"invoice_id": "INV-1", "amount_cents": 10000, "quarter": "Q1", "status": "paid"},
                {"invoice_id": "INV-2", "amount_cents": 20000, "quarter": "Q1", "status": "paid"},
            ],
            "has_more": True,
            "vendor_id": "V1000",
        },
        "error": None,
    }
    payload_dec = {
        "action": "call", "tool": "list_invoices",
        "args": {"vendor_id": "V1000", "quarter": "Q1", "page": 1},
    }
    for phase, payload in (("observation", payload_obs), ("decision", payload_dec)):
        outs = [json.dumps(inj(phase, 0, dict(payload))[0], sort_keys=True, default=str)
                for _ in range(3)]
        assert len(set(outs)) == 1, f"{fault}/{phase} corrupts nondeterministically"


def test_corpus_is_byte_reproducible(world, tasks):
    """
    Two builds of the same corpus must produce identical span ids.

    Span ids are not just labels: they are rendered into every judge prompt as
    the handle the model answers with. When they were random uuids, a fresh
    build produced different prompts for identical traces, the response cache
    could never hit across processes, and every re-run paid full API cost
    again. Determinism here is what makes the judge arm cheap to reproduce.
    """
    from culprit.bench.corpus import build_corpus

    def fingerprint():
        _, cases = build_corpus(n_per_kind=2, target_steps=(0, 1))
        return [s.span_id for c in cases for s in c.trace.spans]

    assert fingerprint() == fingerprint()
