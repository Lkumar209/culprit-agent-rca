"""
Experiment 3 -- does the method know when to say nothing?

Top-1 accuracy is measured on runs that genuinely failed. But a localizer in
production gets pointed at whatever a user flags, including runs that were
fine, and a tool that always names a culprit is a tool that manufactures
suspects. This measures the opposite behaviour on two negative populations:

* `recovered` -- a fault fired but the agent reached the right answer anyway.
  The span really is corrupted; it just did not matter. Blaming it is
  defensible but unhelpful, so the target behaviour is abstention.
* `honest`    -- no fault at all. Blaming anything here is a straight false
  positive.

Heuristics cannot abstain: they always have a "last span" to point at. That is
the asymmetry this experiment is designed to expose.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from culprit.bench.corpus import build_corpus
from culprit.bench.metrics import aggregate, table
from culprit.bench.run import default_methods, run_methods, save
from culprit.localizers.counterfactual import ExhaustiveReplay

if __name__ == "__main__":
    world, cases = build_corpus(n_per_kind=10, target_steps=(0, 1, 2, 3))
    methods = default_methods() + [ExhaustiveReplay(n_samples=3)]

    recs, all_rows = [], []
    for status in ("honest", "recovered"):
        for signal in ("fixed", "changed"):
            rows = run_methods(world, cases, methods, repair_success_prob=1.0,
                               signal=signal, statuses=(status,), progress=False)
            all_rows += rows
            for m in methods:
                rs = [r for r in rows if r.method == m.name]
                if not rs:
                    continue
                recs.append({
                    "population": status, "signal": signal, "method": m.name, "n": len(rs),
                    "abstained": sum(r.abstained for r in rs) / len(rs),
                    "named_a_suspect": sum(not r.abstained for r in rs) / len(rs),
                    "replays": sum(r.replays for r in rs) / len(rs),
                })
            print(f"  done: {status} / {signal}", flush=True)

    print()
    print(table(recs, ["population", "signal", "method", "n", "abstained", "named_a_suspect", "replays"],
                title="Abstention on runs that did not fail (higher `abstained` is better)"))
    path = save("exp03_abstention", {"populations": ["honest", "recovered"]}, all_rows, extra={"grid": recs})
    print(f"saved -> {path}")
