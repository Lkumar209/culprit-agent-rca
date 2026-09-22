"""
Experiment 2 -- how much does a fallible repair oracle cost you?

Experiment 1 gave counterfactual replay a perfect repair: when it decided to
fix a span, the fix was always correct. In production the repaired value comes
from a model asked "what should this step have returned?", and that proposal is
sometimes wrong. This sweeps that probability down and watches what breaks.

The second axis is the outcome signal. `fixed` compares the replayed answer
against the gold answer, which a benchmark has and a user does not. `changed`
only asks whether the answer moved. If `changed` holds up, the method is
deployable against real traces with no labels at all -- which is the whole
question for a Phoenix user.
"""

import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from culprit.bench.corpus import build_corpus
from culprit.bench.metrics import aggregate, table
from culprit.bench.run import run_methods, save
from culprit.localizers.counterfactual import BisectReplay, ExhaustiveReplay

PROBS = (1.0, 0.9, 0.7, 0.5, 0.3)
SIGNALS = ("fixed", "changed")

if __name__ == "__main__":
    world, cases = build_corpus(n_per_kind=10, target_steps=(0, 1, 2, 3))
    methods = [ExhaustiveReplay(), ExhaustiveReplay(n_samples=3), BisectReplay()]
    all_rows, recs = [], []

    for signal in SIGNALS:
        for p in PROBS:
            rows = run_methods(world, cases, methods, repair_success_prob=p,
                               signal=signal, seed=0, progress=False)
            for r in rows:
                all_rows.append(r)
            for rec in aggregate(rows, by=("method",)):
                rec["signal"], rec["p_repair"] = signal, p
                recs.append(rec)
            print(f"  done: signal={signal} p_repair={p}", flush=True)

    print()
    print(table(sorted(recs, key=lambda r: (r["signal"], -r["p_repair"], r["method"])),
                ["signal", "p_repair", "method", "n", "top1", "mrr", "replays"],
                title="Localization accuracy vs repair-oracle reliability"))
    path = save("exp02_repair_sweep", {"probs": PROBS, "signals": SIGNALS}, all_rows,
                extra={"grid": recs})
    print(f"saved -> {path}")
