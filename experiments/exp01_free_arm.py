"""
Experiment 1 -- the zero-cost arm.

Heuristic baselines against counterfactual replay, no model calls anywhere.
This establishes the floor (what looking at the trace gets you) and the
ceiling (what intervening gets you with a perfect repair oracle), and isolates
the silent/overt split that motivates the whole project.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from culprit.bench.corpus import build_corpus, summarize
from culprit.bench.run import by_fault, by_visibility, default_methods, headline, run_methods, save

CONFIG = dict(world_seed=7, task_seed=11, n_per_kind=10, target_steps=(0, 1, 2, 3),
              repair_success_prob=1.0, signal="fixed", seed=0)

if __name__ == "__main__":
    print("building corpus ...")
    world, cases = build_corpus(
        world_seed=CONFIG["world_seed"], task_seed=CONFIG["task_seed"],
        n_per_kind=CONFIG["n_per_kind"], target_steps=CONFIG["target_steps"],
    )
    info = summarize(cases)
    print(json.dumps(info, indent=2) if (json := __import__("json")) else "")

    rows = run_methods(world, cases, default_methods(),
                       repair_success_prob=CONFIG["repair_success_prob"],
                       signal=CONFIG["signal"], seed=CONFIG["seed"])
    print()
    print(headline(rows))
    print(by_visibility(rows))
    path = save("exp01_free_arm", CONFIG, rows, extra={"corpus": info})
    print(f"saved -> {path}")
