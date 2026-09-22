"""
Experiment 6 -- can you buy a better judge with instrumentation instead of a
better model?

The contradiction result says inspection is bounded by whether the trace
contains evidence, not by the capability of the reader. That is a diagnosis,
and a diagnosis implies a treatment: if the ceiling is missing evidence, then
*adding evidence to the trace* should raise it, without touching the judge, the
prompt or the model.

The treatment tested here is redundancy. `list_invoices` is instrumented to
return a server-side `n_matching` and `total_matching_cents` alongside the rows
-- computed from the true result set, so a fault that rewrites the rows leaves
them untouched. A silent corruption then disagrees with a summary in its own
response, and an unverifiable value becomes a visible inconsistency.

**Prediction, registered before running.** On the four observation-phase faults
where the judge is weakest, accuracy should rise substantially with
instrumentation on. Counterfactual replay should be unchanged -- it never needed
the evidence, because it generates its own.

The comparison holds everything else fixed: same world seed, same tasks, same
faults, same injection points, same judge, same prompt. Only the tool's response
shape differs.
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from collections import defaultdict

from culprit.bench.corpus import build_corpus
from culprit.bench.metrics import table, wilson
from culprit.bench.run import run_methods, save
from culprit.llm import LLMClient
from culprit.localizers.counterfactual import BisectReplay
from culprit.localizers.judges import TraceJudge

# The faults where the judge is weakest: observation-phase, and the corrupted
# value is not contradicted by anything else in the trace.
FAULTS = ("stale_amounts", "unit_shift", "truncated_page", "empty_result")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-kind", type=int, default=10)
    ap.add_argument("--budget", type=float, default=0.60)
    ap.add_argument("--model", default="claude-haiku-4-5")
    args = ap.parse_args()

    llm = LLMClient(model=args.model, budget_usd=args.budget)
    if not llm.available:
        print("no ANTHROPIC_API_KEY found")
        return 1

    all_rows, recs = [], []
    for label, instrumented in (("baseline", False), ("instrumented", True)):
        world, cases = build_corpus(
            n_per_kind=args.n_per_kind, target_steps=(0, 1, 2, 3),
            faults=FAULTS, redundant_summaries=instrumented,
        )
        n_failed = sum(c.status == "failed" for c in cases)
        print(f"{label:13} corpus: {n_failed} failed traces", flush=True)

        rows = run_methods(world, cases, [TraceJudge(), BisectReplay()],
                           repair_success_prob=1.0, signal="changed",
                           llm=llm, progress=False)
        for r in rows:
            r.case_id = f"{label}|{r.case_id}"
            all_rows.append(r)

        by = defaultdict(list)
        for r in rows:
            by[(r.method, r.fault)].append(r.hit)
            by[(r.method, "ALL")].append(r.hit)
        for (method, fault), hits in by.items():
            k, n = sum(hits), len(hits)
            lo, hi = wilson(k, n)
            recs.append({"arm": label, "method": method, "fault": fault,
                         "n": n, "top1": k / n, "ci95": f"[{lo:.2f},{hi:.2f}]"})
        print(f"              spent ${llm.spent:.3f} "
              f"({llm.n_calls} new, {llm.n_cached} cached)", flush=True)

    print()
    order = {"ALL": 0, "stale_amounts": 1, "unit_shift": 2,
             "truncated_page": 3, "empty_result": 4}
    judge = sorted([r for r in recs if r["method"] == "llm_trace_judge"],
                   key=lambda r: (order.get(r["fault"], 9), r["arm"]))
    print(table(judge, ["fault", "arm", "n", "top1", "ci95"],
                title="LLM judge: does instrumenting the tool raise the ceiling?"))
    replay = sorted([r for r in recs if r["method"] == "cf_bisect" and r["fault"] == "ALL"],
                    key=lambda r: r["arm"])
    print(table(replay, ["fault", "arm", "n", "top1", "ci95"],
                title="Counterfactual replay over the same traces (control)"))

    save("exp06_instrumentation",
         {"model": args.model, "faults": FAULTS, "n_per_kind": args.n_per_kind},
         all_rows, extra={"grid": recs, "spent_usd": llm.spent})
    print(f"spent ${llm.spent:.3f} total")
    print("saved -> experiments/results/exp06_instrumentation.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
