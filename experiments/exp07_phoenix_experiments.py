"""
Experiment 7 -- run the whole benchmark through Phoenix Datasets + Experiments.

Every earlier experiment used a custom runner. This one uses the product: the
failed-trace corpus is uploaded as a Phoenix **dataset**, each localizer runs as
a Phoenix **experiment** against it, and the metrics are Phoenix **evaluators**
attached to each run. The comparison then lives in the Phoenix UI instead of in
a printed table.

Two datasets are created, because the two things worth measuring need different
ground truth:

* `culprit-failed-traces` -- runs that failed, with the known culprit span.
  Scored for accuracy, rank and cost.
* `culprit-healthy-traces` -- runs that did *not* fail. Scored for abstention,
  where the correct behaviour is to name nobody.

Run it after `phoenix serve`, then open the dataset in the UI and compare the
experiments side by side.
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from phoenix.client import Client

from culprit.bench.corpus import build_corpus
from culprit.llm import LLMClient
from culprit.localizers.counterfactual import BisectReplay, ExhaustiveReplay
from culprit.localizers.heuristics import (
    FirstErrorLocalizer, LastSpanLocalizer, OutputAnomalyLocalizer, RandomLocalizer,
)
from culprit.localizers.judges import TraceJudge
from culprit.phoenix_experiments import (
    ABSTENTION_EVALUATORS, ACCURACY_EVALUATORS, build_dataset, make_task,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:6006")
    ap.add_argument("--n-per-kind", type=int, default=10)
    ap.add_argument("--limit", type=int, default=120,
                    help="examples per dataset; keeps the UI browsable")
    ap.add_argument("--suffix", default="", help="appended to dataset names")
    ap.add_argument("--no-judge", action="store_true",
                    help="skip the LLM judge (no API key needed)")
    args = ap.parse_args()

    client = Client(base_url=args.url)
    world, cases = build_corpus(n_per_kind=args.n_per_kind, target_steps=(0, 1, 2, 3))
    by_id = {c.case_id: c for c in cases}

    failed = [c for c in cases if c.status == "failed"][: args.limit]
    healthy = [c for c in cases if c.status in ("honest", "recovered")][: args.limit // 3]

    print(f"uploading {len(failed)} failed traces as a Phoenix dataset ...")
    ds_failed = build_dataset(failed, f"culprit-failed-traces{args.suffix}", args.url)
    print(f"uploading {len(healthy)} non-failing traces as a Phoenix dataset ...")
    ds_healthy = build_dataset(
        healthy, f"culprit-healthy-traces{args.suffix}", args.url,
        description="Agent runs that did NOT fail. Correct behaviour is to name nobody.",
    )

    llm = None
    methods = [
        RandomLocalizer(), LastSpanLocalizer(), FirstErrorLocalizer(),
        OutputAnomalyLocalizer(), ExhaustiveReplay(), BisectReplay(),
    ]
    if not args.no_judge:
        llm = LLMClient()
        if llm.available:
            methods.insert(4, TraceJudge())
        else:
            print("  (no ANTHROPIC_API_KEY -- skipping the judge)")

    print(f"\nrunning {len(methods)} localizers as Phoenix experiments on the failed set")
    for m in methods:
        client.experiments.run_experiment(
            dataset=ds_failed,
            task=make_task(m, world, by_id, signal="changed", llm=llm),
            evaluators=ACCURACY_EVALUATORS,
            experiment_name=m.name,
            experiment_description=f"Root-cause localization with {m.name}",
            experiment_metadata={"method": m.name, "signal": "changed"},
            print_summary=False,
        )
        print(f"  done: {m.name}", flush=True)

    print(f"\nrunning abstention experiments on the non-failing set")
    for m in (FirstErrorLocalizer(), OutputAnomalyLocalizer(), BisectReplay()):
        client.experiments.run_experiment(
            dataset=ds_healthy,
            task=make_task(m, world, by_id, signal="changed", llm=llm),
            evaluators=ABSTENTION_EVALUATORS,
            experiment_name=f"{m.name}-abstention",
            experiment_description=f"Does {m.name} stay quiet on runs that did not fail?",
            experiment_metadata={"method": m.name, "population": "non-failing"},
            print_summary=False,
        )
        print(f"  done: {m.name}-abstention", flush=True)

    print()
    print("compare the experiments here:")
    print(f"  {args.url}/datasets/{ds_failed.id}/experiments")
    print(f"  {args.url}/datasets/{ds_healthy.id}/experiments")
    if llm is not None and llm.available:
        print(f"\njudge spend this run: ${llm.spent:.4f} ({llm.n_cached} cached)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
