"""
Experiment 5 -- do the findings survive a real LLM agent?

Every other result in this project was measured on traces produced by a
rule-based policy. That was a deliberate trade: a scripted agent is correct on
every unfaulted task, so each failure is attributable to the injected fault and
the localization labels mean something. But it leaves the obvious question
open, and it is the one a reviewer asks first.

This runs the identical pipeline with Claude Haiku 4.5 making the agent's
decisions. Same tools, same tasks, same faults, same loop guard and step
budget -- only `_propose` differs, so any change in the results is
attributable to the policy rather than the scaffolding.

Three things to watch beyond top-1 accuracy:

* **Does the agent notice?** The scripted policy is maximally credulous: it
  trusts whatever a tool returns. A real model might see that a quarterly total
  looks implausible and re-query. If it does, the recovery rate rises and
  silent faults are less dangerous than the scripted arm implied.
* **Does replay still work?** Counterfactual localization assumes replaying the
  agent forward is meaningful. A model agent is far less predictable than a
  rule-based one, and the intervention has to survive that.
* **Does the judge's silent/overt parity hold?** That was the finding that
  refuted the project's starting hypothesis, and it should be re-checked on
  traces the model itself produced.

Cost is real here: replay re-runs the agent, so every probe is several API
calls. The corpus is correspondingly small and the error bars are wide. This
is a supporting result, not a headline.
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from collections import Counter

from culprit.bench.corpus import build_corpus, summarize
from culprit.bench.metrics import aggregate, table, wilson
from culprit.bench.run import blame_direction, by_visibility, headline, run_methods, save
from culprit.llm import LLMClient
from culprit.llm_policy import LLMPolicy
from culprit.localizers.counterfactual import BisectReplay
from culprit.localizers.heuristics import (
    FirstErrorLocalizer, LastSpanLocalizer, OutputAnomalyLocalizer, RandomLocalizer,
)
from culprit.localizers.judges import TraceJudge


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-kind", type=int, default=3)
    ap.add_argument("--max-failed", type=int, default=40)
    ap.add_argument("--budget", type=float, default=1.60)
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--corpus-only", action="store_true",
                    help="build and cost the corpus, then stop before localizing")
    args = ap.parse_args()

    llm = LLMClient(model=args.model, budget_usd=args.budget, max_tokens=300)
    if not llm.available:
        print("no ANTHROPIC_API_KEY found")
        return 1
    policy = LLMPolicy(llm)

    print(f"building an LLM-agent corpus (model={args.model}, cap=${args.budget:.2f}) ...")
    world, cases = build_corpus(
        n_per_kind=args.n_per_kind, target_steps=(0, 1, 2),
        policy=policy, max_failed=args.max_failed,
    )
    info = summarize(cases)
    honest = [c for c in cases if c.status == "honest"]
    n_honest_ok = sum(c.trace.success for c in honest)

    print(f"\n  corpus: {info['by_status']}")
    print(f"  honest solve rate: {n_honest_ok}/{len(honest)}"
          f"  <- must be high, or failures are not attributable to the fault")
    print(f"  unparseable model replies: {policy.n_unparseable}")
    print(f"  failed by visibility: {info['failed_by_visibility']}")
    print(f"  spent so far: ${llm.spent:.3f} ({llm.n_calls} calls, {llm.n_cached} cached)")

    # The recovery rate is the interesting agent-level number: it says whether a
    # real model routes around corruption that the scripted policy swallowed.
    fired = [c for c in cases if c.fault is not None]
    rec = Counter(c.status for c in fired)
    denom = rec["failed"] + rec["recovered"]
    if denom:
        print(f"  fault recovery rate: {rec['recovered']}/{denom} "
              f"({rec['recovered'] / denom:.1%}) -- scripted arm was 459/1006 (45.6%)")

    if args.corpus_only:
        return 0

    methods = [
        RandomLocalizer(), LastSpanLocalizer(), FirstErrorLocalizer(),
        OutputAnomalyLocalizer(), TraceJudge(), BisectReplay(),
    ]
    print(f"\nlocalizing {info['by_status'].get('failed', 0)} failed traces ...")
    rows = run_methods(
        world, cases, methods, repair_success_prob=1.0, signal="changed",
        llm=llm, policy=policy, progress=True,
    )
    print(f"  spent ${llm.spent:.3f} total ({llm.n_calls} calls, {llm.n_cached} cached)")

    print()
    print(headline(rows))
    print(by_visibility(rows))
    print(blame_direction(rows))
    save("exp05_llm_agent",
         {"model": args.model, "policy": "LLMPolicy", "n_per_kind": args.n_per_kind},
         rows, extra={"corpus": info, "spent_usd": llm.spent,
                      "honest_solve_rate": f"{n_honest_ok}/{len(honest)}",
                      "unparseable": policy.n_unparseable})
    print("saved -> experiments/results/exp05_llm_agent.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
