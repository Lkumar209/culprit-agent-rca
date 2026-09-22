"""
Experiment 4 -- the LLM-judge baseline.

This is the comparison a reviewer asks for first: if you can just show the
trace to a model, why intervene at all? Two judge shapes are tested against
the same corpus every other method saw.

The hypothesis being tested is specific. A silent fault produces a span that
is *locally unremarkable* -- well-formed data, plausible values, no error
status -- so there is nothing in that span for a judge to find. If that is
right, judges should track `first_error`: respectable on overt faults, poor on
silent ones, regardless of how capable the model is. If instead judges do well
on silent faults, they are reconstructing the arithmetic of the run from the
trace, and counterfactual replay is a more expensive way to buy the same
answer.

`--dry-run` costs the sweep with the token counter before committing to it.
Run it first. Every response is cached on disk, so a re-run after a scoring
change is free.
"""

import argparse, json, random, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from collections import defaultdict

from culprit.bench.corpus import build_corpus, stratified
from culprit.bench.metrics import aggregate, table, wilson
from culprit.bench.run import blame_direction, by_visibility, headline, run_methods, save
from culprit.llm import LLMClient
from culprit.localizers.counterfactual import ExhaustiveReplay, GuidedReplay
from culprit.localizers.heuristics import FirstErrorLocalizer
from culprit.localizers.judges import SpanJudge, TraceJudge, judge_scorer, render_trace, SYSTEM, TRACE_PROMPT


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="cost the sweep, spend nothing")
    ap.add_argument("--span-judge-n", type=int, default=150, help="traces for the per-span judge")
    ap.add_argument("--budget", type=float, default=2.0)
    ap.add_argument("--model", default="claude-haiku-4-5")
    args = ap.parse_args()

    world, cases = build_corpus(n_per_kind=10, target_steps=(0, 1, 2, 3))
    failed = [c for c in cases if c.status == "failed"]
    sub = stratified(failed, args.span_judge_n)
    llm = LLMClient(model=args.model, budget_usd=args.budget)

    if args.dry_run:
        if not llm.available:
            print("no ANTHROPIC_API_KEY found -- cannot count tokens.")
            print("put one in culprit/.env as ANTHROPIC_API_KEY=sk-ant-...")
            return 1
        print(f"costing the sweep on {llm.model} ...")
        # Measure a handful of real prompts and extrapolate.
        tj, sj = [], []
        for c in failed[:8]:
            cands = [s for s in c.trace.candidates if s.name != "agent.answer"]
            tj.append(llm.count_tokens(TRACE_PROMPT.format(
                question=c.trace.task.question, answer=c.trace.final_answer,
                n=len(cands), spans=render_trace(c.trace, cands)), system=SYSTEM))
            sj.append(sum(len(cands) for _ in [0]) * 0 + len(cands))
        avg_tj = sum(tj) / len(tj)
        avg_spans = sum(sj) / len(sj)
        # A per-span prompt is roughly the span plus <=3 context spans.
        avg_sj = avg_tj / max(1.0, avg_spans) * 4.0
        out_tok = 60
        from culprit.llm import price
        c_trace = len(failed) * price(llm.model, int(avg_tj), out_tok)
        c_span = len(sub) * avg_spans * price(llm.model, int(avg_sj), out_tok)
        print(f"  mean spans/trace      : {avg_spans:.1f}")
        print(f"  mean trace-judge input: {avg_tj:.0f} tokens")
        print(f"  trace judge  : {len(failed)} calls  ~${c_trace:.2f}")
        print(f"  span judge   : {len(sub)} traces x {avg_spans:.1f} spans "
              f"= {len(sub) * avg_spans:.0f} calls  ~${c_span:.2f}")
        print(f"  guided replay: reuses cached judge calls  ~$0.00")
        print(f"  TOTAL ESTIMATE: ${c_trace + c_span:.2f}   (budget cap ${args.budget:.2f})")
        return 0

    if not llm.available:
        print("no ANTHROPIC_API_KEY found (env or culprit/.env) -- cannot run the judge arm.")
        return 1

    # --- full corpus: trace judge vs the cheap and causal references ---
    print(f"trace judge over {len(failed)} failed traces ...")
    rows = run_methods(world, cases, [TraceJudge(), FirstErrorLocalizer()],
                       signal="changed", llm=llm, progress=True)
    print(f"  spent ${llm.spent:.3f} over {llm.n_calls} calls ({llm.n_cached} cached)")

    # --- subsample: per-span judge and the judge-guided hybrid ---
    print(f"span judge + guided replay over {len(sub)} traces ...")
    sub_ids = {c.case_id for c in sub}
    sub_cases = [c for c in cases if c.case_id in sub_ids]
    rows_sub = run_methods(
        world, sub_cases,
        [SpanJudge(), TraceJudge(), GuidedReplay(judge_scorer(TraceJudge()), name="cf_guided_judge"),
         ExhaustiveReplay(), FirstErrorLocalizer()],
        signal="changed", llm=llm, progress=True,
    )
    print(f"  total spent ${llm.spent:.3f} over {llm.n_calls} calls ({llm.n_cached} cached)")

    print()
    print('=' * 72)
    print('FULL CORPUS (546 failed traces): whole-trace judge vs heuristic')
    print('=' * 72)
    print(headline(rows))
    print(by_visibility(rows))
    print(blame_direction(rows))
    print('=' * 72)
    print('SUBSAMPLE: per-span judge, judge-guided replay, causal reference')
    print('=' * 72)
    print(headline(rows_sub))
    print(by_visibility(rows_sub))
    print(blame_direction(rows_sub))
    save("exp04_judge_full", {"model": args.model, "scope": "full corpus"}, rows,
         extra={"spent_usd": llm.spent, "n_calls": llm.n_calls})
    save("exp04_judge_sub", {"model": args.model, "scope": f"{len(sub)}-trace stratified"}, rows_sub,
         extra={"spent_usd": llm.spent, "n_calls": llm.n_calls})
    print(f"saved -> experiments/results/exp04_judge_{{full,sub}}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
