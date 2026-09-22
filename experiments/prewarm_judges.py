"""
Populate the judge response cache in parallel.

The judge arm is ~1,200 independent API calls. Issued sequentially that is
roughly two hours of wall clock, nearly all of it waiting on the network. This
script issues the same calls from a thread pool and writes each response to the
same on-disk cache the experiment reads, so `exp04_judge_arm.py` afterwards runs
from cache in seconds.

Nothing about the experiment's logic changes, and no call is made twice: the
cache is keyed by model plus exact prompt, so re-running this after an
interruption only fetches what is missing. That property is what made it safe
to stop a half-finished sequential run and restart here -- the 275 responses it
had already paid for were kept.
"""

import argparse, sys, pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from culprit.bench.corpus import build_corpus, stratified
from culprit.llm import BudgetExceeded, LLMClient
from culprit.localizers.judges import (
    SPAN_PROMPT, SYSTEM, TRACE_PROMPT, render_span, render_trace,
)


def trace_prompt(case):
    cands = [s for s in case.trace.candidates if s.name != "agent.answer"]
    return TRACE_PROMPT.format(
        question=case.trace.task.question, answer=case.trace.final_answer,
        n=len(cands), spans=render_trace(case.trace, cands),
    )


def span_prompts(case, context_spans: int = 3):
    cands = [s for s in case.trace.candidates if s.name != "agent.answer"]
    for i, s in enumerate(cands):
        yield SPAN_PROMPT.format(
            question=case.trace.task.question, answer=case.trace.final_answer,
            context=render_trace(case.trace, cands[max(0, i - context_spans):i]) or "(this is the first step)",
            span=render_span(s),
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--span-judge-n", type=int, default=70)
    ap.add_argument("--budget", type=float, default=2.5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default="claude-haiku-4-5")
    args = ap.parse_args()

    world, cases = build_corpus(n_per_kind=10, target_steps=(0, 1, 2, 3))
    failed = [c for c in cases if c.status == "failed"]
    sub = stratified(failed, args.span_judge_n)

    prompts = [trace_prompt(c) for c in failed]
    for c in sub:
        prompts += list(span_prompts(c))
    print(f"{len(failed)} failed traces -> {len(prompts)} judge prompts "
          f"({len(failed)} trace-level, {len(prompts) - len(failed)} span-level)")

    llm = LLMClient(model=args.model, budget_usd=args.budget)
    if not llm.available:
        print("no ANTHROPIC_API_KEY found")
        return 1

    done = failed_n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(llm.complete, p, SYSTEM): p for p in prompts}
        for fut in as_completed(futures):
            try:
                fut.result()
            except BudgetExceeded as exc:
                print(f"\nBUDGET CAP HIT: {exc}")
                break
            except Exception as exc:
                failed_n += 1
                if failed_n <= 5:
                    print(f"  call failed: {type(exc).__name__}: {str(exc)[:120]}")
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(prompts)}  spent ${llm.spent:.3f}  "
                      f"(cached {llm.n_cached}, new {llm.n_calls})", flush=True)

    print(f"\ndone: {done}/{len(prompts)}  failures={failed_n}")
    print(f"spent ${llm.spent:.4f}  new calls={llm.n_calls}  cache hits={llm.n_cached}")
    if llm.n_calls:
        print(f"mean tokens/call: in {llm.in_tokens / llm.n_calls:.0f}  "
              f"out {llm.out_tokens / llm.n_calls:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
