# culprit

**Root-cause localization for failed agent traces in [Arize Phoenix](https://github.com/Arize-ai/phoenix).**

When a multi-step agent run goes wrong, Phoenix shows you *that* it failed and
gives you the whole trace tree. It does not tell you *which step* caused it.
`culprit` answers that question: it reads OpenInference spans from a self-hosted
Phoenix instance, finds the span that actually caused the failure, and writes the
verdict back as a span annotation you can see in the UI.

The method is causal rather than descriptive. Instead of asking a model which
span *looks* wrong, `culprit` repairs a span, replays the run forward from it,
and checks whether the outcome moves. A span that changes the outcome is
implicated; a span that does not is a bystander, however suspicious it looks.

---

## What this establishes

On 547 failed agent runs with known ground truth, **counterfactual intervention
identifies the causal span in 1.000 of cases at 3.74 replays, while every
inspection-based method tops out near 0.54** — including a whole-trace LLM judge
that is six times better than chance. The gap holds on both fault classes, on
two independent agent implementations, and comes with an ability inspection does
not have: staying quiet on runs that did not fail.

What it does *not* establish is why inspection plateaus there. The judge is not
short of information — it finds silent faults as readily as overt ones — and its
errors are broad and unsystematic rather than a clean cause-versus-symptom
confusion. That mechanism is open.

## Why this is not a solved problem

The obvious approach is to scroll the trace and blame the span with the error on
it. That heuristic gets **78.9% of overt failures and exactly 0.0% of silent
ones** — and silent failures are 77% of the corpus (`n=547`).

The reason is that in a multi-step agent the symptom and the cause are usually
different spans. A tool that returns a truncated page of results, a stale
number, or a plausible-but-wrong entity id produces a span that is *locally
unremarkable*: well-formed data, sensible values, `status: OK`. The damage only
becomes visible several steps later, in the final answer. Anything that scores
spans by how they look is searching the wrong place.

---

## Results

Benchmark: 547 failed runs of a tool-using agent over a synthetic business
domain, 9 fault types, median 11 spans per trace, one injected fault per run so
the correct answer is known by construction.

**Top-1 localization accuracy** (`experiments/exp01_free_arm.py`, `exp04_judge_arm.py`):

| method | top-1 | silent faults | overt faults | replays | LLM calls |
|---|---|---|---|---|---|
| `last_span` | 0.000 | 0.000 | 0.000 | 0 | 0 |
| `random` | 0.080 | 0.073 | 0.106 | 0 | 0 |
| `earliest_tool` | 0.102 | 0.059 | 0.252 | 0 | 0 |
| `output_anomaly` | 0.165 | 0.139 | 0.252 | 0 | 0 |
| `first_error` | 0.177 | **0.000** | 0.789 | 0 | 0 |
| `llm_trace_judge` | 0.543 | **0.545** | 0.537 | 0 | 1 |
| `cf_exhaustive` | **1.000** | 1.000 | 1.000 | 4.67 | 0 |
| `cf_bisect` | **1.000** | 1.000 | 1.000 | **3.74** | 0 |

Four things in that table are worth more than the headline:

**1. Span-local inspection fails on silent faults; trace-level inspection does
not.** `first_error` scores exactly zero on the 424 silent failures, and that is
not a rigged baseline — `output_anomaly`, a domain-agnostic statistical detector
comparing each tool call against the other calls of that tool, reaches only
0.139. But the whole-trace judge shows **no silent/overt gap at all** (0.545 vs
0.537). So what span-local methods lack is not visibility of the fault; it is
context. A silent fault is invisible *in its own span* and recoverable *from the
whole trace*. The per-span judge sitting between them (0.413, and 0.388 on
silent) is the dose-response.

**2. The LLM judge is not blind to silent faults — this refuted the project's
starting hypothesis.** I predicted judges would track `first_error`: fine on
overt faults, useless on silent ones. Instead the whole-trace judge scores
*identically* on both (0.545 silent vs 0.537 overt). It is not keying on error
markers; it reconstructs the run's arithmetic from the trace. A judge is a
genuinely good cheap localizer — roughly 3× the best heuristic for one API call.

**3. Judges and heuristics fail differently**, which is why the error's
direction and distance are measured and not just its rate:

| method | blames downstream | blames upstream | typical miss distance |
|---|---|---|---|
| `first_error` | 100% | 0% | 7.72 spans |
| `llm_trace_judge` | 57.6% | 42.4% | 3.26 spans |

The distinction is in the *shape* of the error, not its size. `first_error`
fails **systematically**: it always blames downstream, because with no error
span to find it falls back to the end of the trace. The judge fails
**unsystematically** — scattered in both directions, typically three spans off
in a trace averaging 11.7 candidates. Neither is close. A judge that is wrong
is not nearly right; it is wrong in a way you cannot correct by looking harder,
which is what a systematic bias would let you do.

(Reported as mean *absolute* distance. The signed mean is +1.06, which looks
like "off by one" and is an artifact of opposite errors cancelling — worth
stating because it is exactly the statistic that would have flattered the
judge.)

**4. Per-span judging is *worse* than whole-trace judging** (0.413 vs 0.476 on
the same 63 traces). Judging a span in isolation removes exactly the context
needed to notice that a plausible value is wrong.

### Cost, and why bisection wins

`cf_exhaustive` stops at the earliest causal span, so it pays for the culprit's
depth. `cf_bisect` is O(log n) and does not. Faults here are injected at steps
0-3, which puts culprits near the front and flatters the linear scan, so the
comparison is repeated on a corpus with deeper injection points:

| method | culprits at steps 0-3 | culprits at steps 2-7 |
|---|---|---|
| `cf_exhaustive` | 4.67 replays | 7.70 |
| `cf_bisect` | **3.74 replays** | **4.17** |
| `cf_guided_anomaly` | 6.82 replays | 7.90 |

**Bisection is the one to use**: cheapest, depth-independent, equally accurate.

**Guided replay does not earn its cost — a negative result.** Ordering probes by
suspicion and stopping at the first confirmation is *wrong*: when a fault
propagates, repairing any downstream span that carries the corruption also
changes the outcome, so a confirmation proves only that the culprit is at or
before that span. Treating it as a verdict cost 23 points of top-1 (1.000 →
0.768). Corrected — confirmation as an upper bound, then bisect below it, with a
full-bisection fallback when the shortlist misses — accuracy returns to 1.000
but costs 5.7-7.1 replays, more than plain bisection. This held with the LLM
judge as scorer too. A scorer only pays if it is cheaper than the search it
replaces, and bisection is already very cheap.

### Repair reliability is the binding constraint

The 1.000 figures assume a perfect repair oracle. Sweeping that downward
(`experiments/exp02_repair_sweep.py`) is the honest picture:

| repair success | `cf_bisect` | `cf_exhaustive` | `cf_exhaustive` ×3 samples |
|---|---|---|---|
| 1.0 | 1.000 | 1.000 | 1.000 |
| 0.9 | 0.848 | 0.887 | 0.998 |
| 0.7 | 0.536 | 0.667 | **0.976** |
| 0.5 | 0.325 | 0.512 | 0.888 |
| 0.3 | 0.161 | 0.324 | 0.676 |

Accuracy is dominated by repair quality, not search strategy. Bisection degrades
fastest — its efficiency comes from committing to each probe, which is exactly
what hurts when probes are noisy. Sampling each span three times restores 0.667
→ 0.976 at p=0.7 for ~2.1× the replays. **If you can only tune one thing, tune
the repair, not the search.**

### The label-free signal, with a caveat that matters

Deciding "did the answer *change*" needs no ground truth and is what a real
Phoenix user can compute; deciding "did the answer become *correct*" needs the
gold answer. On this benchmark the two agree **exactly**, at every repair
probability. But they agree *because the benchmark injects exactly one fault per
run*, so any intervention that changes the answer also corrects it. On a trace
with several interacting problems they would diverge, and that case is untested.
Read this as "label-free evaluation is viable here", not "label-free evaluation
is free".

### The findings replicate on a real LLM agent

`experiments/exp05_llm_agent.py` re-runs the pipeline with Claude Haiku 4.5
making the agent's decisions -- same tools, tasks, faults and guards, only the
decision-making replaced. 44 failed traces, 5/5 unfaulted tasks solved.

| method | scripted agent (n=547) | LLM agent (n=44) |
|---|---|---|
| `cf_bisect` | 1.000 @ 3.74 replays | **1.000 @ 3.73 replays** |
| `llm_trace_judge` | 0.543 | 0.500 |
| `first_error` | 0.177 (silent **0.000**) | 0.136 (silent **0.000**) |
| `output_anomaly` | 0.165 | 0.159 |

The method transfers at identical cost and every ordering is preserved. The LLM
agent also recovered from 55.6% of injected faults against the scripted
policy's 45.6% -- a real model routes around corruption a credulous rule-based
one swallows, meaning the main arm if anything *overstates* how damaging silent
faults are.

### It knows when to say nothing

Pointed at runs that did not fail (`experiments/exp03_abstention.py`),
counterfactual methods abstain on 97.8-100% of them. Every heuristic names a
suspect 100% of the time, because a heuristic always has a "last span" to point
at.

---

## Quickstart

```bash
pip install -e ".[server,dev]"
phoenix serve                      # self-hosted Phoenix on :6006
culprit doctor                     # check the connection
culprit demo --project culprit-demo
```

`demo` manufactures agent runs with known faults, ships them to Phoenix as
OpenInference spans, reads them back through the ordinary Phoenix client,
localizes each failure, and annotates the guilty span. Then open
`http://localhost:6006` and filter by the `culprit` annotation.

Reproduce the experiments:

```bash
python experiments/exp01_free_arm.py
python experiments/exp02_repair_sweep.py
python experiments/exp03_abstention.py
python experiments/exp04_judge_arm.py --dry-run   # costs the judge sweep first
```

---

## How it works

```
agent run ──► OpenInference spans ──► Phoenix (self-hosted)
                                          │
                                   phoenix_io.load_traces
                                          ▼
                              ┌─── localizer ───┐
   heuristics (free) ─────────┤                 │
   LLM judges (1 or N calls) ─┤   ranked spans  │
   counterfactual replay ─────┘                 │
                              └────────┬────────┘
                                       ▼
                        phoenix_io.write_verdict  ──► span annotation
```

The intervention primitive is in `replay.py`. For a span `k`: repair its output,
replay the run forward, compare the final answer. Two search strategies sit on
top:

- **Exhaustive** — probe spans in trace order, stop at the first that moves the
  outcome. Earliest-first is not an arbitrary tie-break: when a fault
  propagates, repairing *any* downstream span that carries the corruption also
  changes the answer, so several spans are causally implicated and only the
  earliest is the root cause.
- **Bisection** — binary search on the monotone prefix probe. Writing
  `Q(k) = "shielding steps 0..k fixes the run"`, if the fault is at step `j`
  then `Q` is false below `j` and true at or above it, so `Q` is a step function
  and the culprit is where it flips. `O(log n)` replays instead of `O(n)`.
  Monotonicity is asserted for every fault type in the test suite.

`GuidedReplay` composes the two families: a cheap scorer (a judge, a heuristic,
anything) proposes an order, and replays verify it. A wrong guess costs one
extra replay, never a wrong answer.

---

## What this requires, and what it cannot do

**Replay needs a re-executable environment.** A trace records what *did* happen;
a counterfactual asks what *would* have happened, and no amount of stored
telemetry answers that. You must be able to run the agent's tools again. That is
the honest precondition, and it is an explicit protocol
(`culprit.environment.Environment`) rather than a buried assumption:

```python
class Environment(Protocol):
    def execute(self, tool: str, args: dict) -> tuple[dict, str | None]: ...
    def replayable(self, tool: str) -> bool: ...
```

Read-only tools — lookups, retrieval, search — qualify. A tool that charges a
card does not, and `culprit` cannot tell the difference, so the caller declares
it. Without an `Environment`, only the heuristic and judge localizers run.

**Other limits, stated plainly:**

- The agent under test is a scripted policy over a synthetic domain, not a real
  LLM agent. The fault taxonomy is drawn from real production failure modes, but
  whether these results transfer to a live agent is untested.
- Exactly one fault is injected per run. Real failures sometimes have several
  interacting causes; nothing here addresses that.
- The repair oracle is simulated by a success probability. In production the
  repaired value comes from a model, and its real reliability is unmeasured —
  which is precisely why the sweep in `exp02` exists rather than a single number.
- The benchmark's persistent-environment model means a re-executed tool stays
  broken. That choice is load-bearing: replaying into a *healthy* environment
  would make every span upstream of the fault look like a fix, and the method
  would degenerate into always blaming span 1.

---

## Benchmark validity

The numbers rest on properties that would fail silently if broken, so each has a
test in `tests/test_benchmark_validity.py`:

- **The agent is correct without faults** — 40/40 tasks reach the gold answer.
  Otherwise failures would not be attributable to the injected fault.
- **The policy reads observations, not the reference plan** — a policy that
  followed the gold plan would sail through every fault and the benchmark would
  measure nothing.
- **Repairing the labelled culprit fixes the run**, for all 9 fault types. If it
  did not, the label would not be the cause and "accuracy" would be agreement
  with a mislabel.
- **`Q(k)` is monotone**, for all 9 fault types — bisection is only correct if it
  is.

Ground truth is stored in span metadata so it is visible in the Phoenix UI while
debugging, which makes leakage a live risk. `tests/test_no_leakage.py` asserts
that the stripped view carries no labels and that nothing a judge is shown
mentions the fault, the label, or the gold answer. A leak there would turn every
accuracy number into a 1.0 with nothing else complaining.

Phoenix round-trip fidelity is checked directly: localizing traces read back out
of Phoenix agrees with localizing them in-process on **100% of cases**.

---

## Layout

| path | what it is |
|---|---|
| `src/culprit/world.py` | synthetic business domain, deterministic from a seed |
| `src/culprit/tools.py` | the agent's toolset and the multi-hop task generator |
| `src/culprit/agent.py` | the instrumented agent and its policy |
| `src/culprit/faults.py` | 9-fault injection taxonomy, silent vs overt |
| `src/culprit/replay.py` | the counterfactual intervention primitive |
| `src/culprit/environment.py` | the re-execution protocol |
| `src/culprit/localizers/` | heuristics, LLM judges, counterfactual methods |
| `src/culprit/tracing.py` | OpenInference export to Phoenix |
| `src/culprit/phoenix_io.py` | read spans back, write verdicts as annotations |
| `src/culprit/bench/` | corpus construction, metrics, experiment runner |
| `experiments/` | the four experiments and their saved results |

## License

MIT
