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

## Why this use case

2026 is the year agents went to production. Every observability vendor now ships
agent tracing, and every framework emits spans. The result is that when a
multi-step agent gets an answer wrong, an engineer has *more* telemetry than
ever and still no answer to the only question they have: **which step broke it?**

Tracing tells you what happened. It does not tell you what was *responsible*.
Those are different questions, and the gap between them is where debugging time
actually goes. This project is an attempt to close it, and to find out what the
limits are.

## Phoenix workflows explored

Both, deliberately -- the two answer different questions and the project needed
both answers.

**Observability.** The agent is instrumented with OpenInference semantic
conventions and exports over OTLP to a self-hosted Phoenix. Localization then
reads those spans back through `phoenix.client` and writes its verdict as a
**span annotation** (`annotator_kind="CODE"`), so the accusation lands on the
span it accuses, inside the UI the engineer already has open. Round-trip
fidelity is verified: localizing traces read back out of Phoenix agrees with
localizing them in-process on 100% of cases.

**Experimentation.** The benchmark itself runs as Phoenix **datasets**,
**experiments** and **evaluators** (`experiments/exp07_phoenix_experiments.py`).
547 failed traces become dataset examples -- input is what a localizer may see,
output is the known culprit span, metadata carries the fault type, visibility
and phase for slicing in the UI. Each of the eight localizers is an experiment
task. `correct_span`, `reciprocal_rank`, `blame_distance` and `replay_cost` are
evaluators. A second dataset of *non-failing* runs scores abstention, where the
correct behaviour is to name nobody.

```
culprit-failed-traces (120 examples)
  experiment            correct_span   reciprocal_rank   replay_cost
  cf_bisect             1.000          1.000             3.75
  cf_exhaustive         1.000          1.000             4.89
  llm_trace_judge       0.558          0.649             0.00
  first_error           0.192          0.361             0.00
  last_span             0.000          0.217             0.00

culprit-healthy-traces (40 examples)
  experiment            stayed_quiet
  cf_bisect             1.000
  output_anomaly        0.300
  first_error           0.000
```

## What this establishes

On 547 failed agent runs with known ground truth, **counterfactual intervention
identifies the causal span in 1.000 of cases at 3.74 replays, while every
inspection-based method tops out near 0.54** — including a whole-trace LLM judge
that is six times better than chance. The gap holds on both fault classes, on
two independent agent implementations, and comes with an ability inspection does
not have: staying quiet on runs that did not fail.

**Why inspection plateaus there is a tested mechanism.** It is bounded by
whether the trace contains a **contradiction** — not by fault visibility (there
is no silent/overt gap) and not by which side of the agent/tool boundary the
fault sits on. Two observation-phase faults, both silent, identical except that
one's response contradicts its own request, score **1.000 and 0.000**. Replay is
1.000 on both. The prediction was registered before the fault testing it was
written.

| fault | phase | contradiction? | judge | replay |
|---|---|---|---|---|
| `hallucinated_arg` | decision | yes | 0.982 | 1.000 |
| `contradictory_echo` | observation | yes | **1.000** | 1.000 |
| `stale_amounts` | observation | no | **0.000** | 1.000 |

A judge cannot detect a wrong number that nothing contradicts, and no better
judge will, because the information is not in the trace. Intervention is not
bounded this way — it generates evidence by re-running the agent instead of
searching for evidence already recorded.

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

### You can buy a better judge with instrumentation, not a bigger model

If inspection is bounded by evidence rather than by the reader, adding evidence
should raise the ceiling with the model untouched. `list_invoices` was
instrumented to return a server-side `n_matching` and `total_matching_cents`
beside the rows — a fault that rewrites the rows leaves them untouched, so the
response contradicts itself. Same model, same prompt, same everything else:

| fault | baseline | instrumented |
|---|---|---|
| **all four** | 0.344 | **0.478** |
| `unit_shift` | 0.042 | **0.271** |
| `truncated_page` | 0.362 | **0.550** |
| `empty_result` | 0.593 | 0.659 |
| `stale_amounts` | 0.000 | 0.036 |
| *replay (control)* | *1.000* | *1.000* |

**A 39% relative improvement from a response field.** `stale_amounts` refines
the claim: the evidence was added and the judge still could not use it, because
checking that contradiction means summing rows across pages and comparing to a
total at 5–25% drift. **A contradiction has to be cheap to check, not merely
present** — emit summaries that expose corruption by counting or magnitude, not
ones needing exact cross-span arithmetic.

This is the most actionable result here: a team whose tools are not safely
re-executable, and so cannot adopt replay, can still buy most of a judge upgrade
for the price of a response field.

### It knows when to say nothing

Pointed at runs that did not fail (`experiments/exp03_abstention.py`),
counterfactual methods abstain on 97.8-100% of them. Every heuristic names a
suspect 100% of the time, because a heuristic always has a "last span" to point
at.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[server,judges,dev]"

phoenix serve                       # self-hosted Phoenix on :6006
culprit doctor                      # check the connection
culprit demo --project culprit-demo
```

Extras: `server` pulls in Phoenix itself (skip it if you already run one),
`judges` adds the Anthropic client for the LLM-judge baselines, `dev` adds
pytest. The core package needs none of them.

`demo` prints, for example:

```
exporting 36 traces (30 failed, 6 healthy) to project 'culprit-demo' ...
reading them back through the Phoenix client ...
  36 traces recovered from Phoenix
localizing with cf_exhaustive_x3 (signal=changed) ...

  failed traces localized : 30/30  (100.0%)
  healthy traces abstained: 6/6
```

`demo` manufactures agent runs with known faults, ships them to Phoenix as
OpenInference spans, reads them back through the ordinary Phoenix client,
localizes each failure, and annotates the guilty span.

### What lands in Phoenix

Open `http://localhost:6006`, pick the project, and select a failed trace. The
guilty span carries a `culprit` annotation in the span detail pane — the
verdict sits on the span it accuses, next to that span's own input and output.
Read back through the client:

```python
>>> from phoenix.client import Client
>>> c = Client(base_url="http://localhost:6006")
>>> df = c.spans.get_spans_dataframe(project_identifier="culprit-demo")
>>> ann = c.spans.get_span_annotations_dataframe(
...     span_ids=[str(i) for i in df.index], project_identifier="culprit-demo")
>>> ann.iloc[0]
annotation_name                                               culprit
annotator_kind                                                   CODE
result.label                                                root_cause
result.score                                                      1.0
result.explanation    repairing this span changed the outcome; 22 spans probed
```

`annotator_kind="CODE"` matters: Phoenix distinguishes these from human and
LLM annotations, so a replay verdict is filterable and never confused with a
judgement.

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

A separate suite (`tests/test_results_reproduce.py`) guards the *published
numbers* rather than the benchmark's internal soundness — they are different
properties, and only the first one originally had tests. It rebuilds the corpus
and asserts its composition still matches the record saved alongside the
results, re-runs the zero-cost localizers and asserts their top-1 still matches
to three decimals, and asserts no held-out probe fault has leaked into the
evaluation corpus. Each assertion was verified to fail when the bug it guards
is reintroduced.

---

## Learnings worth passing on

Six things this cost me time to find out, in rough order of how much they would
save someone else.

**1. Instrument your tools to make corruption self-contradicting.** The single
most actionable result here. Having `list_invoices` return a server-side
`n_matching` and `total_matching_cents` beside its rows lifted a judge from
0.344 to 0.478 -- a 39% relative improvement bought with a response field, not a
model upgrade. If a tool returns data nothing else can check, a reader has
nothing to work with.

**2. The contradiction has to be cheap to check, not merely present.** Adding
the summary did nothing for `stale_amounts` (0.000 -> 0.036), because catching
it means summing rows across pages and comparing to a total at 5-25% drift.
Emit summaries that expose corruption by *counting* or by *magnitude*. Exact
cross-span arithmetic is evidence a model will not use.

**3. Do not build your own experiment runner.** I wrote ~150 lines that ran a
task over a corpus, scored it, stored results and compared methods -- which is
exactly what Phoenix Experiments does. Moving to datasets + experiments +
evaluators gave per-example drill-down (click the trace where `first_error`
failed and see what it said instead) and made adding a method one more
experiment instead of a new column in a script. I should have started there.

**4. A mean signed error will lie to you.** I reported that judges were "off by
about one span" from a mean offset of +1.06 and nearly published it. Mean
*absolute* offset is 3.26. The signed mean was small because errors in opposite
directions cancelled. It also mattered: "nearly right" implies you can cheaply
search the neighbourhood, and you cannot.

**5. Cache keys must not contain anything random.** Span ids were `uuid4`, and
they get rendered into every judge prompt. Identical traces therefore produced
different prompts on every run, the response cache never hit across processes,
and ~$1 of API calls was wasted before a `cached 0, new 100` line gave it away.
Deterministic ids made the corpus byte-reproducible and the judge arm then
re-ran for $0.00.

**6. Test that your results still reproduce, not just that your code works.**
Adding a diagnostic fault silently enrolled it in the evaluation corpus; 547
traces became 616 and every published figure stopped reproducing. 36 passing
tests said nothing, because they all checked that the benchmark was *correct*
and none checked that it still produced *the reported result*. Those are
different properties. `tests/test_results_reproduce.py` now covers the second,
and was verified by reintroducing the bug.

## Layout

| path | what it is |
|---|---|
| `src/culprit/world.py` | synthetic business domain, deterministic from a seed |
| `src/culprit/tools.py` | the agent's toolset and the multi-hop task generator |
| `src/culprit/agent.py` | the instrumented agent and its policy |
| `src/culprit/faults.py` | 9-fault injection taxonomy (+1 held-out probe) |
| `src/culprit/replay.py` | the counterfactual intervention primitive |
| `src/culprit/environment.py` | the re-execution protocol |
| `src/culprit/localizers/` | heuristics, LLM judges, counterfactual methods |
| `src/culprit/llm.py` | Anthropic client: disk cache, hard budget cap |
| `src/culprit/llm_policy.py` | the LLM agent arm — same scaffolding, model decisions |
| `src/culprit/tracing.py` | OpenInference export to Phoenix |
| `src/culprit/phoenix_io.py` | read spans back, write verdicts as annotations |
| `src/culprit/bench/` | corpus construction, metrics, experiment runner |
| `src/culprit/cli.py` | `culprit doctor` / `demo` / `analyze` |
| `experiments/` | the six experiments and their saved results |
| `docs/phoenix-ui.md` | how to capture the UI screenshot |

`experiments/prewarm_judges.py` populates the judge response cache in parallel;
run it before `exp04` to turn ~2 hours of sequential API calls into ~10 minutes.
The experiment itself then runs from cache.

### Held-out probe

`contradictory_echo` is excluded from `MAIN_FAULTS` and never enters the
evaluation corpus. It was engineered to be maximally detectable by an
inspecting judge in order to test one hypothesis, so sweeping it into the main
results would inflate the judge's score with a fault chosen for that property.
Request it by name (`build_corpus(faults=("contradictory_echo",))`).

## License

MIT
