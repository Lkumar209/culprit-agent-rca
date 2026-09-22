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

## Why this is not a solved problem

The obvious approach is to scroll the trace and blame the span with the error on
it. On a benchmark of 546 failed agent runs with known ground truth, that
heuristic gets **78.9% of overt failures and exactly 0.0% of silent ones** — and
silent failures are 77% of the corpus.

The reason is that in a multi-step agent the symptom and the cause are usually
different spans. A tool that returns a truncated page of results, a stale
number, or a plausible-but-wrong entity id produces a span that is *locally
unremarkable*: well-formed data, sensible values, `status: OK`. The damage only
becomes visible several steps later, in the final answer. Anything that scores
spans by how they look is searching the wrong place.

---

## Results

Benchmark: 546 failed runs of a tool-using agent over a synthetic business
domain, 9 fault types, median 11 spans per trace, one injected fault per run so
the correct answer is known by construction.

**Top-1 localization accuracy** (`experiments/exp01_free_arm.py`):

| method | top-1 | silent faults | overt faults | replays/trace |
|---|---|---|---|---|
| `last_span` | 0.000 | 0.000 | 0.000 | 0 |
| `random` | 0.099 | 0.099 | 0.138 | 0 |
| `earliest_tool` | 0.103 | 0.059 | 0.252 | 0 |
| `first_error` | 0.178 | **0.000** | 0.789 | 0 |
| `cf_bisect` | **1.000** | 1.000 | 1.000 | 3.7 |
| `cf_exhaustive` | **1.000** | 1.000 | 1.000 | 4.7 |

The replay numbers above assume a *perfect* repair oracle. That is a ceiling,
not a product claim. Sweeping repair reliability downward
(`experiments/exp02_repair_sweep.py`) is the honest picture:

| repair success | `cf_bisect` | `cf_exhaustive` | `cf_exhaustive` ×3 samples |
|---|---|---|---|
| 1.0 | 1.000 | 1.000 | 0.998 |
| 0.9 | 0.844 | 0.907 | 0.998 |
| 0.7 | 0.586 | 0.722 | 0.960 |
| 0.5 | 0.357 | 0.544 | 0.855 |
| 0.3 | 0.161 | 0.328 | 0.641 |

Three things fall out of this:

1. **Repair quality is the binding constraint**, not the search strategy.
2. **Bisection is cheap but brittle.** It holds at a flat ~3.7 replays per trace
   regardless of trace length, but its efficiency comes from committing to each
   probe, which is exactly what hurts when probes are noisy.
3. **Sampling buys most of it back.** Probing each span three times restores
   0.72 → 0.96 at p=0.7, for roughly 2.3× the replays.

**The label-free signal works.** Deciding "did the answer *change*" needs no
ground truth and is what a real Phoenix user can compute; deciding "did the
answer become *correct*" needs the gold answer. The label-free signal costs
about 5 points (0.952 vs 1.000 at p=1.0) — so the method is deployable against
unlabeled production traces.

**It knows when to say nothing** (`experiments/exp03_abstention.py`). Pointed at
runs that did not fail, counterfactual methods abstain on 97.6–100% of them.
Every heuristic names a suspect 100% of the time, because a heuristic always has
a "last span" to point at.

> The LLM-judge baseline (`experiments/exp04_judge_arm.py`) is implemented but
> **has not been run** — it needs an API key. Until it is, this repo makes no
> claim about how a model-based judge compares.

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
