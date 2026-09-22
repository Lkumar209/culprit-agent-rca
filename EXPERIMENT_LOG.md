# Experiment log

Chronological record of every run, including the ones that exposed a design
flaw rather than a result. This is an audit trail, not a writeup — polished
numbers live in `README.md` and the report.

Budget: Anthropic API key, judge arm only. Every run below is zero-spend
unless a cost is recorded.

Columns per entry: what was run, why, what it cost, where the output lives, and
what design change came out of it.

---

## 2026-09-21 — Phoenix foundation probe

- **Purpose:** Confirm a self-hosted Phoenix accepts OpenInference spans over
  OTLP and hands them back through the client, before building anything on top
  of that assumption.
- **Run:** `phoenix serve` (v20.15.0) + 2 hand-built traces exported to
  `culprit-smoke`.
- **Cost:** $0.
- **Result:** 18 spans round-tripped with correct span kinds and parent/child
  structure. The faulted `tool.list_invoices` span came back with
  `status_code: OK` — the silent-fault phenomenon showing up in the telemetry
  exactly as predicted.
- **Design change:** None. Confirmed the whole approach could be Phoenix-native
  rather than a standalone script.

## 2026-09-21 — Honest-run sanity check

- **Purpose:** The benchmark is only meaningful if the agent is correct when
  nothing is wrong. Anything less and failures are not attributable to the
  injected fault.
- **Run:** 40 tasks, no injector.
- **Result:** 40/40 reached the gold answer. Initial plan lengths were 2–5 tool
  calls and pagination never triggered.
- **Design change:** Two world parameters changed — `PAGE_SIZE` 5 → 3 and
  invoices per vendor/quarter 1–3 → 1–7. Without pagination the
  `truncated_page` fault had nowhere to bite, and traces were too short for
  localization to be a real problem. After: 25/40 tasks need multiple pages,
  plan lengths 2–8.

## 2026-09-21 — First fault sweep (exposed a degenerate failure mode)

- **Purpose:** Check that all 9 fault types fire and produce failures.
- **Run:** 9 faults × 40 tasks × 6 injection points.
- **Result:** All 9 fired. **But** `tool_error` and `dropped_constraint`
  produced 161 failures each with a median of 49 spans and no final answer —
  the agent was retrying forever until the step cap.
- **Design change:** This would have poisoned the corpus: trace length, not the
  fault, would have been the dominant signal any localizer keyed off. Added two
  guards to the policy, both of which real agents have:
  1. a loop guard — stop retrying a call already attempted twice;
  2. a step budget — after 12 observations, answer from partial evidence.
  After: `tool_error` 112 failures at median 9 spans with 49 recoveries;
  every fault now produces wrong answers rather than no answer.

## 2026-09-21 — Causal oracle validation

- **Purpose:** The central assumption. If repairing the labelled culprit does
  not fix the run, the label is not the cause and top-1 accuracy measures
  agreement with a mislabel.
- **Run:** all 9 faults, p_repair = 1.0, exhaustive single-span interventions.
- **Result:** Repairing the true culprit fixed the run in **100% of cases,
  every fault type**. Selectivity was near-perfect (0.00 other spans also fix)
  for 7 of 9 faults; `tool_error` 1.00 and `arg_typo` 0.80 (the retry span),
  `dropped_constraint` 7.32.
- **Design change:** Two, both found here:
  1. `hallucinated_arg` initially failed — the repaired *decision* was being
     re-corrupted by the environment injector on its way out. A decision fault
     is the agent's own error, so the repair has to be the final word at that
     step (`honor_decision`). Without this, no decision-phase fault is
     localizable at all.
  2. `dropped_constraint`'s 7.32 confirmed that several spans are genuinely
     causally implicated when a fault propagates, which is why the decision
     rule has to be *earliest* fixing span, not *any* fixing span.

## 2026-09-21 — exp01, the zero-cost arm

- **Purpose:** Establish the floor (inspection) and the ceiling (intervention).
- **Run:** 546 failed traces × 6 methods. `experiments/results/exp01_free_arm.json`.
- **Cost:** $0, ~40s wall clock.
- **Result:** Counterfactual replay 1.000 top-1 (bisect 3.74 replays,
  exhaustive 4.67). Heuristics 0.000–0.178. The headline split:
  `first_error` scores **0.789 on overt faults and 0.000 on silent ones**,
  which are 423 of the 546 failures.
- **Design change:** None. This is the result the project exists to produce.

## 2026-09-21 — exp02, repair-oracle sweep

- **Purpose:** exp01's 1.000 assumes a perfect repair. That is a ceiling, not a
  product claim. Sweep repair reliability and the outcome signal.
- **Run:** 5 repair probabilities × 2 signals × 3 methods × 546 traces.
- **Cost:** $0.
- **Result:** Accuracy is dominated by repair quality, not search strategy
  (exhaustive 1.000 → 0.328 as p goes 1.0 → 0.3). Bisection degrades fastest
  (0.844 → 0.161) — its efficiency comes from committing to each probe, which
  is what hurts when probes are noisy. 3× sampling restores 0.722 → 0.960 at
  p=0.7. The label-free `changed` signal costs ~5 points versus gold-based
  `fixed`, so the method is deployable on unlabeled traces.
- **Design change:** Added `ExhaustiveReplay(n_samples=k)` after seeing the
  single-probe false-negative rate — a missed probe is unrecoverable, because
  the scan moves on and blames something downstream.

## 2026-09-21 — exp03, abstention (exposed a signal-definition bug)

- **Purpose:** A localizer pointed at a run that did not fail should decline to
  name a suspect. Measure that on healthy and recovered traces.
- **Run:** 40 honest + 460 recovered traces × 7 methods × 2 signals.
- **Result (first pass):** Counterfactual methods abstained 97.6–100% under the
  `changed` signal but **0%** under `fixed`.
- **Design change:** That 0% was a bug, not a finding. On a run that already
  succeeded, "is the answer correct after repair?" is trivially true at the
  first probe, so the method confidently blamed span 1 for a failure that never
  happened. Fixed by requiring the answer to have *moved* as well as be
  correct. Free on genuinely failed traces (the original answer is wrong, so
  reaching gold always implies a change); exp01 was unchanged, as predicted.
- **Result (after):** counterfactual methods abstain 97.6–100% under both
  signals; every heuristic names a suspect 100% of the time.

## 2026-09-21 — Phoenix round-trip fidelity

- **Purpose:** Every number above was computed in-process. Check that
  localizing traces read back out of Phoenix gives the same answers.
- **Run:** 48 traces exported, read back, localized both ways, compared
  per-case.
- **Result:** First attempt recovered only 28/48 traces and scored 82.1%.
  Cause: OTLP export and Phoenix ingestion are both async, so the read returned
  *partial* traces — which silently corrupts localization, because replay
  reconstructs the agent's step structure from the spans it can see. Added an
  ingestion poll. After: 48/48 recovered, and in-process vs via-Phoenix agree
  on **100% of cases** (both 85.0% on this smaller demo corpus; the gap from
  exp02's 95.2% is corpus composition, not the round-trip).
- **Design change:** `_await_ingest` poll before any read in the demo path.

## 2026-09-21 — Nondeterministic corruption (found by asking *where* a method fails)

- **Purpose:** Counterfactual replay scored 0.952 under the label-free signal
  and 1.000 under the gold-based one. Rather than report the 5-point gap as a
  cost of going label-free, characterise it: which traces does it miss?
- **Run:** `cf_exhaustive`, p_repair 1.0, label-free signal, 546 failed traces,
  grouped by fault and by signed offset from the true culprit.
- **Cost:** $0.
- **Result:** All 26 misses came from a **single fault type** —
  `stale_amounts`, 96.3% of that fault — and every one blamed a span
  *upstream* of the culprit (offsets -2, -4, -6; zero downstream). A miss
  pattern that clean is not a method limitation.
- **Cause:** `stale_amounts` drew its per-invoice drift from the injector's
  shared stateful RNG, so corrupting the same call twice produced different
  amounts. Replay therefore ran in a *different* broken environment than the
  original run: the final answer moved for reasons unrelated to the
  intervention, and the "did the answer change" signal fired on whichever span
  was probed first — always the earliest, hence the uniformly upstream misses.
- **Design change:** The per-invoice factor is now derived from a stable key
  (`fault | seed | invoice_id`) instead of a stateful RNG. A persistently
  broken tool returns the same wrong thing every time it is called, which is
  what the persistent-environment model is supposed to mean. Added
  `test_corruption_is_deterministic` over all 9 faults and both phases.
- **Effect on results:** The label-free signal went 0.952 → **1.000**, equal to
  the gold-based signal. The reported "label-free costs ~5 points" was entirely
  this artifact. The deployability claim gets *stronger*: on this benchmark the
  signal a real Phoenix user can compute is as good as the one requiring gold
  answers. exp01 was unaffected, as predicted — the gold-based signal requires
  reaching the correct answer, which a spurious change does not produce.
- **Note:** This is the third time an experiment exposed a flaw in the harness
  rather than a property of a method. It is also the clearest argument for the
  offset metric: top-1 accuracy alone said "95%, good enough" and hid a bug
  that the direction-of-error breakdown made obvious in one table.

## 2026-09-21 — Nondeterministic span ids (a cache that could never hit)

- **Purpose:** Run the judge arm. Stopped a sequential run at 226/546 to
  parallelise it, expecting the 275 already-paid responses to be reused.
- **Result:** The prewarm reported `cached 0, new 100`. The cache was hitting
  nothing.
- **Cause:** Span ids were random uuid4s regenerated on every corpus build, and
  they are rendered into every judge prompt as the handle the model answers
  with. Identical traces therefore produced different prompts on every build,
  so the cache could never hit across processes.
- **Cost:** ~$1.03 wasted — $0.55 on the killed sequential run, $0.42 on the
  first prewarm, both unusable. My error: the cache's docstring claimed
  re-runs were free and I never verified a cross-process hit before relying
  on it.
- **Design change:** Span ids now derive from a stable `run_id`; the corpus is
  byte-reproducible, asserted by `test_corpus_is_byte_reproducible`. This was a
  reproducibility bug independent of cost — the benchmark was not reproducible
  at all. Verified the fix by checking cache-file existence from a *fresh*
  process (60/60 hits) rather than asserting it again.

## 2026-09-21 — Strengthening the baselines before trusting the headline

- **Purpose:** `first_error` scoring 0.000 on silent faults is true but easy to
  dismiss — of course an error-hunting heuristic finds nothing when there is no
  error. The claim that *inspection itself* cannot see a silent fault needs a
  heuristic that genuinely tries.
- **Run:** Added `output_anomaly`, a domain-agnostic detector that scores each
  tool call against the other calls of the same tool in the same trace
  (z-score on numeric leaves, plus empty-collection detection). It knows
  nothing about the domain or the fault taxonomy.
- **Cost:** $0.
- **Result:** 0.139 on silent faults — better than `first_error`'s 0.000, and
  the best MRR of any heuristic (0.412), but still far below replay. The
  information is not in the span.
- **Design change:** Added to the default baseline set. Closes the "you rigged
  the baselines" objection.

## 2026-09-21 — exp04, the LLM-judge arm (hypothesis refuted)

- **Purpose:** Test whether a model reading the trace can localise the culprit,
  and specifically whether it goes blind on silent faults the way `first_error`
  does.
- **Run:** Claude Haiku 4.5. Whole-trace judge over all 547 failures; per-span
  judge and judge-guided replay over a 63-trace stratified subsample. 1,220
  prompts prewarmed 8-way parallel; the experiment itself then ran from cache.
- **Cost:** $2.011 for the prewarm, **$0.000 for the experiment run** (1,346
  cache hits) — the determinism fix paying for itself immediately.
- **Result:** **The hypothesis was wrong.** The whole-trace judge scores 0.543
  overall and, critically, 0.545 on silent faults vs 0.537 on overt — no gap at
  all. It is not keying on error markers; it reconstructs the run's arithmetic
  from the trace. Two supporting findings: the per-span judge is *worse* than
  the whole-trace judge (0.413 vs 0.476 on identical traces), which supports the
  locality argument even as it undercuts the blindness one; and the direction
  metric shows judges fail by ~1 span in either direction (mean offset +1.06)
  where `first_error` fails by +7.72, always downstream.
- **Design change:** None to the code. The README's framing changed: the judge
  is a genuinely good cheap localizer, not a strawman, and the honest claim is
  about the accuracy/cost frontier rather than judge blindness.

## 2026-09-21 — Guided replay: a wrong algorithm, then a confound

- **Purpose:** Use a cheap scorer to order replay probes and cut cost.
- **Result 1 (a real bug):** Guided replay scored 0.768 against the unguided
  scan's 1.000. Probing in suspicion order breaks the invariant that makes the
  scan correct: when a fault propagates, repairing *any* downstream span that
  carries the corruption also changes the outcome, so a confirmation proves only
  that the culprit is at or before that span — never that it *is* that span.
- **Design change 1:** A confirmation is now an upper bound, followed by
  bisection below it. Diagnosis then showed all 127 remaining misses were "no
  confirmation within max_probes", where the method fell back to the scorer's
  unverified ranking — so that fallback became a full bisection instead.
  Accuracy returned to 1.000.
- **Result 2 (a confound in my own benchmark):** Even corrected, guidance cost
  *more* than plain bisection (5.7-7.1 vs 3.74 replays), with the LLM judge as
  scorer too. Depth analysis showed why: faults are injected at steps 0-3, so
  62% of culprits are found within 4 probes and a left-to-right scan is already
  near-optimal. That is a property of my injection schedule, not of guidance.
- **Design change 2:** Re-ran on a corpus with deeper injection points (steps
  2-7). Exhaustive degraded 4.67 → 7.70 replays; bisection held at 3.74 → 4.17;
  guidance still did not win. **Conclusion: prefix bisection is the method to
  ship** — cheapest, depth-independent, equally accurate — and guided replay is
  a reported negative result rather than the headline it was meant to be.

## 2026-09-22 — exp05, a real LLM agent (the largest gap, closed)

- **Purpose:** Every prior result was measured on traces produced by rule-based
  Python. Re-run the identical pipeline with Claude Haiku 4.5 making the
  agent's decisions, changing only `_propose` so the loop guard, step budget
  and best-effort fallback stay identical and any difference is attributable
  to the policy rather than the scaffolding.
- **Cost:** ~$0.63 total for the arm — $0.05 pilot, $0.48 corpus, $0.105 for
  localization. Replay came out at **$0.0000 marginal** because replaying a
  prefix reproduces observation histories the agent already walked, so the
  prompts were already cached.
- **Two problems found before any result was trusted:**
  1. *A grading artifact.* The model answered `$1,669.00` where gold was
     `1669.00`, and the pilot scored 6/8. Those are correct answers in a
     different surface form, and counting them as failures would have put
     traces in the corpus labelled "failed" with no injected fault to localize
     — every localizer scored against a culprit that does not exist. Added
     `normalize_answer`; the scripted arm's numbers were bit-identical
     afterwards, confirming it as a no-op there. Solve rate went to 8/8.
  2. *A biased corpus.* With `max_failed` set, the sweep stopped early in task
     order, so the first 40 failures all came from one task kind and the
     recovery rate read 0/40. Interleaved the task kinds; recovery went to
     55/99 and all four kinds are represented.
- **Result — the findings replicate.** `cf_bisect` holds at **1.000 top-1 at
  3.73 replays** against the scripted arm's 1.000 at 3.74. `first_error` again
  scores **exactly 0.000** on silent faults. Every method's ordering is
  preserved. Honest solve rate 5/5; 4 unparseable replies in ~870 decisions.
- **Result — the agent is more robust than the simulation.** The LLM agent
  recovered from 55.6% of injected faults against the scripted policy's 45.6%.
  A real model sometimes routes around corruption a credulous rule-based agent
  swallows. This means the main arm, if anything, *overstates* how damaging
  silent faults are — the right direction for a limitation to point.
- **Result — one comparison that cannot be settled here.** On LLM-agent traces
  the judge scored 0.424 silent vs 0.727 overt, where the scripted arm had no
  gap at all. With n=11 overt the intervals are [0.27,0.59] and [0.43,0.90] and
  overlap heavily. Recorded as underpowered and open, not as a reversal.

## 2026-09-22 — Auditing the headline against the evidence

- **Purpose:** Before publishing, check that the framing I had been using --
  "localization is bounded by information, not method quality" -- is actually
  what the data shows.
- **Cost:** $0.
- **Result: the framing was wrong, and one reported statistic was misleading.**
  1. *The information claim is contradicted by my own result.* If silent faults
     were unfindable for lack of information, the whole-trace judge should do
     worse on them. It does not -- 0.545 silent vs 0.537 overt, no gap. What
     span-local methods lack is **context**, not fault visibility: a silent
     fault is invisible in its own span and recoverable from the whole trace.
     The per-span judge (0.413, 0.388 on silent) sitting between `first_error`
     (0.000) and the trace judge (0.545) is the dose-response.
  2. *"The judge is nearly right when wrong" was an artifact.* I had reported a
     mean signed offset of +1.06 and read it as "off by about one span". Mean
     **absolute** offset on misses is **3.26** (median 3.0), in traces
     averaging 11.7 candidate spans. The signed mean was small only because
     errors in opposite directions cancelled -- 57.6% downstream, 42.4%
     upstream. The judge is not nearly right; it is broadly and
     unsystematically wrong.
  3. *Within-k windows are cheap.* `within +/-3 = 0.872` looks strong until the
     control: random reaches 0.692 at the same window, because +/-3 covers most
     of a short trace. Only the exact-match number (judge 0.543 vs random
     0.089) carries weight.
- **Design change:** Headline rewritten in `README.md` and `REPORT.md` to state
  the measured frontier (1.000 at 3.74 replays vs an inspection plateau near
  0.54, across both fault classes and both agent implementations, plus
  abstention) and to state explicitly that the *mechanism* for the 0.54 plateau
  is **not** established. Direction-of-error tables now report mean absolute
  distance, with the signed mean shown as the statistic that would have
  flattered the judge.
- **Note:** This is the seventh time an analysis changed a claim rather than
  confirming it, and the only one where the flaw was in my interpretation
  rather than in the harness. The correction makes the headline narrower and
  the report more defensible: the strong result was never the mechanism, it was
  the frontier.
