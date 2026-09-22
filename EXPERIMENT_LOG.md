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

## PENDING — exp04, the LLM-judge arm

- **Purpose:** The baseline a reviewer asks for first. If a model can just read
  the trace and name the culprit, intervention is an expensive way to buy the
  same answer. Hypothesis: a silent fault leaves a locally unremarkable span,
  so judges should track `first_error` — respectable on overt faults, poor on
  silent ones — largely regardless of model capability.
- **Status:** Implemented, not run. Needs an `ANTHROPIC_API_KEY`. Planned:
  Claude Haiku 4.5, whole-trace judge over all 546 failures, per-span judge
  over a 150-trace stratified subsample, plus the judge-guided hybrid (which
  reuses cached judge calls and so costs nothing extra). Run
  `--dry-run` first — it prices the sweep with the token counter before
  spending anything.
