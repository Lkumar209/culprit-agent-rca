"""
Guard the published numbers against code drift.

Every other test here checks that the benchmark is internally sound. These
check something different and, as it turned out, more easily broken: that the
code still produces the numbers the README and the report actually claim.

This exists because of a specific near-miss. Adding a diagnostic fault to
`FAULT_CATALOG` silently enrolled it in the evaluation corpus, because
`build_corpus` defaulted to the whole catalog. The corpus went from 547 failed
traces to 616, every published figure stopped reproducing, and a fault that had
been engineered to be maximally detectable by an inspecting judge began
contaminating that judge's headline score. Nothing failed. No test complained.
It was found by reading the code weeks' worth of commits later.

The lesson is that "the benchmark is correct" and "the benchmark still produces
the reported result" are different properties, and only the first one had tests.
"""

import json
import pathlib

import pytest

from culprit.bench.corpus import build_corpus, summarize
from culprit.bench.run import run_methods
from culprit.faults import FAULT_CATALOG, MAIN_FAULTS, PROBE_FAULTS
from culprit.localizers.heuristics import (
    EarliestToolLocalizer, FirstErrorLocalizer, LastSpanLocalizer, OutputAnomalyLocalizer,
)

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "results"

# The published configuration. If an experiment's config changes, this must
# change with it -- deliberately, and in the same commit.
PUBLISHED = dict(n_per_kind=10, target_steps=(0, 1, 2, 3))


@pytest.fixture(scope="module")
def published_exp01():
    path = RESULTS / "exp01_free_arm.json"
    if not path.exists():
        pytest.skip("exp01 results not present")
    return json.loads(path.read_text())


def test_probe_faults_are_not_in_the_evaluation_corpus():
    """
    A probe must be requested by name, never swept in by default.

    `contradictory_echo` was built to be maximally detectable by an inspecting
    judge in order to test one hypothesis. In the main corpus it would move the
    judge's score using a fault chosen for that property, which is exactly the
    shape of result a reviewer should distrust.
    """
    for probe in PROBE_FAULTS:
        assert probe in FAULT_CATALOG, f"{probe} should still be a usable fault"
        assert probe not in MAIN_FAULTS, f"{probe} must not be in the default corpus"

    _, cases = build_corpus(n_per_kind=2, target_steps=(0, 1))
    used = {c.fault for c in cases if c.fault}
    assert not (used & set(PROBE_FAULTS)), f"probe fault leaked into the corpus: {used}"


def test_corpus_composition_matches_published(published_exp01):
    """The corpus behind every published figure must still be that corpus."""
    _, cases = build_corpus(**PUBLISHED)
    now = summarize(cases)
    was = published_exp01["extra"]["corpus"]

    assert now["by_status"] == was["by_status"], (
        f"corpus drifted: published {was['by_status']}, now {now['by_status']}"
    )
    assert now["failed_by_visibility"] == was["failed_by_visibility"]
    assert now["failed_by_fault"] == was["failed_by_fault"]


def test_published_heuristic_numbers_still_reproduce(published_exp01):
    """
    Re-run the zero-cost methods and compare against the saved rows.

    Only the free localizers are re-run: they need no API key and no replays,
    so this stays fast enough to run on every commit. They are also enough to
    catch the failure mode that matters -- a change to the world, the tasks, the
    agent or the fault taxonomy moves these numbers immediately.
    """
    world, cases = build_corpus(**PUBLISHED)
    methods = [
        LastSpanLocalizer(), FirstErrorLocalizer(),
        EarliestToolLocalizer(), OutputAnomalyLocalizer(),
    ]
    rows = run_methods(world, cases, methods, repair_success_prob=1.0,
                       signal="changed", progress=False)

    def top1(rs, method):
        sel = [r for r in rs if (r["method"] if isinstance(r, dict) else r.method) == method]
        hits = [r["hit"] if isinstance(r, dict) else r.hit for r in sel]
        return sum(hits) / len(hits), len(hits)

    for m in methods:
        got, n_now = top1(rows, m.name)
        was, n_was = top1(published_exp01["rows"], m.name)
        assert n_now == n_was, f"{m.name}: case count changed, {n_was} -> {n_now}"
        assert abs(got - was) < 5e-4, (
            f"{m.name}: published top-1 {was:.3f}, now {got:.3f} -- "
            "either restore the behaviour or re-run the experiments and update the docs"
        )
