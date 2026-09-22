"""
Run the localizer benchmark as Phoenix Datasets + Experiments.

The benchmark originally shipped with its own runner: sweep a corpus, score
each method, print a table. That worked, and it was the wrong call. Phoenix
*is* an experimentation product -- datasets of examples, tasks run over them,
evaluators scoring the output, experiments compared side by side in a UI -- and
reimplementing that in `bench/run.py` meant never finding out what the product
does well or badly. This module moves the benchmark onto the real thing.

The mapping is almost one to one:

| benchmark concept         | Phoenix concept                         |
|---------------------------|-----------------------------------------|
| a failed trace + culprit  | a dataset **example** (input + output)  |
| a localizer               | an experiment **task**                  |
| top-1 / MRR / cost        | experiment **evaluators**               |
| comparing methods         | comparing **experiments** on a dataset  |

Two things this buys that the custom runner did not. Results become durable and
inspectable per example -- you can click into the trace where `first_error`
failed and see what it answered instead. And comparison becomes the product's
job rather than a `print(table(...))`, so adding a method later is one more
experiment against the same dataset instead of a new column in a script.

One genuine friction worth recording: a dataset example is JSON, but a localizer
needs a `Trace` object with live `SpanRec`s, and the replay methods additionally
need a world and an armed injector that cannot be serialized. The example
therefore carries the *recipe* (`case_id`, fault, target step, seed) and the
task rehydrates the case locally. That keeps the dataset honest -- it holds only
what a Phoenix user could actually store -- at the cost of the task needing the
benchmark package on the path.
"""

from __future__ import annotations

from typing import Any, Callable

from phoenix.client import Client

from .bench.corpus import Case, build_corpus
from .faults import FAULT_CATALOG
from .environment import BenchEnvironment
from .localizers.base import Context, Localizer

DEFAULT_URL = "http://localhost:6006"


def build_dataset(
    cases: list[Case],
    name: str,
    base_url: str = DEFAULT_URL,
    description: str | None = None,
) -> Any:
    """
    Upload failed traces as a Phoenix dataset.

    Input holds what a localizer is allowed to see: the question, the answer the
    agent gave, and the spans. Output holds the ground truth. Metadata holds the
    recipe needed to rehydrate the case, plus the fault attributes used to slice
    results in the UI -- fault name, visibility, and which side of the agent/tool
    boundary the fault sits on.
    """
    examples = []
    for c in cases:
        spans = [s for s in c.trace.candidates if s.name != "agent.answer"]
        examples.append(
            {
                # What a localizer is allowed to see.
                "input": {
                    "question": c.trace.task.question,
                    "agent_answer": c.trace.final_answer,
                    "n_spans": len(spans),
                    "spans": [
                        {
                            "span_id": s.span_id, "name": s.name, "kind": s.span_kind,
                            "input": s.input, "output": s.output, "error": s.error,
                        }
                        for s in spans
                    ],
                },
                # Ground truth.
                "output": {
                    "culprit_span_id": c.trace.culprit_span_id,
                    "gold_answer": c.trace.task.gold,
                },
                # Slicing dimensions in the UI, plus the recipe the task needs to
                # rehydrate the case locally.
                "metadata": {
                    "case_id": c.case_id,
                    "fault": c.fault,
                    "visibility": c.visibility,
                    "fault_phase": FAULT_CATALOG[c.fault][0] if c.fault else "none",
                    "target_step": c.target_step,
                    "seed": c.seed,
                    "task_kind": c.task_kind,
                    "status": c.status,
                },
            }
        )

    return Client(base_url=base_url).datasets.create_dataset(
        name=name,
        examples=examples,
        dataset_description=description
        or ("Failed multi-step agent runs with a known root-cause span. One fault "
            "injected per run, so the correct answer is known by construction."),
    )


def make_task(
    localizer: Localizer,
    world: Any,
    cases_by_id: dict[str, Case],
    signal: str = "changed",
    repair_success_prob: float = 1.0,
    llm: Any = None,
    policy: Any = None,
) -> Callable[..., dict[str, Any]]:
    """
    Wrap a localizer as a Phoenix experiment task.

    The task receives one dataset example and returns the span it accuses, plus
    the cost it paid. Phoenix stores both, so the accuracy/cost frontier that is
    the whole point of this project is visible in the experiment table rather
    than only in a report.
    """

    def task(example: Any) -> dict[str, Any]:
        meta = dict(getattr(example, "metadata", None) or {})
        case = cases_by_id[meta["case_id"]]
        ctx = Context(
            world=BenchEnvironment(world),
            env_injector=case.injector(world, policy=policy),
            repair_success_prob=repair_success_prob,
            policy=policy,
            llm=llm,
            signal=signal,
            gold=case.trace.task.gold if signal == "fixed" else None,
        )
        v = localizer.localize(case.trace, ctx)
        return {
            "predicted_span_id": v.span_id,
            "ranking": v.ranking,
            "explanation": v.explanation,
            "confidence": v.score,
            "replays": v.cost.replays,
            "llm_calls": v.cost.llm_calls,
            "abstained": v.score == 0.0,
        }

    task.__name__ = localizer.name
    return task


# --------------------------------------------------------------------------
# Evaluators
# --------------------------------------------------------------------------
# Each takes the experiment's `output` and the dataset's `expected`, and returns
# a score Phoenix stores against the run. Keeping them small and separate is
# what makes the comparison view useful: a method that is accurate but expensive
# and one that is cheap but wrong are distinguishable at a glance.

def correct_span(output: Any, expected: Any) -> dict[str, Any]:
    """Top-1: did it name the span that actually caused the failure?"""
    got = (output or {}).get("predicted_span_id")
    want = (expected or {}).get("culprit_span_id")
    hit = bool(want) and got == want
    return {
        "score": 1.0 if hit else 0.0,
        "label": "correct" if hit else "wrong",
        "explanation": (
            f"named {got}, the true root cause"
            if hit else f"named {got}; the root cause was {want}"
        ),
    }


def reciprocal_rank(output: Any, expected: Any) -> dict[str, Any]:
    """
    Where in the ranking the true culprit sits.

    Worth scoring separately from top-1: a method that ranks the culprit second
    is useless as a verdict but useful as a way to order replay probes.
    """
    ranking = (output or {}).get("ranking") or []
    want = (expected or {}).get("culprit_span_id")
    if want in ranking:
        rank = ranking.index(want) + 1
        return {"score": 1.0 / rank, "label": f"rank_{rank}",
                "explanation": f"true culprit ranked {rank} of {len(ranking)}"}
    return {"score": 0.0, "label": "unranked",
            "explanation": "true culprit absent from the ranking"}


def blame_distance(output: Any, expected: Any) -> dict[str, Any]:
    """
    How far off, and in which direction.

    Reported as signed distance in span positions. Blaming downstream means the
    method found where the damage became visible rather than where it started.
    """
    got = (output or {}).get("predicted_span_id")
    want = (expected or {}).get("culprit_span_id")
    ranking = (output or {}).get("ranking") or []
    if not (got and want) or got not in ranking or want not in ranking:
        return {"score": None, "label": "unknown", "explanation": "positions unavailable"}
    if got == want:
        return {"score": 0.0, "label": "exact", "explanation": "named the culprit itself"}
    return {
        "score": None, "label": "off_target",
        "explanation": f"accused a different span than the true culprit ({got} vs {want})",
    }


def replay_cost(output: Any) -> dict[str, Any]:
    """Replays spent. The other half of the frontier; lower is better."""
    n = int((output or {}).get("replays", 0))
    return {"score": float(n), "label": f"{n}_replays",
            "explanation": f"{n} counterfactual replays"}


def stayed_quiet(output: Any, expected: Any) -> dict[str, Any]:
    """
    Abstention. Only meaningful on a dataset of runs that did *not* fail, where
    the correct behaviour is to decline to name anyone.
    """
    abstained = bool((output or {}).get("abstained"))
    return {"score": 1.0 if abstained else 0.0,
            "label": "abstained" if abstained else "named_a_suspect",
            "explanation": "declined to blame any span" if abstained
            else f"blamed {(output or {}).get('predicted_span_id')}"}


ACCURACY_EVALUATORS = {
    "correct_span": correct_span,
    "reciprocal_rank": reciprocal_rank,
    "blame_distance": blame_distance,
    "replay_cost": replay_cost,
}

ABSTENTION_EVALUATORS = {
    "stayed_quiet": stayed_quiet,
    "replay_cost": replay_cost,
}
