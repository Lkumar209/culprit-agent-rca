"""
`culprit` command line.

    culprit demo      end-to-end against a live Phoenix instance
    culprit analyze   localize the failures in a Phoenix project
    culprit doctor    check that Phoenix is reachable and writable

`demo` is the one to run first: it manufactures agent runs with known faults,
ships them to Phoenix as OpenInference spans, reads them back through the same
client a user would, localizes each failure, and annotates the guilty span.
Everything it does is something a Phoenix user could do to their own project;
the only privileged thing is that it knows the right answer, so it can tell you
how often the tool was right.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from .bench.corpus import build_corpus
from .environment import BenchEnvironment
from .localizers.base import Context
from .localizers.counterfactual import BisectReplay, ExhaustiveReplay
from .phoenix_io import load_traces, write_verdict
from .tracing import export_trace, flush

DEFAULT_URL = "http://localhost:6006"


def cmd_doctor(args: argparse.Namespace) -> int:
    import httpx

    try:
        r = httpx.get(f"{args.url}/healthz", timeout=5)
        print(f"phoenix at {args.url}: HTTP {r.status_code}")
    except Exception as exc:
        print(f"phoenix at {args.url}: unreachable ({exc})")
        print("start one with:  phoenix serve")
        return 1
    from phoenix.client import Client

    try:
        Client(base_url=args.url).spans.get_spans_dataframe(project_identifier="default")
        print("client read: ok")
    except Exception as exc:
        print(f"client read: {type(exc).__name__}: {exc}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    print(f"building corpus (n_per_kind={args.n}) ...")
    world, cases = build_corpus(n_per_kind=args.n, target_steps=(0, 1, 2))
    failed = [c for c in cases if c.status == "failed"][: args.limit]
    healthy = [c for c in cases if c.status == "honest"][: max(2, args.limit // 5)]
    batch = failed + healthy
    print(f"exporting {len(batch)} traces ({len(failed)} failed, {len(healthy)} healthy) "
          f"to project {args.project!r} ...")
    for c in batch:
        export_trace(c.trace, project_name=args.project,
                     extra_metadata={"culprit.case_id": c.case_id})
    flush(args.project)

    print("reading them back through the Phoenix client ...")
    traces = _await_ingest(args, expected=len(batch),
                           expected_spans=sum(len(c.trace.spans) for c in batch))
    by_case = {c.case_id: c for c in batch}
    print(f"  {len(traces)} traces recovered from Phoenix")

    method = BisectReplay() if args.method == "bisect" else ExhaustiveReplay(n_samples=args.samples)
    print(f"localizing with {method.name} (signal={args.signal}) ...")

    stats: Counter[str] = Counter()
    for tr in traces:
        case = by_case.get(_case_id_of(tr))
        if case is None:
            continue
        ctx = Context(
            world=BenchEnvironment(world),
            env_injector=case.injector(world),
            repair_success_prob=args.p_repair,
            signal=args.signal,
            gold=case.trace.task.gold if args.signal == "fixed" else None,
        )
        v = method.localize(tr, ctx)
        if case.status == "honest":
            stats["healthy_abstained" if v.score == 0.0 else "healthy_false_positive"] += 1
            continue
        stats["hit" if v.span_id == tr.culprit_span_id else "miss"] += 1
        if v.span_id and v.score > 0:
            write_verdict(
                v.span_id, v.score, "root_cause", v.explanation, base_url=args.url,
                metadata={"method": method.name, "replays": v.cost.replays},
            )

    total = stats["hit"] + stats["miss"]
    print()
    print(f"  failed traces localized : {stats['hit']}/{total}"
          f"  ({stats['hit'] / total:.1%})" if total else "  no failed traces")
    print(f"  healthy traces abstained: {stats['healthy_abstained']}/"
          f"{stats['healthy_abstained'] + stats['healthy_false_positive']}")
    print()
    print(f"open {args.url}/projects and filter by annotation 'culprit' to see the verdicts")
    return 0


def _await_ingest(args: argparse.Namespace, expected: int, expected_spans: int,
                  timeout: float = 60.0) -> list:
    """
    Poll until Phoenix has ingested everything we sent.

    OTLP export and Phoenix ingestion are both asynchronous, so a read issued
    straight after `flush()` returns whatever happens to have landed. Reading
    early does not merely return fewer traces -- it returns *partial* ones,
    which silently corrupts localization because replay reconstructs the
    agent's step structure from the spans it can see.
    """
    import time

    deadline = time.time() + timeout
    last = 0
    while time.time() < deadline:
        traces = load_traces(args.project, base_url=args.url, keep_ground_truth=True)
        n_spans = sum(len(t.spans) for t in traces)
        if len(traces) >= expected and n_spans >= expected_spans:
            return traces
        if n_spans != last:
            last = n_spans
            print(f"  ... {len(traces)}/{expected} traces, {n_spans}/{expected_spans} spans")
        time.sleep(1.0)
    print(f"  warning: timed out with {len(traces)}/{expected} traces; results will be partial")
    return traces


def _case_id_of(trace) -> str | None:
    """The case id stamped into the root span metadata at export time."""
    return trace.meta.get("culprit.case_id")


def cmd_analyze(args: argparse.Namespace) -> int:
    traces = load_traces(args.project, base_url=args.url)
    failures = [t for t in traces if not t.success]
    print(f"{len(traces)} traces in {args.project!r}; {len(failures)} failed")
    print("analyze requires a replay Environment for your own tools -- see "
          "culprit.environment.Environment. Without one, only the heuristic "
          "localizers can run.")
    from .localizers.heuristics import FirstErrorLocalizer

    loc, ctx = FirstErrorLocalizer(), Context(signal="changed")
    for t in failures[: args.limit]:
        v = loc.localize(t, ctx)
        print(f"  {t.trace_id[:12]}  {v.span_id}  {v.explanation}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="culprit", description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL, help="Phoenix base URL")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="check the Phoenix connection")
    d.set_defaults(func=cmd_doctor)

    m = sub.add_parser("demo", help="end-to-end run against Phoenix")
    m.add_argument("--project", default="culprit-demo")
    m.add_argument("-n", type=int, default=4, help="tasks per kind")
    m.add_argument("--limit", type=int, default=40, help="max failed traces")
    m.add_argument("--method", choices=("bisect", "exhaustive"), default="exhaustive")
    m.add_argument("--samples", type=int, default=3)
    m.add_argument("--signal", choices=("fixed", "changed"), default="changed")
    m.add_argument("--p-repair", dest="p_repair", type=float, default=1.0)
    m.set_defaults(func=cmd_demo)

    a = sub.add_parser("analyze", help="localize failures in a Phoenix project")
    a.add_argument("--project", required=True)
    a.add_argument("--limit", type=int, default=20)
    a.set_defaults(func=cmd_analyze)

    args = ap.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
