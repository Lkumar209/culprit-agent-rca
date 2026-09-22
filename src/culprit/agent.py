"""
The benchmark agent: a multi-hop, tool-using agent over the synthetic world.

The single most important property of the policy in this module is that it
**derives every action from what it has observed**, never from the task's
reference plan. That sounds obvious, but it is the difference between a real
benchmark and a vacuous one: a policy that replayed the gold plan would sail
through an injected fault unchanged, always produce the gold answer, and every
localization number downstream would be measuring nothing. Because the policy
reads observations, a corrupted `vendor_id` at step 1 genuinely poisons every
subsequent call, exactly as it would in production.

Two policies share one interface:

* `ScriptedPolicy` -- deterministic rules, zero API cost. This is the default
  arm so the whole pipeline is reproducible without spending anything.
* `LLMPolicy` -- a real model over OpenRouter, same tool surface, for the arm
  that checks the findings survive contact with an actual model.

A run emits a flat list of `SpanRec`s. That list is the unit of analysis for
every localizer, and it maps one-to-one onto the OpenInference spans that get
exported to Phoenix, so a localizer developed against the benchmark works
unmodified against spans pulled back out of a live Phoenix instance.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Protocol

from .tools import Task, ToolError, call_tool, normalize_answer
from .world import QUARTERS, World

MAX_STEPS = 24

# A fault injector sees every decision and every observation and may rewrite it.
# Signature: (phase, step_index, payload) -> (payload, was_faulted)
Injector = Callable[[str, int, dict[str, Any]], tuple[dict[str, Any], bool]]


@dataclass
class SpanRec:
    """One span. Mirrors the OpenInference span exported to Phoenix."""

    span_id: str
    parent_id: str | None
    name: str
    span_kind: str  # AGENT | LLM | TOOL
    index: int
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    # Ground truth, benchmark-only. Never shown to a localizer.
    faulted: bool = False
    fault_name: str | None = None

    def redacted(self) -> dict[str, Any]:
        """What a localizer is allowed to see -- no ground-truth leakage."""
        d = asdict(self)
        d.pop("faulted", None)
        d.pop("fault_name", None)
        return d


@dataclass
class Trace:
    trace_id: str
    task: Task
    spans: list[SpanRec]
    final_answer: str | None
    success: bool
    # Ground truth
    culprit_span_id: str | None = None
    fault_name: str | None = None
    # Root-span metadata, populated when the trace is loaded back from Phoenix.
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def candidates(self) -> list[SpanRec]:
        """Spans a localizer may blame: everything but the root."""
        return [s for s in self.spans if s.span_kind != "AGENT"]


class Policy(Protocol):
    def decide(self, task: Task, observations: list[dict[str, Any]]) -> dict[str, Any]: ...


# --------------------------------------------------------------------------
# Scripted policy
# --------------------------------------------------------------------------

def _sig_of(tool: Any, args: dict[str, Any]) -> str:
    """Stable identity for a tool call, used by the loop guard."""
    return json.dumps({"t": tool, "a": args}, sort_keys=True, default=str)


_ORDER_RE = re.compile(r"\bORD-\d+\b")
_QUARTER_RE = re.compile(r"\bQ[1-4]\b")


class ScriptedPolicy:
    """
    Rule-based agent. Reads the question, then works strictly off observations.

    It is deliberately a *competent but not paranoid* agent: it follows
    `has_more` when it is told there is more, retries a tool once on an overt
    error, and otherwise trusts what tools hand it. That trust is what makes
    silent corruption dangerous, and it is an accurate model of how most
    production agents behave.
    """

    name = "scripted"

    def __init__(
        self, retry_on_error: bool = True, max_attempts: int = 2, step_budget: int = 12
    ) -> None:
        self.retry_on_error = retry_on_error
        self.max_attempts = max_attempts
        self.step_budget = step_budget

    def decide(self, task: Task, observations: list[dict[str, Any]]) -> dict[str, Any]:
        """Propose an action, then refuse to bang on a door that is already shut."""
        # Iteration cap. Every production agent framework has one, and
        # "agent burned its step budget and answered from partial data" is
        # among the most common real failures -- a fault that makes the
        # agent re-propose a call it can never complete lands here.
        if len(observations) >= self.step_budget:
            return {
                "action": "finish",
                "answer": self._best_effort(task, observations),
                "gave_up": True,
            }

        action = self._propose(task, observations)
        if action.get("action") != "call":
            return action

        sig = _sig_of(action.get("tool"), action.get("args", {}))
        attempts = sum(1 for o in observations if _sig_of(o.get("tool"), o.get("args", {})) == sig)
        if attempts >= self.max_attempts:
            # Give up and answer from whatever evidence is in hand. Production
            # agents bail; they do not retry forever. Without this guard a
            # persistent tool fault yields a degenerate 49-span retry loop,
            # which would dominate the corpus and make trace length -- rather
            # than the fault itself -- the thing localizers key off.
            return {
                "action": "finish",
                "answer": self._best_effort(task, observations),
                "gave_up": True,
            }
        return action

    def _best_effort(self, task: Task, observations: list[dict[str, Any]]) -> str:
        """The answer the agent commits to when it has stopped making progress."""
        pages = [o for o in observations if o.get("ok") and o.get("tool") == "list_invoices"]

        def amounts(wq: str | None) -> list[int]:
            out: list[int] = []
            for p in pages:
                if wq is None or p["args"].get("quarter") == wq:
                    out += [int(i["amount_cents"]) for i in p["result"].get("invoices", [])]
            return out

        qm = _QUARTER_RE.search(task.question)
        quarter = qm.group(0) if qm else None
        if task.kind == "invoice_count":
            return str(len(amounts(quarter)))
        if task.kind == "quarter_spend":
            return f"{sum(amounts(quarter)) / 100:.2f}"
        if task.kind == "cap_check":
            return "NO"
        if task.kind == "top_quarter":
            totals = {q: sum(amounts(q)) for q in QUARTERS}
            return max(QUARTERS, key=lambda k: (totals[k], k))
        return ""

    def _propose(self, task: Task, observations: list[dict[str, Any]]) -> dict[str, Any]:
        q = task.question
        order_id = (_ORDER_RE.search(q) or [None])[0] if _ORDER_RE.search(q) else None
        quarter_m = _QUARTER_RE.search(q)
        quarter = quarter_m.group(0) if quarter_m else None

        done = [o for o in observations if o.get("ok")]
        by_tool: dict[str, list[dict[str, Any]]] = {}
        for o in done:
            by_tool.setdefault(o["tool"], []).append(o)

        # Retry an overt failure once before moving on.
        if self.retry_on_error and observations:
            last = observations[-1]
            if not last.get("ok") and not last.get("is_retry"):
                return {"action": "call", "tool": last["tool"], "args": last["args"], "is_retry": True}

        # 1. Resolve the order -> vendor.
        if "find_order" not in by_tool:
            return {"action": "call", "tool": "find_order", "args": {"order_id": order_id}}
        vendor_id = by_tool["find_order"][-1]["result"].get("vendor_id")

        # 2. cap_check needs the vendor's category and the matching policy cap.
        if task.kind == "cap_check":
            if "get_vendor" not in by_tool:
                return {"action": "call", "tool": "get_vendor", "args": {"vendor_id": vendor_id}}
            category = by_tool["get_vendor"][-1]["result"].get("category")
            if "get_policy_cap" not in by_tool:
                return {"action": "call", "tool": "get_policy_cap", "args": {"category": category}}

        # 3. Sweep invoices. top_quarter sweeps all four quarters; others one.
        wanted = list(QUARTERS) if task.kind == "top_quarter" else [quarter]
        pages = by_tool.get("list_invoices", [])
        for wq in wanted:
            seen = [p for p in pages if p["args"].get("quarter") == wq]
            if not seen:
                return {
                    "action": "call",
                    "tool": "list_invoices",
                    "args": {"vendor_id": vendor_id, "quarter": wq, "page": 1},
                }
            if seen[-1]["result"].get("has_more"):
                return {
                    "action": "call",
                    "tool": "list_invoices",
                    "args": {
                        "vendor_id": vendor_id,
                        "quarter": wq,
                        "page": int(seen[-1]["args"].get("page", 1)) + 1,
                    },
                }

        def amounts_for(wq: str | None) -> list[int]:
            out: list[int] = []
            for p in pages:
                if wq is None or p["args"].get("quarter") == wq:
                    out += [int(i["amount_cents"]) for i in p["result"].get("invoices", [])]
            return out

        # 4. Compute and answer.
        if task.kind == "invoice_count":
            return {"action": "finish", "answer": str(len(amounts_for(quarter)))}

        if task.kind == "top_quarter":
            totals = {wq: sum(amounts_for(wq)) for wq in QUARTERS}
            return {"action": "finish", "answer": max(QUARTERS, key=lambda k: (totals[k], k))}

        amts = amounts_for(quarter)
        if "sum_amounts" not in by_tool:
            return {"action": "call", "tool": "sum_amounts", "args": {"amounts": amts}}
        total = int(by_tool["sum_amounts"][-1]["result"].get("total_cents", 0))

        if task.kind == "quarter_spend":
            return {"action": "finish", "answer": f"{total / 100:.2f}"}

        if task.kind == "cap_check":
            cap = int(by_tool["get_policy_cap"][-1]["result"].get("cap_cents", 0))
            return {"action": "finish", "answer": "YES" if total > cap else "NO"}

        return {"action": "finish", "answer": ""}


# --------------------------------------------------------------------------
# Run loop
# --------------------------------------------------------------------------

def _id_factory(run_id: str | None):
    """
    Span-id generator for one run.

    Ids are derived from a stable `run_id` rather than drawn at random,
    because they are not merely labels: they are rendered into every judge
    prompt as the handle the model answers with. Random ids meant a fresh
    corpus build produced different prompts for identical traces, so the
    response cache could never hit across processes and every re-run paid the
    full API cost again. Deterministic ids make the whole benchmark
    byte-reproducible.

    `run_id=None` keeps the old random behaviour for ad-hoc use.
    """
    if run_id is None:
        return lambda i: uuid.uuid4().hex[:16]
    return lambda i: hashlib.sha256(f"{run_id}|{i}".encode()).hexdigest()[:16]


def run_agent(
    world: World,
    task: Task,
    policy: Policy | None = None,
    injector: Injector | None = None,
    run_id: str | None = None,
) -> Trace:
    """
    Execute one task and return its trace.

    The injector is offered every policy decision (`phase="decision"`) and every
    tool observation (`phase="observation"`) and may rewrite either. It is the
    only source of corruption; with `injector=None` the run is honest and, if
    the world and policy are sane, ends at the gold answer.
    """
    policy = policy or ScriptedPolicy()
    new_id = _id_factory(run_id)
    _n = iter(range(10_000))
    _sid = lambda: new_id(next(_n))
    trace_id = new_id(-1) if run_id is not None else uuid.uuid4().hex
    root = SpanRec(
        span_id=_sid(), parent_id=None, name="agent.run", span_kind="AGENT", index=0,
        input={"question": task.question, "task_id": task.task_id},
    )
    spans = [root]
    observations: list[dict[str, Any]] = []
    culprit_span_id: str | None = None
    fault_name: str | None = None
    final_answer: str | None = None

    for step in range(MAX_STEPS):
        # --- policy decision ------------------------------------------------
        decision = policy.decide(task, observations)
        d_span = SpanRec(
            span_id=_sid(), parent_id=root.span_id, name="agent.decide", span_kind="LLM",
            index=len(spans),
            input={"question": task.question, "n_observations": len(observations)},
        )
        if injector is not None:
            decision, hit = injector("decision", step, decision)
            if hit:
                d_span.faulted = True
                culprit_span_id = culprit_span_id or d_span.span_id
        d_span.output = {"decision": decision}
        spans.append(d_span)

        if decision.get("action") == "finish":
            final_answer = str(decision.get("answer", ""))
            spans.append(
                SpanRec(
                    span_id=_sid(), parent_id=root.span_id, name="agent.answer",
                    span_kind="LLM", index=len(spans),
                    input={"question": task.question}, output={"answer": final_answer},
                )
            )
            break

        # --- tool execution -------------------------------------------------
        tool, args = decision.get("tool"), decision.get("args", {})
        t_span = SpanRec(
            span_id=_sid(), parent_id=root.span_id, name=f"tool.{tool}", span_kind="TOOL",
            index=len(spans), input={"tool": tool, "args": args},
        )
        try:
            result = call_tool(world, tool, args)
            err = None
        except ToolError as exc:
            result, err = {}, str(exc)

        obs: dict[str, Any] = {
            "tool": tool, "args": args, "ok": err is None, "result": result,
            "error": err, "is_retry": bool(decision.get("is_retry")),
        }
        if injector is not None:
            payload, hit = injector("observation", step, obs)
            obs = payload
            if hit:
                t_span.faulted = True
                culprit_span_id = culprit_span_id or t_span.span_id

        t_span.output = {"ok": obs["ok"], "result": obs["result"]}
        t_span.error = obs["error"]
        spans.append(t_span)
        observations.append(obs)

    root.output = {"answer": final_answer, "n_spans": len(spans)}
    success = final_answer is not None and normalize_answer(final_answer) == normalize_answer(task.gold)

    if injector is not None:
        fault_name = getattr(injector, "fault_name", None)
        for s in spans:
            if s.faulted:
                s.fault_name = fault_name

    return Trace(
        trace_id=trace_id, task=task, spans=spans, final_answer=final_answer,
        success=success, culprit_span_id=culprit_span_id, fault_name=fault_name,
    )
