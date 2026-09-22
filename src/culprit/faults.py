"""
Fault injection: the source of ground truth.

Every faulted run corrupts exactly one step, and we record which span that
was. That recorded span is the label -- which is the entire reason this
benchmark can report a localization *accuracy* instead of an anecdote.

The taxonomy is split along the axis that turns out to matter most:

* **Overt** faults announce themselves. The tool raises, the span carries an
  error, and the symptom is co-located with the cause. Any heuristic that
  looks for a red span finds these.
* **Silent** faults return well-formed, plausible data that is simply wrong.
  Nothing in the faulted span looks unusual in isolation; the symptom only
  surfaces later, in the final answer. These are the cases where "which span
  looks wrong?" and "which span *is* wrong?" come apart, and they are the
  reason this project exists.

A faulted run does not always fail. A fault the agent happens to route around
produces a *recovered* trace -- kept deliberately, because a localizer that
confidently blames a span in a run that actually succeeded is a localizer that
would cry wolf in production.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .world import World

# fault name -> (phase, visibility)
FAULT_CATALOG: dict[str, tuple[str, str]] = {
    # Observation-phase: the tool hands back something wrong.
    "wrong_entity": ("observation", "silent"),
    "stale_amounts": ("observation", "silent"),
    "unit_shift": ("observation", "silent"),
    "truncated_page": ("observation", "silent"),
    "empty_result": ("observation", "silent"),
    "tool_error": ("observation", "overt"),
    # A held-out probe, not part of the main corpus. Observation-phase like
    # its neighbours above, but engineered to contradict its own request:
    # the response echoes a different vendor_id than the one asked for. It
    # tests whether an inspecting judge is limited by which side of the
    # agent/tool boundary a fault sits on, or by whether the trace contains
    # a contradiction to notice.
    "contradictory_echo": ("observation", "silent"),
    # Decision-phase: the agent asks for the wrong thing.
    "hallucinated_arg": ("decision", "silent"),
    "arg_typo": ("decision", "overt"),
    "dropped_constraint": ("decision", "silent"),
}

# Diagnostic probes. These exist to test a specific hypothesis and are
# deliberately NOT part of the evaluation corpus: `contradictory_echo` was
# engineered to be maximally detectable by an inspecting judge, so including it
# in the main sweep would inflate the judge's score with a fault chosen for that
# property. `build_corpus` defaults to MAIN_FAULTS; a probe has to be requested
# by name.
PROBE_FAULTS = ("contradictory_echo",)
MAIN_FAULTS = tuple(f for f in FAULT_CATALOG if f not in PROBE_FAULTS)

SILENT = tuple(n for n, (_, v) in FAULT_CATALOG.items() if v == "silent")
OVERT = tuple(n for n, (_, v) in FAULT_CATALOG.items() if v == "overt")


@dataclass
class Injector:
    """
    A single-fault injector.

    Fires at the first eligible opportunity at or after `target_step`, then
    keeps firing for repeats of that same call signature -- a tool that is
    broken stays broken, so the agent's retry hits the same wall it did the
    first time. `culprit` ground truth is always the *first* span it hit.
    """

    fault_name: str
    world: World
    target_step: int = 0
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.fired = False
        self.signature: tuple[Any, ...] | None = None
        self.phase, self.visibility = FAULT_CATALOG[self.fault_name]

    # -- eligibility --------------------------------------------------------
    def _eligible(self, phase: str, step: int, payload: dict[str, Any]) -> bool:
        if phase != self.phase or step < self.target_step:
            return False
        f = self.fault_name

        if phase == "decision":
            if payload.get("action") != "call":
                return False
            tool = payload.get("tool")
            if f == "hallucinated_arg":
                return tool == "list_invoices" and payload.get("args", {}).get("vendor_id")
            if f == "arg_typo":
                return tool in ("find_order", "get_vendor")
            if f == "dropped_constraint":
                return tool == "list_invoices" and payload.get("args", {}).get("quarter")
            return False

        # observation phase
        if not payload.get("ok"):
            return False
        tool, res = payload.get("tool"), payload.get("result", {})
        if f == "wrong_entity":
            return tool == "find_order"
        if f in ("stale_amounts", "unit_shift"):
            return tool == "list_invoices" and bool(res.get("invoices"))
        if f == "truncated_page":
            return tool == "list_invoices" and len(res.get("invoices", [])) >= 2
        if f == "empty_result":
            return tool == "list_invoices" and bool(res.get("invoices"))
        if f == "tool_error":
            return tool in ("find_order", "list_invoices", "get_vendor")
        if f == "contradictory_echo":
            return tool == "list_invoices" and bool(res.get("invoices"))
        return False

    def _signature_of(self, phase: str, payload: dict[str, Any]) -> tuple[Any, ...]:
        if phase == "decision":
            return ("d", payload.get("tool"), repr(sorted(payload.get("args", {}).items())))
        return ("o", payload.get("tool"), repr(sorted(payload.get("args", {}).items())))

    # -- corruption ---------------------------------------------------------
    def _corrupt(self, phase: str, payload: dict[str, Any]) -> dict[str, Any]:
        f = self.fault_name
        p = dict(payload)

        if phase == "decision":
            args = dict(p.get("args", {}))
            if f == "hallucinated_arg":
                alt = self.world.confusable_with(args["vendor_id"])
                if alt:
                    args["vendor_id"] = alt
            elif f == "arg_typo":
                for k in ("order_id", "vendor_id"):
                    if k in args and isinstance(args[k], str) and args[k]:
                        s = args[k]
                        # Transpose the last two characters: a typo, not a wrecking ball.
                        args[k] = s[:-2] + s[-1] + s[-2] if len(s) >= 2 else s + "X"
                        break
            elif f == "dropped_constraint":
                args["quarter"] = None
            p["args"] = args
            return p

        res = dict(p.get("result", {}))
        if f == "wrong_entity":
            alt = self.world.confusable_with(res.get("vendor_id", ""))
            if alt:
                res["vendor_id"] = alt
        elif f == "stale_amounts":
            # Plausible drift: last quarter's numbers, off by 5-25%.
            #
            # The per-invoice factor is derived from a *stable* key rather than
            # a shared stateful RNG. A broken tool returns the same wrong thing
            # every time you call it, and replay depends on that: with a
            # stateful RNG the same call corrupted twice produced different
            # amounts, so replaying changed the final answer for reasons that
            # had nothing to do with the intervention. The label-free signal
            # then fired on whichever span happened to be probed first, and
            # this fault alone accounted for every miss it recorded.
            inv = [dict(i) for i in res.get("invoices", [])]
            for i in inv:
                r = random.Random(f"{f}|{self.seed}|{i.get('invoice_id')}")
                factor = 1 + r.uniform(-0.25, -0.05)
                i["amount_cents"] = max(1, int(i["amount_cents"] * factor))
            res["invoices"] = inv
        elif f == "unit_shift":
            inv = [dict(i) for i in res.get("invoices", [])]
            for i in inv:
                i["amount_cents"] = i["amount_cents"] // 100  # cents reported as dollars
            res["invoices"] = inv
        elif f == "truncated_page":
            inv = list(res.get("invoices", []))
            res["invoices"] = inv[: max(1, len(inv) // 2)]
            res["has_more"] = False  # the lie that makes it silent
        elif f == "empty_result":
            res["invoices"] = []
            res["has_more"] = False
        elif f == "contradictory_echo":
            # Serve another vendor's invoices AND echo that vendor's id, so
            # the response visibly disagrees with the request that produced
            # it. The data is as wrong as `stale_amounts`; the difference is
            # that the trace now contains the evidence.
            asked = str(p.get("args", {}).get("vendor_id", ""))
            alt = self.world.confusable_with(asked) if asked else None
            if alt:
                rows = self.world.invoices_for(alt, p.get("args", {}).get("quarter"))
                res["vendor_id"] = alt
                res["invoices"] = [
                    {"invoice_id": i.invoice_id, "quarter": i.quarter,
                     "amount_cents": i.amount_cents, "status": i.status}
                    for i in rows[:3]
                ]
                res["has_more"] = False
        elif f == "tool_error":
            p["ok"] = False
            p["error"] = "upstream 503: service temporarily unavailable"
            p["result"] = {}
            return p
        p["result"] = res
        return p

    def clone_armed(self) -> "Injector":
        """
        A copy that reproduces this injector's corruption on replay.

        Counterfactual replay has to re-run the agent in the *same broken
        environment*, not a healthy one. If the environment were healthy on
        replay, repairing any span at all would let the run sail past the
        original fault, every span would look like a fix, and the method would
        always blame the first span. Carrying the armed signature forward is
        what keeps a persistent fault persistent -- a stale cache is still
        stale the second time you call it.
        """
        twin = make_injector(self.fault_name, self.world, self.target_step, self.seed)
        twin.fired = self.fired
        twin.signature = self.signature
        return twin

    # -- injector protocol --------------------------------------------------
    def __call__(self, phase: str, step: int, payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        sig = self._signature_of(phase, payload)
        if self.fired:
            # Keep corrupting repeats of the same call: broken stays broken.
            if sig == self.signature:
                return self._corrupt(phase, payload), True
            return payload, False
        if self._eligible(phase, step, payload):
            self.fired = True
            self.signature = sig
            return self._corrupt(phase, payload), True
        return payload, False


def make_injector(fault_name: str, world: World, target_step: int = 0, seed: int = 0) -> Injector:
    if fault_name not in FAULT_CATALOG:
        raise ValueError(f"unknown fault: {fault_name!r}")
    return Injector(fault_name=fault_name, world=world, target_step=target_step, seed=seed)
