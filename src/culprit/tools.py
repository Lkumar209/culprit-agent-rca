"""
The agent's toolset, and the tasks it is asked to solve.

Tools are pure functions of the world plus their arguments. Keeping them pure
matters for the replay engine: re-executing a tool call during counterfactual
replay has to be safe and has to return what it would have returned the first
time, otherwise "the outcome changed when I re-ran step k" would be
confounded by side effects rather than by the fault.

Tool results are plain JSON-able dicts because that is what actually crosses
the wire in a real agent, and because the fault injector corrupts results at
exactly that boundary -- a faulted result has to be indistinguishable, in
shape, from an honest one.

`list_invoices` returns a `has_more` flag. The honest implementation sets it
truthfully; the truncation fault sets it to False while dropping the tail.
That one flag is the difference between a bug the agent could in principle
notice and one it cannot.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from .world import PAGE_SIZE, QUARTERS, World

ToolFn = Callable[..., dict[str, Any]]


class ToolError(Exception):
    """Raised for malformed arguments or missing entities -- an *overt* failure."""


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------

def find_order(world: World, order_id: str) -> dict[str, Any]:
    o = world.orders.get(str(order_id).strip())
    if o is None:
        raise ToolError(f"no such order: {order_id!r}")
    return {
        "order_id": o.order_id,
        "vendor_id": o.vendor_id,
        "placed_on": o.placed_on,
        "item_count": o.item_count,
    }


def get_vendor(world: World, vendor_id: str) -> dict[str, Any]:
    v = world.vendors.get(str(vendor_id).strip())
    if v is None:
        raise ToolError(f"no such vendor: {vendor_id!r}")
    return {
        "vendor_id": v.vendor_id,
        "name": v.name,
        "region": v.region,
        "category": v.category,
    }


def list_invoices(
    world: World, vendor_id: str, quarter: str | None = None, page: int = 1
) -> dict[str, Any]:
    if str(vendor_id).strip() not in world.vendors:
        raise ToolError(f"no such vendor: {vendor_id!r}")
    if quarter is not None and quarter not in QUARTERS:
        raise ToolError(f"bad quarter: {quarter!r}")
    page = int(page)
    if page < 1:
        raise ToolError(f"page must be >= 1, got {page}")

    rows = world.invoices_for(str(vendor_id).strip(), quarter)
    start = (page - 1) * PAGE_SIZE
    chunk = rows[start : start + PAGE_SIZE]
    return {
        "vendor_id": vendor_id,
        "quarter": quarter,
        "page": page,
        "invoices": [
            {
                "invoice_id": i.invoice_id,
                "quarter": i.quarter,
                "amount_cents": i.amount_cents,
                "status": i.status,
            }
            for i in chunk
        ],
        "has_more": start + PAGE_SIZE < len(rows),
    }


def get_policy_cap(world: World, category: str) -> dict[str, Any]:
    cap = world.policy_caps_cents.get(str(category).strip())
    if cap is None:
        raise ToolError(f"no policy for category: {category!r}")
    return {"category": category, "cap_cents": cap}


def sum_amounts(world: World, amounts: list[int]) -> dict[str, Any]:
    if not isinstance(amounts, list):
        raise ToolError(f"amounts must be a list, got {type(amounts).__name__}")
    try:
        total = sum(int(a) for a in amounts)
    except (TypeError, ValueError) as exc:
        raise ToolError(f"non-numeric amount in {amounts!r}") from exc
    return {"total_cents": total, "n": len(amounts)}


TOOLS: dict[str, ToolFn] = {
    "find_order": find_order,
    "get_vendor": get_vendor,
    "list_invoices": list_invoices,
    "get_policy_cap": get_policy_cap,
    "sum_amounts": sum_amounts,
}

TOOL_SPECS = [
    {
        "name": "find_order",
        "description": "Look up a purchase order by id. Returns the vendor_id that fulfilled it.",
        "args": {"order_id": "string, e.g. ORD-4412"},
    },
    {
        "name": "get_vendor",
        "description": "Look up a vendor by id. Returns name, region and spend category.",
        "args": {"vendor_id": "string, e.g. V1003"},
    },
    {
        "name": "list_invoices",
        "description": (
            "List a vendor's invoices, one page at a time. Filter by quarter when given. "
            "Check `has_more` and request the next page if it is true."
        ),
        "args": {
            "vendor_id": "string",
            "quarter": "one of Q1..Q4, or null for all quarters",
            "page": "integer, 1-based",
        },
    },
    {
        "name": "get_policy_cap",
        "description": "Get the spend cap (in cents) for a spend category.",
        "args": {"category": "one of logistics, software, facilities, consulting"},
    },
    {
        "name": "sum_amounts",
        "description": "Sum a list of integer cent amounts.",
        "args": {"amounts": "list of integers (cents)"},
    },
]


def call_tool(world: World, name: str, args: dict[str, Any]) -> dict[str, Any]:
    fn = TOOLS.get(name)
    if fn is None:
        raise ToolError(f"unknown tool: {name!r}")
    try:
        return fn(world, **args)
    except TypeError as exc:  # wrong/missing kwargs -- an overt arg fault
        raise ToolError(f"bad arguments for {name}: {exc}") from exc


# --------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------

def normalize_answer(s: Any) -> str:
    """
    Canonical form for comparing an agent's answer against gold.

    A model asked for "dollars to two decimal places" may answer `$1,669.00`.
    That is the right answer in a different surface form, and scoring it as a
    failure would be a grading artifact with real consequences here: the trace
    would enter the corpus labelled "failed" while containing no injected fault
    to localize, so every localizer would be scored against a culprit that does
    not exist. Formatting compliance is not what this benchmark measures.
    """
    if s is None:
        return ""
    t = str(s).strip().replace("$", "").replace(",", "").strip().rstrip(".")
    if re.fullmatch(r"-?\d+(\.\d+)?", t):
        return f"{float(t):.2f}" if "." in t else str(int(t))
    return t.upper()


@dataclass(frozen=True)
class Task:
    task_id: str
    kind: str
    question: str
    gold: str
    # The reference plan: the tool calls a correct agent makes, in order.
    # Used by the scripted policy and by the honest-run sanity check.
    plan: tuple[dict[str, Any], ...]

    def to_json(self) -> str:
        return json.dumps(
            {"task_id": self.task_id, "kind": self.kind, "question": self.question},
            sort_keys=True,
        )


def _pages_for(world: World, vendor_id: str, quarter: str | None) -> list[dict[str, Any]]:
    """The sequence of list_invoices calls a diligent agent makes (follows has_more)."""
    calls, page = [], 1
    while True:
        calls.append(
            {"tool": "list_invoices", "args": {"vendor_id": vendor_id, "quarter": quarter, "page": page}}
        )
        if not list_invoices(world, vendor_id, quarter, page)["has_more"]:
            break
        page += 1
    return calls


def build_tasks(world: World, seed: int = 11, n_per_kind: int = 10) -> list[Task]:
    """Generate checkable multi-hop tasks with their reference plans."""
    import random

    rng = random.Random(seed)
    order_ids = sorted(world.orders)
    tasks: list[Task] = []

    def _mk(kind: str, idx: int, question: str, gold: str, plan: list[dict[str, Any]]) -> Task:
        return Task(
            task_id=f"{kind}-{idx:03d}", kind=kind, question=question, gold=gold, plan=tuple(plan)
        )

    # 1. quarter_spend -- find_order -> list_invoices(+pages) -> sum
    for i in range(n_per_kind):
        oid = rng.choice(order_ids)
        q = rng.choice(QUARTERS)
        vid = world.orders[oid].vendor_id
        rows = world.invoices_for(vid, q)
        gold_cents = sum(r.amount_cents for r in rows)
        plan = [{"tool": "find_order", "args": {"order_id": oid}}]
        plan += _pages_for(world, vid, q)
        plan += [{"tool": "sum_amounts", "args": {"amounts": [r.amount_cents for r in rows]}}]
        tasks.append(
            _mk(
                "quarter_spend", i,
                f"What was the total amount invoiced in {q} by the vendor that fulfilled "
                f"order {oid}? Answer in dollars, to two decimal places.",
                f"{gold_cents / 100:.2f}",
                plan,
            )
        )

    # 2. cap_check -- adds get_vendor + get_policy_cap, 5-6 hops
    for i in range(n_per_kind):
        oid = rng.choice(order_ids)
        q = rng.choice(QUARTERS)
        vid = world.orders[oid].vendor_id
        cat = world.vendors[vid].category
        rows = world.invoices_for(vid, q)
        total = sum(r.amount_cents for r in rows)
        cap = world.policy_caps_cents[cat]
        plan = [
            {"tool": "find_order", "args": {"order_id": oid}},
            {"tool": "get_vendor", "args": {"vendor_id": vid}},
            {"tool": "get_policy_cap", "args": {"category": cat}},
        ]
        plan += _pages_for(world, vid, q)
        plan += [{"tool": "sum_amounts", "args": {"amounts": [r.amount_cents for r in rows]}}]
        tasks.append(
            _mk(
                "cap_check", i,
                f"Did the vendor that fulfilled order {oid} exceed the policy spend cap "
                f"for its category during {q}? Answer YES or NO.",
                "YES" if total > cap else "NO",
                plan,
            )
        )

    # 3. invoice_count -- shortest chain, a control for trace length
    for i in range(n_per_kind):
        oid = rng.choice(order_ids)
        q = rng.choice(QUARTERS)
        vid = world.orders[oid].vendor_id
        rows = world.invoices_for(vid, q)
        plan = [{"tool": "find_order", "args": {"order_id": oid}}]
        plan += _pages_for(world, vid, q)
        tasks.append(
            _mk(
                "invoice_count", i,
                f"How many invoices did the vendor that fulfilled order {oid} submit in {q}? "
                f"Answer with a single integer.",
                str(len(rows)),
                plan,
            )
        )

    # 4. top_quarter -- four list sweeps, the longest traces in the corpus
    for i in range(n_per_kind):
        oid = rng.choice(order_ids)
        vid = world.orders[oid].vendor_id
        totals = {q: sum(r.amount_cents for r in world.invoices_for(vid, q)) for q in QUARTERS}
        best = max(QUARTERS, key=lambda q: (totals[q], q))
        plan = [{"tool": "find_order", "args": {"order_id": oid}}]
        for q in QUARTERS:
            plan += _pages_for(world, vid, q)
        tasks.append(
            _mk(
                "top_quarter", i,
                f"In which quarter did the vendor that fulfilled order {oid} invoice the most? "
                f"Answer with one of Q1, Q2, Q3, Q4.",
                best,
                plan,
            )
        )

    return tasks
