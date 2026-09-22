"""
The synthetic business world the benchmark agent operates over.

Everything in this module is deterministic given a seed. That is the whole
point: the benchmark needs a *programmatically checkable* answer for every
task, so that "did the agent fail?" is a fact rather than a judgement call.
If task success were itself an LLM judgement, every number downstream would
inherit that judge's noise and the localization results would be unfalsifiable.

Three design decisions worth recording:

1. **Amounts are integer cents.** Float money would introduce rounding
   disagreements that look exactly like an injected fault, contaminating the
   ground truth. The `unit_mismatch` fault deliberately exploits the
   cents/dollars distinction, so the underlying representation has to be
   unambiguous.

2. **`list_invoices` is paginated.** Silent truncation is one of the most
   realistic production agent failures -- the tool returns a well-formed page
   and the agent never notices there was a second one. Pagination has to exist
   in the honest world for the `truncated_tool_result` fault to be a
   corruption of normal behaviour rather than an obviously broken response.

3. **Vendor names are deliberately confusable** (shared prefixes, Inc./LLC
   variants). `wrong_entity_resolution` needs a plausible wrong answer; if the
   distractor were obviously unrelated, any judge would catch it and the hard
   cases would stop being hard.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

QUARTERS = ("Q1", "Q2", "Q3", "Q4")
REGIONS = ("NA", "EMEA", "APAC")
CATEGORIES = ("logistics", "software", "facilities", "consulting")

# Confusable stems: each stem gets 2-3 vendors with different suffixes, so the
# world always contains a plausible near-miss for any vendor the agent resolves.
_STEMS = (
    "Northwind", "Northgate", "Summit", "Summitline", "Meridian", "Meridien",
    "Cobalt", "Cobalt Bay", "Ironwood", "Iron Harbor", "Vantage", "Vantiq",
)
_SUFFIXES = ("Inc.", "LLC", "Group", "Partners")

PAGE_SIZE = 3


@dataclass(frozen=True)
class Vendor:
    vendor_id: str
    name: str
    region: str
    category: str


@dataclass(frozen=True)
class Order:
    order_id: str
    vendor_id: str
    placed_on: str
    item_count: int


@dataclass(frozen=True)
class Invoice:
    invoice_id: str
    vendor_id: str
    quarter: str
    amount_cents: int
    status: str


@dataclass
class World:
    """An immutable-by-convention snapshot of the synthetic org."""

    vendors: dict[str, Vendor] = field(default_factory=dict)
    orders: dict[str, Order] = field(default_factory=dict)
    invoices: dict[str, Invoice] = field(default_factory=dict)
    policy_caps_cents: dict[str, int] = field(default_factory=dict)

    # --- indexes -------------------------------------------------------
    def invoices_for(self, vendor_id: str, quarter: str | None = None) -> list[Invoice]:
        out = [
            inv
            for inv in self.invoices.values()
            if inv.vendor_id == vendor_id and (quarter is None or inv.quarter == quarter)
        ]
        # Stable ordering so pagination is reproducible across runs.
        return sorted(out, key=lambda i: i.invoice_id)

    def vendors_by_name_prefix(self, prefix: str) -> list[Vendor]:
        p = prefix.lower().strip()
        return sorted(
            (v for v in self.vendors.values() if v.name.lower().startswith(p)),
            key=lambda v: v.vendor_id,
        )

    def confusable_with(self, vendor_id: str) -> str | None:
        """A different vendor sharing this one's name stem -- the plausible wrong answer."""
        target = self.vendors[vendor_id]
        stem = target.name.rsplit(" ", 1)[0]
        for v in self.vendors.values():
            if v.vendor_id != vendor_id and v.name.rsplit(" ", 1)[0] == stem:
                return v.vendor_id
        # Fall back to any vendor in the same category: still plausible.
        for v in self.vendors.values():
            if v.vendor_id != vendor_id and v.category == target.category:
                return v.vendor_id
        return None


def build_world(seed: int = 7, n_vendors: int = 24, n_orders: int = 60) -> World:
    """Construct the deterministic world. Same seed always yields the same org."""
    rng = random.Random(seed)
    w = World()

    names: list[str] = []
    for stem in _STEMS:
        for suffix in rng.sample(_SUFFIXES, k=2):
            names.append(f"{stem} {suffix}")
    rng.shuffle(names)

    for i in range(n_vendors):
        vid = f"V{1000 + i}"
        w.vendors[vid] = Vendor(
            vendor_id=vid,
            name=names[i % len(names)],
            region=rng.choice(REGIONS),
            category=rng.choice(CATEGORIES),
        )

    vendor_ids = list(w.vendors)
    for i in range(n_orders):
        oid = f"ORD-{4400 + i}"
        vid = rng.choice(vendor_ids)
        w.orders[oid] = Order(
            order_id=oid,
            vendor_id=vid,
            placed_on=f"2026-0{rng.randint(1, 9)}-{rng.randint(10, 28)}",
            item_count=rng.randint(1, 40),
        )

    # Every vendor gets 1-7 invoices per quarter, so a good share of vendor/quarter
    # pairs spill past PAGE_SIZE and need real pagination -- which is what gives the
    # `truncated_tool_result` fault somewhere to bite.
    inv_n = 0
    for vid in vendor_ids:
        for q in QUARTERS:
            for _ in range(rng.randint(1, 7)):
                iid = f"INV-{9000 + inv_n}"
                inv_n += 1
                w.invoices[iid] = Invoice(
                    invoice_id=iid,
                    vendor_id=vid,
                    quarter=q,
                    amount_cents=rng.randrange(50_00, 900_00, 25),
                    status=rng.choice(("paid", "paid", "paid", "pending")),
                )

    for cat in CATEGORIES:
        w.policy_caps_cents[cat] = rng.randrange(800_00, 2_000_00, 1000)

    return w
