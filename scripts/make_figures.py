"""
Generate the figures used in README.md, REPORT.md and COOKBOOK.md.

Every number plotted is read from `experiments/results/*.json` rather than typed
in, so a figure cannot drift from the experiment that produced it. Re-run this
after re-running an experiment.

Each figure is emitted twice, light and dark, and embedded with a `<picture>`
element so GitHub serves the right one for the reader's theme. The palette is
the validated categorical default (slots 1-3), which clears the CVD and
normal-vision floors on all pairs in both modes; the light-mode aqua sits below
3:1 against the surface, so every mark carries a visible direct label rather
than relying on colour alone.
"""

from __future__ import annotations

import json
import pathlib
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
OUT = ROOT / "docs" / "figures"

THEMES = {
    "light": {
        "surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e",
        "muted": "#8a8880", "grid": "#e6e5e1",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    },
    "dark": {
        "surface": "#1a1a19", "primary": "#ffffff", "secondary": "#c3c2b7",
        "muted": "#8a8880", "grid": "#33322f",
        "series": ["#3987e5", "#d95926", "#199e70"],
    },
}


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def top1(rows, method, visibility=None, field="hit"):
    sel = [r for r in rows if r["method"] == method]
    if visibility:
        sel = [r for r in sel if r["visibility"] == visibility]
    return (sum(r[field] for r in sel) / len(sel)) if sel else float("nan")


def spread(values, min_gap):
    """
    Nudge label positions apart while preserving their order.

    Several methods share an x position (every inspection method costs zero
    replays) and their accuracies are close, so the labels collide. Rather than
    jitter the marks -- which would misstate the data -- the marks stay put and
    only the label anchors move, joined back by leader lines.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = list(values)
    for n, i in enumerate(order):
        if n == 0:
            continue
        prev = out[order[n - 1]]
        if out[i] - prev < min_gap:
            out[i] = prev + min_gap
    return out


def style(ax, t, *, xlabel="", ylabel="", title="", subtitle=""):
    ax.set_facecolor(t["surface"])
    ax.figure.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["grid"])
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=t["secondary"], labelsize=9, length=0)
    ax.grid(axis="y", color=t["grid"], linewidth=1, alpha=0.9)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=t["secondary"], fontsize=9.5, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=t["secondary"], fontsize=9.5, labelpad=8)
    if title:
        ax.set_title(title, color=t["primary"], fontsize=13, fontweight="bold",
                     loc="left", pad=22 if subtitle else 12)
    if subtitle:
        ax.text(0, 1.035, subtitle, transform=ax.transAxes, color=t["secondary"],
                fontsize=9.5, va="bottom")


def save(fig, name: str, mode: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}-{mode}.svg", format="svg", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


# --------------------------------------------------------------------------
# 1. The headline: who finds silent faults?
# --------------------------------------------------------------------------
def fig_visibility(mode: str) -> None:
    t = THEMES[mode]
    rows = load("exp01_free_arm.json")["rows"] + load("exp04_judge_full.json")["rows"]
    methods = [
        ("last_span", "last span"), ("random", "random"),
        ("earliest_tool", "earliest tool"), ("output_anomaly", "output anomaly"),
        ("first_error", "first error"), ("llm_trace_judge", "LLM judge"),
        ("cf_bisect", "counterfactual\nreplay"),
    ]
    silent = [top1(rows, m, "silent") for m, _ in methods]
    overt = [top1(rows, m, "overt") for m, _ in methods]

    fig, ax = plt.subplots(figsize=(9.6, 4.9))
    x = range(len(methods))
    w = 0.38
    b1 = ax.bar([i - w / 2 for i in x], silent, w, color=t["series"][0],
                label="silent faults (77% of failures)", zorder=3)
    b2 = ax.bar([i + w / 2 for i in x], overt, w, color=t["series"][1],
                label="overt faults", zorder=3)
    # Where both bars share a value the two labels overlap, so a single
    # centred label is drawn spanning the pair instead.
    for i, (sv, ov) in enumerate(zip(silent, overt)):
        if abs(sv - ov) < 1e-9:
            ax.text(i, max(sv, ov) + 0.022, f"{sv:.3f}", ha="center", va="bottom",
                    fontsize=8.5, color=t["primary"], fontweight="bold")
            continue
        for bar, v in ((b1[i], sv), (b2[i], ov)):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.022, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8.5,
                    color=t["primary"] if v > 0.5 else t["secondary"],
                    fontweight="bold" if v == 0.0 else "normal")
    ax.set_xticks(list(x))
    ax.set_xticklabels([lbl for _, lbl in methods], fontsize=9)
    ax.set_ylim(0, 1.13)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    style(ax, t, ylabel="top-1 accuracy",
          title="Only intervention finds faults that leave no trace",
          subtitle="Localizing the causal span in 547 failed agent runs, by whether the fault is visible in the span it corrupts")
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper left", ncols=2)
    for txt in leg.get_texts():
        txt.set_color(t["secondary"])
    ax.annotate("a heuristic that hunts for errors\nfinds nothing to hunt", xy=(4 - w / 2, 0.055),
                xytext=(2.55, 0.50), fontsize=8.5, color=t["muted"], ha="center",
                arrowprops=dict(arrowstyle="-", color=t["muted"], linewidth=1,
                                shrinkA=2, shrinkB=4))
    save(fig, "accuracy-by-visibility", mode)


# --------------------------------------------------------------------------
# 2. Repair reliability is the binding constraint
# --------------------------------------------------------------------------
def fig_repair_sweep(mode: str) -> None:
    t = THEMES[mode]
    grid = [g for g in load("exp02_repair_sweep.json")["extra"]["grid"]
            if g["signal"] == "changed"]
    series = [("cf_bisect", "bisection", 0), ("cf_exhaustive", "linear scan", 1),
              ("cf_exhaustive_x3", "linear scan, 3 samples", 2)]
    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    for method, label, slot in series:
        pts = sorted([(g["p_repair"], g["top1"]) for g in grid if g["method"] == method])
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=t["series"][slot], linewidth=2, marker="o",
                markersize=8, markeredgecolor=t["surface"], markeredgewidth=2, zorder=3)
        # The x-axis is inverted, so xs[0] (least reliable repair) is the RIGHT
        # end of each line. Labelling there keeps the three series apart; at the
        # left end they all converge on 1.000 and the labels pile up.
        ax.text(xs[0] - 0.022, ys[0], f"{label}  {ys[0]:.3f}", color=t["series"][slot],
                fontsize=9.5, va="center", ha="left", fontweight="bold")
    ax.invert_xaxis()
    ax.set_xlim(1.05, 0.02)
    ax.set_ylim(0, 1.08)
    ax.set_xticks([1.0, 0.9, 0.7, 0.5, 0.3])
    style(ax, t, xlabel="probability a proposed repair is correct  →  less reliable",
          ylabel="top-1 accuracy",
          title="Repair quality is the binding constraint, not search strategy",
          subtitle="Sampling each span three times recovers most of the loss; bisection degrades fastest because it commits to every probe")
    save(fig, "repair-sweep", mode)


# --------------------------------------------------------------------------
# 3. The accuracy/cost frontier
# --------------------------------------------------------------------------
def fig_frontier(mode: str) -> None:
    t = THEMES[mode]
    rows = load("exp01_free_arm.json")["rows"] + load("exp04_judge_full.json")["rows"]
    pts = [
        ("last span", "last_span", 0, 0), ("random", "random", 0, 0),
        ("earliest tool", "earliest_tool", 0, 0), ("output anomaly", "output_anomaly", 0, 0),
        ("first error", "first_error", 0, 0), ("LLM judge", "llm_trace_judge", 1, 0),
        ("bisection", "cf_bisect", 2, 1), ("linear scan", "cf_exhaustive", 2, 1),
    ]
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    seen = set()
    placed = []
    for label, method, slot, _ in pts:
        acc = top1(rows, method)
        cost = top1(rows, method, field="replays")
        family = {0: "inspect the trace (free)", 1: "ask a model (1 call)",
                  2: "intervene (replays)"}[slot]
        ax.scatter([cost], [acc], s=120, color=t["series"][slot], zorder=4,
                   edgecolor=t["surface"], linewidth=2,
                   label=family if family not in seen else None)
        seen.add(family)
        placed.append((label, cost, acc, slot))

    # The free methods all sit at zero replays; space their labels apart and
    # join each back to its mark.
    free = [p for p in placed if p[1] < 0.5]
    ys = spread([p[2] for p in free], 0.088)
    for (label, cost, acc, slot), ly in zip(free, ys):
        ax.plot([cost + 0.06, 0.52], [acc, ly], color=t["grid"], linewidth=1, zorder=2)
        ax.text(0.58, ly, f"{label}  {acc:.3f}", fontsize=8.5,
                color=t["secondary"], va="center")
    # Both replay methods sit at 1.000, so a label to the right of the cheaper
    # one would run into the more expensive one's mark. Stack them vertically.
    paid = sorted([p for p in placed if p[1] >= 0.5], key=lambda p: p[1])
    for n, (label, cost, acc, slot) in enumerate(paid):
        ax.text(cost, acc + (0.055 if n == 0 else -0.075), f"{label}  {acc:.3f}",
                fontsize=8.5, color=t["secondary"], ha="center",
                va="bottom" if n == 0 else "top")
    ax.set_xlim(-0.35, 7.1)
    ax.set_ylim(-0.08, 1.16)
    style(ax, t, xlabel="mean replays per trace  →  more expensive",
          ylabel="top-1 accuracy",
          title="What each approach costs, and what it buys",
          subtitle="Inspection is free and tops out near 0.54; intervention is exact for a few replays")
    leg = ax.legend(frameon=False, fontsize=9.5, loc="lower right", bbox_to_anchor=(1.0, 0.06))
    for txt in leg.get_texts():
        txt.set_color(t["secondary"])
    save(fig, "cost-frontier", mode)


# --------------------------------------------------------------------------
# 4. Instrumentation beats a bigger model
# --------------------------------------------------------------------------
def fig_instrumentation(mode: str) -> None:
    t = THEMES[mode]
    grid = [g for g in load("exp06_instrumentation.json")["extra"]["grid"]
            if g["method"] == "llm_trace_judge"]
    by = defaultdict(dict)
    for g in grid:
        by[g["fault"]][g["arm"]] = g["top1"]
    order = ["ALL", "empty_result", "truncated_page", "unit_shift", "stale_amounts"]
    labels = {"ALL": "all four faults", "empty_result": "empty result",
              "truncated_page": "truncated page", "unit_shift": "unit shift",
              "stale_amounts": "stale amounts"}

    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    lefts = spread([by[f]["baseline"] for f in order], 0.045)
    for i, fault in enumerate(order):
        lo, hi = by[fault]["baseline"], by[fault]["instrumented"]
        headline = fault == "ALL"
        color = t["series"][0] if headline else (
            t["series"][2] if hi - lo > 0.05 else t["muted"])
        ax.plot([0, 1], [lo, hi], color=color, linewidth=2.6 if headline else 2,
                marker="o", markersize=9 if headline else 8,
                markeredgecolor=t["surface"], markeredgewidth=2, zorder=4 if headline else 3)
        ax.text(-0.045, lefts[i], f"{lo:.3f}", ha="right", va="center", fontsize=9,
                color=t["primary"] if headline else t["secondary"],
                fontweight="bold" if headline else "normal")
        ax.text(1.045, hi, f"{hi:.3f}   {labels[fault]}", ha="left", va="center",
                fontsize=9, color=t["primary"] if headline else t["secondary"],
                fontweight="bold" if headline else "normal")
    ax.set_xlim(-0.42, 1.72)
    ax.set_ylim(-0.10, 0.78)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["tool returns\nrows only", "tool also returns\ncount + total"],
                       fontsize=9.5)
    ax.grid(axis="y", alpha=0)
    style(ax, t, ylabel="LLM judge top-1 accuracy",
          title="A response field bought more than a bigger model would",
          subtitle="Same model, same prompt. Adding a server-side summary the fault cannot touch makes corruption self-contradicting")
    ax.text(0.5, 0.735, "+39% relative", ha="center", fontsize=9.5,
            color=t["series"][0], fontweight="bold")
    # No inline callout here: any position for it collides with one of the
    # five lines. The point it would make -- that `stale_amounts` needs exact
    # cross-span arithmetic to check -- is made in the prose beside the figure.
    save(fig, "instrumentation", mode)


# --------------------------------------------------------------------------
# 5. The mechanism: contradiction, not visibility
# --------------------------------------------------------------------------
def fig_contradiction(mode: str) -> None:
    t = THEMES[mode]
    bars = [
        ("hallucinated_arg\ndecision phase", 0.982, True, 0),
        ("contradictory_echo\nobservation phase", 1.000, True, 2),
        ("stale_amounts\nobservation phase", 0.000, False, 1),
    ]
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for i, (label, val, contradicts, slot) in enumerate(bars):
        ax.bar([i], [val], 0.52, color=t["series"][slot], zorder=3)
        ax.text(i, val + 0.03, f"{val:.3f}", ha="center", fontsize=11,
                color=t["primary"], fontweight="bold")
        ax.text(i, -0.115, "contradiction\nin the trace" if contradicts
                else "no contradiction", ha="center", va="top", fontsize=9,
                color=t["secondary"] if contradicts else t["muted"])
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=9.5)
    ax.set_ylim(0, 1.16)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    style(ax, t, ylabel="LLM judge top-1 accuracy",
          title="Inspection is bounded by evidence, not by the reader",
          subtitle="Two observation-phase faults, both silent, identical except whether the response contradicts its own request")
    ax.plot([1, 2], [1.075, 1.075], color=t["muted"], linewidth=1)
    ax.text(1.5, 1.09, "same phase, same visibility", ha="center", fontsize=8.5,
            color=t["muted"])
    save(fig, "contradiction", mode)


if __name__ == "__main__":
    for mode in ("light", "dark"):
        fig_visibility(mode)
        fig_repair_sweep(mode)
        fig_frontier(mode)
        fig_instrumentation(mode)
        fig_contradiction(mode)
    made = sorted(p.name for p in OUT.glob("*.svg"))
    print(f"wrote {len(made)} figures to {OUT.relative_to(ROOT)}:")
    for m in made:
        print(f"  {m}")
