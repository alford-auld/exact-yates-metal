"""The Yates kernel is bandwidth-bound, once high-stride passes coalesce.

Regenerate:  python docs/figures/kernel-bandwidth.py
             (run from docs/figures/)
Data:        ../../bench/results/bench.json            (bench/bench.py)
             ../../bench/results/coalescing_tuning.json (bench/tune_coalescing.py)
"""

import json
import os

import numpy as np
from orx_figstyle import (BASELINE, PALETTE, TEXT, family, figure_grid,
                          panel_labels, save, use_style)

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "..", "..", "bench", "results", "bench.json")
COAL = os.path.join(HERE, "..", "..", "bench", "results", "coalescing_tuning.json")

DTYPES = [("uint32", PALETTE["blue"]), ("uint64", PALETTE["orange"]),
          ("float32", PALETTE["green"])]


def main():
    use_style()
    with open(BENCH) as fh:
        bench = json.load(fh)
    with open(COAL) as fh:
        coal = json.load(fh)

    fig, (ax_a, ax_b) = figure_grid(1, 2, width=TEXT, ratio=0.44)

    # (a) achieved fraction of a device-to-device copy measured on the same
    #     machine at the same working-set size
    ax_a.axhline(100, color=BASELINE, linewidth=0.8, zorder=1)
    ax_a.annotate("copy bandwidth", xy=(10.2, 100), xytext=(0, 2),
                  textcoords="offset points", ha="left", va="bottom",
                  fontsize=6, color=BASELINE)
    for name, colour in DTYPES:
        rows = bench["dtypes"][name]["sizes"]
        ax_a.plot([r["n"] for r in rows],
                  [r["fraction_of_copy"] * 100 for r in rows],
                  color=colour, linewidth=1.1, marker="o", markersize=2.2,
                  label=name, zorder=3)
    ax_a.set_xlabel("Transform size $n$  ($N = 2^n$)")
    ax_a.set_ylabel("Achieved / copy bandwidth (%)")
    ax_a.set_ylim(0, 115)
    ax_a.set_xticks(range(10, 30, 4))
    ax_a.legend(loc="lower left", fontsize=6.5, frameon=False, ncol=3,
                columnspacing=1.0, handlelength=1.4)

    # (b) what that took: the pre-fix greedy policy vs the shipped one.
    #     Policy is the primary comparison so it goes on the axis, where there
    #     is room for a readable name; n is the secondary series.
    POLICIES = [("old greedy 3p/128B", "minimise passes\n(128 B runs)"),
                ("shipped policy", "run length\nscaled to stride")]
    sizes = [e for e in coal["sizes"] if e["n"] in (28, 29)]
    shades = family("blue", len(sizes))
    x = np.arange(len(POLICIES))
    width = 0.34
    for i, e in enumerate(sizes):
        vals, passes = [], []
        for key, _ in POLICIES:
            c = next(c for c in e["candidates"] if c["label"] == key)
            vals.append(c["fraction_of_copy"] * 100)
            passes.append(c["passes"])
        bars = ax_b.bar(x + (i - (len(sizes) - 1) / 2) * width, vals, width,
                        label=f"$n$={e['n']} ({e['bytes'] / 2**30:.0f} GiB)",
                        color=shades[i], edgecolor="black", linewidth=0.4)
        ax_b.bar_label(bars,
                       labels=[f"{v:.0f}%" for v in vals],
                       fontsize=5.5, padding=1.5)

    ax_b.set_xticks(x, [label for _, label in POLICIES])
    ax_b.set_ylabel("Achieved / copy bandwidth (%)")
    ax_b.set_ylim(0, 148)
    ax_b.set_yticks(range(0, 121, 20))
    ax_b.tick_params(axis="x", length=0)
    ax_b.grid(axis="x", visible=False)
    ax_b.legend(loc="upper center", fontsize=6, frameon=False, ncol=2,
                handlelength=1.1, columnspacing=1.0)

    panel_labels([ax_a, ax_b])
    save(fig, os.path.join(HERE, "kernel-bandwidth"))


if __name__ == "__main__":
    main()
