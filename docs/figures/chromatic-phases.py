"""Where the chromatic-number runtime goes, and how it scales.

Regenerate:  python docs/figures/chromatic-phases.py
             (run from docs/figures/)
Data:        ../../bench/results/chromatic.json and chromatic_pre_montgomery.json,
             produced by bench/chromatic/bench.py --out bench/results/chromatic.json
"""

import json
import os

import numpy as np
from orx_figstyle import (BASELINE, MUTED, PALETTE, TEXT, figure_grid,
                          panel_labels, save, use_style)

HERE = os.path.dirname(os.path.abspath(__file__))
AFTER = os.path.join(HERE, "..", "..", "bench", "results", "chromatic.json")
BEFORE = os.path.join(HERE, "..", "..", "bench", "results", "chromatic_pre_montgomery.json")

# Grey for the component that is never the story; blue for the butterfly, which
# is the component this whole project exists to make fast; orange for the one
# that actually dominates.
PHASES = [
    ("indicator", "indicator_fraction", MUTED),
    ("subset-zeta (the butterfly)", "zeta_fraction", PALETTE["blue"]),
    ("k-search", "k_search_fraction", PALETTE["orange"]),
]


def main():
    use_style()
    with open(AFTER) as fh:
        after = json.load(fh)
    with open(BEFORE) as fh:
        before = json.load(fh)

    rows = after["sizes"]
    n = np.array([r["n"] for r in rows])

    fig, (ax_a, ax_b) = figure_grid(1, 2, width=TEXT, ratio=0.44)

    # (a) composition -- fractions, so the axis legitimately spans 0..100%
    bottom = np.zeros(len(rows))
    for label, key, colour in PHASES:
        vals = np.array([r[key] for r in rows]) * 100.0
        ax_a.bar(n, vals, bottom=bottom, width=0.85, label=label,
                 color=colour, edgecolor="black", linewidth=0.3)
        bottom += vals
    ax_a.set_xlabel("Vertices $n$")
    ax_a.set_ylabel("Share of end-to-end time (%)")
    ax_a.set_ylim(0, 100)
    ax_a.set_xticks(range(10, 30, 4))
    ax_a.grid(axis="x", visible=False)

    # (b) absolute scaling, before and after the Montgomery change
    bn = np.array([r["n"] for r in before["sizes"]])
    bt = np.array([r["total_s"] for r in before["sizes"]]) * 1e3
    at = np.array([r["total_s"] for r in rows]) * 1e3
    ax_b.semilogy(bn, bt, color=BASELINE, linewidth=1.1, marker="s",
                  markersize=2.2, label="generic 64-bit %")
    ax_b.semilogy(n, at, color=PALETTE["blue"], linewidth=1.1, marker="o",
                  markersize=2.2, label="Montgomery")
    ax_b.set_xlabel("Vertices $n$")
    ax_b.set_ylabel("End-to-end time (ms, log)")
    ax_b.set_xticks(range(10, 30, 4))
    ax_b.legend(loc="upper left", fontsize=6, frameon=False)

    ax_b.annotate(f"{at[-1]:.0f} ms\nat $n$={n[-1]}", xy=(n[-1], at[-1]),
                  xytext=(-4, 2), textcoords="offset points", ha="right",
                  va="bottom", fontsize=6.5, color=PALETTE["blue"])

    panel_labels([ax_a, ax_b])
    # One legend for the whole figure, below the panels: in a 100%-stacked
    # chart every bar is full height, so a legend inside (a) necessarily covers
    # the thin segments it is there to explain.
    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3,
               fontsize=6.5, frameon=False, handlelength=1.4,
               columnspacing=1.4, borderaxespad=0.0)
    save(fig, os.path.join(HERE, "chromatic-phases"))


if __name__ == "__main__":
    main()
