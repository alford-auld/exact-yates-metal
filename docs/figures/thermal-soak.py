"""Sustained-load throughput on a passively cooled MacBook Air.

Regenerate:  python docs/figures/thermal-soak.py
             (run from docs/figures/)
Data:        ../../bench/results/thermal.json, produced by
             bench/thermal.py --minutes 7 --out bench/results/thermal.json
"""

import json
import os

from orx_figstyle import BASELINE, PALETTE, TEXT, figure, save, use_style

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "bench", "results", "thermal.json")


def main():
    use_style()
    with open(DATA) as fh:
        d = json.load(fh)

    buckets = d["buckets"]
    t = [b["t_s"] for b in buckets]
    gbps = [b["gb_per_s"] for b in buckets]
    first = gbps[0]

    fig, ax = figure(width=TEXT, ratio=0.42)

    # The first bucket is the reference the slowdown is measured against, so it
    # is a reference line: grey, not a palette colour.
    ax.axhline(first, color=BASELINE, linewidth=0.8, zorder=1)
    ax.annotate("first bucket", xy=(t[-1], first), xytext=(-2, 3),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=6.5, color=BASELINE)

    ax.plot(t, gbps, color=PALETTE["blue"], linewidth=1.2, marker="o",
            markersize=2.4, zorder=3)

    final = gbps[-1]
    ax.annotate(f"{(first / final - 1) * 100:.1f}% slower\nafter {t[-1] / 60:.0f} min",
                xy=(t[-1], final), xytext=(-6, 8), textcoords="offset points",
                ha="right", va="bottom", fontsize=7, color=PALETTE["blue"],
                backgroundcolor="white")

    ax.set_xlabel("Elapsed time under continuous load (s)")
    ax.set_ylabel("Sustained throughput (GB/s)")
    ax.set_xlim(0, t[-1] * 1.02)
    lo, hi = min(gbps), max(gbps)
    pad = (hi - lo) * 0.35
    ax.set_ylim(lo - pad, hi + pad)      # not truncated to zero: the claim is
                                         # the *shape* of the decline, and the
                                         # y-range is stated in the caption
    save(fig, os.path.join(HERE, "thermal-soak"))


if __name__ == "__main__":
    main()
