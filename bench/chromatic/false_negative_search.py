#!/usr/bin/env python3
"""Search for instances where a zero residue is a false negative.

    .venv/bin/python bench/chromatic/false_negative_search.py

This is the evidence behind the claims in apps/chromatic/README.md:

1. A false negative against mod 2^64 *is* constructible inside the memory
   ceiling: c_k is multiplicative over disjoint unions, so the search is a
   knapsack over small components at a fixed k.  It finds 7 x K_4 at k=30 --
   28 vertices, v2(c_30) = 70.
2. A false negative that corrupts the *reported chromatic number* is not,
   because that needs 2^64 | c_chi, and the best ratio v2(c_chi)/n over the
   families searched is 7/8, implying n >= 74.

Both findings are pinned as tests in tests/chromatic/test_modular.py; this
script is how they were obtained and how to re-derive them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import apps.chromatic as ch  # noqa: E402
from bench.harness import write_json  # noqa: E402


def v2(x: int):
    if x == 0:
        return None
    k = 0
    while x % 2 == 0:
        x //= 2
        k += 1
    return k


def component_library(max_n: int, per_size: int, seed: int):
    rng = np.random.default_rng(seed)
    lib = []
    for n in range(2, max_n + 1):
        lib += [ch.complete_graph(n), ch.path(n), ch.empty_graph(n),
                ch.turan(n, 2), ch.turan(n, 3)]
        if n >= 3:
            lib.append(ch.cycle(n))
        for s in range(per_size):
            lib.append(ch.random_graph(n, rng.uniform(0.2, 0.95), 7000 * n + s))
    return [g for g in lib if g.n >= 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--component-max-n", type=int, default=9)
    ap.add_argument("--random-per-size", type=int, default=30)
    ap.add_argument("--max-k", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="bench/results/chromatic_false_negative.json")
    args = ap.parse_args()

    ceiling = ch.max_feasible_n()
    lib = component_library(args.component_max_n, args.random_per_size, args.seed)
    print(f"components: {len(lib)} graphs on 2..{args.component_max_n} vertices")
    print(f"memory ceiling n={ceiling}\n")

    # --- 1. knapsack for the largest v2(c_k) achievable within the ceiling ---
    best = None
    for k in range(2, args.max_k + 1):
        per_size = {}
        for g in lib:
            c = ch.c_k_exact_reference(g, k)
            val = v2(c)
            if val is None:
                continue
            if g.n not in per_size or val > per_size[g.n][0]:
                per_size[g.n] = (val, g)
        if not per_size:
            continue
        dp = [0] * (ceiling + 1)
        pick = [None] * (ceiling + 1)
        for cap in range(1, ceiling + 1):
            for sz, (val, g) in per_size.items():
                if sz <= cap and dp[cap - sz] + val > dp[cap]:
                    dp[cap], pick[cap] = dp[cap - sz] + val, (sz, g)
        if best is None or dp[ceiling] > best[0]:
            best = (dp[ceiling], k, pick, per_size)

    total_v2, k_best, pick, per_size = best
    cap, comps = ceiling, []
    while cap > 0 and pick[cap]:
        sz, g = pick[cap]
        comps.append(g.name)
        cap -= sz
    print(f"best total v2(c_k) within n<={ceiling}: {total_v2}  at k={k_best}")
    print(f"  components: {comps}")
    print(f"  defeats mod 2^64: {total_v2 >= 64}\n")

    # --- 2. can a false negative land at k = chi? ---
    rates = []
    for g in lib:
        chi = ch.chromatic_number(g, mode="exact").chromatic_number
        c = ch.c_k_exact_reference(g, chi)
        val = v2(c)
        if val:
            rates.append((val / g.n, val, g.n, chi, g.name))
    rates.sort(reverse=True)
    top = rates[0]
    needed = int(np.ceil(64 / top[0]))
    print("best v2(c_chi)/n  (what would corrupt the reported chi):")
    for r, val, n, chi, nm in rates[:5]:
        print(f"   rate={r:.3f}  v2={val}  n={n}  chi={chi}  {nm}")
    print(f"\n   need rate >= 64/{ceiling} = {64/ceiling:.3f} to fit within the ceiling")
    print(f"   best observed = {top[0]:.3f} -> would need n >= {needed}")
    print(f"   corrupting instance constructible within the ceiling: {needed <= ceiling}")

    write_json(args.out, {
        "schema": 1, "command": " ".join(["python", *sys.argv]),
        "memory_ceiling_n": ceiling, "components_searched": len(lib),
        "best_v2_within_ceiling": total_v2, "best_v2_k": k_best,
        "best_v2_components": comps,
        "defeats_mod_2_64": total_v2 >= 64,
        "best_v2_chi_rate": top[0], "best_v2_chi_witness": top[4],
        "vertices_needed_to_corrupt_chi": needed,
        "chi_corruption_possible_within_ceiling": needed <= ceiling,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
