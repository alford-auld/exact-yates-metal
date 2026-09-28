#!/usr/bin/env python3
"""Refute the conjecture v2(c_chi) <= n, and locate the smallest counterexample.

    .venv/bin/python bench/chromatic/v2_conjecture_search.py \
        --out bench/results/chromatic_v2_conjecture.json

apps/chromatic/README.md recorded `v2(c_chi) <= n` as a conjecture, "verified
over the survey and never violated", with K_66 (64 of 66) and seven disjoint
K_4 (21 of 28) as the tight witnesses.  Both are closed-form clique families.

It is false.  A connected 8-vertex graph has c_chi = 1536 = 2^9 * 3, so
v2 = 9 > 8 = n.  This script establishes the exact boundary:

  * exhaustively over ALL labeled graphs on n <= 7 vertices the bound holds
    (attained with equality at n = 7, never exceeded);
  * at n = 8 it fails.

Nothing in the solver relies on the conjecture -- the default mode is CRT and
unconditional -- but it was published as evidence, so it is now published as
refuted, with the witness.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


def _v2(x: int) -> int:
    k = 0
    while x and x % 2 == 0:
        x //= 2
        k += 1
    return k


def analyse(n: int, adj: list[int]):
    """(chi, c_chi) by inclusion-exclusion over exact Python integers."""
    icount = [0] * (1 << n)
    for s in range(1 << n):
        if s == 0:
            icount[s] = 1
            continue
        v = (s & -s).bit_length() - 1
        rest = s & ~(1 << v)
        icount[s] = icount[rest] + icount[rest & ~adj[v]]

    def ck(k: int) -> int:
        tot = 0
        for S in range(1 << n):
            t = icount[S] ** k
            tot += t if ((n - bin(S).count("1")) % 2 == 0) else -t
        return tot

    for k in range(1, n + 1):
        c = ck(k)
        if c > 0:
            return k, c
    raise AssertionError("no k with c_k > 0")


def _adj_from_bits(n: int, pairs, bits: int) -> list[int]:
    adj = [0] * n
    b = bits
    while b:
        i = (b & -b).bit_length() - 1
        u, v = pairs[i]
        adj[u] |= 1 << v
        adj[v] |= 1 << u
        b &= b - 1
    return adj


def exhaustive(n: int):
    """Max of v2(c_chi) - n over every labeled graph on n vertices."""
    pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    worst, witness = -99, None
    for bits in range(1 << len(pairs)):
        adj = _adj_from_bits(n, pairs, bits)
        chi, c = analyse(n, adj)
        e = _v2(c) - n
        if e > worst:
            worst, witness = e, {"bits": bits, "chi": chi, "c_chi": c,
                                 "v2": _v2(c)}
    return worst, witness, 1 << len(pairs)


#: The counterexample, as an explicit edge list so it does not depend on the
#: random-graph generator staying bit-identical.
COUNTEREXAMPLE_EDGES = [(0, 4), (0, 5), (1, 2), (1, 5), (1, 7), (2, 3), (2, 5),
                        (2, 6), (2, 7), (3, 5), (3, 6), (4, 6), (5, 7), (6, 7)]
COUNTEREXAMPLE_N = 8


def counterexample():
    adj = [0] * COUNTEREXAMPLE_N
    for u, v in COUNTEREXAMPLE_EDGES:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    chi, c = analyse(COUNTEREXAMPLE_N, adj)
    return {"n": COUNTEREXAMPLE_N, "edges": COUNTEREXAMPLE_EDGES, "chi": chi,
            "c_chi": c, "v2": _v2(c), "exceeds_n_by": _v2(c) - COUNTEREXAMPLE_N}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-exhaustive-n", type=int, default=7,
                    help="n=7 is 2^21 graphs and takes ~100 s")
    args = ap.parse_args()

    ce = counterexample()
    print(f"counterexample: n={ce['n']}, chi={ce['chi']}, "
          f"c_chi={ce['c_chi']} = 2^{ce['v2']} * {ce['c_chi'] >> ce['v2']}, "
          f"v2={ce['v2']} > n={ce['n']}")
    assert ce["v2"] > ce["n"], "the counterexample no longer refutes the bound"

    holds_through, results = 0, {}
    for n in range(3, args.max_exhaustive_n + 1):
        t0 = time.time()
        worst, witness, count = exhaustive(n)
        results[str(n)] = {"graphs": count, "max_v2_minus_n": worst,
                           "witness": witness}
        print(f"n={n}: all {count} labeled graphs in {time.time()-t0:.0f}s, "
              f"max v2(c_chi) - n = {worst:+d}", flush=True)
        if worst <= 0:
            holds_through = n

    out = {"schema": 1,
           "command": "python bench/chromatic/v2_conjecture_search.py",
           "conjecture": "v2(c_chi) <= n",
           "refuted": True,
           "holds_exhaustively_through_n": holds_through,
           "smallest_counterexample_n": ce["n"],
           "counterexample": ce,
           "exhaustive": results}
    print(f"\nconjecture holds exhaustively for n <= {holds_through}, "
          f"fails at n = {ce['n']}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
