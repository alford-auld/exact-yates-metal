#!/usr/bin/env python3
"""f(n) = max over graphs on n vertices of v2(c_chi(G)), computed exactly.

    .venv/bin/python bench/chromatic/max_v2_table.py --max-n 9 \
        --out bench/results/chromatic_max_v2.json

The conjecture v2(c_chi) <= n is false (see v2_conjecture_search.py: it holds
exhaustively through n = 7 and fails at n = 8).  That conjecture was the only
route to an *unconditional* statement that a mod-2^64 false negative cannot
corrupt chi within the memory ceiling: forced <= 25 is a theorem, and
v2(c_chi) <= n <= 29 would have capped the total at 29 against the 64 needed.

With it gone, the useful replacement is not another conjecture but the actual
function.  f(n) is exactly computable for small n, and its growth rate
extrapolated to n = 29 is a far better statement than "no bound is known".

Two facts make it tractable:

* v2(c_chi) is isomorphism-invariant, so the search space is *unlabeled*
  graphs: 853 connected on 7 vertices, 11117 on 8, 261080 on 9 -- versus
  2^36 labeled on 9.  `geng` from nauty enumerates them.
* c_k is multiplicative over disjoint unions, so the disconnected case is a
  knapsack over the connected table.  It needs v2(c_k(G)) for every k >= chi(G),
  not only at chi(G), because a component sits at k = max_i chi_i.

Padding with isolated vertices never changes v2 (K_1 contributes (2^k - 1),
odd), so max over |V| = n and max over |V| <= n coincide.

Requires `geng` on PATH (brew install nauty).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import subprocess
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


def parse_graph6(line: str):
    """graph6 -> (n, adjacency bitmasks).  Only n < 63 is needed here."""
    data = [ord(c) - 63 for c in line.strip()]
    n, rest = data[0], data[1:]
    bits = []
    for byte in rest:
        bits.extend((byte >> i) & 1 for i in range(5, -1, -1))
    adj = [0] * n
    idx = 0
    for j in range(1, n):
        for i in range(j):
            if idx < len(bits) and bits[idx]:
                adj[i] |= 1 << j
                adj[j] |= 1 << i
            idx += 1
    return n, adj


def profile(n: int, adj: list[int], kmax: int):
    """(chi, {k: v2(c_k)}) for k = chi..kmax, via a coefficient histogram.

    c_k = sum_S (-1)^(n-|S|) i(S)^k groups by the value of i(S): collect
    coef[v] = #even-parity S with i(S)=v minus #odd-parity, then
    c_k = sum_v coef[v] * v^k.  There are far fewer distinct v than subsets.
    """
    icount = [0] * (1 << n)
    icount[0] = 1
    for s in range(1, 1 << n):
        v = (s & -s).bit_length() - 1
        rest = s & ~(1 << v)
        icount[s] = icount[rest] + icount[rest & ~adj[v]]

    coef = collections.defaultdict(int)
    for s in range(1 << n):
        coef[icount[s]] += 1 if ((n - bin(s).count("1")) % 2 == 0) else -1
    items = [(v, c) for v, c in coef.items() if c and v]

    out, chi = {}, None
    for k in range(1, kmax + 1):
        c = sum(c * v ** k for v, c in items)
        if chi is None:
            if c > 0:
                chi = k
                out[k] = _v2(c)
        else:
            out[k] = _v2(c)
    return chi, out


def connected_table(max_n: int, geng: str):
    """best[k][(m, attains)] = max v2(c_k(G)) over connected G with |G| = m.

    `attains` is True for the sub-table restricted to chi(G) == k, which is
    what a component sitting at the overall chi must satisfy.
    """
    best = collections.defaultdict(lambda: -1)
    witness = {}
    for m in range(1, max_n + 1):
        t0 = time.time()
        if m == 1:
            graphs = [(1, [0])]
        else:
            proc = subprocess.run([geng, "-c", "-q", str(m)],
                                  capture_output=True, text=True, check=True)
            graphs = [parse_graph6(l) for l in proc.stdout.splitlines() if l.strip()]
        for nn, adj in graphs:
            chi, prof = profile(nn, adj, max_n)
            for k, v in prof.items():
                for attains in ({False, True} if chi == k else {False}):
                    key = (k, m, attains)
                    if v > best[key]:
                        best[key] = v
                        witness[key] = {"n": m, "chi": chi, "k": k, "v2": v}
        print(f"  m={m}: {len(graphs)} connected graphs in {time.time()-t0:.1f}s",
              flush=True)
    return best, witness


def f_of_n(max_n: int, best) -> dict:
    """Knapsack: components of total size <= n, all chi_i <= K, one attaining K."""
    out = {}
    for n in range(1, max_n + 1):
        bestf, arg = -1, None
        for K in range(1, n + 1):
            # unbounded knapsack over components with chi <= K
            dp = [0] * (n + 1)
            for w in range(1, n + 1):
                for m in range(1, w + 1):
                    v = best[(K, m, False)]
                    if v >= 0 and dp[w - m] + v > dp[w]:
                        dp[w] = dp[w - m] + v
            for m in range(1, n + 1):
                v = best[(K, m, True)]
                if v < 0:
                    continue
                total = v + dp[n - m]
                if total > bestf:
                    bestf, arg = total, {"chi": K, "attaining_component_size": m}
        out[str(n)] = {"f": bestf, "argmax": arg}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=9)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    geng = shutil.which("geng") or "/opt/homebrew/bin/geng"
    if not os.path.exists(geng):
        sys.exit("geng not found; brew install nauty")

    print(f"enumerating connected unlabeled graphs through n={args.max_n}")
    best, witness = connected_table(args.max_n, geng)
    table = f_of_n(args.max_n, best)

    print(f"\n f(n) = max over graphs on n vertices of v2(c_chi):")
    print(f"{'n':>3} {'f(n)':>5} {'f(n)-n':>7}  argmax")
    for n in range(1, args.max_n + 1):
        e = table[str(n)]
        print(f"{n:>3} {e['f']:>5} {e['f']-n:>+7}  {e['argmax']}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"schema": 1,
                       "command": "python bench/chromatic/max_v2_table.py",
                       "max_n": args.max_n,
                       "f": table,
                       "connected_witnesses": {f"k{k}_m{m}_att{a}": w
                                               for (k, m, a), w in witness.items()}},
                      fh, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
