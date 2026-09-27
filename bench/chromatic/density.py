#!/usr/bin/env python3
"""Cost of exact mode against graph *density*, at fixed n.

    .venv/bin/python bench/chromatic/density.py --out bench/results/chromatic_density.json

bench/chromatic/bench.py sweeps n at a fixed density of 0.5, which is the wrong
axis for the cost of `mode="exact"`.  The CRT prime count grows as
`k * log2(i(V)) / 30`, and the two factors pull against each other with density:
sparse graphs have many independent sets (large i(V)) but usually small chi,
dense graphs the reverse.  Whether the worst case is sparse, dense or in the
middle is an empirical question, and this is the measurement that answers it.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import yates  # noqa: E402
import apps.chromatic as ch  # noqa: E402
from bench.harness import cooldown, thermal_snapshot, time_sustained, write_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=26)
    ap.add_argument("--densities", default="0.05,0.1,0.15,0.2,0.3,0.5,0.7,0.9")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--duration", type=float, default=1.5)
    ap.add_argument("--warmup", type=float, default=0.4)
    ap.add_argument("--cooldown", type=float, default=2.0)
    ap.add_argument("--out", default="bench/results/chromatic_density.json")
    args = ap.parse_args()

    n = args.n
    ch.check_feasible(n)
    lim, host = yates.limits(), yates.host_info()
    print(f"density sweep at n={n} ({(1 << n) * ch.BYTES_PER_SUBSET / 2**20:.0f} MiB "
          f"of subset state), mlx {lim.mlx_version}, {host.get('Chip','?')}")
    hdr = (f"{'p':>5} {'edges':>6} {'i(V)':>12} {'log2':>5} {'chi':>4} {'bounds':>9} "
           f"{'#k':>3} {'#p@chi':>7} {'exact ms':>10} {'2^64 ms':>9} {'ratio':>6}")
    print(hdr)
    print("-" * len(hdr), flush=True)

    rows = []
    for p in [float(x) for x in args.densities.split(",")]:
        g = ch.random_graph(n, p, args.seed)
        r = ch.chromatic_number(g, mode="exact")
        i_v = r.num_independent_sets
        primes = ch.primes_needed_for_exact(i_v, r.chromatic_number, n=n)
        ts = dict(duration_s=args.duration, warmup_s=args.warmup, min_samples=2)

        t_exact = time_sustained(
            lambda g=g: mx.array(ch.chromatic_number(
                g, mode="exact", time_it=False).chromatic_number), **ts)
        t_fast = time_sustained(
            lambda g=g: mx.array(ch.chromatic_number(
                g, mode="mod2_64", time_it=False).chromatic_number), **ts)

        row = {
            "p": p, "n": n, "num_edges": g.num_edges, "i_of_V": i_v,
            "log2_i_of_V": i_v.bit_length() - 1,
            "chi": r.chromatic_number, "lower": r.lower_bound,
            "upper": r.upper_bound, "k_tested": len(r.tested),
            "primes_at_chi": primes,
            "exact_s": t_exact.sustained_s, "mod2_64_s": t_fast.sustained_s,
            "exact_over_fast": t_exact.sustained_s / t_fast.sustained_s,
        }
        rows.append(row)
        print(f"{p:5.2f} {g.num_edges:6d} {i_v:12d} {row['log2_i_of_V']:5d} "
              f"{r.chromatic_number:4d} {f'[{r.lower_bound},{r.upper_bound}]':>9} "
              f"{len(r.tested):3d} {primes:7d} {t_exact.sustained_s*1e3:10.1f} "
              f"{t_fast.sustained_s*1e3:9.1f} {row['exact_over_fast']:6.2f}x",
              flush=True)
        mx.clear_cache()
        cooldown(args.cooldown, f"after p={p}")

    worst = max(rows, key=lambda r: r["exact_s"])
    best = min(rows, key=lambda r: r["exact_s"])
    print(f"\n  exact mode spans {best['exact_s']*1e3:.0f} ms (p={best['p']}) to "
          f"{worst['exact_s']*1e3:.0f} ms (p={worst['p']}) -- "
          f"{worst['exact_s']/best['exact_s']:.1f}x across density at fixed n")
    write_json(args.out, {
        "schema": 1, "command": " ".join(["python", *sys.argv]),
        "device": lim.as_dict(), "host": host, "settings": vars(args),
        "thermal": thermal_snapshot(), "rows": rows,
        "spread": worst["exact_s"] / best["exact_s"],
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
