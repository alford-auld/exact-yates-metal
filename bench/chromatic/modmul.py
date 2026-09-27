#!/usr/bin/env python3
"""How much does the modular multiply cost, and what does Montgomery buy?

    .venv/bin/python bench/chromatic/modmul.py

The pointwise k-th power is the only modular multiplication in the algorithm.
With a 64-bit `%` in the square-and-multiply loop it is compute-bound; with
Montgomery reduction it returns to the memory-bandwidth floor.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import yates  # noqa: E402
import apps.chromatic as ch  # noqa: E402
from apps.chromatic.count import power_terms  # noqa: E402
from bench.harness import cooldown, time_sustained, write_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=26)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--duration", type=float, default=1.5)
    ap.add_argument("--cooldown", type=float, default=2.0)
    ap.add_argument("--out", default="bench/results/chromatic_modmul.json")
    args = ap.parse_args()

    n, k = args.n, args.k
    g = ch.random_graph(n, 0.5, 7)
    counts = ch.independent_counts(g)
    mx.eval(counts)
    p = ch.random_primes(1, seed=7)[0]
    ts = dict(duration_s=args.duration, warmup_s=0.4, min_samples=2, max_samples=5)
    moved = (1 << n) * 4 + (1 << n) * 8        # read uint32, write uint64

    print(f"n={n}, k={k}, {moved / 2**20:.0f} MiB moved per pass, p={p}\n")
    rows = []
    cases = [
        ("mod 2^64 (wrapping, no reduction)", dict(modulus=None)),
        ("mod p, generic 64-bit %", dict(modulus=p, force_generic_mod=True)),
        ("mod p, Montgomery", dict(modulus=p)),
    ]
    for label, kw in cases:
        t = time_sustained(lambda kw=kw: power_terms(counts, k, n, signed=True, **kw),
                           **ts)
        gbps = moved / t.sustained_s / 1e9
        rows.append({"label": label, "ms": t.sustained_s * 1e3, "gb_per_s": gbps})
        print(f"  {label:36s} {t.sustained_s * 1e3:7.2f} ms   {gbps:6.1f} GB/s")
        cooldown(args.cooldown)

    a = ch.indicator_gpu(g)
    mx.eval(a)
    t = time_sustained(lambda: yates.zeta_sub(a), **ts)
    plan = yates.describe_plan(n, 1 << n, mx.uint32)
    zeta_gbps = 2 * plan["num_passes"] * (1 << n) * 4 / t.sustained_s / 1e9
    print(f"\n  {'zeta (uint32, %d passes), for scale' % plan['num_passes']:36s} "
          f"{t.sustained_s * 1e3:7.2f} ms   {zeta_gbps:6.1f} GB/s")

    speedup = rows[1]["ms"] / rows[2]["ms"]
    print(f"\n  Montgomery vs generic %: {speedup:.2f}x")
    write_json(args.out, {
        "schema": 1, "command": " ".join(["python", *sys.argv]),
        "n": n, "k": k, "prime": p, "bytes_moved": moved,
        "device": yates.limits().as_dict(), "host": yates.host_info(),
        "cases": rows, "zeta_ms": t.sustained_s * 1e3, "zeta_gb_per_s": zeta_gbps,
        "montgomery_speedup": speedup,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
