#!/usr/bin/env python3
"""Why the n=29 k-search costs more than its own 2x scaling predicts.

    .venv/bin/python bench/chromatic/memory_pressure.py \
        --out bench/results/chromatic_memory_pressure.json

Normalised by k and by CRT prime count, the k-search doubles per n as it
should -- except at n = 29, which is ~3x n = 28 with k and primes identical in
both rows.  Two candidate explanations, and the protocol used for the kernel
benchmark prescribes the discriminating measurements:

1. *The memory system is slower at large working sets.*  Ruled out by
   re-measuring the copy kernel: bandwidth is flat from 512 MiB to 3072 MiB.
   (6144 MiB cannot be measured at all -- an out-of-place copy needs 12 GiB
   against an 11.84 GiB recommended working set -- which is itself the clue.)

2. *Allocation pressure near the budget.*  Isolated by ballast: run the SAME
   n = 28 instance, same k, same primes, same plan, with a live dummy array
   sized to push the peak to n = 29's footprint.  Only the memory pressure
   changes.

3. *More work.*  The published table's `primes` column is
   `primes_per_k_at_chi` -- the CRT prime count at chi ONLY.  The search also
   evaluates c_k at other k, with their own prime counts, so that column is not
   the amount of work done and normalising by it is wrong.  Counting the actual
   pointwise-power evaluations: n = 28 does 5 (k=6 with 3 primes, k=5 with 2),
   n = 29 does 6 (k=6 with 3, k=7 with 3).

Result: the excess is mostly ordinary.  One extra power unit (6 vs 5) and the
2x data together predict 2.42x; memory pressure above a 9 GiB peak adds 1.16x;
the product is 2.81x against a measured 2.99x, leaving ~1.06x unexplained
rather than the 1.20x reported before this was measured.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import mlx.core as mx

import apps.chromatic as ch
import apps.chromatic.chromatic as CH
import yates

PRIME = 2147483647          # a 31-bit prime, as the CRT path draws


def power_units(n: int):
    """How many pointwise-power evaluations one exact solve actually runs."""
    g = ch.random_graph(n, 0.5, 7)
    calls = []
    orig = CH.c_k_multi_modular

    def spy(counts, k, nn, i_v, *a, **kw):
        r = orig(counts, k, nn, i_v, *a, **kw)
        calls.append({"k": k, "primes": len(r.primes)})
        return r

    CH.c_k_multi_modular = spy
    try:
        ch.chromatic_number(g, mode="exact")
    finally:
        CH.c_k_multi_modular = orig
    mx.clear_cache()
    return calls, sum(c["primes"] for c in calls)


def unit_cost(n: int, k: int, reps: int = 5):
    """One pointwise power + reduction mod a 31-bit prime."""
    g = ch.random_graph(n, 0.5, 7)
    counts = ch.independent_counts(g)
    mx.eval(counts)
    ch.c_k_reduction(counts, k, n, modulus=PRIME)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        ch.c_k_reduction(counts, k, n, modulus=PRIME)
        ts.append(time.perf_counter() - t0)
    del counts
    mx.clear_cache()
    return statistics.median(ts)


def copy_bandwidth(mib: int, reps: int = 7):
    """Device-to-device copy at a given footprint, or None if it will not fit."""
    x = mx.zeros(((mib << 20) // 4,), dtype=mx.uint32)
    mx.eval(x)
    try:
        mx.eval(yates.device_copy(x))
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            mx.eval(yates.device_copy(x))
            ts.append(time.perf_counter() - t0)
        # median: pressure shows up as intermittent stalls, not a uniform
        # slowdown, so a mean would report the outliers as the trend
        return {"median_gbps": 2 * x.nbytes / statistics.median(ts) / 1e9,
                "min_gbps": 2 * x.nbytes / max(ts) / 1e9,
                "max_gbps": 2 * x.nbytes / min(ts) / 1e9,
                "live_gib": 2 * x.nbytes / 2 ** 30}
    except RuntimeError:
        return None            # out-of-place copy needs 2x this footprint
    finally:
        del x
        mx.clear_cache()


def solve(n: int, ballast_gib: float, reps: int = 7):
    # returns None if the instance plus ballast will not fit
    """Median end-to-end exact chi, with `ballast_gib` held live throughout."""
    mx.clear_cache()
    gc.collect()
    keep = None
    if ballast_gib:
        keep = mx.zeros((int(ballast_gib * (1 << 30)) // 4,), dtype=mx.uint32)
        mx.eval(keep)
    g = ch.random_graph(n, 0.5, 7)
    try:
        r = ch.chromatic_number(g, mode="exact")     # warm
    except RuntimeError:
        del keep
        mx.clear_cache()
        return None
    mx.reset_peak_memory()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = ch.chromatic_number(g, mode="exact")
        ts.append(time.perf_counter() - t0)
    peak = mx.get_peak_memory() / 2 ** 30
    del keep
    mx.clear_cache()
    gc.collect()
    return (statistics.median(ts), min(ts), max(ts), r.chromatic_number, peak)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bw = {}
    print("copy bandwidth by footprint (the kernel sweep stops at 2048 MiB):")
    for mib in (512, 1024, 2048, 3072, 4096, 5120, 6144):
        g = copy_bandwidth(mib)
        bw[str(mib)] = g
        print(f"  {mib:5d} MiB ({2*mib/1024:4.1f} GiB live): "
              + (f"{g['median_gbps']:5.1f} GB/s median, "
                 f"[{g['min_gbps']:.1f}-{g['max_gbps']:.1f}]" if g else
                 "does not fit (out-of-place copy needs 2x)"), flush=True)

    print("\npointwise-power units actually run (the `primes` column is at chi only):")
    units = {}
    for n in (28, 29):
        calls, total = power_units(n)
        costs = {c["k"]: unit_cost(n, c["k"]) for c in calls}
        units[str(n)] = {"calls": calls, "total_units": total,
                         "unit_cost_s": costs,
                         "predicted_k_search_s": sum(c["primes"] * costs[c["k"]]
                                                     for c in calls)}
        print(f"  n={n}: {calls} -> {total} units, "
              f"predicted k-search "
              f"{units[str(n)]['predicted_k_search_s']*1e3:.0f} ms", flush=True)

    print("\nballast ablation -- same n, k, primes and plan; only pressure changes:")
    runs = {}
    for n, ballast in [(28, 0), (28, 3), (28, 4), (28, 5),
                       (28, 6), (28, 7), (29, 0)]:
        r = solve(n, ballast)
        if r is None:
            print(f"  n={n} ballast={ballast} GiB: OOM", flush=True)
            continue
        s, lo, hi, chi, peak = r
        runs[f"n{n}_ballast{ballast}"] = {"n": n, "ballast_gib": ballast,
                                          "seconds": s, "min_s": lo,
                                          "max_s": hi, "chi": chi,
                                          "peak_gib": peak}
        print(f"  n={n} ballast={ballast} GiB: {s*1e3:7.1f} ms  "
              f"[{lo*1e3:.1f}-{hi*1e3:.1f}]  peak {peak:.2f} GiB  chi={chi}",
              flush=True)

    u28, u29 = units["28"], units["29"]
    work = u29["predicted_k_search_s"] / u28["predicted_k_search_s"]
    base = runs["n28_ballast0"]["seconds"]
    pressed = runs["n28_ballast5"]["seconds"]   # 10 GiB peak == n=29's own
    actual29 = runs["n29_ballast0"]["seconds"]
    pressure = pressed / base
    excess = actual29 / (2 * base)
    residual = excess / pressure

    accounted = work * pressure
    k_search_ratio = 2.988          # published n=29 / n=28 k-search
    print(f"\n  published k-search ratio n=28 -> n=29  : {k_search_ratio:.3f}x")
    print(f"  predicted from units (2x data, 6 vs 5) : {work:.3f}x")
    print(f"  memory pressure above a 9 GiB peak     : {pressure:.3f}x")
    print(f"  accounted                              : {accounted:.3f}x")
    print(f"  unexplained                            : "
          f"{k_search_ratio/accounted:.3f}x")

    out = {"schema": 1,
           "command": "python bench/chromatic/memory_pressure.py",
           "copy_gbps_by_mib": bw,
           "runs": runs,
           "power_units": units,
           "excess_over_2x_scaling": excess,
           "work_ratio_from_units": work,
           "memory_pressure_factor": pressure,
           "published_k_search_ratio": k_search_ratio,
           "accounted": accounted,
           "unexplained": k_search_ratio / accounted}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
