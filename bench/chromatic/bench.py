#!/usr/bin/env python3
"""Chromatic-number benchmark: wall clock by n, split by phase.

Every number in the chromatic section of README.md comes from this script:

    .venv/bin/python bench/chromatic/bench.py --out bench/results/chromatic.json

The split that matters is indicator / zeta / k-search.  The butterfly -- the
thing the whole Yates kernel exists for -- is expected to be a *small* fraction
of the total, and reporting that honestly is the point of this benchmark: the
algorithm runs exactly one subset-zeta over the cube, and then does a pointwise
power plus a reduction for each candidate k.

Uses the same steady-state protocol as bench/bench.py: a measurement window per
configuration, the median of its second half, and an idle cooldown between
configurations because this machine is passively cooled.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import yates  # noqa: E402
import apps.chromatic as ch  # noqa: E402
from apps.chromatic.count import power_terms  # noqa: E402
from bench.harness import (cooldown, thermal_snapshot, time_sustained,  # noqa: E402
                           write_json)


def make_graph(n: int, family: str, seed: int) -> ch.Graph:
    if family == "random":
        return ch.random_graph(n, 0.5, seed)
    if family == "dense":
        return ch.random_graph(n, 0.8, seed)
    if family == "sparse":
        return ch.random_graph(n, 0.15, seed)
    raise ValueError(f"unknown family {family!r}")


def one_k_test(counts: mx.array, n: int, k: int, modulus):
    """One colourability test: the pointwise power plus the alternating sum.

    Timed per-k rather than as a whole binary search, because on many instances
    the clique bound already equals chi and the search performs zero iterations
    -- which would make a "k search" column read as free rather than as
    not-executed.  Total search cost is this figure times the number of k the
    search actually visits, both of which are reported.
    """
    def run():
        terms = power_terms(counts, k, n, modulus, signed=True)
        return mx.sum(terms)
    return run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="bench/results/chromatic.json")
    ap.add_argument("--min-n", type=int, default=10)
    ap.add_argument("--max-n", type=int, default=None,
                    help="default: this machine's detected memory ceiling")
    ap.add_argument("--family", default="random",
                    choices=["random", "dense", "sparse"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--duration", type=float, default=1.5)
    ap.add_argument("--warmup", type=float, default=0.4)
    ap.add_argument("--cooldown", type=float, default=2.0)
    ap.add_argument("--max-samples", type=int, default=5,
                    help="cap iterations; one pass already takes seconds at large n")
    ap.add_argument("--dp-limit", type=int, default=22,
                    help="largest n at which the NumPy DP indicator is also timed")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.duration, args.warmup, args.cooldown, args.max_samples = 0.2, 0.1, 0.0, 2

    lim, host = yates.limits(), yates.host_info()
    ceiling = ch.max_feasible_n()
    top = args.max_n if args.max_n is not None else ceiling

    print("=" * 100)
    print(f"Chromatic number by inclusion-exclusion | mlx {lim.mlx_version}")
    print(f"  {host.get('Model Identifier','?')} {host.get('Chip','?')} | "
          f"{host.get('Memory','?')} RAM | macOS {host.get('macOS','?')}")
    print(f"  GPU {lim.device_name}, working set {lim.max_recommended_working_set_size/2**30:.2f} GiB")
    print(f"  memory ceiling n={ceiling} ({ch.BYTES_PER_SUBSET} B/subset), "
          f"benchmarking n={args.min_n}..{top}, family={args.family}")
    print("=" * 100, flush=True)

    results = {
        "schema": 1,
        "command": " ".join(["python", *sys.argv]),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mlx_version": lim.mlx_version, "device": lim.as_dict(), "host": host,
        "settings": vars(args), "memory_ceiling_n": ceiling,
        "bytes_per_subset": ch.BYTES_PER_SUBSET,
        "thermal_before": thermal_snapshot(),
        "sizes": [],
    }

    hdr = (f"{'n':>3} {'m':>5} {'chi':>4} {'i(V)':>9} {'mem':>7} | "
           f"{'indic':>8} {'zeta':>9} {'k-search':>10} {'total':>10} | "
           f"{'ind%':>6} {'zeta%':>6} {'ks%':>6} {'#k':>3} {'#p':>3}")
    print(hdr)
    print("-" * len(hdr), flush=True)

    for n in range(args.min_n, top + 1):
        g = make_graph(n, args.family, args.seed)
        try:
            ch.check_feasible(n)
        except MemoryError as exc:
            print(f"  n={n}: {exc}")
            break

        ts = dict(duration_s=args.duration, warmup_s=args.warmup,
                  min_samples=2, max_samples=args.max_samples)

        # Phases are measured as nested prefixes of the real computation and the
        # inner ones subtracted, so the three shares sum to the measured total by
        # construction.  Timing each phase standalone and extrapolating would not:
        # the number of CRT primes depends on k, so a per-k figure measured at
        # k=chi over-counts the smaller k the binary search also visits.
        t_ind = time_sustained(lambda: ch.indicator_gpu(g), **ts)
        t_counts = time_sustained(lambda: ch.independent_counts(g), **ts)
        t_total = time_sustained(
            lambda: mx.array(ch.chromatic_number(g, mode="exact",
                                                 time_it=False).chromatic_number),
            **ts)

        t_ind_bl = time_sustained(
            lambda: ch.indicator_gpu(g, section="branchless"), **ts)

        indicator_s = t_ind.sustained_s
        zeta_s = max(t_counts.sustained_s - indicator_s, 0.0)
        total = t_total.sustained_s
        search_s = max(total - t_counts.sustained_s, 0.0)

        counts = ch.independent_counts(g)
        mx.eval(counts)
        i_v = int(counts[-1].item())
        b = ch.bounds(g)
        lo, hi = int(b["lower"]), int(b["upper"])
        prime = ch.random_primes(1, seed=args.seed)[0]
        r = ch.chromatic_number(g, mode="exact")
        num_k = len(r.tested)
        primes_per_k = ch.primes_needed_for_exact(i_v, r.chromatic_number)
        t_ks = time_sustained(one_k_test(counts, n, r.chromatic_number, prime), **ts)

        dp_s = None
        if n <= args.dp_limit:
            t0 = time.perf_counter()
            ch.indicator_numpy_dp(g)
            dp_s = time.perf_counter() - t0

        row = {
            "n": n, "num_edges": g.num_edges, "chi": r.chromatic_number,
            "i_of_V": i_v, "lower": lo, "upper": hi, "k_tested": num_k,
            "primes_per_k_at_chi": primes_per_k,
            "bytes": (1 << n) * ch.BYTES_PER_SUBSET,
            "indicator_direct_s": indicator_s,
            "indicator_branchless_s": t_ind_bl.sustained_s,
            "indicator_numpy_dp_s": dp_s,
            "indicator_plus_zeta_s": t_counts.sustained_s,
            "zeta_s": zeta_s,
            "k_search_s": search_s,
            "one_k_test_s": t_ks.sustained_s,
            "total_s": total,
            "indicator_fraction": indicator_s / total,
            "zeta_fraction": zeta_s / total,
            "k_search_fraction": search_s / total,
            "drift": t_total.drift_ratio,
        }
        results["sizes"].append(row)
        print(f"{n:>3} {g.num_edges:>5} {r.chromatic_number:>4} {i_v:>9} "
              f"{(1 << n) * ch.BYTES_PER_SUBSET / 2**20:>6.0f}M | "
              f"{indicator_s*1e3:>7.2f}m {zeta_s*1e3:>8.2f}m "
              f"{search_s*1e3:>9.2f}m {total*1e3:>9.2f}m | "
              f"{row['indicator_fraction']*100:>5.1f}% {row['zeta_fraction']*100:>5.1f}% "
              f"{row['k_search_fraction']*100:>5.1f}% {num_k:>3} {primes_per_k:>3}",
              flush=True)

        del counts
        mx.clear_cache()
        cooldown(args.cooldown, f"after n={n}")

    # Instances where the clique bound is useless and the search must work.
    print("\nNamed hard instances (clique bound does not reach chi):", flush=True)
    named = [("M_5 (Mycielskian)", ch.mycielskian(5)),
             ("Kneser(7,2)", ch.kneser(7, 2)),
             ("Kneser(8,2)", ch.kneser(8, 2))]
    results["named"] = []
    for label, g in named:
        if g.n > top:
            print(f"  {label}: n={g.n} above the benchmarked range, skipped")
            continue
        r = ch.chromatic_number(g, mode="exact")
        ts = dict(duration_s=args.duration, warmup_s=args.warmup,
                  min_samples=2, max_samples=args.max_samples)
        t_ind = time_sustained(lambda g=g: ch.indicator_gpu(g), **ts)
        t_counts = time_sustained(lambda g=g: ch.independent_counts(g), **ts)
        t_tot = time_sustained(
            lambda g=g: mx.array(ch.chromatic_number(g, mode="exact",
                                                     time_it=False).chromatic_number),
            **ts)
        counts = ch.independent_counts(g); mx.eval(counts)
        prime = ch.random_primes(1, seed=args.seed)[0]
        t_k = time_sustained(one_k_test(counts, g.n, r.chromatic_number, prime), **ts)
        zeta_s = max(t_counts.sustained_s - t_ind.sustained_s, 0.0)
        search_s = max(t_tot.sustained_s - t_counts.sustained_s, 0.0)
        entry = {
            "label": label, "n": g.n, "chi": r.chromatic_number,
            "lower": r.lower_bound, "upper": r.upper_bound,
            "k_tested": len(r.tested), "i_of_V": r.num_independent_sets,
            "indicator_s": t_ind.sustained_s, "zeta_s": zeta_s,
            "k_search_s": search_s, "one_k_test_s": t_k.sustained_s,
            "primes_per_k_at_chi": ch.primes_needed_for_exact(
                r.num_independent_sets, r.chromatic_number),
            "total_s": t_tot.sustained_s,
            "indicator_fraction": t_ind.sustained_s / t_tot.sustained_s,
            "zeta_fraction": zeta_s / t_tot.sustained_s,
            "k_search_fraction": search_s / t_tot.sustained_s,
        }
        results["named"].append(entry)
        print(f"  {label:20s} n={g.n:2d} chi={r.chromatic_number} "
              f"bounds=[{r.lower_bound},{r.upper_bound}] k_tested={len(r.tested)} | "
              f"indicator {t_ind.sustained_s*1e3:7.2f}ms  zeta {zeta_s*1e3:7.2f}ms  "
              f"k-search {search_s*1e3:8.2f}ms  total {t_tot.sustained_s*1e3:8.2f}ms  "
              f"| zeta {entry['zeta_fraction']*100:.1f}% ks {entry['k_search_fraction']*100:.1f}%",
              flush=True)
        del counts
        mx.clear_cache()
        cooldown(args.cooldown, f"after {label}")

    results["thermal_after"] = thermal_snapshot()
    results["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(args.out, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
