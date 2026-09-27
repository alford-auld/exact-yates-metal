#!/usr/bin/env python3
"""Continuous-load soak: does this passively cooled machine throttle?

bench/bench.py idles between configurations, so it deliberately never lets the
GPU get hot.  This script does the opposite: it runs one configuration back to
back with no cooldown at all for --minutes, and reports throughput per time
bucket, so sustained behaviour under continuous load is visible rather than
assumed.

    .venv/bin/python bench/thermal.py --minutes 6 --out bench/results/thermal.json
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yates  # noqa: E402
from bench.harness import thermal_snapshot, write_json  # noqa: E402

DTYPES = {"uint32": mx.uint32, "uint64": mx.uint64, "float32": mx.float32}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=6.0)
    ap.add_argument("--bucket", type=float, default=15.0, help="seconds per report bucket")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--dtype", default="uint32", choices=sorted(DTYPES))
    ap.add_argument("--elems", type=int, default=1 << 26)
    ap.add_argument("--variant", default="WHT")
    ap.add_argument("--out", default="bench/results/thermal.json")
    args = ap.parse_args()

    dtype = DTYPES[args.dtype]
    lim, host = yates.limits(), yates.host_info()
    total = max(1 << args.n, args.elems)
    rng = np.random.default_rng(0)
    if dtype is mx.float32:
        host_arr = rng.standard_normal(total).astype(np.float32)
    elif dtype is mx.uint32:
        host_arr = rng.integers(0, 1 << 32, size=total, dtype=np.uint32)
    else:
        host_arr = rng.integers(0, (1 << 64) - 1, size=total, dtype=np.uint64)
    a = mx.reshape(mx.array(host_arr), (total >> args.n, 1 << args.n))
    mx.eval(a)
    del host_arr

    itemsize = a.itemsize
    plan = yates.describe_plan(args.n, a.size, dtype)
    traffic = 2 * plan["num_passes"] * a.size * itemsize

    print(f"soak: {args.variant} {args.dtype} n={args.n} "
          f"{a.size*itemsize/2**20:.0f} MiB working set, {plan['num_passes']} passes, "
          f"no cooldown, {args.minutes:.0f} min")
    print(f"  {host.get('Chip','?')} / {lim.device_name} / mlx {lim.mlx_version}")
    print(f"  thermal before: {thermal_snapshot().get('pmset_therm','?')!r}")
    print(f"\n{'t (s)':>7} {'iters':>6} {'median ms':>10} {'GB/s':>8} {'vs first bucket':>16}")

    for _ in range(3):                       # warm up / trigger JIT
        mx.eval(yates.transform(a, args.variant))
    mx.synchronize()

    start = time.perf_counter()
    deadline = start + args.minutes * 60
    buckets, cur, cur_end = [], [], start + args.bucket
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        mx.eval(yates.transform(a, args.variant))
        mx.synchronize()
        cur.append(time.perf_counter() - t0)
        if time.perf_counter() >= cur_end:
            med = statistics.median(cur)
            buckets.append({"t_s": cur_end - start, "iters": len(cur),
                            "median_s": med, "gb_per_s": traffic / med / 1e9})
            cur, cur_end = [], cur_end + args.bucket
    if cur:
        med = statistics.median(cur)
        buckets.append({"t_s": time.perf_counter() - start, "iters": len(cur),
                        "median_s": med, "gb_per_s": traffic / med / 1e9})

    first = buckets[0]["median_s"]
    for b in buckets:
        b["vs_first"] = b["median_s"] / first
        print(f"{b['t_s']:7.0f} {b['iters']:6d} {b['median_s']*1e3:10.2f} "
              f"{b['gb_per_s']:8.1f} {b['vs_first']:15.3f}x")

    worst = max(buckets, key=lambda b: b["vs_first"])
    last = buckets[-1]
    print(f"\n  worst bucket {worst['vs_first']:.3f}x of the first (at t={worst['t_s']:.0f}s)")
    print(f"  final bucket {last['vs_first']:.3f}x of the first")
    after = thermal_snapshot()
    print(f"  thermal after: {after.get('pmset_therm','?')!r}")

    write_json(args.out, {
        "schema": 1,
        "command": " ".join(["python", *sys.argv]),
        "settings": vars(args), "device": lim.as_dict(), "host": host,
        "plan": plan, "traffic_bytes": traffic,
        "buckets": buckets,
        "worst_slowdown": worst["vs_first"], "final_slowdown": last["vs_first"],
        "thermal_after": after,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
