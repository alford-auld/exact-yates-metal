#!/usr/bin/env python3
"""Yates kernel benchmark: sustained bandwidth as a fraction of a device copy.

Every number in README.md comes from this script.  Re-run it with:

    .venv/bin/python bench/bench.py --out bench/results/bench.json

Method
------
* The denominator is a pure device-to-device copy kernel (yates/kernel.metal,
  section "copy") measured on this machine, in this session, with the same
  steady-state protocol -- not a vendor bandwidth figure.  It is re-measured at
  *every* working-set size, because the achievable copy bandwidth on this
  machine varies by a factor of two between a 16 MiB and a 1 GiB footprint;
  a single global denominator would be meaningless.
* Each configuration runs back to back for --duration seconds and the reported
  time is the median of the second half of that window, so it is a sustained
  number.  Peak (fastest single iteration) is recorded alongside only as a
  throttling indicator.  --cooldown seconds of idle separate configurations,
  because this machine is passively cooled.
* Small n are batched up to --min-elems so that every configuration measures
  memory bandwidth rather than kernel launch overhead, and so that the
  working set is constant across n.
* Traffic is 2 * passes * elements * sizeof(elem), with `passes` taken from the
  tiering plan actually dispatched.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yates  # noqa: E402
from bench.harness import (  # noqa: E402
    cooldown, thermal_snapshot, time_sustained, write_json,
)

DTYPES = {"uint32": mx.uint32, "uint64": mx.uint64, "float32": mx.float32}


class InputPool:
    """Random data generated once per dtype and resliced for every n.

    Regenerating multi-GiB host arrays per configuration would dominate the
    wall clock and pollute the thermal state the benchmark is trying to observe.
    """

    def __init__(self, dtype, largest_elems: int, rng):
        self.dtype = dtype
        if dtype is mx.float32:
            host = rng.standard_normal(largest_elems).astype(np.float32)
        elif dtype is mx.uint32:
            host = rng.integers(0, 1 << 32, size=largest_elems, dtype=np.uint32)
        else:
            host = rng.integers(0, (1 << 64) - 1, size=largest_elems, dtype=np.uint64)
        self.base = mx.array(host)
        mx.eval(self.base)
        del host

    def view(self, n: int, min_elems: int) -> mx.array:
        """A standalone, materialised, row-contiguous (batch, 2^n) array.

        The materialisation must happen here rather than inside the timed
        region.  mx.contiguous() on an already-contiguous prefix slice is a real
        device copy whenever the parent buffer has more than 16 KiB of slack --
        intended MLX behaviour (PR #1270), so that a small slice can release a
        large parent allocation, not a bug.  Charging that copy to the kernel
        under test would halve every reported bandwidth.  The threshold is
        pinned by tests/test_platform_contracts.py.
        """
        total = min(max(1 << n, min_elems), self.base.size)
        total = (total >> n) << n              # whole rows only
        shape = (total >> n, 1 << n)
        if total == self.base.size:
            v = mx.reshape(self.base, shape)   # already contiguous, free
        else:
            v = mx.contiguous(mx.reshape(self.base[:total], shape))
        mx.eval(v)
        return v

    def close(self):
        self.base = None
        mx.clear_cache()


def max_n_for(itemsize: int, headroom: float) -> int:
    """Largest n that fits detected unified memory with headroom.

    The transform is out of place, so input and output are live together.
    """
    lim = yates.limits()
    per_array = min(int(lim.max_recommended_working_set_size * headroom) // 2,
                    lim.max_buffer_length)
    n = 0
    while ((1 << (n + 1)) * itemsize) <= per_array:
        n += 1
    return n


def pick_copy_vec(a: mx.array, args) -> tuple:
    """Fastest elements-per-thread for the copy kernel, chosen once per dtype."""
    best, table = None, {}
    for vec in (1, 2, 4, 8):
        if a.size % vec:
            continue
        t = time_sustained(lambda: yates.device_copy(a, vec=vec), args.duration, args.warmup)
        gbps = 2 * a.size * a.itemsize / t.sustained_s / 1e9
        table[vec] = gbps
        print(f"    copy vec={vec}: {gbps:7.1f} GB/s sustained", flush=True)
        if best is None or gbps > best[1]:
            best = (vec, gbps)
    return best[0], table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="bench/results/bench.json")
    ap.add_argument("--duration", type=float, default=2.0,
                    help="steady-state measurement window per configuration (s)")
    ap.add_argument("--warmup", type=float, default=0.5)
    ap.add_argument("--cooldown", type=float, default=3.0,
                    help="idle seconds between configurations; passive cooling")
    ap.add_argument("--min-n", type=int, default=10)
    ap.add_argument("--max-n", type=int, default=None,
                    help="default: detected from unified memory with --headroom")
    ap.add_argument("--headroom", type=float, default=0.5,
                    help="fraction of the recommended working set to use")
    ap.add_argument("--min-elems", type=int, default=1 << 26,
                    help="minimum working set in elements; small n are batched up")
    ap.add_argument("--dtypes", default="uint32,uint64,float32")
    ap.add_argument("--variant", default="WHT")
    ap.add_argument("--quick", action="store_true", help="smoke test only")
    args = ap.parse_args()

    if args.quick:
        args.duration, args.warmup, args.cooldown = 0.4, 0.15, 0.0

    yates.device.require_metal()
    lim, host = yates.limits(), yates.host_info()

    print("=" * 96)
    print(f"Yates kernel benchmark | mlx {lim.mlx_version} | variant {args.variant}")
    print(f"  {host.get('Model Identifier','?')} {host.get('Chip','?')} | "
          f"{host.get('Total Number of Cores','?')} CPU cores | {host.get('Memory','?')} RAM | "
          f"macOS {host.get('macOS','?')}")
    print(f"  GPU {lim.device_name} ({lim.architecture}) simd_width={lim.simd_width} "
          f"max_threads/tg={lim.max_threads_per_threadgroup} tg_mem={lim.max_threadgroup_memory}B")
    print(f"  working set limit {lim.max_recommended_working_set_size/2**30:.2f} GiB, "
          f"max buffer {lim.max_buffer_length/2**30:.2f} GiB")
    print(f"  {args.duration}s window / {args.warmup}s warmup / {args.cooldown}s cooldown, "
          f"min working set {args.min_elems} elems")
    print("=" * 96, flush=True)

    rng = np.random.default_rng(0)
    results = {
        "schema": 3,
        "command": " ".join(["python", *sys.argv]),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mlx_version": lim.mlx_version,
        "device": lim.as_dict(), "host": host, "settings": vars(args),
        "thermal_before": thermal_snapshot(),
        "dtypes": {},
    }

    for dt_name in [d.strip() for d in args.dtypes.split(",")]:
        dtype = DTYPES[dt_name]
        itemsize = mx.zeros((1,), dtype=dtype).itemsize
        top_n = args.max_n if args.max_n is not None else max_n_for(itemsize, args.headroom)
        print(f"\n### {dt_name} ({itemsize} B/elem), n = {args.min_n}..{top_n}", flush=True)

        largest = max(1 << top_n, args.min_elems)
        print(f"  allocating {largest*itemsize/2**30:.2f} GiB of random input", flush=True)
        pool = InputPool(dtype, largest, rng)

        probe = pool.view(args.min_n, args.min_elems)
        print(f"  choosing copy-kernel width @ {probe.size*itemsize/2**20:.0f} MiB",
              flush=True)
        vec, vec_table = pick_copy_vec(probe, args)
        print(f"  using vec={vec}", flush=True)
        del probe
        mx.clear_cache()
        cooldown(args.cooldown, "after copy width probe")

        entry = {"itemsize": itemsize, "max_n": top_n, "copy_vec": vec,
                 "copy_vec_table": vec_table, "sizes": []}
        first_ref = None

        for n in range(args.min_n, top_n + 1):
            try:
                a = pool.view(n, args.min_elems)
            except Exception as exc:
                print(f"  n={n}: allocation failed ({exc}); stopping {dt_name}", flush=True)
                break

            nbytes = a.size * itemsize
            plan = yates.describe_plan(n, a.size, dtype)
            traffic = 2 * plan["num_passes"] * nbytes

            # denominator, measured on this exact working set
            tc = time_sustained(lambda: yates.device_copy(a, vec=vec), args.duration, args.warmup)
            copy_gbps = 2 * nbytes / tc.sustained_s / 1e9

            tt = time_sustained(lambda: yates.transform(a, args.variant),
                                args.duration, args.warmup)
            gbps = traffic / tt.sustained_s / 1e9
            elems_per_s = a.size / tt.sustained_s

            row = {
                "n": n, "batch": a.size >> n, "array_bytes": nbytes,
                "passes": plan["num_passes"], "tile_bytes": plan["tile_bytes"],
                "stages_by_tier": plan["stages_by_tier"],
                "traffic_bytes": traffic,
                "copy": {"timing": tc.as_dict(), "gb_per_s": copy_gbps},
                "timing": tt.as_dict(), "gb_per_s": gbps,
                "fraction_of_copy": gbps / copy_gbps,
                "elements_per_s": elems_per_s,
                "butterflies_per_s": elems_per_s * n / 2,
            }
            entry["sizes"].append(row)
            print(f"  n={n:2d} b={row['batch']:>6d} {nbytes/2**20:7.0f} MiB "
                  f"p={plan['num_passes']} tile={plan['tile_bytes']:>5d} | "
                  f"copy {copy_gbps:6.1f} | xform {tt.sustained_s*1e3:8.2f} ms "
                  f"{gbps:6.1f} GB/s | {gbps/copy_gbps*100:5.1f}% of copy | "
                  f"drift {tt.drift_ratio:.3f}", flush=True)

            if first_ref is None:
                first_ref = (n, tt.sustained_s)
            del a
            mx.clear_cache()
            cooldown(args.cooldown, f"after n={n}")

        if first_ref:
            n0, t0 = first_ref
            a = pool.view(n0, args.min_elems)
            tt = time_sustained(lambda: yates.transform(a, args.variant),
                                args.duration, args.warmup)
            entry["thermal_recheck"] = {"n": n0, "first_s": t0,
                                        "last_s": tt.sustained_s,
                                        "slowdown": tt.sustained_s / t0}
            print(f"  thermal re-check n={n0}: {t0*1e3:.2f} ms at start -> "
                  f"{tt.sustained_s*1e3:.2f} ms at end ({tt.sustained_s/t0:.3f}x)", flush=True)
            del a
            mx.clear_cache()

        pool.close()
        results["dtypes"][dt_name] = entry

    results["thermal_after"] = thermal_snapshot()
    results["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(args.out, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
