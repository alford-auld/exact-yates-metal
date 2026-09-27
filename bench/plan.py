#!/usr/bin/env python3
"""Print the tiering plan the dispatcher would choose, without running anything.

    .venv/bin/python bench/plan.py --dtype uint32 --min-n 1 --max-n 30
"""

from __future__ import annotations

import argparse
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yates  # noqa: E402

DTYPES = {"uint32": mx.uint32, "uint64": mx.uint64, "float32": mx.float32}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="uint32", choices=sorted(DTYPES))
    ap.add_argument("--min-n", type=int, default=1)
    ap.add_argument("--max-n", type=int, default=30)
    ap.add_argument("--elems", type=int, default=None,
                    help="total elements; default 2^n (batch of one)")
    args = ap.parse_args()

    dtype = DTYPES[args.dtype]
    lim = yates.limits()
    print(f"device {lim.device_name}  simd_width={lim.simd_width}  "
          f"max_threads/tg={lim.max_threads_per_threadgroup}  "
          f"tg_mem={lim.max_threadgroup_memory}B  mlx {lim.mlx_version}")
    print(f"dtype {args.dtype}\n")
    print(f"{'n':>3} {'tile':>6} {'pass':>4} {'tiers simd/tg/reg':>18}  "
          f"{'traffic/N/elem':>14}  passes")
    for n in range(args.min_n, args.max_n + 1):
        total = args.elems or (1 << n)
        p = yates.describe_plan(n, total, dtype)
        t = p["stages_by_tier"]
        desc = " ".join(
            f"{q['stages']}c{q['logc']}r{q['logr']}g{q['logg']}x{q['elems_per_thread']}"
            for q in p["passes"]
        )
        print(f"{n:>3} {p['tile_bytes']:>6} {p['num_passes']:>4} "
              f"{t['simd_shuffle']:>5}/{t['threadgroup']:>3}/{t['register']:>3}   "
              f"{2*p['num_passes']:>14}  {desc}")
    print("\nlegend: [a,b) = stage range, c=log2 columns, r=log2 rows resident, "
          "g=log2 tiles/threadgroup, x=elements per thread")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
