#!/usr/bin/env python3
"""Measure how the contiguous-run length of a high-stride pass affects bandwidth.

This is the experiment behind the `_COAL_*` constants in yates/kernel.py.  At
n=29 (uint32, 2 GiB) the default 3-pass plan with 128-byte runs reached 57% of
copy bandwidth; forcing 2 KiB runs, at the cost of a fourth device pass, reached
95%.  Re-run with:

    .venv/bin/python bench/tune_coalescing.py

Each candidate is checked against the default plan's output before it is timed.
"""
import sys; sys.path.insert(0,'.')
import mlx.core as mx, numpy as np, itertools, time
import yates
from yates import kernel as K
from bench.harness import time_sustained, cooldown

LOGW = 5
def run_plan(a, passes, variant='WHT'):
    flat = mx.reshape(a, (-1,))
    codes = K.encode_matrix(variant)
    def go():
        f = flat
        for ps in passes: f = K._run_pass(f, ps, codes, LOGW, (0,0,0))
        return f
    return go

def mk(s,p,logc,maxthread=10,tile_log=13):
    logr = min(p, maxthread-logc)
    assert logr>=0 and p+logc<=tile_log, (s,p,logc)
    return K.Pass(s=s,p=p,logc=logc,logr=logr,logg=0)

for n in (28, 29):
    N = 1<<n
    a = mx.array(np.random.default_rng(0).integers(0,1<<32,size=N,dtype=np.uint32)); mx.eval(a)
    nb = N*4
    ref = np.array(yates.transform(a,'WHT', naive=False))
    tc = time_sustained(lambda: yates.device_copy(a, vec=1), 1.5, 0.5)
    cg = 2*nb/tc.sustained_s/1e9
    print(f"\n=== n={n}  {nb/2**20:.0f} MiB   copy {cg:.1f} GB/s ===")

    cands = {}
    cands['default(tile32k)'] = yates.plan_passes(n, N, 4, tile_bytes=32768)
    cands['tile16k'] = yates.plan_passes(n, N, 4, tile_bytes=16384)
    cands['tile8k']  = yates.plan_passes(n, N, 4, tile_bytes=8192)
    # hand plans: wide columns on the high-stride passes
    if n == 28:
        cands['13/8/ c9 4+3'] = [mk(0,13,0), mk(13,8,5), mk(21,4,9), mk(25,3,9)]
        cands['13/8/ c8 4+3'] = [mk(0,13,0), mk(13,8,5), mk(21,4,8), mk(25,3,8)]
        cands['13/8/ c7 6+1'] = [mk(0,13,0), mk(13,8,5), mk(21,6,7), mk(27,1,7)]
        cands['13/ c7 8+7']   = [mk(0,13,0), mk(13,6,7), mk(19,6,7), mk(25,3,7)]
    else:
        cands['13/8/ c9 4+4'] = [mk(0,13,0), mk(13,8,5), mk(21,4,9), mk(25,4,9)]
        cands['13/8/ c8 4+4'] = [mk(0,13,0), mk(13,8,5), mk(21,4,8), mk(25,4,8)]
        cands['13/8/ c7 6+2'] = [mk(0,13,0), mk(13,8,5), mk(21,6,7), mk(27,2,7)]
        cands['13/ c7 6+6+4'] = [mk(0,13,0), mk(13,6,7), mk(19,6,7), mk(25,4,7)]

    for label, ps in cands.items():
        got = np.array(mx.reshape(run_plan(a, ps)(), (N,)))
        ok = np.array_equal(got, ref)
        t = time_sustained(run_plan(a, ps), 1.5, 0.5)
        traffic = 2*len(ps)*nb
        g = traffic/t.sustained_s/1e9
        print(f"  {label:20s} passes={len(ps)} {t.sustained_s*1e3:8.1f} ms  "
              f"{g:6.1f} GB/s  {g/cg*100:5.1f}% of copy  correct={ok}")
    del a; mx.clear_cache(); time.sleep(3)
