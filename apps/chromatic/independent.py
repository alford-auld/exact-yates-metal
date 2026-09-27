"""The independence indicator a(S), on the GPU and in NumPy.

``a(S) = 1`` iff the vertex set with bitmask ``S`` spans no edge, i.e. iff
``adj[v] & S == 0`` for every ``v`` in ``S``.

Three implementations:

* :func:`indicator_gpu` -- one Metal thread per subset, O(2^n * n) work and no
  dependencies.  This is the production path.
* :func:`indicator_numpy_dp` -- the lowbit dynamic program, O(2^n) *total* work:
  ``a(S) = a(S \\ {v}) and (adj[v] & S == 0)`` for ``v = lowest bit of S``.
  Asymptotically better than the direct form, and the timing comparison the
  brief asks for.
* :func:`indicator_numpy_direct` -- the obvious O(2^n * n) NumPy form, used as
  the oracle in tests.

Why the direct form wins on the GPU even though the DP does less work: the DP
recurrence is sequential in ``popcount(S)``, and ``mx.fast.metal_kernel`` is
out-of-place, so a GPU DP would need one full-array pass per level -- O(2^n * n)
of *memory traffic* to save O(2^n * n) of *ALU*, on a kernel that is already
memory-bound.  Measured both ways in ``bench/chromatic/``.
"""

from __future__ import annotations

import functools
import os
from typing import Optional

import mlx.core as mx
import numpy as np

from .graph import Graph, bits

_METAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "independent.metal")


@functools.lru_cache(maxsize=1)
def _sections() -> dict:
    with open(_METAL_PATH) as fh:
        text = fh.read()
    out, name, buf = {}, None, []
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("// ===== @section ") and stripped.endswith("====="):
            if name is not None:
                out[name] = "".join(buf)
            name, buf = stripped.split()[3], []
        elif name is not None:
            buf.append(line)
    if name is not None:
        out[name] = "".join(buf)
    return out


@functools.lru_cache(maxsize=4)
def _kernel(section: str):
    return mx.fast.metal_kernel(
        name=f"indep_{section}",
        input_names=["adj"],
        output_names=["out"],
        source=_sections()[section],
        ensure_row_contiguous=True,
    )


def indicator_gpu(g: Graph, dtype=mx.uint32, section: str = "direct") -> mx.array:
    """``a(S)`` for every ``S`` in ``[0, 2^n)``, as a device array of 0/1.

    ``uint32`` is the right element type: the subsequent subset-zeta of ``a``
    counts independent sets, and that count is at most ``2^n``, so it stays
    exact in 32 bits for every ``n`` the memory ceiling allows.
    """
    n = g.n
    size = 1 << n
    if n == 0:
        return mx.ones((1,), dtype=dtype)
    adj = mx.array(g.adj_array())
    threads = min(256, size)
    (out,) = _kernel(section)(
        inputs=[adj],
        template=[("T", dtype)],
        grid=(size, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(size,)],
        output_dtypes=[dtype],
    )
    return out


def indicator_numpy_direct(g: Graph, dtype=np.uint32) -> np.ndarray:
    """O(2^n * n) NumPy oracle: n vectorised passes over the whole cube."""
    n = g.n
    size = 1 << n
    if n == 0:
        return np.ones(1, dtype=dtype)
    s = np.arange(size, dtype=np.uint32)
    bad = np.zeros(size, dtype=bool)
    for v in range(n):
        in_s = ((s >> np.uint32(v)) & np.uint32(1)).astype(bool)
        bad |= in_s & ((np.uint32(g.adj[v]) & s) != 0)
    return (~bad).astype(dtype)


def indicator_numpy_dp(g: Graph, dtype=np.uint32) -> np.ndarray:
    """O(2^n) lowbit dynamic program.

    Every non-empty ``S`` is written uniquely as ``S = 2^v | (T << (v+1))`` where
    ``v`` is its lowest set bit.  Then ``S \\ {v} = T << (v+1)``, whose own lowest
    set bit is greater than ``v``, so iterating ``v`` downward from ``n-1``
    always reads an entry already written.  Total work is
    ``sum_v 2^(n-1-v) = 2^n - 1``.
    """
    n = g.n
    size = 1 << n
    a = np.zeros(size, dtype=bool)
    a[0] = True
    if n == 0:
        return a.astype(dtype)
    for v in range(n - 1, -1, -1):
        t = np.arange(1 << (n - 1 - v), dtype=np.uint32) << np.uint32(v + 1)
        s = t | np.uint32(1 << v)
        a[s] = a[t] & ((np.uint32(g.adj[v]) & s) == 0)
    return a.astype(dtype)


def independent_sets(g: Graph):
    """Every independent set as a bitmask.  O(2^n); tests and tiny graphs only."""
    return [s for s in range(1 << g.n) if g.is_independent(s)]


def count_independent_sets(g: Graph) -> int:
    """i(V): the total number of independent sets, including the empty set.

    Used to bound ``c_k <= i(V)^k``, which is what decides how many primes a
    fully exact CRT reconstruction needs.
    """
    return int(np.asarray(indicator_gpu(g)).sum())
