"""MLX wrapper, dispatch and tiering policy for the Yates butterfly kernel."""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import mlx.core as mx

from . import device
from .reference import check_dtype, check_length, ring_bits
from .variants import get

_METAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel.metal")


@functools.lru_cache(maxsize=1)
def _sections() -> dict:
    """Split kernel.metal on its ``// ===== @section NAME =====`` markers."""
    with open(_METAL_PATH, "r") as fh:
        text = fh.read()
    out, name, buf = {}, None, []
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("// ===== @section ") and stripped.endswith("====="):
            if name is not None:
                out[name] = "".join(buf)
            name = stripped.split()[3]
            buf = []
        elif name is not None:
            buf.append(line)
    if name is not None:
        out[name] = "".join(buf)
    missing = {"header", "pass", "copy"} - set(out)
    if missing:
        raise RuntimeError(f"kernel.metal is missing sections: {sorted(missing)}")
    return out


@functools.lru_cache(maxsize=1)
def _pass_kernel():
    s = _sections()
    return mx.fast.metal_kernel(
        name="yates_pass",
        input_names=["inp"],
        output_names=["out"],
        source=s["pass"],
        header=s["header"],
        ensure_row_contiguous=True,
    )


@functools.lru_cache(maxsize=1)
def _copy_kernel():
    return mx.fast.metal_kernel(
        name="yates_copy",
        input_names=["inp"],
        output_names=["out"],
        source=_sections()["copy"],
        ensure_row_contiguous=True,
    )


# --------------------------------------------------------------------------
# matrix encoding
# --------------------------------------------------------------------------


def encode_matrix(variant) -> Tuple[Tuple[str, int], ...]:
    """Matrix entries as (sign, magnitude) template pairs, all non-negative.

    MLX pastes template integers straight into the generated Metal identifier,
    so a literal ``-1`` would emit ``custom_kernel_..._-1_...`` which is not a
    valid C++ name.  Sign and magnitude are carried separately instead.
    """
    names = ("00", "01", "10", "11")
    out: List[Tuple[str, int]] = []
    for nm, c in zip(names, get(variant).entries):
        c = int(c)
        out.append((f"S{nm}", 1 if c < 0 else 0))
        out.append((f"A{nm}", abs(c)))
    return tuple(out)


# --------------------------------------------------------------------------
# tiering policy
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Pass:
    """One device-memory pass covering transform stages ``[s, s+p)``."""

    s: int      # first index bit transformed by this pass
    p: int      # number of stages fused into this pass
    logc: int   # log2 contiguous columns per tile (coalescing width)
    logr: int   # log2 rows resident per tile at once
    logg: int   # log2 independent tiles per threadgroup

    @property
    def threads_per_group(self) -> int:
        return 1 << (self.logc + self.logr + self.logg)

    @property
    def elements_per_thread(self) -> int:
        return 1 << (self.p - self.logr)

    def tiers(self, log_simd_width: int) -> dict:
        """How many stages land in each tier, for introspection and tests."""
        nb_simd = min(max(log_simd_width - self.logc, 0), self.logr)
        return {
            "simd_shuffle": nb_simd,
            "threadgroup": self.logr - nb_simd,
            "register": self.p - self.logr,
        }

    def threadgroup_bytes(self, itemsize: int, log_simd_width: int) -> int:
        nb_simd = min(max(log_simd_width - self.logc, 0), self.logr)
        if self.logr <= nb_simd:
            return itemsize  # buffer elided
        return itemsize << (self.p + self.logc + self.logg)


# Coalescing policy for strided passes.
#
# A pass at bit offset s reads rows 2^s elements apart, C contiguous elements
# each.  When that stride is large, every row of a tile lands on a different
# DRAM page and TLB entry, and short runs stop amortising the page activation:
# measured on this machine, the n=29 uint32 transform runs at 57% of copy
# bandwidth with 128-byte runs and 95% with 2 KiB runs, even though the longer
# runs force an extra device pass.  So the required run length scales with the
# stride, clamped to a measured-useful range.  See bench/tune_coalescing.py.
_COAL_STRIDE_DIVISOR = 8192   # run bytes ~= stride bytes / this
_COAL_MIN_RUN_BYTES = 128     # one cache line; enough at small strides
_COAL_MAX_RUN_BYTES = 2048    # beyond this the extra passes cost more than they save


def _coalescing_logc(s: int, itemsize: int, tile_elems_log: int,
                     max_thread_log: int) -> int:
    """log2 of the contiguous columns a pass at offset ``s`` should use."""
    if s == 0:
        return 0                      # tiles are already contiguous
    stride_bytes = itemsize << s
    run = min(max(stride_bytes // _COAL_STRIDE_DIVISOR, _COAL_MIN_RUN_BYTES),
              _COAL_MAX_RUN_BYTES)
    logc = max((run // itemsize).bit_length() - 1, 0)
    return min(logc, s, max_thread_log, tile_elems_log - 1)


def _ilog2(v: int) -> int:
    return v.bit_length() - 1


def _v2(v: int) -> int:
    """Number of trailing zero bits (2-adic valuation) of a positive int."""
    return (v & -v).bit_length() - 1 if v else 0


def plan_passes(
    n: int,
    total_elems: int,
    itemsize: int,
    lim: Optional[device.DeviceLimits] = None,
    tile_bytes: Optional[int] = None,
) -> List[Pass]:
    """Decompose ``n`` stages into device passes.

    Greedy: each pass fuses as many stages as the threadgroup tile holds, which
    minimises the number of passes and hence total device traffic.
    """
    lim = lim or device.limits()
    if tile_bytes is None:
        tile_bytes = choose_tile_bytes(n, total_elems, itemsize, lim)
    tile_bytes = min(tile_bytes, lim.max_threadgroup_memory)

    tile_elems_log = _ilog2(tile_bytes // itemsize)
    max_thread_log = _ilog2(lim.max_threads_per_threadgroup)
    target_tg_log = min(max_thread_log, 8)

    passes: List[Pass] = []
    s = 0
    while s < n:
        remaining = n - s
        logc = _coalescing_logc(s, itemsize, tile_elems_log, max_thread_log)
        p = min(remaining, tile_elems_log - logc)
        # A short final pass has tile memory to spare: spend it on a wider
        # contiguous run so the strided traffic still coalesces.
        if s > 0 and p < tile_elems_log - logc:
            logc = min(s, tile_elems_log - p, max_thread_log,
                       max((_COAL_MAX_RUN_BYTES // itemsize).bit_length() - 1, 0))
            p = min(remaining, tile_elems_log - logc)
        if p <= 0:
            raise RuntimeError(
                f"cannot fit any stage in a {tile_bytes} byte tile for "
                f"{itemsize}-byte elements"
            )

        logr = min(p, max_thread_log - logc)
        if logr < 0:
            raise RuntimeError("coalescing width exceeds the threadgroup size limit")

        # G tiles share a threadgroup, so G must divide the tile count; the
        # batch dimension need not be a power of two.
        num_tiles = total_elems >> (p + logc)
        logg = min(
            max(target_tg_log - logc - logr, 0),
            max_thread_log - logc - logr,
            tile_elems_log - p - logc,
            _v2(num_tiles),
        )
        passes.append(Pass(s=s, p=p, logc=logc, logr=logr, logg=max(logg, 0)))
        s += p
    return passes


def naive_passes(n: int, total_elems: int, lim: Optional[device.DeviceLimits] = None) -> List[Pass]:
    """One device pass per stage, with every optimisation tier switched off.

    ``p=1`` fuses nothing, ``logc=0`` uses no coalescing columns and ``logr=0``
    puts the single butterfly entirely inside one thread's registers, so
    ``NB_SIMD == 0`` (no shuffles) and the threadgroup buffer is elided.  This
    is the textbook strided device-memory butterfly, and it is the reference
    the tiered path is validated against in tests/test_naive_vs_tiered.py.
    """
    lim = lim or device.limits()
    max_thread_log = _ilog2(lim.max_threads_per_threadgroup)
    out = []
    for j in range(n):
        num_tiles = total_elems >> 1
        logg = min(max(min(max_thread_log, 8), 0), _v2(num_tiles))
        out.append(Pass(s=j, p=1, logc=0, logr=0, logg=logg))
    return out


def choose_tile_bytes(
    n: int, total_elems: int, itemsize: int, lim: Optional[device.DeviceLimits] = None
) -> int:
    """Smallest tile that achieves the minimum achievable pass count.

    Bigger tiles fuse more stages per pass (less device traffic) but cost
    occupancy, so we take the smallest tile that is not paying extra passes.
    """
    lim = lim or device.limits()
    best = None
    cand = 1024
    while cand <= lim.max_threadgroup_memory:
        if cand // itemsize >= 2:
            try:
                k = len(plan_passes(n, total_elems, itemsize, lim, tile_bytes=cand))
            except RuntimeError:
                k = None
            if k is not None and (best is None or k < best[0]):
                best = (k, cand)
        cand *= 2
    if best is None:
        raise RuntimeError("no viable threadgroup tile size for this configuration")
    return best[1]


def describe_plan(n: int, total_elems: int, dtype, tile_bytes: Optional[int] = None) -> dict:
    """Human-readable tiering decision, used by tests and the benchmark."""
    lim = device.limits()
    itemsize = mx.zeros((1,), dtype=dtype).itemsize
    if tile_bytes is None:
        tile_bytes = choose_tile_bytes(n, total_elems, itemsize, lim)
    passes = plan_passes(n, total_elems, itemsize, lim, tile_bytes)
    logw = _ilog2(lim.simd_width)
    tiers = {"simd_shuffle": 0, "threadgroup": 0, "register": 0}
    for ps in passes:
        for k, v in ps.tiers(logw).items():
            tiers[k] += v
    return {
        "n": n,
        "tile_bytes": tile_bytes,
        "passes": [
            {
                "stages": f"[{ps.s},{ps.s + ps.p})",
                "s": ps.s, "p": ps.p, "logc": ps.logc, "logr": ps.logr, "logg": ps.logg,
                "threads_per_group": ps.threads_per_group,
                "elems_per_thread": ps.elements_per_thread,
                "threadgroup_bytes": ps.threadgroup_bytes(itemsize, logw),
                "tiers": ps.tiers(logw),
            }
            for ps in passes
        ],
        "num_passes": len(passes),
        "stages_by_tier": tiers,
        "device_traffic_bytes": 2 * len(passes) * total_elems * itemsize,
    }


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------


def _run_pass(
    a: mx.array,
    ps: Pass,
    mcodes: Sequence[Tuple[str, int]],
    logw: int,
    nrm: Tuple[int, int, int],
    stream=None,
) -> mx.array:
    total = a.size
    num_tiles = total >> (ps.p + ps.logc)
    num_groups = num_tiles >> ps.logg
    tg = ps.threads_per_group
    template = [("T", a.dtype)]
    template += list(mcodes)
    template += [
        ("SOFF", ps.s), ("P", ps.p), ("LOGC", ps.logc),
        ("LOGR", ps.logr), ("LOGG", ps.logg), ("LOGW", logw),
        ("NRM", nrm[0]), ("NRME", nrm[1]), ("NRMH", nrm[2]),
    ]
    (out,) = _pass_kernel()(
        inputs=[a],
        template=template,
        grid=(num_groups * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[a.shape],
        output_dtypes=[a.dtype],
        stream=stream,
    )
    return out


def transform(
    x: mx.array,
    variant,
    axis: int = -1,
    normalize: bool = False,
    tile_bytes: Optional[int] = None,
    naive: bool = False,
    stream=None,
) -> mx.array:
    """Apply ``M^(x)n`` along ``axis`` with the templated Metal kernel.

    Args:
      x: input array; the transform axis must have length ``2**n``.
      variant: a name (``"WHT"``, ``"ZETA_SUB"``, ...), a :class:`Variant`, or a
        raw 2x2 integer matrix.
      axis: transform axis, default the last.
      normalize: scale the result by ``2**(-n/2)``, giving the orthonormal
        isometry ``H/sqrt(N)``.  Float only, and folded into the last pass.
      tile_bytes: override the threadgroup tile budget (benchmark knob).
      naive: use one unfused device pass per stage with every tier disabled.
        Slow by construction; it exists so the tiered path has an independent
        on-GPU reference (see :func:`naive_passes`).
    """
    device.require_metal()
    check_dtype(x.dtype)
    v = get(variant)

    moved = axis not in (-1, x.ndim - 1)
    a = mx.swapaxes(x, axis, -1) if moved else x
    shape = a.shape
    size = shape[-1] if shape else 1
    n = check_length(size)

    if normalize and ring_bits(x.dtype):
        raise TypeError(
            "normalize=True scales by 2**(-n/2), which is not a ring element of "
            f"{x.dtype}; use float32 for the orthonormal isometry"
        )

    nrm = (1, n // 2, n % 2) if normalize else (0, 0, 0)

    if n == 0:
        out = a if not normalize else a * (1.0 if not nrm[2] else 2.0**-0.5)
        return mx.swapaxes(out, axis, -1) if moved else out

    flat = mx.reshape(mx.contiguous(a), (-1,))
    lim = device.limits()
    logw = _ilog2(lim.simd_width)
    mcodes = encode_matrix(v)
    if naive:
        passes = naive_passes(n, flat.size, lim)
    else:
        passes = plan_passes(n, flat.size, x.itemsize, lim, tile_bytes)

    for i, ps in enumerate(passes):
        last = i == len(passes) - 1
        flat = _run_pass(flat, ps, mcodes, logw, nrm if last else (0, 0, 0), stream)

    out = mx.reshape(flat, shape)
    return mx.swapaxes(out, axis, -1) if moved else out


# convenience wrappers ------------------------------------------------------

def wht(x, axis=-1, normalize=False, **kw):
    """Walsh-Hadamard transform (unnormalized unless ``normalize``)."""
    return transform(x, "WHT", axis=axis, normalize=normalize, **kw)


def zeta_sub(x, axis=-1, **kw):
    """Subset-sum (zeta) transform; the polar-code kernel."""
    return transform(x, "ZETA_SUB", axis=axis, **kw)


def mobius_sub(x, axis=-1, **kw):
    """Subset Mobius inversion; exact inverse of :func:`zeta_sub`."""
    return transform(x, "MOB_SUB", axis=axis, **kw)


def zeta_sup(x, axis=-1, **kw):
    """Superset-sum (zeta) transform."""
    return transform(x, "ZETA_SUP", axis=axis, **kw)


def mobius_sup(x, axis=-1, **kw):
    """Superset Mobius inversion; exact inverse of :func:`zeta_sup`."""
    return transform(x, "MOB_SUP", axis=axis, **kw)


def device_copy(x: mx.array, vec: int = 4, stream=None) -> mx.array:
    """Straight device-to-device copy kernel: the bandwidth denominator."""
    device.require_metal()
    flat = mx.reshape(mx.contiguous(x), (-1,))
    if flat.size % vec:
        raise ValueError(f"copy size {flat.size} not divisible by vec={vec}")
    (out,) = _copy_kernel()(
        inputs=[flat],
        template=[("T", x.dtype), ("VEC", vec)],
        grid=(flat.size // vec, 1, 1),
        threadgroup=(min(256, flat.size // vec), 1, 1),
        output_shapes=[flat.shape],
        output_dtypes=[flat.dtype],
        stream=stream,
    )
    return mx.reshape(out, x.shape)
