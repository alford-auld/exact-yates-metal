"""Obviously-correct reference implementations -- the test oracles.

Two independent oracles are provided:

* :func:`reference_transform` -- pure MLX, one naive stage at a time.
* :func:`numpy_transform`     -- pure NumPy, same algorithm, no MLX involved.
* :func:`dense_transform`     -- builds the full ``2^n x 2^n`` Kronecker power
  and does a dense matrix-vector product.  O(4^n); only for tiny n, but it
  depends on nothing in the Yates factorisation at all.

None of these is fast.  That is the point.
"""

from __future__ import annotations

from typing import Sequence

import mlx.core as mx
import numpy as np

from .variants import get

#: Element types this package supports.
UNSIGNED_DTYPES = (mx.uint32, mx.uint64)
SUPPORTED_DTYPES = (mx.uint32, mx.uint64, mx.float32)

_RING_BITS = {mx.uint32: 32, mx.uint64: 64}


def ring_bits(dtype) -> int:
    """Width of the modular ring for an exact dtype, or 0 for floating point."""
    return _RING_BITS.get(dtype, 0)


def check_dtype(dtype) -> None:
    if dtype not in SUPPORTED_DTYPES:
        raise TypeError(
            f"unsupported element type {dtype}; supported: "
            + ", ".join(str(d) for d in SUPPORTED_DTYPES)
        )


def log2_exact(n: int) -> int:
    """``log2(n)`` for a power of two, else raise."""
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(
            f"transform length must be a positive power of two, got {n}"
        )
    return n.bit_length() - 1


def check_length(size: int) -> int:
    """Validate the transform axis length and return ``n = log2(size)``."""
    try:
        return log2_exact(size)
    except ValueError:
        raise ValueError(
            f"transform axis has length {size}, which is not a power of two; "
            "the Yates algorithm requires N = 2**n (pad the input if needed)"
        ) from None


# --------------------------------------------------------------------------
# scalar constants in the target ring
# --------------------------------------------------------------------------


def ring_scalar(c: int, dtype) -> mx.array:
    """The ring element ``c`` as an MLX scalar of ``dtype``.

    For unsigned dtypes negative constants are reduced mod ``2**bits``, which
    is exactly what wrapping unsigned arithmetic computes.
    """
    bits = ring_bits(dtype)
    if bits:
        # numpy round-trip: mx.array() rejects Python ints >= 2**63
        np_dt = np.uint32 if bits == 32 else np.uint64
        return mx.array(np.array(c % (1 << bits), dtype=np_dt))
    return mx.array(float(c), dtype=dtype)


def _axpy(c: int, v: mx.array, acc):
    """``acc + c*v`` with the multiply elided for c in {0, 1, -1}."""
    if c == 0:
        return acc
    if c == 1:
        term = v
    elif c == -1:
        term = mx.zeros_like(v) - v          # wrapping negation; exact mod 2^k
    else:
        term = ring_scalar(c, v.dtype) * v
    return term if acc is None else acc + term


# --------------------------------------------------------------------------
# MLX reference
# --------------------------------------------------------------------------


def reference_transform(x: mx.array, variant, axis: int = -1) -> mx.array:
    """Naive MLX Yates transform: ``n`` stages, one slice pair per stage."""
    v = get(variant)
    m00, m01, m10, m11 = v.entries
    check_dtype(x.dtype)

    x = mx.swapaxes(x, axis, -1) if axis not in (-1, x.ndim - 1) else x
    shape = x.shape
    size = shape[-1] if shape else 1
    n = check_length(size)

    a = mx.reshape(x, (-1, size))
    batch = a.shape[0]

    for j in range(n):
        # index i = hi * 2^(j+1) + t * 2^j + lo,  bit j of i is t
        a = mx.reshape(a, (batch, size >> (j + 1), 2, 1 << j))
        x0 = a[:, :, 0, :]
        x1 = a[:, :, 1, :]
        y0 = _axpy(m01, x1, _axpy(m00, x0, None))
        y1 = _axpy(m11, x1, _axpy(m10, x0, None))
        if y0 is None:
            y0 = mx.zeros_like(x0)
        if y1 is None:
            y1 = mx.zeros_like(x1)
        a = mx.reshape(mx.stack([y0, y1], axis=2), (batch, size))

    out = mx.reshape(a, shape)
    return mx.swapaxes(out, axis, -1) if axis not in (-1, x.ndim - 1) else out


# --------------------------------------------------------------------------
# NumPy reference (fully independent of MLX)
# --------------------------------------------------------------------------


def numpy_transform(x: np.ndarray, variant, axis: int = -1) -> np.ndarray:
    """Naive NumPy Yates transform.  Wraps modularly for unsigned dtypes."""
    v = get(variant)
    m00, m01, m10, m11 = v.entries
    x = np.swapaxes(x, axis, -1)
    shape = x.shape
    size = shape[-1]
    n = check_length(size)
    dt = x.dtype

    def mul(c, arr):
        if c == 1:
            return arr
        if c == -1:
            return (dt.type(0) - arr) if dt.kind == "u" else -arr
        return dt.type(c % (1 << (8 * dt.itemsize))) * arr if dt.kind == "u" else dt.type(c) * arr

    a = x.reshape(-1, size).copy()
    batch = a.shape[0]
    with np.errstate(over="ignore"):
        for j in range(n):
            a = a.reshape(batch, size >> (j + 1), 2, 1 << j)
            x0, x1 = a[:, :, 0, :].copy(), a[:, :, 1, :].copy()
            y0 = np.zeros_like(x0) if m00 == 0 and m01 == 0 else None
            if y0 is None:
                y0 = (mul(m00, x0) if m00 else np.zeros_like(x0))
                if m01:
                    y0 = y0 + mul(m01, x1)
            y1 = (mul(m10, x0) if m10 else np.zeros_like(x0))
            if m11:
                y1 = y1 + mul(m11, x1)
            a[:, :, 0, :] = y0
            a[:, :, 1, :] = y1
            a = a.reshape(batch, size)
    return np.swapaxes(a.reshape(shape), axis, -1)


# --------------------------------------------------------------------------
# dense Kronecker power (independent of the Yates factorisation)
# --------------------------------------------------------------------------


def kron_power(variant, n: int, dtype=np.int64) -> np.ndarray:
    """The explicit ``2^n x 2^n`` matrix ``M^(x)n``."""
    m = np.array(get(variant).m, dtype=dtype)
    out = np.ones((1, 1), dtype=dtype)
    for _ in range(n):
        out = np.kron(out, m)
    return out


def dense_transform(x: np.ndarray, variant) -> np.ndarray:
    """``M^(x)n @ x`` via an explicit dense matrix.  O(4^n); tiny n only."""
    n = check_length(x.shape[-1])
    a = kron_power(variant, n, dtype=x.dtype if x.dtype.kind == "f" else np.int64)
    flat = x.reshape(-1, x.shape[-1])
    with np.errstate(over="ignore"):
        res = flat.astype(a.dtype) @ a.T
    return res.reshape(x.shape)


# --------------------------------------------------------------------------
# naive O(4^n) set convolutions, for the convolution-theorem tests
# --------------------------------------------------------------------------


def naive_convolution(f: np.ndarray, g: np.ndarray, op: str) -> np.ndarray:
    """``h[k] = sum_{i op j == k} f[i] g[j]`` for op in {xor, or, and}."""
    size = f.shape[-1]
    check_length(size)
    idx = np.arange(size)
    if op == "xor":
        k = idx[:, None] ^ idx[None, :]
    elif op == "or":
        k = idx[:, None] | idx[None, :]
    elif op == "and":
        k = idx[:, None] & idx[None, :]
    else:
        raise ValueError(f"unknown op {op!r}")
    with np.errstate(over="ignore"):
        prod = (f[:, None] * g[None, :]).ravel()
    out = np.zeros(size, dtype=prod.dtype)
    np.add.at(out, k.ravel(), prod)
    return out
