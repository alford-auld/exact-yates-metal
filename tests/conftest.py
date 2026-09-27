"""Shared fixtures.  Every run prints the MLX version and machine it measured."""

import os
import sys

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yates  # noqa: E402

UINT_DTYPES = [(mx.uint32, np.uint32, 32), (mx.uint64, np.uint64, 64)]
ALL_VARIANTS = list(yates.VARIANTS)


def pytest_report_header(config):
    lim = yates.limits()
    host = yates.host_info()
    return [
        f"mlx {lim.mlx_version} | metal available: {mx.metal.is_available()}",
        f"gpu  {lim.device_name} ({lim.architecture}) simd_width={lim.simd_width} "
        f"max_threads/tg={lim.max_threads_per_threadgroup} "
        f"tg_mem={lim.max_threadgroup_memory}B (via {lim.source})",
        f"host {host.get('Model Identifier','?')} {host.get('Chip','?')} "
        f"{host.get('Total Number of Cores','?')} cores, {host.get('Memory','?')} RAM, "
        f"macOS {host.get('macOS','?')}",
    ]


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(20240927)


def np_of(a: mx.array) -> np.ndarray:
    mx.eval(a)
    return np.array(a)


def rand_uint(rng, shape, np_dtype) -> np.ndarray:
    """Full-range random unsigned values (the whole ring, not a small subset)."""
    bits = 32 if np_dtype is np.uint32 else 64
    hi = rng.integers(0, 1 << 32, size=shape, dtype=np.uint64)
    if bits == 32:
        return hi.astype(np.uint32)
    lo = rng.integers(0, 1 << 32, size=shape, dtype=np.uint64)
    return (hi << np.uint64(32)) | lo


def exact_transform_python(x: np.ndarray, variant, modulus: int) -> np.ndarray:
    """Yates transform in arbitrary-precision Python ints, reduced at the end.

    Independent of NumPy's and Metal's wraparound; this is what "exact mod 2^k"
    is being checked against.
    """
    m = yates.get(variant).m
    vals = [int(v) for v in x.ravel()]
    n = (len(vals)).bit_length() - 1
    for j in range(n):
        step = 1 << j
        for base in range(0, len(vals), step << 1):
            for off in range(step):
                i0, i1 = base + off, base + off + step
                a, b = vals[i0], vals[i1]
                vals[i0] = m[0][0] * a + m[0][1] * b
                vals[i1] = m[1][0] * a + m[1][1] * b
    return np.array([v % modulus for v in vals], dtype=object).reshape(x.shape)


def as_signed(a: np.ndarray, bits: int) -> np.ndarray:
    """Reinterpret an unsigned ring element as a two's-complement integer.

    Walsh coefficients of a +/-1 vector are small, so the ring representation is
    an exact stand-in for signed arithmetic as long as |value| < 2^(bits-1).
    """
    v = a.astype(object)
    half = 1 << (bits - 1)
    return np.array([int(x) - (1 << bits) if int(x) >= half else int(x) for x in v.ravel()],
                    dtype=np.int64).reshape(a.shape)


def pm1_ring(bits_array: np.ndarray, np_dtype) -> np.ndarray:
    """(-1)^b encoded in an unsigned ring: 1 stays 1, -1 becomes 2^k - 1."""
    bits = 32 if np_dtype is np.uint32 else 64
    return np.where(bits_array == 0, np_dtype(1), np_dtype((1 << bits) - 1)).astype(np_dtype)
