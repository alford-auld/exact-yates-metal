"""Group 4: Parseval.  With the unnormalized WHT, ||Hx||^2 == 2^n ||x||^2."""

import mlx.core as mx
import numpy as np
import pytest

import yates
from kernel_helpers import np_of, rand_uint


@pytest.mark.parametrize("n", list(range(0, 15)))
def test_parseval_float32(n, rng):
    x = rng.standard_normal(1 << n).astype(np.float32)
    y = np_of(yates.transform(mx.array(x), "WHT")).astype(np.float64)
    lhs = float((y ** 2).sum())
    rhs = float((1 << n) * (x.astype(np.float64) ** 2).sum())
    assert np.isclose(lhs, rhs, rtol=1e-5), f"n={n}: {lhs} vs {rhs}"


@pytest.mark.parametrize("n", list(range(0, 15)))
def test_parseval_exact_mod_2_64(n, rng):
    """Parseval holds exactly in Z/2^64: squares and sums both wrap consistently."""
    x = rand_uint(rng, (1 << n,), np.uint64)
    y = np_of(yates.transform(mx.array(x), "WHT"))
    with np.errstate(over="ignore"):
        lhs = np.uint64((y.astype(object) ** 2 % (1 << 64)).sum() % (1 << 64))
        rhs = np.uint64(((1 << n) * (x.astype(object) ** 2).sum()) % (1 << 64))
    assert lhs == rhs, f"n={n}: {lhs} vs {rhs}"


@pytest.mark.parametrize("n", [0, 1, 6, 12])
def test_normalized_wht_is_an_isometry(n, rng):
    """H/sqrt(N) preserves the Euclidean norm -- the preconditioning use case."""
    x = rng.standard_normal((4, 1 << n)).astype(np.float32)
    y = np_of(yates.transform(mx.array(x), "WHT", normalize=True)).astype(np.float64)
    nx = np.linalg.norm(x.astype(np.float64), axis=-1)
    ny = np.linalg.norm(y, axis=-1)
    assert np.allclose(nx, ny, rtol=1e-5), f"n={n}: {nx} vs {ny}"


@pytest.mark.parametrize("n", [3, 8, 12])
def test_parseval_batched(n, rng):
    x = rng.standard_normal((6, 1 << n)).astype(np.float32)
    y = np_of(yates.transform(mx.array(x), "WHT")).astype(np.float64)
    assert np.allclose(
        (y ** 2).sum(-1), (1 << n) * (x.astype(np.float64) ** 2).sum(-1), rtol=1e-5
    )


@pytest.mark.parametrize("n", list(range(0, 13)))
def test_parseval_exact_mod_2_32(n, rng):
    x = rand_uint(rng, (1 << n,), np.uint32)
    y = np_of(yates.transform(mx.array(x), "WHT"))
    lhs = int((y.astype(object) ** 2).sum()) % (1 << 32)
    rhs = int((1 << n) * (x.astype(object) ** 2).sum()) % (1 << 32)
    assert lhs == rhs, f"n={n}: {lhs} vs {rhs}"
