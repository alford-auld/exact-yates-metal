"""Group 6: the transpose identity <Tx, y> == <x, (T^T)y>, for every variant.

This validates both the forward kernel and the VJP, since the backward pass is
literally the transposed-generator kernel.
"""

import mlx.core as mx
import numpy as np
import pytest

import yates
from conftest import ALL_VARIANTS, UINT_DTYPES, np_of, rand_uint


@pytest.mark.parametrize("name", ALL_VARIANTS)
@pytest.mark.parametrize("n", [0, 1, 2, 5, 9, 13, 16])
def test_transpose_identity_float32(name, n, rng):
    x = rng.standard_normal(1 << n).astype(np.float32)
    y = rng.standard_normal(1 << n).astype(np.float32)
    tname = yates.TRANSPOSE[name]

    lhs = float(np_of(yates.transform(mx.array(x), name)).astype(np.float64) @ y.astype(np.float64))
    rhs = float(x.astype(np.float64) @ np_of(yates.transform(mx.array(y), tname)).astype(np.float64))
    scale = max(1.0, abs(lhs), abs(rhs))
    assert np.isclose(lhs, rhs, rtol=1e-5, atol=1e-5 * scale), f"{name} n={n}: {lhs} vs {rhs}"


@pytest.mark.parametrize("name", ALL_VARIANTS)
@pytest.mark.parametrize("mx_dt,np_dt,bits", UINT_DTYPES)
@pytest.mark.parametrize("n", [1, 6, 11])
def test_transpose_identity_exact_in_the_ring(name, mx_dt, np_dt, bits, n, rng):
    """<Tx, y> == <x, T^T y> exactly in Z/2^k, with no tolerance at all."""
    x = rand_uint(rng, (1 << n,), np_dt)
    y = rand_uint(rng, (1 << n,), np_dt)
    tname = yates.TRANSPOSE[name]
    mod = 1 << bits

    lhs = int((np_of(yates.transform(mx.array(x), name)).astype(object) * y.astype(object)).sum()) % mod
    rhs = int((x.astype(object) * np_of(yates.transform(mx.array(y), tname)).astype(object)).sum()) % mod
    assert lhs == rhs, f"{name} n={n} {bits}-bit: {lhs} != {rhs}"


@pytest.mark.parametrize("name", ALL_VARIANTS)
@pytest.mark.parametrize("n", [0, 1, 4, 8])
def test_transpose_table_agrees_with_the_dense_matrix(name, n):
    """(M^(x)n)^T == (M^T)^(x)n, checked on the explicit Kronecker powers."""
    a = yates.kron_power(name, n)
    at = yates.kron_power(yates.TRANSPOSE[name], n)
    assert np.array_equal(a.T, at)


@pytest.mark.parametrize("name", ALL_VARIANTS)
@pytest.mark.parametrize("n", [1, 5, 10])
def test_vjp_equals_the_transposed_transform(name, n, rng):
    """mx.grad of <T x, g> must be exactly the transposed kernel applied to g."""
    g_np = rng.standard_normal(1 << n).astype(np.float32)
    g = mx.array(g_np)
    x = mx.array(rng.standard_normal(1 << n).astype(np.float32))

    grad = np_of(mx.grad(lambda a: mx.sum(yates.yates_transform(a, name) * g))(x))
    want = np_of(yates.transform(g, yates.TRANSPOSE[name]))
    scale = max(1.0, float(np.abs(want).max()))
    assert np.allclose(grad, want, rtol=1e-5, atol=1e-5 * scale)
    assert yates.adjoint_of(name) == yates.TRANSPOSE[name]


@pytest.mark.parametrize("name", ALL_VARIANTS)
def test_adjoint_is_an_involution(name):
    assert yates.TRANSPOSE[yates.TRANSPOSE[name]] == name
