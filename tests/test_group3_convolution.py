"""Group 3: convolution theorems against naive O(4^n) implementations.

XOR-convolution is diagonalized by the WHT, OR-convolution by ZETA_SUB and
AND-convolution by ZETA_SUP.  All three are checked exactly over Z/2^32 with
full-range random inputs -- the naive reference wraps the same way the kernel
does, so a wrong kernel cannot hide behind "overflow".
"""

import mlx.core as mx
import numpy as np
import pytest

import yates
from conftest import np_of, rand_uint

N_RANGE = list(range(0, 11))


@pytest.mark.parametrize("n", N_RANGE)
def test_xor_convolution_diagonalized_by_wht(n, rng):
    f = rand_uint(rng, (1 << n,), np.uint32)
    g = rand_uint(rng, (1 << n,), np.uint32)

    F = yates.transform(mx.array(f), "WHT")
    G = yates.transform(mx.array(g), "WHT")
    # N*h = WHT(WHT(f) . WHT(g)); stated without the division so it is exact in
    # the ring (1/N does not exist mod 2^32).
    n_times_h = np_of(yates.transform(F * G, "WHT"))

    with np.errstate(over="ignore"):
        want = np.uint32(1 << n) * yates.naive_convolution(f, g, "xor")
    assert np.array_equal(n_times_h, want)


@pytest.mark.parametrize("n", N_RANGE)
def test_or_convolution_diagonalized_by_zeta_sub(n, rng):
    f = rand_uint(rng, (1 << n,), np.uint32)
    g = rand_uint(rng, (1 << n,), np.uint32)
    F = yates.transform(mx.array(f), "ZETA_SUB")
    G = yates.transform(mx.array(g), "ZETA_SUB")
    h = np_of(yates.transform(F * G, "MOB_SUB"))
    assert np.array_equal(h, yates.naive_convolution(f, g, "or"))


@pytest.mark.parametrize("n", N_RANGE)
def test_and_convolution_diagonalized_by_zeta_sup(n, rng):
    f = rand_uint(rng, (1 << n,), np.uint32)
    g = rand_uint(rng, (1 << n,), np.uint32)
    F = yates.transform(mx.array(f), "ZETA_SUP")
    G = yates.transform(mx.array(g), "ZETA_SUP")
    h = np_of(yates.transform(F * G, "MOB_SUP"))
    assert np.array_equal(h, yates.naive_convolution(f, g, "and"))


@pytest.mark.parametrize("n", [0, 1, 4, 8, 10])
def test_zeta_sub_really_is_the_subset_sum(n, rng):
    """f(S) = sum over T subset of S of f(T), checked by enumeration."""
    f = rng.integers(0, 1 << 20, size=(1 << n,), dtype=np.uint64).astype(np.uint32)
    got = np_of(yates.transform(mx.array(f), "ZETA_SUB"))
    want = np.array(
        [sum(int(f[t]) for t in range(1 << n) if (t & s) == t) % (1 << 32)
         for s in range(1 << n)], dtype=np.uint32)
    assert np.array_equal(got, want)


@pytest.mark.parametrize("n", [0, 1, 4, 8, 10])
def test_zeta_sup_really_is_the_superset_sum(n, rng):
    f = rng.integers(0, 1 << 20, size=(1 << n,), dtype=np.uint64).astype(np.uint32)
    got = np_of(yates.transform(mx.array(f), "ZETA_SUP"))
    want = np.array(
        [sum(int(f[t]) for t in range(1 << n) if (t & s) == s) % (1 << 32)
         for s in range(1 << n)], dtype=np.uint32)
    assert np.array_equal(got, want)


@pytest.mark.parametrize("n", [3, 7, 10])
def test_xor_convolution_in_float32(n, rng):
    f = rng.standard_normal(1 << n).astype(np.float32)
    g = rng.standard_normal(1 << n).astype(np.float32)
    F = yates.transform(mx.array(f), "WHT")
    G = yates.transform(mx.array(g), "WHT")
    h = np_of(yates.transform(F * G, "WHT")) / (1 << n)
    want = yates.naive_convolution(f.astype(np.float64), g.astype(np.float64), "xor")
    assert np.allclose(h, want, rtol=1e-4, atol=1e-4 * max(1.0, float(np.abs(want).max())))


@pytest.mark.parametrize("op,fwd,inv", [
    ("or", "ZETA_SUB", "MOB_SUB"),
    ("and", "ZETA_SUP", "MOB_SUP"),
])
@pytest.mark.parametrize("n", [0, 1, 4, 8, 10])
def test_set_convolutions_exact_in_z_mod_2_64(op, fwd, inv, n, rng):
    f = rand_uint(rng, (1 << n,), np.uint64)
    g = rand_uint(rng, (1 << n,), np.uint64)
    F = yates.transform(mx.array(f), fwd)
    G = yates.transform(mx.array(g), fwd)
    h = np_of(yates.transform(F * G, inv))
    assert np.array_equal(h, yates.naive_convolution(f, g, op))


@pytest.mark.parametrize("n", [0, 1, 4, 8, 10])
def test_xor_convolution_exact_in_z_mod_2_64(n, rng):
    f = rand_uint(rng, (1 << n,), np.uint64)
    g = rand_uint(rng, (1 << n,), np.uint64)
    F = yates.transform(mx.array(f), "WHT")
    G = yates.transform(mx.array(g), "WHT")
    n_times_h = np_of(yates.transform(F * G, "WHT"))
    with np.errstate(over="ignore"):
        want = np.uint64(1 << n) * yates.naive_convolution(f, g, "xor")
    assert np.array_equal(n_times_h, want)


@pytest.mark.parametrize("op,fwd,inv", [
    ("or", "ZETA_SUB", "MOB_SUB"),
    ("and", "ZETA_SUP", "MOB_SUP"),
])
@pytest.mark.parametrize("n", [3, 7, 10])
def test_set_convolutions_float32(op, fwd, inv, n, rng):
    f = rng.standard_normal(1 << n).astype(np.float32)
    g = rng.standard_normal(1 << n).astype(np.float32)
    F = yates.transform(mx.array(f), fwd)
    G = yates.transform(mx.array(g), fwd)
    h = np_of(yates.transform(F * G, inv)).astype(np.float64)
    want = yates.naive_convolution(f.astype(np.float64), g.astype(np.float64), op)
    assert np.allclose(h, want, rtol=1e-3,
                       atol=1e-3 * max(1.0, float(np.abs(want).max())))
