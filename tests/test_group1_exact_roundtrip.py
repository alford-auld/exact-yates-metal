"""Group 1: exact round trip over Z/2^32 and Z/2^64.  Zero tolerance.

The zeta and Mobius generators are unipotent (det 1, Smith normal form the
identity), so they are automorphisms of Z/2^k.  Composing a zeta with its
Mobius inverse must reproduce the input bit-for-bit, with no headroom.
"""

import mlx.core as mx
import numpy as np
import pytest

import yates
from conftest import UINT_DTYPES, np_of, rand_uint

PAIRS = [("ZETA_SUB", "MOB_SUB"), ("ZETA_SUP", "MOB_SUP")]
N_RANGE = list(range(0, 21))


@pytest.mark.parametrize("fwd,inv", PAIRS)
@pytest.mark.parametrize("mx_dt,np_dt,bits", UINT_DTYPES)
@pytest.mark.parametrize("n", N_RANGE)
def test_zeta_mobius_roundtrip_is_bitwise_exact(fwd, inv, mx_dt, np_dt, bits, n, rng):
    x = rand_uint(rng, (1 << n,), np_dt)
    a = mx.array(x)
    back = yates.transform(yates.transform(a, fwd), inv)
    assert np.array_equal(np_of(back), x), f"{inv}({fwd}(x)) != x for n={n}, {bits} bits"

    # ...and in the other order: both are automorphisms.
    back2 = yates.transform(yates.transform(a, inv), fwd)
    assert np.array_equal(np_of(back2), x)


@pytest.mark.parametrize("fwd,inv", PAIRS)
@pytest.mark.parametrize("mx_dt,np_dt,bits", UINT_DTYPES)
def test_roundtrip_exact_on_ring_extremes(fwd, inv, mx_dt, np_dt, bits, rng):
    """All-ones, all-zeros and alternating patterns, where wraparound is certain."""
    n = 12
    full = (1 << bits) - 1
    patterns = [
        np.full(1 << n, full, dtype=np_dt),
        np.zeros(1 << n, dtype=np_dt),
        np.array([full if i % 2 else 0 for i in range(1 << n)], dtype=np_dt),
        np.array([np_dt(1) << np_dt(i % bits) for i in range(1 << n)], dtype=np_dt),
    ]
    for x in patterns:
        back = yates.transform(yates.transform(mx.array(x), fwd), inv)
        assert np.array_equal(np_of(back), x)


@pytest.mark.parametrize("mx_dt,np_dt,bits", UINT_DTYPES)
def test_guaranteed_bits_api_reports_full_width_for_unipotent(mx_dt, np_dt, bits):
    for name in ("ZETA_SUB", "MOB_SUB", "ZETA_SUP", "MOB_SUP"):
        assert yates.is_unipotent(name)
        assert yates.smith_normal_form_2x2(yates.get(name).m) == (1, 1)
        for n in (0, 1, 7, 20):
            assert yates.bit_loss(name, n) == 0
            assert yates.guaranteed_bits(name, n, bits) == bits


@pytest.mark.parametrize("mx_dt,np_dt,bits", UINT_DTYPES)
@pytest.mark.parametrize("n", [1, 5, 9])
def test_wraparound_matches_arbitrary_precision_arithmetic(mx_dt, np_dt, bits, n, rng):
    """The kernel really computes mod 2^k, not "mod 2^k when nothing overflows"."""
    from conftest import exact_transform_python

    x = rand_uint(rng, (1 << n,), np_dt)
    for name in yates.VARIANTS:
        want = exact_transform_python(x, name, 1 << bits)
        got = np_of(yates.transform(mx.array(x), name))
        assert np.array_equal(got.astype(object), want), f"{name} n={n} {bits}-bit"


@pytest.mark.parametrize("n", [3, 10, 14])
def test_batched_roundtrip_is_exact(n, rng):
    x = rand_uint(rng, (5, 3, 1 << n), np.uint32)
    back = yates.transform(yates.transform(mx.array(x), "ZETA_SUB"), "MOB_SUB")
    assert np.array_equal(np_of(back), x)


@pytest.mark.parametrize("fwd,inv", PAIRS)
@pytest.mark.parametrize("n", [0, 1, 4, 10, 14])
def test_roundtrip_float32(fwd, inv, n, rng):
    """float32 round trip: exact for small integers, within tolerance otherwise.

    Unlike the ring case this is *not* a bit-for-bit guarantee -- float32 has no
    exactness contract -- so integer-valued inputs whose partial sums stay below
    2^24 are checked exactly, and random inputs only to tolerance.
    """
    exact_in = rng.integers(0, 16, size=(3, 1 << n)).astype(np.float32)
    back = yates.transform(yates.transform(mx.array(exact_in), fwd), inv)
    assert np.array_equal(np_of(back), exact_in), "integer-valued float32 must be exact"

    x = rng.standard_normal((3, 1 << n)).astype(np.float32)
    back = np_of(yates.transform(yates.transform(mx.array(x), fwd), inv))
    assert np.allclose(back, x, rtol=1e-4, atol=1e-4 * max(1.0, 2.0 ** (n / 2)))
