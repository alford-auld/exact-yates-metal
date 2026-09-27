"""Group 2: the Hadamard bit-loss contract, and that the bound is tight.

SNF(H_2) = diag(1, 2), so H_(2^n) has elementary divisors 2^j with multiplicity
binomial(n, j) and det +/- 2^(n 2^(n-1)).  WHT o WHT = 2^n I exactly, so over
Z/2^k a forward-then-inverse WHT determines the input only modulo 2^(k-n).
"""

import math

import mlx.core as mx
import numpy as np
import pytest

import yates
from conftest import UINT_DTYPES, np_of, rand_uint


def test_snf_and_elementary_divisors_match_the_stated_contract():
    assert yates.smith_normal_form_2x2(yates.get("WHT").m) == (1, 2)
    assert not yates.is_unipotent("WHT")
    for n in range(0, 12):
        mult = yates.elementary_divisor_multiplicities("WHT", n)
        assert mult == {2**j: math.comb(n, j) for j in range(n + 1)}
        # det H_(2^n) = +/- 2^(n * 2^(n-1))
        log2_det = sum(j * c for j, c in enumerate(math.comb(n, j) for j in range(n + 1)))
        assert log2_det == n * 2 ** (n - 1) if n else log2_det == 0
        assert yates.bit_loss("WHT", n) == n


@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize("n", [0, 1, 5, 12, 20])
def test_guaranteed_bits_api(bits, n):
    assert yates.guaranteed_bits("WHT", n, bits) == bits - n


@pytest.mark.parametrize("n", list(range(0, 21)))
def test_wht_of_wht_is_N_times_input_over_Z(n, rng):
    """With 64-bit headroom the round trip is the exact integer identity N*x.

    uint64 is the two's-complement carrier: intermediate Walsh coefficients are
    negative, but every value stays below 2^63 in magnitude, so the unsigned
    result is the exact integer result.
    """
    x = rng.integers(0, 1 << 20, size=(1 << n,), dtype=np.uint64)
    y = yates.transform(yates.transform(mx.array(x), "WHT"), "WHT")
    want = (np.uint64(1 << n) * x)
    assert np.array_equal(np_of(y), want)
    # headroom check: the claim above is only meaningful if nothing wrapped
    assert int(want.max()) < (1 << 63)


@pytest.mark.parametrize("mx_dt,np_dt,bits", UINT_DTYPES)
@pytest.mark.parametrize("n", [1, 4, 8, 13, 17])
def test_wht_roundtrip_recovers_exactly_k_minus_n_low_bits(mx_dt, np_dt, bits, n, rng):
    x = rand_uint(rng, (1 << n,), np_dt)
    y = np_of(yates.transform(yates.transform(mx.array(x), "WHT"), "WHT"))

    # y == 2^n * x  (mod 2^bits): the low n bits must be zero...
    assert np.all(y % np_dt(1 << n) == 0)
    # ...and dividing them out recovers x modulo 2^(bits-n), no more, no less.
    keep = bits - n
    recovered = (y >> np_dt(n)).astype(np_dt)
    assert np.array_equal(recovered, x & np_dt((1 << keep) - 1))
    assert yates.guaranteed_bits("WHT", n, bits) == keep


@pytest.mark.parametrize("mx_dt,np_dt,bits", UINT_DTYPES)
@pytest.mark.parametrize("n", [1, 3, 7, 11])
def test_bound_is_tight_exhibit_input_losing_exactly_n_bits(mx_dt, np_dt, bits, n, rng):
    """Two inputs differing only in bit (bits-n) are indistinguishable after the
    round trip -- so at least n bits are lost -- while every lower bit survives,
    so not one bit more than n is lost."""
    x = rand_uint(rng, (1 << n,), np_dt)
    collide = x.copy()
    collide[0] = np_dt(int(x[0]) ^ (1 << (bits - n)))  # flip the lowest lost bit

    rt = lambda v: np_of(yates.transform(yates.transform(mx.array(v), "WHT"), "WHT"))
    y, yc = rt(x), rt(collide)

    assert not np.array_equal(x, collide), "the two inputs must actually differ"
    assert np.array_equal(y, yc), (
        f"n={n}: inputs differing in bit {bits - n} must be indistinguishable "
        "after WHT o WHT, i.e. at least n bits are lost"
    )

    # Flipping any bit strictly below bits-n IS visible: no extra loss.
    lower = x.copy()
    lower[0] = np_dt(int(x[0]) ^ (1 << (bits - n - 1)))
    assert not np.array_equal(rt(lower), y), (
        f"n={n}: bit {bits - n - 1} must survive the round trip; "
        "the loss is exactly n bits, not more"
    )


@pytest.mark.parametrize("n", [0, 1, 5, 10, 14])
def test_wht_of_wht_is_N_times_input_in_float32(n, rng):
    """float32 loses precision rather than low bits; with exact inputs it is exact."""
    x = rng.integers(0, 64, size=1 << n).astype(np.float32)
    y = np_of(yates.transform(yates.transform(mx.array(x), "WHT"), "WHT"))
    assert np.array_equal(y, np.float32(1 << n) * x)

    z = rng.standard_normal(1 << n).astype(np.float32)
    w = np_of(yates.transform(yates.transform(mx.array(z), "WHT"), "WHT"))
    assert np.allclose(w, (1 << n) * z, rtol=1e-4, atol=1e-3 * (1 << n) * 2.0 ** (n / 2) / 64)


def test_float32_has_no_exactness_contract():
    """bit_loss / guaranteed_bits describe the ring only; ring_bits(float32) is 0."""
    assert yates.ring_bits(mx.float32) == 0
    assert yates.ring_bits(mx.uint32) == 32
    assert yates.ring_bits(mx.uint64) == 64
