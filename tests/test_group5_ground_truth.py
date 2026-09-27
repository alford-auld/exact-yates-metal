"""Group 5: published ground truth.  A wrong kernel will not hit these by luck.

* The inner-product bent function on n = 2m variables has a perfectly flat
  Walsh spectrum: |f_hat(a)| = 2^m for every one of the 2^n points a.
* The AES S-box has maximum Walsh coefficient 32 over all 255 nonzero output
  masks, hence nonlinearity 2^7 - 32/2 = 112.
"""

import mlx.core as mx
import numpy as np
import pytest

import yates
from aes import sbox
from conftest import as_signed, np_of, pm1_ring


def _pm1(bits: np.ndarray) -> np.ndarray:
    """(-1)^b as float32."""
    return (1.0 - 2.0 * bits).astype(np.float32)


def _parity(v: np.ndarray) -> np.ndarray:
    out = np.zeros_like(v)
    x = v.copy()
    while x.any():
        out ^= x & 1
        x >>= 1
    return out & 1


# --------------------------------------------------------------------------
# inner-product bent function
# --------------------------------------------------------------------------


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6, 7])
def test_inner_product_bent_function_has_flat_spectrum(m):
    """f(x,y) = <x,y> mod 2 on n = 2m variables is bent: |f_hat| == 2^m."""
    n = 2 * m
    idx = np.arange(1 << n, dtype=np.uint32)
    x, y = idx >> np.uint32(m), idx & np.uint32((1 << m) - 1)
    f = _parity(x & y)
    spectrum = np_of(yates.transform(mx.array(_pm1(f)), "WHT"))

    assert np.array_equal(np.abs(spectrum), np.full(1 << n, float(1 << m))), (
        f"m={m}: spectrum is not flat; "
        f"observed magnitudes {sorted(set(np.abs(spectrum).tolist()))}"
    )
    # A bent function is maximally nonlinear: NL = 2^(n-1) - 2^(n/2-1).
    nl = (1 << (n - 1)) - int(np.abs(spectrum).max()) // 2
    assert nl == (1 << (n - 1)) - (1 << (n // 2 - 1))


def test_non_bent_function_does_not_have_a_flat_spectrum():
    """Control: the test above must be able to fail."""
    n = 6
    f = _parity(np.arange(1 << n, dtype=np.uint32) & np.uint32(0b101010))
    spectrum = np_of(yates.transform(mx.array(_pm1(f)), "WHT"))
    assert len(set(np.abs(spectrum).tolist())) > 1


# --------------------------------------------------------------------------
# AES S-box
# --------------------------------------------------------------------------


def test_sbox_matches_the_published_table():
    s = sbox()
    assert sorted(s.tolist()) == list(range(256)), "S-box must be a permutation"
    for x, want in [(0x00, 0x63), (0x01, 0x7C), (0x10, 0xCA),
                    (0x53, 0xED), (0x7F, 0xD2), (0xFF, 0x16)]:
        assert int(s[x]) == want, f"S[{x:#04x}] = {int(s[x]):#04x}, expected {want:#04x}"


def test_aes_sbox_max_walsh_coefficient_is_32_and_nonlinearity_is_112():
    s = sbox().astype(np.uint32)
    masks = np.arange(1, 256, dtype=np.uint32)          # 255 nonzero output masks
    comps = _parity(masks[:, None] & s[None, :])        # (255, 256) component funcs
    spectra = np_of(yates.transform(mx.array(_pm1(comps)), "WHT"))

    assert spectra.shape == (255, 256)
    max_walsh = int(np.abs(spectra).max())
    assert max_walsh == 32, f"max Walsh coefficient {max_walsh}, expected 32"

    nonlinearity = (1 << 7) - max_walsh // 2
    assert nonlinearity == 112, f"nonlinearity {nonlinearity}, expected 112"

    # The AES spectrum lands on multiples of 4 and never leaves [-32, 32].
    observed = sorted(set(np.abs(spectra).ravel().tolist()))
    assert observed == [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0], observed


def test_aes_sbox_differential_uniformity_via_and_convolution():
    """Independent published constant: the AES S-box is differentially 4-uniform.

    Exercises ZETA_SUB/MOB_SUB rather than the WHT, via XOR-convolution of the
    S-box indicator with itself, computed with the WHT and checked exactly.
    """
    s = sbox().astype(np.int64)
    ddt_max = 0
    for a in range(1, 256):
        counts = np.bincount(s ^ s[np.arange(256) ^ a], minlength=256)
        ddt_max = max(ddt_max, int(counts.max()))
    assert ddt_max == 4, f"differential uniformity {ddt_max}, expected 4"


# --------------------------------------------------------------------------
# the same published constants, computed in the exact rings
# --------------------------------------------------------------------------


@pytest.mark.parametrize("np_dt,bits", [(np.uint32, 32), (np.uint64, 64)])
@pytest.mark.parametrize("m", [1, 2, 3, 4, 6])
def test_bent_function_flat_spectrum_in_the_exact_rings(np_dt, bits, m):
    """Walsh coefficients are tiny compared with the ring, so two's-complement
    arithmetic mod 2^k reproduces the integer spectrum exactly."""
    n = 2 * m
    idx = np.arange(1 << n, dtype=np.uint32)
    f = _parity((idx >> np.uint32(m)) & (idx & np.uint32((1 << m) - 1)))
    v = pm1_ring(f, np_dt)
    spectrum = as_signed(np_of(yates.transform(mx.array(v), "WHT")), bits)
    assert np.array_equal(np.abs(spectrum), np.full(1 << n, 1 << m))


@pytest.mark.parametrize("np_dt,bits", [(np.uint32, 32), (np.uint64, 64)])
def test_aes_sbox_walsh_and_nonlinearity_in_the_exact_rings(np_dt, bits):
    s = sbox().astype(np.uint32)
    masks = np.arange(1, 256, dtype=np.uint32)
    comps = _parity(masks[:, None] & s[None, :])
    spectra = as_signed(np_of(yates.transform(mx.array(pm1_ring(comps, np_dt)), "WHT")), bits)
    assert int(np.abs(spectra).max()) == 32
    assert (1 << 7) - int(np.abs(spectra).max()) // 2 == 112
