"""The modulus design claims in apps/chromatic/count.py, verified not assumed.

The brief suggested the Yates kernel could be run mod p "with a trivially
different generator, or with a post-pass reduction".  These tests show why
neither holds, and why keeping the modulus under 2^31 instead lets the kernel
be reused completely unchanged.
"""

import mlx.core as mx
import numpy as np
import pytest
import yates

import apps.chromatic as ch
from apps.chromatic.count import PRIME_BITS, max_prime_bits


def _twos_complement(word: int) -> int:
    return word - (1 << 64) if word >= (1 << 63) else word


def _exact_mobius_top(y: np.ndarray, n: int) -> int:
    """The alternating sum, in Python big integers."""
    total = 0
    for s, v in enumerate(y):
        total += int(v) if (n - bin(s).count("1")) % 2 == 0 else -int(v)
    return total


# --------------------------------------------------------------------------
# claim 1: a generator change cannot produce modular reduction
# --------------------------------------------------------------------------


def test_no_generator_gives_reduction_mod_p():
    """The butterfly is linear over the element ring; ``x mod p`` is not.

    Concretely: whatever 2x2 integer matrix is used, the kernel's output is
    determined by ring arithmetic in Z/2^64, so two inputs congruent mod 2^64
    give congruent outputs.  Reduction mod p is not a function of the residue
    mod 2^64, so no generator can implement it.
    """
    p = 2147483647
    # a and b differ by exactly 2^64 conceptually: as uint64 they are the same
    # word, yet their images mod p would have to differ if the kernel reduced.
    a = np.array([p + 1, 0], dtype=np.uint64)
    got = int(np.array(yates.transform(mx.array(a), "ZETA_SUB"))[1])
    assert got == (p + 1) % (1 << 64), "kernel works in Z/2^64, as documented"
    assert got != (p + 1) % p, "it is emphatically not reducing mod p"

    # And the four named generators are all the matrix supplies: entries only.
    for name in ("ZETA_SUB", "MOB_SUB", "ZETA_SUP", "MOB_SUP", "WHT"):
        assert set(yates.get(name).entries) <= {-1, 0, 1}


# --------------------------------------------------------------------------
# claim 2: at 62-bit primes a post-pass reduction is already too late
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [16, 20, 24])
def test_a_fused_pass_overflows_long_before_its_boundary(n):
    """The dispatcher fuses many stages per device pass, and Mobius values can
    double per stage, so with a 62-bit modulus an intermediate blows past 2^64
    inside the first pass -- there is no pass boundary early enough to reduce at.
    """
    plan = yates.describe_plan(n, 1 << n, mx.uint64)
    stages_in_first_pass = plan["passes"][0]["p"]
    assert stages_in_first_pass >= 10, plan
    worst_bits_after_first_pass = 62 + stages_in_first_pass
    assert worst_bits_after_first_pass > 64, (
        f"with a 62-bit modulus, {stages_in_first_pass} fused stages reach "
        f"2^{worst_bits_after_first_pass} before the first reduction point")


# --------------------------------------------------------------------------
# claim 3: under 2^31, the unmodified kernel is exact over Z
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [4, 8, 12, 16])
def test_unmodified_uint64_mobius_is_exact_over_Z_within_the_bound(n):
    rng = np.random.default_rng(n)
    p = (1 << PRIME_BITS) - 1
    y = rng.integers(0, p, size=1 << n, dtype=np.uint64)
    assert (1 << n) * p < (1 << 63), "the bound this test is about"

    word = int(np.array(yates.transform(mx.array(y), "MOB_SUB"))[-1])
    assert _twos_complement(word) == _exact_mobius_top(y, n)


def test_the_bound_is_necessary_not_merely_sufficient():
    """Violate ``2^n * max|y| < 2^63`` and the two's-complement reading breaks.

    Random values will not do it: the signs ``(-1)^(n-|S|)`` cancel, so a random
    input lands near ``sqrt(2^n) * max|y|`` rather than the worst case.  Zeroing
    the negatively-signed half makes every surviving term reinforce, which is
    what the ``2^n`` factor in the bound is actually guarding against.
    """
    n = 16
    big = (1 << 50) - 12345
    y = np.array([big if (n - bin(s).count("1")) % 2 == 0 else 0
                  for s in range(1 << n)], dtype=np.uint64)

    exact = _exact_mobius_top(y, n)
    assert exact == big * (1 << (n - 1))
    assert abs(exact) > (1 << 63), "this input deliberately exceeds the bound"

    word = int(np.array(yates.transform(mx.array(y), "MOB_SUB"))[-1])
    assert _twos_complement(word) != exact, "the exact integer is no longer recoverable"
    # the kernel is still exact *mod 2^64* -- that never stops being true
    assert word == exact % (1 << 64)


def test_prime_width_keeps_the_pointwise_modmul_in_64_bits():
    """``p < 2^31`` means ``p*p < 2^62``: one ulong multiply, no 128-bit path."""
    for p in ch.random_primes(8, seed=4):
        assert p < (1 << PRIME_BITS)
        assert p * p < (1 << 62)


@pytest.mark.parametrize("n", range(1, 30))
def test_max_prime_bits_is_safe_at_every_feasible_n(n):
    bits = max_prime_bits(n)
    largest = (1 << bits) - 1          # the biggest modulus that width allows
    assert bits >= 20, "still plenty of entropy for the failure bound"
    assert (1 << n) * largest < (1 << 63), "alternating sum must not overflow"
    assert largest * largest < (1 << 62), "pointwise modmul must not overflow"


def test_i_of_s_fits_uint32_so_the_zeta_needs_no_modulus_at_all():
    """``i(S) <= 2^n``, so the one butterfly the algorithm runs is exact in 32
    bits and is shared unchanged across every modulus and every k."""
    g = ch.empty_graph(20)
    counts = ch.independent_counts(g)
    assert counts.dtype == mx.uint32
    assert int(counts[-1].item()) == 2 ** 20 < 2 ** 32


def test_kernel_is_used_unmodified():
    """No fork, no modular variant: the app calls the shipped entry points."""
    import inspect

    from apps.chromatic import count as C
    src = inspect.getsource(C)
    assert "yates.zeta_sub" in src and "yates.mobius_sub" in src
    assert "kernel.metal" not in src
    assert set(yates.kernel._sections()) == {"header", "pass", "copy"}
