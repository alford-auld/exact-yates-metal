"""Counting k-colourings by inclusion-exclusion, on the Yates kernel.

With ``a(S) = [S is independent]``,

    i(S) = (ZETA_SUB a)(S)                      # independent subsets of S
    c_k  = sum_S (-1)^(n-|S|) i(S)^k            # ordered k-tuples of
                                                # independent sets covering V

and ``c_k`` is exactly the subset-Mobius transform of ``S -> i(S)^k`` evaluated
at ``S = V``.  ``G`` is k-colourable iff ``c_k > 0``.

Choice of modulus -- the thing the brief asks to confirm before choosing
-------------------------------------------------------------------------
The task brief suggests the existing kernel can be run mod p "with a trivially
different generator, or with a post-pass reduction".  Neither is right, and
neither is needed:

* **Not a generator change.** A generator is a 2x2 matrix over the element ring.
  Reduction mod p is a property of the *ring*, not of the matrix, and it is not
  a linear map over Z/2^64, so no choice of four constants produces it.  Working
  mod p would mean a new kernel that carries a modulus and does a conditional
  subtract after each butterfly.
* **Not a post-pass reduction either**, at the prime size the brief proposes.
  The dispatcher fuses up to 13 stages into one device pass, and Mobius values
  can grow by 2x per stage, so with p near 2^62 an intermediate reaches 2^75
  and overflows long before the pass boundary where the reduction would happen.

What actually works, with **no kernel change at all**: keep the primes small
enough that the entire transform is exact over ``Z``, and reduce once at the
end.  Concretely with ``p < 2^31``:

* ``i(S) <= 2^n``, so the zeta runs in ``uint32`` and is exact outright -- and
  it is computed *once*, shared by every modulus.
* ``i(S)^k mod p`` needs products ``< 2^62``: one ``ulong`` multiply, no
  128-bit arithmetic (:mod:`count.metal`).
* The Mobius/alternating sum of ``2^n`` terms each ``< p`` has magnitude
  ``< 2^n * 2^31 <= 2^62 < 2^63`` for every feasible ``n``, so the unmodified
  ``uint64`` kernel returns the *exact integer* in two's complement.  One
  reduction mod p at the very end finishes it.

This is verified directly in
``tests/chromatic/test_modulus_design.py``.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import numpy as np
import yates

from .graph import Graph
from .independent import indicator_gpu

_METAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "count.metal")

#: Largest prime bit-width the "exact over Z, reduce at the end" scheme allows.
#: 31 keeps ``p*p < 2^62`` for the pointwise modmul, and ``2^n * p < 2^63`` for
#: the alternating sum at every n the memory ceiling permits.
PRIME_BITS = 31


def max_prime_bits(n: int) -> int:
    """Bits of headroom available for the modulus at ``n`` vertices."""
    return min(PRIME_BITS, 62 - n)


@functools.lru_cache(maxsize=1)
def _source() -> str:
    with open(_METAL_PATH) as fh:
        text = fh.read()
    marker = "// ===== @section powmod ====="
    return text.split(marker, 1)[1]


@functools.lru_cache(maxsize=1)
def _powmod_kernel():
    return mx.fast.metal_kernel(
        name="chromatic_powmod",
        input_names=["inp", "par"],
        output_names=["out"],
        source=_source(),
        ensure_row_contiguous=True,
    )


# --------------------------------------------------------------------------
# step 1: one zeta over the whole cube, shared by every modulus and every k
# --------------------------------------------------------------------------


def independent_counts(g: Graph) -> mx.array:
    """``i(S)`` for all ``S``: the number of independent subsets of ``S``.

    Exact in ``uint32`` because ``i(S) <= 2^|S| <= 2^n``.  This is the only
    butterfly the whole algorithm runs, and it does not depend on ``k`` or on
    the modulus, so it is computed once and reused.
    """
    return yates.zeta_sub(indicator_gpu(g, dtype=mx.uint32))


# --------------------------------------------------------------------------
# step 2: pointwise k-th power (the only modular multiplication)
# --------------------------------------------------------------------------


def _params(modulus: Optional[int], k: int, n: int) -> mx.array:
    p = 0 if modulus is None else int(modulus)
    return mx.array(np.array([p, k, n], dtype=np.uint64))


def power_terms(counts: mx.array, k: int, n: int, modulus: Optional[int] = None,
                signed: bool = False) -> mx.array:
    """``i(S)^k`` mod ``modulus`` (mod 2^64 when ``modulus`` is None).

    With ``signed=True`` the inclusion-exclusion sign ``(-1)^(n-|S|)`` is folded
    in, so the alternating sum becomes a plain sum.
    """
    if modulus is not None and modulus.bit_length() > max_prime_bits(n):
        raise ValueError(
            f"modulus {modulus} needs {modulus.bit_length()} bits, but only "
            f"{max_prime_bits(n)} are safe at n={n}: the alternating sum of 2^n "
            "terms must stay below 2^63 for the unmodified uint64 kernel to "
            "return the exact integer"
        )
    size = counts.size
    threads = min(256, size)
    (out,) = _powmod_kernel()(
        inputs=[counts, _params(modulus, k, n)],
        template=[("MODULAR", 0 if modulus is None else 1),
                  ("SIGNED", 1 if signed else 0)],
        grid=(size, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(size,)],
        output_dtypes=[mx.uint64],
    )
    return out


# --------------------------------------------------------------------------
# step 3: the alternating sum, two ways
# --------------------------------------------------------------------------


def _reduce_scalar(raw: int, modulus: Optional[int]) -> int:
    """Interpret a uint64 word and reduce it into the target ring."""
    if modulus is None:
        return raw % (1 << 64)                       # already the answer mod 2^64
    signed = raw - (1 << 64) if raw >= (1 << 63) else raw
    return signed % modulus


def c_k_reduction(counts: mx.array, k: int, n: int,
                  modulus: Optional[int] = None) -> int:
    """``c_k`` via the direct alternating reduction (no Mobius transform).

    Only the value at ``S = V`` is wanted, so the full inverse transform is
    unnecessary: fold the sign into the pointwise power and sum.  This is the
    production path -- it needs one ``uint64`` array rather than two.
    """
    terms = power_terms(counts, k, n, modulus, signed=True)
    raw = int(mx.sum(terms).item())
    return _reduce_scalar(raw, modulus)


def c_k_mobius(counts: mx.array, k: int, n: int,
               modulus: Optional[int] = None) -> int:
    """``c_k`` via the full subset-Mobius transform, read at the top index.

    Mathematically identical to :func:`c_k_reduction` and materially more
    expensive; kept because the brief requires the two to agree bitwise, which
    is a strong check on both the kernel and the sign convention.
    """
    terms = power_terms(counts, k, n, modulus, signed=False)
    full = yates.mobius_sub(terms)
    raw = int(full[-1].item())
    return _reduce_scalar(raw, modulus)


def c_k_raw_words(counts: mx.array, k: int, n: int,
                  modulus: Optional[int] = None) -> Tuple[int, int]:
    """The two paths' raw ``uint64`` words, for the bitwise-agreement test."""
    terms_signed = power_terms(counts, k, n, modulus, signed=True)
    reduction = int(mx.sum(terms_signed).item())
    terms = power_terms(counts, k, n, modulus, signed=False)
    mobius = int(yates.mobius_sub(terms)[-1].item())
    return reduction, mobius


# --------------------------------------------------------------------------
# reference implementation (exact integers, no GPU, no modulus)
# --------------------------------------------------------------------------


def c_k_exact_reference(g: Graph, k: int) -> int:
    """``c_k`` as an exact Python integer.  O(2^n * n); the test oracle."""
    n = g.n
    i_of_s = _zeta_reference(g)
    total = 0
    for s in range(1 << n):
        term = i_of_s[s] ** k
        total += term if (n - bin(s).count("1")) % 2 == 0 else -term
    return total


def _zeta_reference(g: Graph) -> list:
    """``i(S)`` by the textbook in-place subset zeta, in Python integers."""
    n = g.n
    a = [1 if g.is_independent(s) else 0 for s in range(1 << n)]
    for v in range(n):
        bit = 1 << v
        for s in range(1 << n):
            if s & bit:
                a[s] += a[s ^ bit]
    return a


def proper_colourings_brute_force(g: Graph, k: int) -> int:
    """Number of proper k-colourings: maps ``V -> [k]`` with no monochromatic edge.

    This is **not** ``c_k``.  ``c_k`` counts ordered k-tuples of independent sets
    whose union is ``V``, which allows the sets to overlap; a proper colouring
    corresponds to a tuple of *disjoint* such sets.  For K_3 with k=4 there are
    24 proper colourings but ``c_4 = 60``.

    The two agree on what matters here and nothing more: a cover by k
    independent sets can always be made disjoint (send each vertex to the first
    set containing it), so ``c_k > 0`` iff a proper k-colouring exists.  That
    equivalence is what the brute-force cross-check tests.  O(k^n).
    """
    import itertools

    n = g.n
    if n == 0:
        return 1
    edges = g.edges
    total = 0
    for assignment in itertools.product(range(k), repeat=n):
        if all(assignment[u] != assignment[v] for u, v in edges):
            total += 1
    return total


def is_k_colourable_brute_force(g: Graph, k: int) -> bool:
    """Does a proper k-colouring exist?  O(k^n) with early exit."""
    import itertools

    n = g.n
    if n == 0:
        return True
    if k == 0:
        return False
    edges = g.edges
    for assignment in itertools.product(range(k), repeat=n):
        if all(assignment[u] != assignment[v] for u, v in edges):
            return True
    return False


def chromatic_number_brute_force(g: Graph, max_k: Optional[int] = None) -> int:
    """chi(G) by trying k = 0, 1, 2, ... exhaustively.  Tiny graphs only."""
    limit = g.n if max_k is None else max_k
    for k in range(limit + 1):
        if is_k_colourable_brute_force(g, k):
            return k
    raise AssertionError("every graph is n-colourable; unreachable")


def covers_brute_force(g: Graph, k: int) -> int:
    """``c_k`` by direct enumeration of k-tuples of independent sets.

    Exactly the quantity the inclusion-exclusion formula computes, obtained
    without any transform.  Costs ``i(V)^k``, so tiny graphs and small k only.
    """
    import itertools

    from .independent import independent_sets

    sets = independent_sets(g)
    full = (1 << g.n) - 1
    if k == 0:
        return 1 if full == 0 else 0
    return sum(1 for tup in itertools.product(sets, repeat=k)
               if _union(tup) == full)


def _union(masks) -> int:
    u = 0
    for m in masks:
        u |= m
    return u
