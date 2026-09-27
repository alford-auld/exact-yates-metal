"""Multi-modular verification, and unconditional CRT reconstruction of c_k.

The single-modulus (2^64) test is *one-sided*: a nonzero residue proves
``c_k != 0``, but a zero residue may be a nonzero ``c_k`` that happens to be a
multiple of 2^64.  This module removes that gap two ways.

**Probabilistic.**  Recompute ``c_k`` modulo ``r`` primes drawn uniformly at
random from ``[2^30, 2^31)``.  If ``c_k != 0`` it has at most
``D = floor(log(c_k) / log(2^30))`` distinct prime divisors in that range, and
the number of primes available is, by Rosser-Schoenfeld,

    M = pi(2^31) - pi(2^30) > 2^31/ln(2^31) - 1.25506 * 2^30/ln(2^30) > 3.5e7

so a single draw hides a nonzero ``c_k`` with probability at most ``D/M``, and
``r`` independent draws with probability at most ``(D/M)^r``.  The assumption
is exactly that the primes are drawn independently of the instance -- true here,
since the seed does not depend on the graph.

**Unconditional.**  ``c_k`` counts ordered k-tuples of independent sets, so
``0 <= c_k <= i(V)^k`` where ``i(V)`` is the total number of independent sets --
a quantity the algorithm already computes.  Once the product of the moduli
exceeds ``i(V)^k``, CRT reconstructs ``c_k`` as an exact integer and there is no
probability left in the statement at all.  This is usually cheap: ``i(V)`` is
typically far below ``2^n`` (the Chvatal graph has 127 independent sets on 12
vertices, so ``c_4 <= 127^4 < 2^28`` needs a single prime).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import mlx.core as mx

from .count import PRIME_BITS, c_k_reduction, max_prime_bits

#: Rigorous lower bound on the number of primes in [2^30, 2^31).
#: pi(x) > x/ln x  and  pi(x) < 1.25506 x/ln x   (Rosser-Schoenfeld).
PRIME_POOL_LOWER_BOUND = int(
    2**31 / math.log(2**31) - 1.25506 * 2**30 / math.log(2**30)
)

_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_prime(n: int) -> bool:
    """Miller-Rabin; the base set is deterministic below 3.3e24."""
    if n < 2:
        return False
    for p in _MR_BASES:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR_BASES:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def random_primes(count: int, seed: int = 0, bits: int = PRIME_BITS) -> List[int]:
    """``count`` distinct primes drawn uniformly at random from ``[2^(bits-1), 2^bits)``."""
    import random

    if count <= 0:
        return []
    rng = random.Random(seed)
    lo, hi = 1 << (bits - 1), (1 << bits) - 1
    out: List[int] = []
    seen = set()
    guard = 0
    while len(out) < count:
        guard += 1
        if guard > 10000 * count:
            raise RuntimeError(f"could not find {count} primes in [{lo},{hi}]")
        cand = rng.randrange(lo, hi) | 1
        if cand in seen or not is_prime(cand):
            continue
        seen.add(cand)
        out.append(cand)
    return out


# --------------------------------------------------------------------------
# CRT
# --------------------------------------------------------------------------


def crt(residues: Sequence[int], moduli: Sequence[int]) -> tuple:
    """Combine residues into ``(value, modulus)`` with ``0 <= value < modulus``."""
    if len(residues) != len(moduli):
        raise ValueError("residues and moduli must have the same length")
    if not moduli:
        return 0, 1
    value, modulus = int(residues[0]) % int(moduli[0]), int(moduli[0])
    for r, m in zip(residues[1:], moduli[1:]):
        m = int(m)
        g = math.gcd(modulus, m)
        if g != 1:
            raise ValueError(f"moduli {modulus} and {m} are not coprime")
        # value + modulus * t == r (mod m)
        t = ((int(r) - value) * pow(modulus, -1, m)) % m
        value += modulus * t
        modulus *= m
    return value % modulus, modulus


def bound_on_c_k(num_independent_sets: int, k: int, n: Optional[int] = None) -> int:
    """An upper bound on ``c_k``, from whichever of two arguments is tighter.

    * ``c_k <= i(V)^k`` -- every coordinate of the tuple is an independent set.
    * ``c_k <= (2^k - 1)^n`` -- a covering is determined by which *nonempty*
      subset of the k slots contains each of the n vertices.  (Not every such
      assignment yields independent slots, hence an inequality; for the edgeless
      graph it is an equality, which is why ``c_k = (2^k-1)^n`` there.)

    The second is the tighter one exactly when ``i(V)`` is large, i.e. on sparse
    graphs -- which are also the instances where the first bound is worst and
    the CRT prime count is highest.  Passing ``n`` is therefore worth one or two
    primes on the expensive cases and never costs anything.
    """
    by_sets = max(1, int(num_independent_sets) ** int(k))
    if n is None:
        return by_sets
    by_slots = max(1, (2 ** int(k) - 1) ** int(n))
    return min(by_sets, by_slots)


def primes_needed_for_exact(num_independent_sets: int, k: int,
                            bits: int = PRIME_BITS,
                            n: Optional[int] = None) -> int:
    """How many primes make the CRT reconstruction of ``c_k`` unconditional."""
    bound = bound_on_c_k(num_independent_sets, k, n)
    prod, count = 1, 0
    smallest = 1 << (bits - 1)
    while prod <= bound:
        prod *= smallest
        count += 1
    return max(1, count)


def failure_probability_bound(num_independent_sets: int, k: int, num_primes: int,
                              bits: int = PRIME_BITS,
                              n: Optional[int] = None) -> float:
    """Upper bound on Pr[all ``num_primes`` residues vanish while ``c_k != 0``]."""
    bound = bound_on_c_k(num_independent_sets, k, n)
    max_divisors = max(1, int(math.log(bound) / math.log(1 << (bits - 1))))
    per_prime = min(1.0, max_divisors / PRIME_POOL_LOWER_BOUND)
    return per_prime ** num_primes


def forced_two_adic_valuation(chi: int) -> int:
    """``v2(chi!) = chi - popcount(chi)``: the part of ``v2(c_chi)`` forced by
    symmetry, and hence a lower bound on it.

    **Lemma.** ``chi! | c_chi(G)`` for every graph G.

    *Proof.* Let ``(S_1, ..., S_chi)`` be a covering of V by independent sets
    with ``S_i = S_j`` for some ``i != j``.  Dropping ``S_j`` still covers V, so
    V is covered by ``chi - 1`` independent sets; and a covering by m
    independent sets implies m-colourability (give each vertex the least index
    covering it -- every colour class is a subset of an independent set).  That
    contradicts the minimality of chi.  So at ``k = chi`` the components of a
    covering tuple are pairwise distinct, the coordinate-permutation action of
    ``S_chi`` on the coverings is free, and the orbit-counting gives
    ``chi! | c_chi``.  (Equality for ``K_n``: at ``k = n`` every slot must hold
    a distinct singleton, so ``c_n(K_n) = n!``.)  QED

    Why this matters for the one-sided guarantee: a false negative that corrupts
    the *reported* chromatic number needs ``2^64 | c_chi``.  By Legendre this
    lemma forces only ``chi - popcount(chi)`` of those 64 bits, which is at most
    25 for any chi a 29-vertex machine can reach.  The rest would have to come
    from the unordered cofactor ``c_chi / chi!``.  See
    :func:`first_clique_defeating_modulus` for where the forced part alone
    suffices.
    """
    if chi < 0:
        raise ValueError(f"chi must be non-negative, got {chi}")
    return chi - bin(chi).count("1")


def first_clique_defeating_modulus(bits: int = 64) -> int:
    """Smallest ``n`` with ``2^bits | c_chi(K_n)``, i.e. ``2^bits | n!``.

    For the clique family the lemma is tight (``c_n(K_n) = n!``), so this is
    exact rather than a search: ``n = 66`` for a 64-bit modulus, since
    ``v2(66!) = 66 - 2 = 64`` while ``v2(65!) = v2(64!) = 63``.
    """
    n = 1
    while forced_two_adic_valuation(n) < bits:
        n += 1
    return n


# --------------------------------------------------------------------------
# the multi-modular driver
# --------------------------------------------------------------------------


@dataclass
class ModularResult:
    """Everything known about ``c_k`` after the modular computations."""

    k: int
    primes: List[int]
    residues: List[int]
    residue_mod_2_64: Optional[int] = None
    exact_value: Optional[int] = None         # set when CRT was conclusive
    combined_modulus: int = 1
    bound_on_value: Optional[int] = None
    failure_bound: Optional[float] = None

    @property
    def any_nonzero(self) -> bool:
        """Any nonzero residue, modulo anything, proves ``c_k != 0``.

        Sound in one direction only, which is the whole point: ``c_k`` is a
        count, so a single nonzero residue certifies ``c_k > 0``.
        """
        return any(r != 0 for r in self.residues) or bool(self.residue_mod_2_64)

    @property
    def all_zero(self) -> bool:
        return not self.any_nonzero

    @property
    def consistent(self) -> bool:
        """Do the *prime* residues agree on whether ``c_k`` vanishes?

        All of them zero, or all of them nonzero.  Some prime dividing ``c_k``
        while another does not is possible in principle but has probability
        ``<= D/M`` per prime, so at these sizes it indicates a bug and is
        surfaced rather than silently re-rolled.

        The ``mod 2^64`` residue is deliberately excluded: 2 is not a random
        prime, ``c_k`` routinely carries a large power of it, and a mismatch
        there is a *false negative being caught*, not an inconsistency.  See
        :attr:`mod_2_64_is_false_negative`.
        """
        flags = {r != 0 for r in self.residues}
        return len(flags) <= 1

    @property
    def mod_2_64_is_false_negative(self) -> bool:
        """True when ``c_k == 0 mod 2^64`` but a prime modulus proves otherwise.

        Demonstrably reachable: 7 disjoint copies of K_4 at k=30 (28 vertices)
        has ``v2(c_30) = 70``.
        """
        if self.residue_mod_2_64 is None or not self.residues:
            return False
        return self.residue_mod_2_64 == 0 and any(r != 0 for r in self.residues)

    @property
    def colourable(self) -> bool:
        if self.exact_value is not None:
            return self.exact_value > 0
        return self.any_nonzero

    @property
    def certainty(self) -> str:
        if self.exact_value is not None:
            return "exact (CRT reconstruction)"
        if self.any_nonzero:
            return "exact (nonzero residue is a sound certificate)"
        return f"probabilistic (Pr[wrong] <= {self.failure_bound:.3e})"


def c_k_multi_modular(counts: mx.array, k: int, n: int,
                      num_independent_sets: int,
                      num_primes: int = 4,
                      seed: int = 0,
                      exact: bool = False,
                      include_2_64: bool = True) -> ModularResult:
    """Compute ``c_k`` modulo several primes, and reconstruct it when possible.

    Args:
      counts: ``i(S)`` from :func:`apps.chromatic.count.independent_counts`.
      num_independent_sets: ``i(V)``, used for the bound on ``c_k``.
      num_primes: how many random primes to use when ``exact`` is False.
      exact: use however many primes make CRT unconditional.
      include_2_64: also record the fast ``mod 2^64`` residue.
    """
    bits = max_prime_bits(n)
    if bits < 2:
        raise ValueError(f"n={n} leaves no headroom for a modulus")
    bound = bound_on_c_k(num_independent_sets, k, n)
    if exact:
        num_primes = primes_needed_for_exact(num_independent_sets, k, bits, n)
    primes = random_primes(num_primes, seed=seed, bits=bits)
    residues = [c_k_reduction(counts, k, n, modulus=p) for p in primes]

    res = ModularResult(
        k=k, primes=primes, residues=residues,
        residue_mod_2_64=(c_k_reduction(counts, k, n, modulus=None)
                          if include_2_64 else None),
        bound_on_value=bound,
        failure_bound=failure_probability_bound(num_independent_sets, k,
                                                len(primes), bits, n),
    )
    value, modulus = crt(residues, primes)
    res.combined_modulus = modulus
    if modulus > bound:
        # c_k is a non-negative integer strictly below the combined modulus, so
        # its residue *is* its value.  No probability remains.
        res.exact_value = value
        res.failure_bound = 0.0
    return res
