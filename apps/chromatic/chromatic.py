"""Exact chromatic number by inclusion-exclusion: bounds, search, top-level API.

    chi(G) = min { k : c_k > 0 },    c_k = sum_S (-1)^(n-|S|) i(S)^k

One subset-zeta over the whole cube produces ``i(S)``; after that each candidate
``k`` costs a pointwise power and one reduction.  O*(2^n) for every graph, with
no search over colourings and no dependence on instance hardness.

The guarantee attached to an answer depends on how ``c_k`` was tested; see
:class:`ChromaticResult.guarantee`.  Every result object carries that statement,
and every CLI run prints it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import mlx.core as mx
import yates

from .count import (c_k_mobius, c_k_reduction, independent_counts,
                    max_prime_bits)
from .graph import Graph, bits
from .independent import count_independent_sets
from .modular import ModularResult, c_k_multi_modular

#: Bytes of device memory needed per subset, for the reduction path:
#: ``i(S)`` as uint32 plus the ``uint64`` power terms, both live at once.
BYTES_PER_SUBSET = 4 + 8
#: The cross-check path additionally materialises the full Mobius transform.
BYTES_PER_SUBSET_WITH_MOBIUS = 4 + 8 + 8


def max_feasible_n(headroom: float = 0.6, with_mobius: bool = False) -> int:
    """Largest ``n`` this machine can hold, from the detected unified memory."""
    budget = int(yates.limits().max_recommended_working_set_size * headroom)
    per = BYTES_PER_SUBSET_WITH_MOBIUS if with_mobius else BYTES_PER_SUBSET
    n = 0
    while (1 << (n + 1)) * per <= budget:
        n += 1
    return n


def check_feasible(n: int, headroom: float = 0.6, with_mobius: bool = False) -> None:
    """Refuse oversized inputs with a message, rather than thrashing."""
    limit = max_feasible_n(headroom, with_mobius)
    if n > limit:
        lim = yates.limits()
        per = BYTES_PER_SUBSET_WITH_MOBIUS if with_mobius else BYTES_PER_SUBSET
        raise MemoryError(
            f"n={n} needs {(1 << n) * per / 2**30:.1f} GiB of device memory "
            f"({per} B per subset x 2^{n} subsets); this machine reports a "
            f"{lim.max_recommended_working_set_size / 2**30:.2f} GiB recommended "
            f"working set, so the limit at headroom={headroom} is n={limit}. "
            "This algorithm is O*(2^n) in memory by construction -- there is no "
            "streaming variant to fall back to."
        )


# --------------------------------------------------------------------------
# cheap bounds
# --------------------------------------------------------------------------


def greedy_clique(g: Graph) -> List[int]:
    """A maximal clique by repeated highest-degree choice.  ``|clique| <= chi``.

    A clique lower bound is deliberately weak on the primary test family: every
    Mycielskian is triangle-free, so this returns 2 no matter how large chi is.
    That is the point of using them -- the search cannot be shortcut by bounds.

    The Lovasz theta bound would be far stronger here (theta is exactly the
    kind of bound that does see Mycielskians) but it needs an SDP solve, which
    is neither cheap nor a dependency worth taking for a bound that only seeds
    a binary search over at most n values.
    """
    best: List[int] = []
    for start in range(g.n):
        clique = [start]
        cand = g.adj[start]
        while cand:
            v = max(bits(cand), key=lambda u: bin(g.adj[u] & cand).count("1"))
            clique.append(v)
            cand &= g.adj[v]
        if len(clique) > len(best):
            best = clique
    return best


def dsatur(g: Graph) -> List[int]:
    """DSATUR colouring.  Returns ``colour[v]``; the colour count bounds chi above."""
    n = g.n
    colour = [-1] * n
    sat: List[set] = [set() for _ in range(n)]
    for _ in range(n):
        v = max((u for u in range(n) if colour[u] < 0),
                key=lambda u: (len(sat[u]), g.degree(u)))
        c = 0
        while c in sat[v]:
            c += 1
        colour[v] = c
        for u in bits(g.adj[v]):
            sat[u].add(c)
    return colour


def verify_colouring(g: Graph, colour: List[int]) -> bool:
    return all(colour[u] != colour[v] for u, v in g.edges)


def bounds(g: Graph) -> Dict[str, object]:
    """``(lower, upper)`` with the witnesses that certify each."""
    if g.n == 0:
        return {"lower": 0, "upper": 0, "clique": [], "colouring": []}
    clique = greedy_clique(g)
    colour = dsatur(g)
    if not verify_colouring(g, colour):
        raise AssertionError("DSATUR produced an improper colouring")
    return {
        "lower": len(clique),
        "upper": len(set(colour)),
        "clique": clique,
        "colouring": colour,
    }


# --------------------------------------------------------------------------
# result
# --------------------------------------------------------------------------


@dataclass
class ChromaticResult:
    graph: str
    n: int
    num_edges: int
    chromatic_number: int
    lower_bound: int
    upper_bound: int
    clique: List[int]
    colouring: List[int]
    num_independent_sets: int
    mode: str
    tested: Dict[int, ModularResult] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def unproved_zero_tests(self) -> List[int]:
        """The ``k`` at which "not colourable" was concluded from a zero residue
        that was *not* an exact reconstruction.

        Only these can be wrong.  A *nonzero* residue modulo anything proves
        ``c_k != 0``, and ``c_k`` is a count, so ``c_k > 0`` and ``G`` really is
        k-colourable -- that direction is sound whatever the modulus.  The
        converse is not: ``c_k == 0 (mod m)`` is consistent with a nonzero
        ``c_k`` divisible by ``m``.  So the answer is unconditional exactly when
        every k that was ruled out was ruled out exactly.
        """
        return sorted(k for k, r in self.tested.items()
                      if not r.colourable and r.exact_value is None)

    @property
    def certain(self) -> bool:
        """True when no step of the search rests on an unproved zero."""
        return not self.unproved_zero_tests

    @property
    def guarantee(self) -> str:
        """The statement that must accompany every reported value.

        Derived from the evidence actually obtained, not from the mode that was
        requested: asking for ``multimodular`` but getting enough primes for CRT
        yields an unconditional answer, and says so.
        """
        chi = self.chromatic_number
        if self.certain:
            ruled_out = [k for k, r in self.tested.items() if not r.colourable]
            how = ("no k below chi needed ruling out (the clique bound already "
                   "matches)" if not ruled_out else
                   f"c_k was reconstructed exactly by CRT for k in {ruled_out}")
            return (
                f"chi = {chi}, UNCONDITIONAL. c_{chi} != 0 is a sound certificate, and "
                f"{how}, so the minimality of {chi} is proved rather than sampled."
            )
        unproved = self.unproved_zero_tests
        if self.mode == "mod2_64":
            return (
                f"chi <= {chi}, ONE-SIDED -- this is an upper bound, not a proved "
                f"chromatic number. c_{chi} != 0 proves a {chi}-colouring exists (one is "
                f"attached). But c_k == 0 mod 2^64 at k in {unproved} does not prove "
                "c_k == 0: it is also consistent with a nonzero c_k that is a multiple "
                "of 2^64, so a smaller chi cannot be ruled out. Use mode='multimodular' "
                "or mode='exact' to close the gap."
            )
        worst = max((self.tested[k].failure_bound or 0.0) for k in unproved)
        return (
            f"chi <= {chi}, and achievable (a {chi}-colouring is attached). c_{chi} != 0 is "
            f"a sound certificate; ruling out k in {unproved} rests on multi-modular "
            f"agreement, with Pr[some such c_k is nonzero yet vanished modulo every "
            f"prime drawn] <= {worst:.3e}, assuming the primes were drawn independently "
            "of the instance."
        )

    def summary(self) -> str:
        lines = [
            f"{self.graph}: n={self.n}, m={self.num_edges}, "
            f"independent sets i(V)={self.num_independent_sets}",
            f"  bounds: clique >= {self.lower_bound}, DSATUR <= {self.upper_bound}",
            f"  chi = {self.chromatic_number}   [mode={self.mode}]",
            f"  {self.guarantee}",
        ]
        if self.timings:
            total = sum(self.timings.values())
            parts = "  ".join(f"{k} {v * 1e3:.1f}ms ({v / total * 100:.0f}%)"
                              for k, v in self.timings.items())
            lines.append(f"  time: {total * 1e3:.1f}ms total -- {parts}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# the algorithm
# --------------------------------------------------------------------------


def chromatic_number(
    g: Graph,
    mode: str = "exact",
    num_primes: int = 4,
    seed: int = 0,
    headroom: float = 0.6,
    cross_check_mobius: bool = False,
    time_it: bool = True,
) -> ChromaticResult:
    """chi(G) by inclusion-exclusion, with the guarantee recorded in the result.

    Args:
      mode: ``"exact"`` uses enough primes for unconditional CRT reconstruction;
        ``"multimodular"`` uses ``num_primes`` random primes and reports a
        failure bound; ``"mod2_64"`` is the fast one-sided single-modulus test.
      cross_check_mobius: also compute every ``c_k`` through the full
        subset-Mobius transform and assert bitwise agreement.  Doubles memory.
    """
    if mode not in ("exact", "multimodular", "mod2_64"):
        raise ValueError(f"unknown mode {mode!r}")
    check_feasible(g.n, headroom, with_mobius=cross_check_mobius)

    timings: Dict[str, float] = {}

    def clock(label, fn):
        if not time_it:
            return fn()
        mx.synchronize()
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out) if isinstance(out, mx.array) else None
        mx.synchronize()
        timings[label] = timings.get(label, 0.0) + time.perf_counter() - t0
        return out

    if g.n == 0:
        return ChromaticResult(g.name or "graph", 0, 0, 0, 0, 0, [], [], 1, mode)

    b = clock("bounds", lambda: bounds(g))
    lo, hi = int(b["lower"]), int(b["upper"])

    counts = clock("indicator+zeta", lambda: independent_counts(g))
    # The top entry of the subset-zeta is i(V), the total number of independent
    # sets -- which is exactly the bound c_k <= i(V)^k the CRT sizing needs.
    i_v = int(counts[-1].item())

    tested: Dict[int, ModularResult] = {}

    def colourable(k: int) -> bool:
        if k in tested:
            return tested[k].colourable
        t0 = time.perf_counter()
        if mode == "mod2_64":
            value = c_k_reduction(counts, k, g.n, modulus=None)
            r = ModularResult(k=k, primes=[], residues=[], residue_mod_2_64=value)
        else:
            r = c_k_multi_modular(counts, k, g.n, i_v,
                                  num_primes=num_primes, seed=seed + k,
                                  exact=(mode == "exact"))
            if not r.consistent:
                raise AssertionError(
                    f"multi-modular disagreement at k={k}: residues {r.residues} "
                    f"for primes {r.primes}. At these sizes this indicates a bug, "
                    "not a coincidence -- investigate rather than re-rolling."
                )
        if cross_check_mobius:
            mod = None if mode == "mod2_64" else (r.primes[0] if r.primes else None)
            red = c_k_reduction(counts, k, g.n, modulus=mod)
            mob = c_k_mobius(counts, k, g.n, modulus=mod)
            if red != mob:
                raise AssertionError(
                    f"reduction and full-Mobius paths disagree at k={k}: "
                    f"{red} vs {mob}")
        if time_it:
            timings["k search"] = timings.get("k search", 0.0) + \
                time.perf_counter() - t0
        tested[k] = r
        return r.colourable

    # chi is monotone in k, and DSATUR already proved hi is achievable.
    left, right = lo, hi
    while left < right:
        mid = (left + right) // 2
        if colourable(mid):
            right = mid
        else:
            left = mid + 1
    chi = left
    if chi not in tested:
        colourable(chi)

    return ChromaticResult(
        graph=g.name or "graph", n=g.n, num_edges=g.num_edges,
        chromatic_number=chi, lower_bound=lo, upper_bound=hi,
        clique=list(b["clique"]), colouring=list(b["colouring"]),
        num_independent_sets=i_v, mode=mode, tested=tested, timings=timings,
    )
