"""The inclusion-exclusion identity, and the two ways of evaluating it.

The brief requires that ``c_k`` from the full subset-Mobius transform, read at
``S = V``, equals ``c_k`` from the direct alternating reduction **bitwise**, for
every k and every test graph.  That is the strongest single check on the whole
pipeline: it exercises the kernel, the sign convention and the modular
arithmetic at once.
"""

import mlx.core as mx
import numpy as np
import pytest

import apps.chromatic as ch
from apps.chromatic.count import c_k_raw_words

GRAPHS = [
    ch.empty_graph(1), ch.empty_graph(4), ch.complete_graph(3),
    ch.complete_graph(5), ch.cycle(5), ch.cycle(6), ch.path(7),
    ch.petersen(), ch.grotzsch(), ch.chvatal(), ch.kneser(6, 2),
    ch.random_graph(11, 0.4, 11), ch.random_graph(11, 0.8, 12),
    ch.mycielskian(4), ch.turan(10, 3),
]
KS = [0, 1, 2, 3, 4, 5, 7]


@pytest.mark.parametrize("g", GRAPHS, ids=lambda g: f"{g.name}_n{g.n}")
@pytest.mark.parametrize("k", KS)
def test_mobius_and_reduction_agree_bitwise_mod_2_64(g, k):
    counts = ch.independent_counts(g)
    reduction, mobius = c_k_raw_words(counts, k, g.n, modulus=None)
    assert reduction == mobius, (
        f"{g.name} k={k}: reduction word {reduction} != Mobius word {mobius}")


@pytest.mark.parametrize("g", GRAPHS[:8], ids=lambda g: f"{g.name}_n{g.n}")
@pytest.mark.parametrize("k", [2, 3, 4])
def test_mobius_and_reduction_agree_bitwise_mod_prime(g, k):
    p = ch.random_primes(1, seed=5)[0]
    counts = ch.independent_counts(g)
    reduction, mobius = c_k_raw_words(counts, k, g.n, modulus=p)
    assert reduction == mobius
    assert ch.c_k_reduction(counts, k, g.n, p) == ch.c_k_mobius(counts, k, g.n, p)


@pytest.mark.parametrize("g", GRAPHS, ids=lambda g: f"{g.name}_n{g.n}")
@pytest.mark.parametrize("k", KS)
def test_gpu_matches_exact_integer_reference(g, k):
    counts = ch.independent_counts(g)
    exact = ch.c_k_exact_reference(g, k)
    assert ch.c_k_reduction(counts, k, g.n, None) == exact % (1 << 64)
    p = ch.random_primes(1, seed=3)[0]
    assert ch.c_k_reduction(counts, k, g.n, p) == exact % p


@pytest.mark.parametrize("g", [ch.empty_graph(1), ch.complete_graph(3),
                               ch.cycle(5), ch.path(4), ch.random_graph(6, 0.5, 2),
                               ch.complete_bipartite(2, 2)],
                         ids=lambda g: g.name)
@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_c_k_counts_covers_by_independent_sets(g, k):
    """c_k really is the number of ordered k-tuples of independent sets whose
    union is V -- checked by direct enumeration, no transform involved."""
    assert ch.c_k_exact_reference(g, k) == ch.covers_brute_force(g, k)


@pytest.mark.parametrize("g", [ch.complete_graph(3), ch.cycle(5), ch.petersen(),
                               ch.grotzsch(), ch.random_graph(9, 0.5, 8)],
                         ids=lambda g: g.name)
def test_c_k_positive_exactly_from_chi_onwards(g):
    """c_k = 0 for k < chi and c_k > 0 for k >= chi."""
    chi = ch.chromatic_number_brute_force(g)
    for k in range(0, min(chi + 3, 9)):
        c = ch.c_k_exact_reference(g, k)
        assert (c > 0) == (k >= chi), f"{g.name}: c_{k}={c} but chi={chi}"


def test_c_k_is_multiplicative_over_disjoint_unions():
    """Used by the false-negative construction, so pin it."""
    a, b = ch.complete_graph(4), ch.cycle(5)
    u = ch.disjoint_union(a, b)
    for k in range(1, 6):
        assert ch.c_k_exact_reference(u, k) == \
            ch.c_k_exact_reference(a, k) * ch.c_k_exact_reference(b, k)


def test_empty_graph_closed_form():
    """With no edges every subset is independent, so c_k = (2^k - 1)^n."""
    for n in (1, 3, 6):
        g = ch.empty_graph(n)
        for k in (1, 2, 3, 5):
            assert ch.c_k_exact_reference(g, k) == (2 ** k - 1) ** n


def _surjections(j: int, n: int) -> int:
    """Number of surjections from a j-set onto an n-set, n! * S(j,n)."""
    from math import factorial
    stirling = [[0] * (n + 1) for _ in range(j + 1)]
    stirling[0][0] = 1
    for a in range(1, j + 1):
        for b in range(1, n + 1):
            stirling[a][b] = b * stirling[a - 1][b] + stirling[a - 1][b - 1]
    return factorial(n) * stirling[j][n]


def test_complete_graph_closed_form():
    """In K_n the independent sets are exactly the empty set and the singletons.

    A covering k-tuple therefore picks which slots are non-empty and maps those
    slots *onto* the n vertices (slots may repeat a vertex), giving
    ``sum_j C(k,j) * Surj(j,n)``.  Derived from the structure of K_n rather than
    from the inclusion-exclusion formula, so this is an independent check.
    """
    from math import comb
    for n in (1, 2, 3, 4):
        g = ch.complete_graph(n)
        for k in range(n, n + 4):
            want = sum(comb(k, j) * _surjections(j, n) for j in range(n, k + 1))
            assert ch.c_k_exact_reference(g, k) == want, (n, k)


def test_k_zero_and_the_empty_graph_edge_case():
    assert ch.c_k_exact_reference(ch.empty_graph(0), 0) == 1
    for g in (ch.complete_graph(2), ch.cycle(5)):
        assert ch.c_k_exact_reference(g, 0) == 0
