"""Random graphs cross-checked against exhaustive search over all k^n colourings."""

import pytest

import apps.chromatic as ch

RANDOM_CASES = [(n, p, seed)
                for n in (4, 6, 8, 10, 12)
                for p, seed in ((0.3, 1), (0.5, 2), (0.8, 3))]


@pytest.mark.parametrize("n,p,seed", RANDOM_CASES,
                         ids=[f"n{n}_p{p}_s{s}" for n, p, s in RANDOM_CASES])
def test_random_graph_matches_brute_force(n, p, seed):
    g = ch.random_graph(n, p, seed)
    want = ch.chromatic_number_brute_force(g)
    got = ch.chromatic_number(g, mode="exact")
    assert got.chromatic_number == want, f"{g.name}: {got.chromatic_number} != {want}"
    assert got.certain


@pytest.mark.parametrize("n,p,seed", RANDOM_CASES[:9],
                         ids=[f"n{n}_p{p}_s{s}" for n, p, s in RANDOM_CASES[:9]])
def test_c_k_vanishing_pattern_matches_brute_force(n, p, seed):
    """c_k = 0 for every k < chi and c_k > 0 for every k >= chi."""
    g = ch.random_graph(n, p, seed)
    chi = ch.chromatic_number_brute_force(g)
    counts = ch.independent_counts(g)
    for k in range(0, min(chi + 2, 7)):
        exact = ch.c_k_exact_reference(g, k)
        colourable = ch.is_k_colourable_brute_force(g, k)
        assert (exact > 0) == colourable == (k >= chi), f"{g.name} k={k}"
        prime = ch.random_primes(1, seed=17)[0]
        assert ch.c_k_reduction(counts, k, g.n, prime) == exact % prime


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pruned_search_equals_unpruned_enumeration(n, seed):
    """The oracle itself is validated: backtracking with symmetry breaking must
    give exactly the same answers as enumerating all k^n assignments."""
    g = ch.random_graph(n, 0.5, 300 + 10 * n + seed)
    for k in range(0, n + 1):
        assert ch.is_k_colourable_brute_force(g, k) == \
            ch.is_k_colourable_exhaustive(g, k), f"{g.name} k={k}"


@pytest.mark.parametrize("seed", range(6))
def test_dense_random_graphs_where_chi_is_large(seed):
    g = ch.random_graph(9, 0.9, 500 + seed)
    assert ch.chromatic_number(g, mode="exact").chromatic_number == \
        ch.chromatic_number_brute_force(g)
