"""The independence indicator: three implementations, one answer."""

import numpy as np
import pytest

import apps.chromatic as ch

GRAPHS = [
    ch.empty_graph(0), ch.empty_graph(1), ch.empty_graph(5),
    ch.complete_graph(1), ch.complete_graph(6), ch.cycle(7), ch.path(8),
    ch.petersen(), ch.grotzsch(), ch.chvatal(), ch.kneser(6, 2),
    ch.random_graph(12, 0.3, 1), ch.random_graph(12, 0.7, 2),
    ch.mycielskian(4),
]


@pytest.mark.parametrize("g", GRAPHS, ids=lambda g: f"{g.name}_n{g.n}")
def test_all_indicator_paths_agree_with_definition(g):
    brute = np.array([1 if g.is_independent(s) else 0 for s in range(1 << g.n)],
                     dtype=np.uint32)
    assert np.array_equal(ch.indicator_numpy_direct(g), brute)
    assert np.array_equal(ch.indicator_numpy_dp(g), brute)
    assert np.array_equal(np.array(ch.indicator_gpu(g)), brute)
    assert np.array_equal(np.array(ch.indicator_gpu(g, section="branchless")), brute)


@pytest.mark.parametrize("g", GRAPHS[:8], ids=lambda g: f"{g.name}_n{g.n}")
def test_zeta_of_indicator_counts_independent_subsets(g):
    """i(S) must equal the number of independent T contained in S."""
    counts = np.array(ch.independent_counts(g))
    for s in range(min(1 << g.n, 300)):
        want = sum(1 for t in range(1 << g.n)
                   if (t & s) == t and g.is_independent(t))
        assert int(counts[s]) == want, f"{g.name}: i({s}) wrong"


@pytest.mark.parametrize("g", GRAPHS, ids=lambda g: f"{g.name}_n{g.n}")
def test_i_of_v_is_the_total_independent_set_count(g):
    counts = np.array(ch.independent_counts(g))
    assert int(counts[-1]) == len(ch.independent_sets(g))
    assert ch.count_independent_sets(g) == int(counts[-1])


def test_indicator_fits_uint32():
    """i(S) <= 2^n, so uint32 is exact for every feasible n."""
    g = ch.empty_graph(20)                      # worst case: every subset independent
    counts = np.array(ch.independent_counts(g))
    assert counts.dtype == np.uint32
    assert int(counts[-1]) == 2 ** 20


def test_dp_recurrence_direction():
    """The lowbit DP must read entries already written; a wrong loop order
    would silently produce a different answer on some graph."""
    g = ch.random_graph(10, 0.5, 99)
    assert np.array_equal(ch.indicator_numpy_dp(g), ch.indicator_numpy_direct(g))
