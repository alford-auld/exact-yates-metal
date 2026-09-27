"""Graph representation, DIMACS round trip, and generator properties."""

import numpy as np
import pytest

import apps.chromatic as ch


def test_bitmask_and_matrix_round_trip():
    for g in (ch.petersen(), ch.chvatal(), ch.random_graph(9, 0.5, 4)):
        again = ch.from_matrix(ch.to_matrix(g))
        assert again.adj == g.adj


def test_dimacs_round_trip():
    for g in (ch.grotzsch(), ch.kneser(6, 2), ch.random_graph(11, 0.4, 7)):
        parsed = ch.parse_dimacs(ch.to_dimacs(g))
        assert parsed.n == g.n and parsed.adj == g.adj


def test_dimacs_parsing_details():
    text = """c a comment
p edge 4 3
e 1 2
e 2 3
e 3 4
"""
    g = ch.parse_dimacs(text)
    assert g.n == 4 and sorted(g.edges) == [(0, 1), (1, 2), (2, 3)]


@pytest.mark.parametrize("text,msg", [
    ("e 1 2\n", "problem line"),
    ("p edge 3 1\ne 1 5\n", "outside"),
    ("p edge 3 1\ne 2 2\n", "self-loop"),
    ("c only a comment\n", "no 'p' problem line"),
    ("p edge 3 1\nq 1 2\n", "unrecognised"),
])
def test_dimacs_errors(text, msg):
    with pytest.raises(ValueError, match=msg):
        ch.parse_dimacs(text)


def test_dimacs_is_one_indexed():
    g = ch.parse_dimacs("p edge 2 1\ne 1 2\n")
    assert g.edges == [(0, 1)]


def test_rejects_malformed_graphs():
    with pytest.raises(ValueError, match="self-loop"):
        ch.Graph(2, (0b01, 0b00))
    with pytest.raises(ValueError, match="not symmetric"):
        ch.Graph(2, (0b10, 0b00))
    with pytest.raises(ValueError, match="bitmask limit"):
        ch.Graph(33, tuple([0] * 33))
    with pytest.raises(ValueError, match="symmetric"):
        ch.from_matrix([[0, 1], [0, 0]])


def test_mycielski_properties():
    """Each step doubles+1 the vertices and preserves triangle-freeness.

    The chain stops at M_5 (23 vertices): M_6 has 47, past the 32-vertex
    bitmask limit, and 2^47 subsets is far past any machine anyway.
    """
    for k in range(3, 6):
        g = ch.mycielskian(k)
        prev = ch.mycielskian(k - 1)
        assert g.n == 2 * prev.n + 1
        if k >= 3:
            assert _triangles(g) == 0, f"M_{k} must be triangle-free"
            assert len(ch.greedy_clique(g)) == 2, "clique number stays 2"
    assert ch.mycielskian(3).n == 5 and ch.mycielskian(4).n == 11
    assert ch.mycielskian(5).n == 23


def test_mycielskian_3_is_c5_and_4_is_grotzsch():
    assert sorted(ch.mycielskian(3).edges) == sorted(ch.cycle(5).edges) or \
        _isomorphic_small(ch.mycielskian(3), ch.cycle(5))
    g = ch.grotzsch()
    assert g.n == 11 and g.num_edges == 20


def test_petersen_two_constructions_agree():
    assert _isomorphic_small(ch.petersen(), ch.petersen_standard())


def test_chvatal_published_properties():
    g = ch.chvatal()
    assert g.n == 12 and g.num_edges == 24
    assert all(g.degree(v) == 4 for v in range(12)), "Chvatal is 4-regular"
    assert _triangles(g) == 0, "Chvatal is triangle-free"


def test_kneser_sizes_and_lovasz_formula():
    from math import comb
    for n in range(4, 9):
        for k in (1, 2):
            if n < k:
                continue
            g = ch.kneser(n, k)
            assert g.n == comb(n, k)
            if n >= 2 * k:
                assert ch.kneser_chromatic_number(n, k) == n - 2 * k + 2


def test_disjoint_union_and_complement():
    a, b = ch.complete_graph(3), ch.cycle(4)
    u = ch.disjoint_union(a, b)
    assert u.n == 7 and u.num_edges == a.num_edges + b.num_edges
    g = ch.random_graph(8, 0.5, 1)
    c = g.complement()
    assert g.num_edges + c.num_edges == 8 * 7 // 2


def _triangles(g):
    return sum(1 for u in range(g.n) for v in range(u + 1, g.n)
               for w in range(v + 1, g.n)
               if (g.adj[u] >> v & 1) and (g.adj[u] >> w & 1) and (g.adj[v] >> w & 1))


def _isomorphic_small(a, b):
    """Brute-force isomorphism; only for the tiny graphs used above."""
    import itertools
    if a.n != b.n or a.num_edges != b.num_edges:
        return False
    if a.n > 10:
        raise ValueError("too big for brute-force isomorphism")
    target = {frozenset(e) for e in b.edges}
    for perm in itertools.permutations(range(a.n)):
        if {frozenset((perm[u], perm[v])) for u, v in a.edges} == target:
            return True
    return False
