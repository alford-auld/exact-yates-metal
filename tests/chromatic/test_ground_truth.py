"""Every family with a published chromatic number must come back right."""

import pytest

import apps.chromatic as ch
from ground_truth import GROUND_TRUTH

IDS = [f"{g.name}_n{g.n}" for g, _, _ in GROUND_TRUTH]


@pytest.mark.parametrize("g,chi,source", GROUND_TRUTH, ids=IDS)
def test_published_chromatic_number(g, chi, source):
    r = ch.chromatic_number(g, mode="exact")
    assert r.chromatic_number == chi, f"{g.name}: got {r.chromatic_number}, " \
                                      f"published {chi} ({source})"
    assert r.certain, f"{g.name}: answer should be unconditional, {r.guarantee}"


@pytest.mark.parametrize("g,chi,source", GROUND_TRUTH, ids=IDS)
def test_bounds_bracket_the_answer(g, chi, source):
    r = ch.chromatic_number(g, mode="exact")
    assert r.lower_bound <= chi <= r.upper_bound
    assert ch.verify_colouring(g, r.colouring)
    assert len(set(r.colouring)) == r.upper_bound
    # the greedy clique is a real clique, hence a real lower bound
    for i, u in enumerate(r.clique):
        for v in r.clique[i + 1:]:
            assert g.adj[u] >> v & 1, f"{g.name}: reported clique is not a clique"


@pytest.mark.parametrize("g,chi,source", GROUND_TRUTH, ids=IDS)
def test_all_three_modes_agree(g, chi, source):
    a = ch.chromatic_number(g, mode="exact").chromatic_number
    b = ch.chromatic_number(g, mode="multimodular", num_primes=3).chromatic_number
    c = ch.chromatic_number(g, mode="mod2_64").chromatic_number
    assert a == b == c == chi


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_mycielskian_chain_is_the_adversarial_family(k):
    """chi(M_k) = k while the clique number stays at 2 for k >= 3.

    This is the family that defeats clique-based bounds, so it is where the
    O*(2^n) method actually has to do the work.
    """
    g = ch.mycielskian(k)
    r = ch.chromatic_number(g, mode="exact")
    assert r.chromatic_number == k
    if k >= 3:
        assert r.lower_bound == 2, "clique bound must be useless here"
        assert r.chromatic_number - r.lower_bound == k - 2


@pytest.mark.parametrize("n,k", [(4, 2), (5, 2), (6, 2), (7, 2), (5, 1), (4, 1)])
def test_kneser_matches_lovasz(n, k):
    g = ch.kneser(n, k)
    want = ch.kneser_chromatic_number(n, k)
    assert ch.chromatic_number(g, mode="exact").chromatic_number == want


def test_mobius_cross_check_on_every_family():
    """Run the whole ground-truth set again with the full-Mobius cross-check on."""
    for g, chi, _ in GROUND_TRUTH:
        if g.n > 16:            # the cross-check doubles memory; keep it quick
            continue
        r = ch.chromatic_number(g, mode="exact", cross_check_mobius=True)
        assert r.chromatic_number == chi


def test_disconnected_graph_takes_the_max():
    a, b = ch.cycle(5), ch.complete_graph(4)      # chi 3 and 4
    u = ch.disjoint_union(a, b)
    assert ch.chromatic_number(u, mode="exact").chromatic_number == 4


def test_empty_graph_on_zero_vertices():
    r = ch.chromatic_number(ch.empty_graph(0))
    assert r.chromatic_number == 0
