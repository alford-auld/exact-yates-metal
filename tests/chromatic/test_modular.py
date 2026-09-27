"""Multi-modular verification, CRT reconstruction, and the one-sided guarantee.

Includes the false-negative regression the brief asks for: an instance where
the single-modulus (2^64) test really does report ``c_k == 0`` for a nonzero
``c_k``.
"""

import pytest

import apps.chromatic as ch


def v2(x: int) -> int:
    """2-adic valuation."""
    assert x != 0
    k = 0
    while x % 2 == 0:
        x //= 2
        k += 1
    return k


def union_of(h, t):
    g = h
    for _ in range(t - 1):
        g = ch.disjoint_union(g, h)
    return ch.Graph(g.n, g.adj, f"{t}x{h.name}")


# --------------------------------------------------------------------------
# primes and CRT
# --------------------------------------------------------------------------


def test_is_prime_against_a_sieve():
    limit = 5000
    sieve = [True] * limit
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit, i):
                sieve[j] = False
    for i in range(limit):
        assert ch.is_prime(i) == sieve[i], i
    for p in (2147483647, 1000000007, 2305843009213693951):
        assert ch.is_prime(p)
    for c in (2147483647 * 3, 1000000007 * 2, 3215031751):
        assert not ch.is_prime(c)


def test_random_primes_are_distinct_primes_in_range():
    ps = ch.random_primes(12, seed=1)
    assert len(set(ps)) == 12
    for p in ps:
        assert ch.is_prime(p)
        assert 1 << (ch.PRIME_BITS - 1) <= p < 1 << ch.PRIME_BITS
    assert ch.random_primes(5, seed=1) == ch.random_primes(5, seed=1)
    assert ch.random_primes(5, seed=1) != ch.random_primes(5, seed=2)


@pytest.mark.parametrize("value", [0, 1, 7, 12345678901234567890, 2 ** 100 - 3])
def test_crt_round_trip(value):
    primes = ch.random_primes(5, seed=9)
    residues = [value % p for p in primes]
    got, modulus = ch.crt(residues, primes)
    assert modulus == _product(primes)
    assert got == value % modulus


def test_crt_rejects_non_coprime_moduli():
    with pytest.raises(ValueError, match="coprime"):
        ch.crt([1, 1], [6, 10])


def test_primes_needed_matches_the_bound():
    for i_v, k in [(4, 2), (127, 4), (7407, 5), (2 ** 20, 8)]:
        r = ch.primes_needed_for_exact(i_v, k)
        smallest = 1 << (ch.PRIME_BITS - 1)
        assert smallest ** r > ch.bound_on_c_k(i_v, k)
        assert smallest ** (r - 1) <= ch.bound_on_c_k(i_v, k) or r == 1


def test_failure_bound_shrinks_geometrically():
    b1 = ch.failure_probability_bound(2 ** 20, 8, 1)
    b4 = ch.failure_probability_bound(2 ** 20, 8, 4)
    assert 0 < b4 < b1 < 1e-3
    assert abs(b4 - b1 ** 4) < 1e-30
    assert ch.PRIME_POOL_LOWER_BOUND > 3.5e7


# --------------------------------------------------------------------------
# agreement across moduli
# --------------------------------------------------------------------------

AGREEMENT_GRAPHS = [ch.complete_graph(4), ch.cycle(7), ch.petersen(),
                    ch.grotzsch(), ch.chvatal(), ch.kneser(6, 2),
                    ch.random_graph(12, 0.5, 21), ch.mycielskian(4)]


@pytest.mark.parametrize("g", AGREEMENT_GRAPHS, ids=lambda g: f"{g.name}_n{g.n}")
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_all_residues_agree_on_vanishing(g, k):
    """All r residues must be simultaneously zero or simultaneously nonzero.

    A disagreement at these sizes is a bug, not a probabilistic event.
    """
    counts = ch.independent_counts(g)
    i_v = ch.count_independent_sets(g)
    r = ch.c_k_multi_modular(counts, k, g.n, i_v, num_primes=6, seed=k)
    assert r.consistent, f"{g.name} k={k}: residues {r.residues} for {r.primes}"
    exact = ch.c_k_exact_reference(g, k)
    assert (exact != 0) == r.any_nonzero
    for p, res in zip(r.primes, r.residues):
        assert res == exact % p


@pytest.mark.parametrize("g", AGREEMENT_GRAPHS, ids=lambda g: f"{g.name}_n{g.n}")
@pytest.mark.parametrize("k", [2, 3, 4])
def test_crt_reconstructs_c_k_exactly(g, k):
    counts = ch.independent_counts(g)
    i_v = ch.count_independent_sets(g)
    r = ch.c_k_multi_modular(counts, k, g.n, i_v, exact=True)
    assert r.exact_value is not None
    assert r.exact_value == ch.c_k_exact_reference(g, k)
    assert r.combined_modulus > r.bound_on_value
    assert r.failure_bound == 0.0


def test_bound_on_c_k_is_actually_an_upper_bound():
    for g in AGREEMENT_GRAPHS:
        i_v = ch.count_independent_sets(g)
        for k in (1, 2, 3, 4):
            assert ch.c_k_exact_reference(g, k) <= ch.bound_on_c_k(i_v, k)


# --------------------------------------------------------------------------
# the false-negative regression
# --------------------------------------------------------------------------


def test_disjoint_k4_powers_of_two_are_as_predicted():
    """c_30(K_4) has 2-adic valuation 10, and c_k is multiplicative, so t
    disjoint copies give valuation 10t.  This is the lever for constructing a
    false negative at any modulus 2^b with b <= 10t."""
    c = ch.c_k_exact_reference(ch.complete_graph(4), 30)
    assert v2(c) == 10
    g2 = union_of(ch.complete_graph(4), 2)
    assert ch.c_k_exact_reference(g2, 30) == c ** 2
    assert v2(c ** 2) == 20


def test_false_negative_against_a_2_16_modulus():
    """A *constructed* false negative, small enough to run in the fast suite.

    2 x K_4 (8 vertices) at k=30 has ``v2(c_30) = 20``, so ``c_30 = 0 mod 2^16``
    even though ``c_30`` is enormous.  Exactly the failure mode the mod-2^64
    path has; only the modulus is shrunk so the witness fits in 8 vertices
    instead of 28.
    """
    g = union_of(ch.complete_graph(4), 2)
    assert g.n == 8
    counts = ch.independent_counts(g)
    exact = ch.c_k_exact_reference(g, 30)
    assert exact > 0

    assert ch.c_k_reduction(counts, 30, g.n, modulus=1 << 16) == 0, \
        "this is the false negative: zero residue, nonzero c_k"
    # ...and the prime moduli see through it immediately
    r = ch.c_k_multi_modular(counts, 30, g.n, ch.count_independent_sets(g),
                             num_primes=3, seed=0)
    assert r.any_nonzero and r.consistent
    assert not r.mod_2_64_is_false_negative, \
        "v2(c_30)=20 defeats 2^16 but not 2^64; only the 28-vertex case does that"


@pytest.mark.slow
def test_false_negative_against_the_real_2_64_modulus():
    """The same construction scaled up until it defeats mod 2^64 itself.

    7 x K_4 is 28 vertices -- inside the memory ceiling -- and ``v2(c_30) = 70``,
    so ``c_30 = 0 mod 2^64`` while ``c_30`` has 149 digits.  The single-modulus
    test reports "not 30-colourable" for a graph that is obviously
    30-colourable.
    """
    g = union_of(ch.complete_graph(4), 7)
    assert g.n == 28
    if g.n > ch.max_feasible_n():
        pytest.skip(f"n=28 exceeds this machine's ceiling of {ch.max_feasible_n()}")

    exact = ch.c_k_exact_reference(ch.complete_graph(4), 30) ** 7
    assert v2(exact) == 70 and exact % (1 << 64) == 0

    counts = ch.independent_counts(g)
    assert ch.c_k_reduction(counts, 30, g.n, modulus=None) == 0, \
        "mod 2^64 must report zero here -- that is the whole point"

    i_v = ch.count_independent_sets(g)
    r = ch.c_k_multi_modular(counts, 30, g.n, i_v, num_primes=4, seed=0)
    assert r.any_nonzero, "the prime moduli must not be fooled"
    assert r.consistent, "the primes must agree with each other"
    assert r.mod_2_64_is_false_negative, \
        "the mod-2^64 residue disagreeing with the primes IS the false negative"
    for p, res in zip(r.primes, r.residues):
        assert res == exact % p


def test_no_false_negative_can_corrupt_a_reported_chi_within_the_ceiling():
    """Why the mod-2^64 path is nonetheless safe in practice here.

    To make the *reported chromatic number* wrong, a graph would need
    ``2^64 | c_chi`` -- the false negative has to land at ``k = chi`` itself,
    not at some k the binary search never visits.  Over the families searched
    (complete, Turan, cycles, and 280 random graphs on 2..8 vertices) the best
    ratio ``v2(c_chi)/n`` observed is 7/8 for K_8, which would need n >= 74
    vertices to reach valuation 64 -- far past the ~29-vertex memory ceiling.
    This test pins the ratio for the extremal case found.
    """
    g = ch.complete_graph(8)
    chi = ch.chromatic_number(g, mode="exact").chromatic_number
    assert chi == 8
    assert v2(ch.c_k_exact_reference(g, chi)) == 7
    assert 64 / (7 / 8) > ch.max_feasible_n()


def _product(xs):
    out = 1
    for x in xs:
        out *= x
    return out


# --------------------------------------------------------------------------
# the free-action lemma: chi! | c_chi
# --------------------------------------------------------------------------


def _v2_factorial(n: int) -> int:
    """Legendre's formula, computed the slow way as an independent check."""
    total, p = 0, 2
    while p <= n:
        total += n // p
        p *= 2
    return total


@pytest.mark.parametrize("chi", list(range(0, 80)))
def test_forced_valuation_is_legendre(chi):
    assert ch.forced_two_adic_valuation(chi) == _v2_factorial(chi)


LEMMA_GRAPHS = [
    ch.complete_graph(1), ch.complete_graph(4), ch.complete_graph(6),
    ch.empty_graph(5), ch.path(7), ch.cycle(5), ch.cycle(6),
    ch.turan(8, 3), ch.petersen(), ch.grotzsch(), ch.chvatal(),
    ch.mycielskian(4), ch.complete_bipartite(3, 4),
] + [ch.random_graph(n, p, 900 + n) for n in (5, 6, 7, 8) for p in (0.3, 0.6, 0.9)]


@pytest.mark.parametrize("g", LEMMA_GRAPHS, ids=lambda g: f"{g.name}_n{g.n}")
def test_chi_factorial_divides_c_chi(g):
    """At k = chi the components of a covering tuple are pairwise distinct, so
    S_chi acts freely on the coverings and chi! divides their number."""
    from math import factorial

    chi = ch.chromatic_number(g, mode="exact").chromatic_number
    c = ch.c_k_exact_reference(g, chi)
    assert c > 0
    assert c % factorial(chi) == 0, f"{g.name}: {chi}! does not divide c_{chi}={c}"
    assert v2(c) >= ch.forced_two_adic_valuation(chi)


@pytest.mark.parametrize("g", LEMMA_GRAPHS[:8], ids=lambda g: f"{g.name}_n{g.n}")
def test_lemma_needs_k_equal_to_chi(g):
    """Above chi the action is not free -- coverings may repeat a set -- so the
    divisibility genuinely is a statement about k = chi and not about all k."""
    from math import factorial

    chi = ch.chromatic_number(g, mode="exact").chromatic_number
    divides_above = [ch.c_k_exact_reference(g, k) % factorial(k) == 0
                     for k in range(chi + 1, chi + 4)]
    assert not all(divides_above) or g.n <= 2, (
        f"{g.name}: k! divided c_k for every k above chi, which would make the "
        "restriction to k=chi look unnecessary; check the example")


@pytest.mark.parametrize("n", list(range(1, 8)))
def test_lemma_is_tight_on_cliques(n):
    """c_n(K_n) = n! exactly: every slot must hold a distinct singleton."""
    from math import factorial

    assert ch.c_k_exact_reference(ch.complete_graph(n), n) == factorial(n)


def test_first_clique_defeating_2_64_is_k_66():
    """Because the lemma is tight on cliques, the first clique whose c_chi is
    divisible by 2^64 is exact arithmetic rather than a search: v2(66!) = 64."""
    assert ch.first_clique_defeating_modulus(64) == 66
    assert ch.forced_two_adic_valuation(66) == 64
    assert ch.forced_two_adic_valuation(65) == 63
    assert ch.forced_two_adic_valuation(64) == 63
    assert ch.first_clique_defeating_modulus(16) == 18


def test_forced_part_cannot_reach_64_within_the_memory_ceiling():
    """The sharp replacement for the old 287-graph search.

    A false negative that corrupts the *reported* chi needs 2^64 | c_chi.  The
    lemma forces only chi - popcount(chi) of those bits, which is at most 25 for
    any chi reachable at n <= 29, so at least 39 bits would have to come from
    the unordered cofactor c_chi / chi!.
    """
    ceiling = ch.max_feasible_n()
    forced = max(ch.forced_two_adic_valuation(chi) for chi in range(ceiling + 1))
    assert forced == 25 and ceiling == 29, (forced, ceiling)
    assert 64 - forced == 39


@pytest.mark.parametrize("g", LEMMA_GRAPHS, ids=lambda g: f"{g.name}_n{g.n}")
def test_unordered_cofactor_is_almost_always_odd(g):
    """The residual risk, measured: c_chi / chi! carries very little 2-adic
    valuation of its own.  This is evidence, not a theorem -- so the test
    records a generous ceiling rather than asserting oddness."""
    from math import factorial

    chi = ch.chromatic_number(g, mode="exact").chromatic_number
    cofactor = ch.c_k_exact_reference(g, chi) // factorial(chi)
    assert v2(cofactor) <= 8, (
        f"{g.name}: cofactor valuation {v2(cofactor)} is far above anything "
        "seen in the survey; that would weaken the residual-risk argument")


# --------------------------------------------------------------------------
# the (2^k - 1)^n bound
# --------------------------------------------------------------------------


@pytest.mark.parametrize("g", [ch.empty_graph(4), ch.empty_graph(6), ch.cycle(6),
                               ch.petersen(), ch.chvatal(),
                               ch.random_graph(9, 0.5, 3)],
                         ids=lambda g: g.name)
@pytest.mark.parametrize("k", [1, 2, 3, 4, 6])
def test_slot_bound_is_a_valid_upper_bound(g, k):
    """A covering is determined by which nonempty subset of the k slots holds
    each vertex, so c_k <= (2^k - 1)^n."""
    assert ch.c_k_exact_reference(g, k) <= (2 ** k - 1) ** g.n
    i_v = ch.count_independent_sets(g)
    assert ch.c_k_exact_reference(g, k) <= ch.bound_on_c_k(i_v, k, g.n)


def test_slot_bound_is_tight_on_the_edgeless_graph():
    for n in (1, 3, 5):
        g = ch.empty_graph(n)
        for k in (1, 2, 3):
            assert ch.c_k_exact_reference(g, k) == (2 ** k - 1) ** n
            assert ch.bound_on_c_k(2 ** n, k, n) == (2 ** k - 1) ** n


def test_combined_bound_never_worse_and_sometimes_better():
    better = 0
    for n in range(10, 30):
        for k in range(1, n + 1):
            i_v = 1 << n                      # the worst case: edgeless
            with_n = ch.primes_needed_for_exact(i_v, k, n=n)
            without = ch.primes_needed_for_exact(i_v, k)
            assert with_n <= without
            better += with_n < without
    assert better > 0, "the extra bound should help somewhere in the feasible grid"
