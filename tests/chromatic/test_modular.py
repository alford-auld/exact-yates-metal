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
# the free-action lemma: chi! | c_chi, per connected component
# --------------------------------------------------------------------------


def _v2_factorial(n: int) -> int:
    """Legendre's formula, computed the slow way as an independent check."""
    total, p = 0, 2
    while p <= n:
        total += n // p
        p *= 2
    return total


@pytest.mark.parametrize("chi", list(range(0, 80)))
def test_factorial_valuation_is_legendre(chi):
    assert ch.two_adic_valuation_factorial(chi) == _v2_factorial(chi)


CONNECTED = [
    ch.complete_graph(1), ch.complete_graph(4), ch.complete_graph(6),
    ch.path(7), ch.cycle(5), ch.cycle(6), ch.turan(8, 3), ch.petersen(),
    ch.grotzsch(), ch.chvatal(), ch.mycielskian(4), ch.complete_bipartite(3, 4),
] + [ch.random_graph(n, p, 900 + n) for n in (5, 6, 7, 8) for p in (0.6, 0.9)]

DISCONNECTED = [
    ch.empty_graph(5),
    ch.disjoint_copies(ch.complete_graph(4), 3),
    ch.disjoint_copies(ch.complete_graph(3), 4),
    ch.disjoint_union(ch.cycle(5), ch.complete_graph(4)),
    ch.clique_plus_independent(4, 4),
    ch.clique_plus_independent(6, 3),
    ch.clique_plus_independent(3, 6),
]


@pytest.mark.parametrize("g", CONNECTED + DISCONNECTED,
                         ids=lambda g: f"{g.name}_n{g.n}")
def test_chi_factorial_divides_c_chi_per_component(g):
    """The lemma applies per component, and only to components attaining chi."""
    from math import factorial

    chi = ch.chromatic_number(g, mode="exact").chromatic_number
    c = ch.c_k_exact_reference(g, chi)
    assert c > 0
    for comp in ch.connected_components(g):
        if ch.chromatic_number(comp, mode="exact").chromatic_number == chi:
            assert ch.c_k_exact_reference(comp, chi) % factorial(chi) == 0
    assert v2(c) >= ch.forced_two_adic_valuation(g)


@pytest.mark.parametrize("g", CONNECTED[:8], ids=lambda g: f"{g.name}_n{g.n}")
def test_lemma_needs_k_equal_to_chi(g):
    """Above chi the action is not free -- coverings may repeat a set -- so the
    divisibility is genuinely a statement about k = chi."""
    from math import factorial

    chi = ch.chromatic_number(g, mode="exact").chromatic_number
    divides_above = [ch.c_k_exact_reference(g, k) % factorial(k) == 0
                     for k in range(chi + 1, chi + 4)]
    assert not all(divides_above) or g.n <= 2


@pytest.mark.parametrize("n", list(range(1, 8)))
def test_lemma_is_tight_on_cliques(n):
    """c_n(K_n) = n! exactly: every slot must hold a distinct singleton."""
    from math import factorial

    assert ch.c_k_exact_reference(ch.complete_graph(n), n) == factorial(n)


def test_connected_formula_understates_disconnected_graphs():
    """Why forced_two_adic_valuation takes a graph and not a chi.

    K_4 seven times over has c_4 = (4!)^7, so v2 = 21, of which the
    per-component formula accounts for all 21 and the connected formula
    (v2(chi!) = 3) would account for 3.
    """
    from math import factorial

    g = ch.disjoint_copies(ch.complete_graph(4), 7)
    assert g.n == 28 and len(ch.connected_components(g)) == 7
    c = factorial(4) ** 7                       # multiplicativity; n=28 is too
    assert v2(c) == 21                          # big for the Python reference
    assert ch.forced_two_adic_valuation(g) == 21
    assert ch.two_adic_valuation_factorial(4) == 3


def test_first_clique_defeating_2_64_is_k_66():
    """The lemma is tight on cliques, so this is arithmetic, not a search."""
    assert ch.first_clique_defeating_modulus(64) == 66
    assert ch.two_adic_valuation_factorial(66) == 64
    assert ch.two_adic_valuation_factorial(65) == 63
    assert ch.two_adic_valuation_factorial(64) == 63
    assert ch.first_clique_defeating_modulus(16) == 18


def _forced_leaderboard(n: int):
    """Every (m, chi) with m*chi <= n, ranked by the forced part m*v2(chi!)."""
    return sorted(((m * ch.two_adic_valuation_factorial(chi), m, chi)
                   for chi in range(1, n + 1)
                   for m in range(1, n // chi + 1)),
                  reverse=True)


def test_forced_part_cannot_reach_64_within_the_memory_ceiling():
    """The sharp replacement for the old 287-graph search.

    A component attaining chi needs at least chi vertices, so m*chi <= n and
    the forced part is at most max_{m*chi<=n} m*v2(chi!).
    """
    ceiling = ch.max_feasible_n()
    assert ceiling == 29
    assert ch.max_forced_valuation_within(ceiling) == 25
    assert 64 - 25 == 39


@pytest.mark.parametrize("n", range(1, 30))
def test_disconnected_graphs_never_raise_the_forced_bound(n):
    """"Allowing disconnected graphs does not raise it" is load-bearing for 25.

    The search space is tiny, so enumerate it rather than assert it: for every
    n <= 29 the maximum of m*v2(chi!) over m*chi <= n is attained at m = 1,
    i.e. by a connected graph.  Multiplicity never buys back what the smaller
    chi gives up.
    """
    best = _forced_leaderboard(n)[0][0]
    connected_best = max(ch.two_adic_valuation_factorial(chi)
                         for chi in range(1, n + 1))
    assert best == connected_best
    assert best == ch.max_forced_valuation_within(n)
    assert any(m == 1 for score, m, _ in _forced_leaderboard(n) if score == best)


def test_forced_bound_runners_up_at_the_ceiling():
    """The margin at n = 29, because "does not raise it" is true but not obvious.

    The disconnected runners-up come within 3 of the connected optimum, so the
    claim is not a comfortable one and the numbers belong in a test.
    """
    board = _forced_leaderboard(29)
    assert board[0][0] == 25
    assert {(m, chi) for score, m, chi in board if score == 25} == {(1, 29), (1, 28)}

    best_by_m = {}
    for score, m, chi in board:
        best_by_m.setdefault(m, (score, chi))
    assert best_by_m[1][0] == 25          # connected
    assert best_by_m[2] == (22, 14)       # two components, chi = 14
    assert best_by_m[7] == (21, 4)        # seven disjoint K_4
    assert all(score < 25 for m, (score, _) in best_by_m.items() if m > 1)


@pytest.mark.parametrize("g", CONNECTED + DISCONNECTED,
                         ids=lambda g: f"{g.name}_n{g.n}")
def test_unordered_cofactor_valuation_is_small_per_component(g):
    """The residual risk, measured on the quantity where it is meaningful.

    v2(c_chi) splits as forced + unforced; the unforced part is the valuation
    of the per-component unordered cofactors plus that of the components below
    chi.  Evidence, not a theorem, so the ceiling is generous and the test
    exists to fail loudly if a family breaks the pattern.
    """
    chi = ch.chromatic_number(g, mode="exact").chromatic_number
    unforced = v2(ch.c_k_exact_reference(g, chi)) - ch.forced_two_adic_valuation(g)
    assert unforced >= 0, "the forced part must be a lower bound"
    assert unforced <= 8, (
        f"{g.name}: unforced valuation {unforced} is far above anything in the "
        "survey; that would weaken the residual-risk argument")


#: The smallest graph with v2(c_chi) > n, as an explicit edge list.  Found by
#: bench/chromatic/unforced_survey.py and located exactly by
#: bench/chromatic/v2_conjecture_search.py.
V2_COUNTEREXAMPLE_EDGES = [(0, 4), (0, 5), (1, 2), (1, 5), (1, 7), (2, 3),
                           (2, 5), (2, 6), (2, 7), (3, 5), (3, 6), (4, 6),
                           (5, 7), (6, 7)]


def test_v2_of_c_chi_can_exceed_n():
    """`v2(c_chi) <= n` was published as a conjecture here.  It is FALSE.

    This connected 8-vertex graph has chi = 4 and c_chi = 1536 = 2^9 * 3, so
    v2 = 9 > 8 = n.  Small enough to check by hand.

    The bound does hold exhaustively for every labeled graph on n <= 7 -- all
    2097152 of them at n = 7, where it is attained with equality but never
    exceeded -- so 8 is the smallest n at which it fails.  That boundary is
    established by bench/chromatic/v2_conjecture_search.py.

    Nothing in the solver relies on the conjecture: the default mode is CRT and
    unconditional.  The test exists so the refutation cannot be quietly lost
    again, the way the claim itself was quietly published.
    """
    g = ch.from_edges(8, V2_COUNTEREXAMPLE_EDGES)
    chi = ch.chromatic_number(g, mode="exact").chromatic_number
    c = ch.c_k_exact_reference(g, chi)

    assert chi == 4
    assert c == 1536 == 2 ** 9 * 3
    assert v2(c) == 9
    assert v2(c) > g.n, "the counterexample no longer refutes v2(c_chi) <= n"
    assert len(ch.connected_components(g)) == 1, "and it is connected"


@pytest.mark.parametrize("g", CONNECTED + DISCONNECTED,
                         ids=lambda g: f"{g.name}_n{g.n}")
def test_v2_of_c_chi_stays_far_below_the_64_bit_threshold(g):
    """What actually matters, now that v2(c_chi) <= n is refuted.

    The residual-risk argument needs 39 unforced bits, not a bound of n.  The
    survey's largest observed v2(c_chi) is nowhere near that, and this test
    fails loudly if some family changes that.
    """
    chi = ch.chromatic_number(g, mode="exact").chromatic_number
    assert v2(ch.c_k_exact_reference(g, chi)) <= 2 * g.n + 8


def test_v2_conjecture_witnesses():
    from math import factorial

    assert ch.two_adic_valuation_factorial(66) == 64 <= 66
    assert v2(factorial(4) ** 7) == 21 <= 28


# --------------------------------------------------------------------------
# multiplicativity, and the K_a + E_b closed form
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pair", [
    (ch.complete_graph(4), ch.cycle(5)),
    (ch.path(4), ch.complete_graph(3)),
    (ch.cycle(6), ch.empty_graph(3)),
    (ch.complete_bipartite(2, 2), ch.complete_graph(2)),
], ids=lambda p: f"{p[0].name}+{p[1].name}" if isinstance(p, tuple) else str(p))
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_c_k_multiplicative_over_components_on_the_solver(pair, k):
    """c_k(G1 + G2) = c_k(G1) c_k(G2), checked bitwise through the GPU path.

    A free exact invariant: it exercises the indicator, the zeta and the
    reduction on three different graphs and demands an exact ring identity
    between the results.
    """
    g1, g2 = pair
    u = ch.disjoint_union(g1, g2)
    p = ch.random_primes(1, seed=31)[0]
    got = ch.c_k_reduction(ch.independent_counts(u), k, u.n, p)
    want = (ch.c_k_reduction(ch.independent_counts(g1), k, g1.n, p)
            * ch.c_k_reduction(ch.independent_counts(g2), k, g2.n, p)) % p
    assert got == want
    assert ch.c_k_exact_reference(u, k) == \
        ch.c_k_exact_reference(g1, k) * ch.c_k_exact_reference(g2, k)


def _c_k_clique_plus_independent(a: int, b: int, k: int) -> int:
    """c_k(K_a + E_b), derived here rather than quoted.

    By multiplicativity c_k = c_k(K_a) * c_k(E_b).
    * In E_b every subset is independent, so each of the b vertices
      independently picks any nonempty subset of the k slots: (2^k - 1)^b.
    * In K_a the independent sets are the empty set and the a singletons, so a
      slot holds one of a+1 values and every vertex must be hit:
      sum_i (-1)^i C(a,i) (a+1-i)^k by inclusion-exclusion over missed vertices.
    At k = a this collapses to a!, which is the only case worth quoting.
    """
    from math import comb

    clique = sum((-1) ** i * comb(a, i) * (a + 1 - i) ** k for i in range(a + 1))
    return clique * (2 ** k - 1) ** b


@pytest.mark.parametrize("a,b", [(2, 2), (3, 2), (4, 3), (2, 5), (5, 1), (3, 4)])
@pytest.mark.parametrize("dk", [0, 1, 2])
def test_clique_plus_independent_closed_form(a, b, dk):
    g = ch.clique_plus_independent(a, b)
    k = a + dk
    assert ch.c_k_exact_reference(g, k) == _c_k_clique_plus_independent(a, b, k)
    assert ch.chromatic_number(g, mode="exact").chromatic_number == a
    assert ch.count_independent_sets(g) == (a + 1) * 2 ** b


@pytest.mark.parametrize("a,b", [(3, 3), (4, 2), (5, 2)])
def test_clique_plus_independent_at_k_equals_chi_is_a_factorial(a, b):
    """At k = chi = a the clique contributes exactly a!."""
    from math import factorial

    assert _c_k_clique_plus_independent(a, b, a) == factorial(a) * (2 ** a - 1) ** b
    assert ch.c_k_exact_reference(ch.clique_plus_independent(a, b), a) == \
        factorial(a) * (2 ** a - 1) ** b


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


@pytest.mark.parametrize("k", [2, 3, 4, 6, 10, 16])
def test_slot_bound_tightening_matches_its_closed_form(k):
    """The saving is not empirical: with i(V) = 2^n the ratio of bit-counts is
    exactly log2(2^k - 1)/k = 1 + log2(1 - 2^-k)/k, independent of n, and it
    decays as Theta(2^-k / k)."""
    import math

    closed = 1 + math.log2(1 - 2.0 ** -k) / k
    assert math.isclose(closed, math.log2(2 ** k - 1) / k, rel_tol=1e-12)
    for n in (10, 20, 29):
        ratio = math.log2((2 ** k - 1) ** n) / math.log2((2 ** n) ** k)
        assert math.isclose(ratio, closed, rel_tol=1e-12)


def test_combined_bound_never_worse_and_sometimes_better():
    better = 0
    for n in range(10, 30):
        for k in range(1, n + 1):
            i_v = 1 << n                      # the worst case: edgeless
            with_n = ch.primes_needed_for_exact(i_v, k, n=n)
            without = ch.primes_needed_for_exact(i_v, k)
            assert with_n <= without
            better += with_n < without
    assert better == 42, f"expected the closed form to bite in 42 cases, got {better}"
