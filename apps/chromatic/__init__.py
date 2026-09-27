"""Exact chromatic number by inclusion-exclusion, on the Yates butterfly kernel.

    chi(G) = min { k : c_k > 0 },   c_k = sum_S (-1)^(n-|S|) i(S)^k

where ``i = ZETA_SUB(a)`` counts independent subsets.  O*(2^n) for every graph,
with no search over colourings.

The headline caveat lives in :attr:`ChromaticResult.guarantee` and is printed by
every CLI run: a *nonzero* residue is always a sound certificate of
k-colourability, but a *zero* residue mod 2^64 is not a proof that ``c_k = 0``.
Use ``mode="exact"`` (the default) for an unconditional answer via CRT.
"""

from .graph import (
    Graph, MAX_VERTICES, bits, from_edges, from_matrix, to_matrix,
    parse_dimacs, read_dimacs, to_dimacs,
    empty_graph, complete_graph, complete_bipartite, cycle, path,
    mycielski_step, mycielskian, grotzsch, petersen, petersen_standard,
    chvatal, kneser, kneser_chromatic_number, random_graph, turan,
    disjoint_union,
)
from .independent import (
    indicator_gpu, indicator_numpy_direct, indicator_numpy_dp,
    independent_sets, count_independent_sets,
)
from .count import (
    independent_counts, power_terms, c_k_reduction, c_k_mobius,
    c_k_raw_words, c_k_exact_reference, covers_brute_force,
    proper_colourings_brute_force, is_k_colourable_brute_force,
    is_k_colourable_exhaustive,
    chromatic_number_brute_force, PRIME_BITS, max_prime_bits,
)
from .modular import (
    ModularResult, c_k_multi_modular, crt, is_prime, random_primes,
    bound_on_c_k, primes_needed_for_exact, failure_probability_bound,
    forced_two_adic_valuation, first_clique_defeating_modulus,
    PRIME_POOL_LOWER_BOUND,
)
from .chromatic import (
    ChromaticResult, chromatic_number, bounds, greedy_clique, dsatur,
    verify_colouring, max_feasible_n, check_feasible,
    BYTES_PER_SUBSET, BYTES_PER_SUBSET_WITH_MOBIUS,
)

__all__ = [
    "Graph", "MAX_VERTICES", "bits", "from_edges", "from_matrix", "to_matrix",
    "parse_dimacs", "read_dimacs", "to_dimacs",
    "empty_graph", "complete_graph", "complete_bipartite", "cycle", "path",
    "mycielski_step", "mycielskian", "grotzsch", "petersen",
    "petersen_standard", "chvatal", "kneser", "kneser_chromatic_number",
    "random_graph", "turan", "disjoint_union",
    "indicator_gpu", "indicator_numpy_direct", "indicator_numpy_dp",
    "independent_sets", "count_independent_sets",
    "independent_counts", "power_terms", "c_k_reduction", "c_k_mobius",
    "c_k_raw_words", "c_k_exact_reference", "covers_brute_force",
    "proper_colourings_brute_force", "is_k_colourable_brute_force",
    "is_k_colourable_exhaustive",
    "chromatic_number_brute_force", "PRIME_BITS", "max_prime_bits",
    "ModularResult", "c_k_multi_modular", "crt", "is_prime", "random_primes",
    "bound_on_c_k", "primes_needed_for_exact", "failure_probability_bound",
    "forced_two_adic_valuation", "first_clique_defeating_modulus",
    "PRIME_POOL_LOWER_BOUND",
    "ChromaticResult", "chromatic_number", "bounds", "greedy_clique", "dsatur",
    "verify_colouring", "max_feasible_n", "check_feasible",
    "BYTES_PER_SUBSET", "BYTES_PER_SUBSET_WITH_MOBIUS",
]
