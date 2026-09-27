"""Fixtures for the chromatic-number application tests."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import apps.chromatic as ch  # noqa: E402


def pytest_report_header(config):
    return [f"chromatic: memory ceiling n={ch.max_feasible_n()} "
            f"(n={ch.max_feasible_n(with_mobius=True)} with the Mobius cross-check), "
            f"prime width {ch.PRIME_BITS} bits"]


#: (generator, published chromatic number, source of the published value)
GROUND_TRUTH = [
    (ch.empty_graph(1), 1, "trivial"),
    (ch.empty_graph(6), 1, "no edges"),
    (ch.complete_graph(1), 1, "K_n has chi = n"),
    (ch.complete_graph(2), 2, "K_n has chi = n"),
    (ch.complete_graph(5), 5, "K_n has chi = n"),
    (ch.complete_graph(8), 8, "K_n has chi = n"),
    (ch.complete_bipartite(1, 1), 2, "complete bipartite is 2-chromatic"),
    (ch.complete_bipartite(3, 4), 2, "complete bipartite is 2-chromatic"),
    (ch.complete_bipartite(5, 6), 2, "complete bipartite is 2-chromatic"),
    (ch.cycle(4), 2, "even cycle"),
    (ch.cycle(10), 2, "even cycle"),
    (ch.cycle(5), 3, "odd cycle"),
    (ch.cycle(11), 3, "odd cycle"),
    (ch.path(9), 2, "path"),
    (ch.petersen(), 3, "Petersen graph"),
    (ch.petersen_standard(), 3, "Petersen graph"),
    (ch.chvatal(), 4, "Chvatal graph"),
    (ch.grotzsch(), 4, "Grotzsch graph"),
    (ch.mycielskian(2), 2, "M_k has chi = k"),
    (ch.mycielskian(3), 3, "M_k has chi = k"),
    (ch.mycielskian(4), 4, "M_k has chi = k"),
    (ch.mycielskian(5), 5, "M_k has chi = k"),
    (ch.kneser(5, 2), 3, "Lovasz: chi(K(n,k)) = n-2k+2"),
    (ch.kneser(6, 2), 4, "Lovasz: chi(K(n,k)) = n-2k+2"),
    (ch.kneser(7, 2), 5, "Lovasz: chi(K(n,k)) = n-2k+2"),
    (ch.turan(9, 3), 3, "Turan graph T(n,r) is r-partite"),
    (ch.turan(12, 4), 4, "Turan graph T(n,r) is r-partite"),
]


@pytest.fixture(scope="session")
def ground_truth():
    return GROUND_TRUTH
