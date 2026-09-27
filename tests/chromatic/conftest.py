"""Fixtures and the run banner for the chromatic-number tests."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import apps.chromatic as ch  # noqa: E402

from ground_truth import GROUND_TRUTH  # noqa: E402,F401


def pytest_report_header(config):
    return [f"chromatic: memory ceiling n={ch.max_feasible_n()} "
            f"(n={ch.max_feasible_n(with_mobius=True)} with the Mobius cross-check), "
            f"prime width {ch.PRIME_BITS} bits"]


@pytest.fixture(scope="session")
def ground_truth():
    return GROUND_TRUTH
