"""Fixtures and the run banner for the Yates kernel tests.

Importable helpers live in ``tests/kernel_helpers.py`` -- see the note there for
why they are not in this file.
"""

import os
import sys

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yates  # noqa: E402


def pytest_report_header(config):
    lim = yates.limits()
    host = yates.host_info()
    return [
        f"mlx {lim.mlx_version} | metal available: {mx.metal.is_available()}",
        f"gpu  {lim.device_name} ({lim.architecture}) simd_width={lim.simd_width} "
        f"max_threads/tg={lim.max_threads_per_threadgroup} "
        f"tg_mem={lim.max_threadgroup_memory}B (via {lim.source})",
        f"host {host.get('Model Identifier','?')} {host.get('Chip','?')} "
        f"{host.get('Total Number of Cores','?')} cores, {host.get('Memory','?')} RAM, "
        f"macOS {host.get('macOS','?')}",
    ]


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(20240927)
