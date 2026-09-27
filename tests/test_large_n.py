"""Large-n correctness, where the wide-column high-stride passes are used.

The default tiering plan only reaches C >= 128 columns and four device passes
above n ~ 24, so the rest of the suite never exercises those templates.  These
tests do, using two checks that need no host-side oracle of the same size:

* the exact zeta/Mobius round trip, which must be bitwise identity, and
* agreement with the forced-naive path (one unfused device pass per stage,
  every optimisation tier disabled).

Marked slow: multi-GiB allocations and tens of GiB of device traffic.
Run with:  .venv/bin/python -m pytest -m slow tests/test_large_n.py
"""

import mlx.core as mx
import numpy as np
import pytest

import yates
from kernel_helpers import np_of

pytestmark = pytest.mark.slow


def _fits(n: int, itemsize: int, copies: int = 3) -> bool:
    return (1 << n) * itemsize * copies < yates.limits().max_recommended_working_set_size


@pytest.mark.parametrize("n", [24, 25, 26, 27, 28, 29])
def test_exact_roundtrip_at_large_n(n, rng):
    if not _fits(n, 4):
        pytest.skip(f"n={n} does not fit in the detected working set")
    x = mx.array(rng.integers(0, 1 << 32, size=1 << n, dtype=np.uint32))
    mx.eval(x)
    plan = yates.describe_plan(n, x.size, mx.uint32)
    assert plan["num_passes"] >= 3, plan
    back = yates.transform(yates.transform(x, "ZETA_SUB"), "MOB_SUB")
    assert bool(mx.all(back == x).item()), f"n={n} round trip not exact"
    del x, back
    mx.clear_cache()


@pytest.mark.parametrize("n", [24, 26])
def test_tiered_matches_naive_at_large_n(n, rng):
    """The wide-column plans agree with one-stage-per-pass on the same GPU."""
    if not _fits(n, 4, copies=4):
        pytest.skip(f"n={n} does not fit in the detected working set")
    x = mx.array(rng.integers(0, 1 << 32, size=1 << n, dtype=np.uint32))
    mx.eval(x)
    tiered = yates.transform(x, "WHT")
    naive = yates.transform(x, "WHT", naive=True)
    assert bool(mx.all(tiered == naive).item()), f"n={n} tiered != naive"
    del x, tiered, naive
    mx.clear_cache()


@pytest.mark.parametrize("n", [24, 26])
def test_plan_invariance_at_large_n(n, rng):
    """Different tile budgets pick different C and pass counts; same answer."""
    if not _fits(n, 4, copies=4):
        pytest.skip(f"n={n} does not fit in the detected working set")
    x = mx.array(rng.integers(0, 1 << 32, size=1 << n, dtype=np.uint32))
    mx.eval(x)
    ref = yates.transform(x, "WHT", tile_bytes=32768)
    for tb in (4096, 8192, 16384):
        got = yates.transform(x, "WHT", tile_bytes=tb)
        assert bool(mx.all(got == ref).item()), f"n={n} tile_bytes={tb}"
        del got
    del x, ref
    mx.clear_cache()
