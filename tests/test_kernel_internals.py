"""Kernel-level tests: tier isolation, tiling policy, dtypes, error handling."""

import mlx.core as mx
import numpy as np
import pytest

import yates
from yates import kernel as K
from conftest import ALL_VARIANTS, UINT_DTYPES, np_of, rand_uint

LOGW = (yates.limits().simd_width).bit_length() - 1


def _run(x, variant, passes):
    """Drive an explicit pass list through the kernel (bypasses the planner)."""
    flat = mx.reshape(mx.contiguous(x), (-1,))
    codes = K.encode_matrix(variant)
    for ps in passes:
        flat = K._run_pass(flat, ps, codes, LOGW, (0, 0, 0))
    return mx.reshape(flat, x.shape)


# --------------------------------------------------------------------------
# each tier in isolation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL_VARIANTS)
def test_register_tier_alone(name, rng):
    """logr=0 puts every butterfly of the pass inside one thread's registers."""
    n = 3
    ps = [K.Pass(s=0, p=n, logc=0, logr=0, logg=5)]
    assert ps[0].tiers(LOGW) == {"simd_shuffle": 0, "threadgroup": 0, "register": n}
    x = rand_uint(rng, (64, 1 << n), np.uint32)
    assert np.array_equal(np_of(_run(mx.array(x), name, ps)),
                          yates.numpy_transform(x, name))


@pytest.mark.parametrize("name", ALL_VARIANTS)
def test_simd_shuffle_tier_alone(name, rng):
    """p == logr == log2(simd width) and logc == 0: pure simd_shuffle_xor."""
    n = LOGW
    ps = [K.Pass(s=0, p=n, logc=0, logr=n, logg=0)]
    assert ps[0].tiers(LOGW) == {"simd_shuffle": n, "threadgroup": 0, "register": 0}
    assert ps[0].threadgroup_bytes(4, LOGW) == 4, "no threadgroup buffer should be used"
    x = rand_uint(rng, (64, 1 << n), np.uint32)
    assert np.array_equal(np_of(_run(mx.array(x), name, ps)),
                          yates.numpy_transform(x, name))


@pytest.mark.parametrize("name", ALL_VARIANTS)
def test_threadgroup_tier_alone(name, rng):
    """logc >= log2(simd width) pushes every exchange into threadgroup memory."""
    n = LOGW + 3
    first = K.Pass(s=0, p=LOGW, logc=0, logr=LOGW, logg=0)          # simd only
    second = K.Pass(s=LOGW, p=3, logc=LOGW, logr=3, logg=0)         # threadgroup only
    assert second.tiers(LOGW) == {"simd_shuffle": 0, "threadgroup": 3, "register": 0}
    x = rand_uint(rng, (32, 1 << n), np.uint32)
    assert np.array_equal(np_of(_run(mx.array(x), name, [first, second])),
                          yates.numpy_transform(x, name))


def test_all_three_tiers_are_used_by_the_default_policy():
    plan = yates.describe_plan(20, 1 << 20, mx.uint32)
    tiers = plan["stages_by_tier"]
    assert tiers["simd_shuffle"] > 0, plan
    assert tiers["threadgroup"] > 0, plan
    assert tiers["register"] > 0, plan
    assert sum(tiers.values()) == 20


# --------------------------------------------------------------------------
# naive path vs tiered path
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL_VARIANTS)
@pytest.mark.parametrize("mx_dt,np_dt,bits", UINT_DTYPES)
@pytest.mark.parametrize("n", [0, 1, 2, 7, 13, 16])
def test_naive_path_matches_oracle_and_tiered_path(name, mx_dt, np_dt, bits, n, rng):
    x = rand_uint(rng, (3, 1 << n), np_dt)
    a = mx.array(x)
    naive = np_of(yates.transform(a, name, naive=True))
    assert np.array_equal(naive, yates.numpy_transform(x, name)), "naive vs oracle"
    assert np.array_equal(np_of(yates.transform(a, name)), naive), "tiered vs naive"


def test_naive_passes_disable_every_optimisation_tier():
    for ps in yates.naive_passes(10, 1 << 14):
        t = ps.tiers(LOGW)
        assert ps.p == 1 and t["simd_shuffle"] == 0 and t["threadgroup"] == 0


# --------------------------------------------------------------------------
# tiling policy
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tile_bytes", [1024, 2048, 4096, 8192, 16384, 32768])
@pytest.mark.parametrize("name", ["WHT", "ZETA_SUB"])
def test_result_is_independent_of_the_tile_size(tile_bytes, name, rng):
    """Every tiling of the same transform must produce identical bits."""
    n = 17
    x = rand_uint(rng, (1 << n,), np.uint32)
    ref = yates.numpy_transform(x, name)
    got = np_of(yates.transform(mx.array(x), name, tile_bytes=tile_bytes))
    assert np.array_equal(got, ref), f"tile_bytes={tile_bytes}"


def test_plan_respects_the_measured_device_limits():
    lim = yates.limits()
    logw = (lim.simd_width).bit_length() - 1
    for n in range(1, 29):
        for dtype, itemsize in [(mx.uint32, 4), (mx.uint64, 8), (mx.float32, 4)]:
            passes = yates.plan_passes(n, 1 << max(n, 12), itemsize, lim)
            assert sum(p.p for p in passes) == n
            assert [p.s for p in passes] == list(
                np.cumsum([0] + [p.p for p in passes[:-1]])
            )
            for p in passes:
                assert p.threads_per_group <= lim.max_threads_per_threadgroup
                assert p.threadgroup_bytes(itemsize, logw) <= lim.max_threadgroup_memory
                assert p.logc <= p.s or p.s == 0
                assert 0 <= p.logr <= p.p


def test_pass_count_is_minimal_for_the_chosen_tile():
    """Greedy fusion: no plan with the same tile budget uses fewer passes."""
    lim = yates.limits()
    for n in [14, 20, 24, 28]:
        tb = yates.choose_tile_bytes(n, 1 << n, 4, lim)
        got = len(yates.plan_passes(n, 1 << n, 4, lim, tile_bytes=tb))
        alternatives = [
            len(yates.plan_passes(n, 1 << n, 4, lim, tile_bytes=t))
            for t in (1024, 2048, 4096, 8192, 16384, 32768)
            if t <= lim.max_threadgroup_memory
        ]
        assert got == min(alternatives), f"n={n}: {got} vs best {min(alternatives)}"


# --------------------------------------------------------------------------
# dtypes, shapes, errors
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL_VARIANTS)
@pytest.mark.parametrize("n", [0, 1, 6, 12, 15])
def test_float32_matches_the_oracle(name, n, rng):
    x = rng.standard_normal((2, 1 << n)).astype(np.float32)
    got = np_of(yates.transform(mx.array(x), name)).astype(np.float64)
    want = yates.numpy_transform(x.astype(np.float64), name)
    scale = max(1.0, float(np.abs(want).max()))
    assert np.allclose(got, want, rtol=1e-5, atol=1e-5 * scale)


@pytest.mark.parametrize("shape,axis", [
    ((7, 1 << 10), -1), ((3, 5, 1 << 8), -1), ((1 << 9, 6), 0),
    ((4, 1 << 7, 3), 1), ((1, 1 << 12), -1), ((1 << 3,), -1), ((5, 1), -1),
    ((3, 1 << 4, 5), -2),
])
def test_batching_and_axis_selection(shape, axis, rng):
    x = rand_uint(rng, shape, np.uint32)
    for name in ("WHT", "ZETA_SUB", "MOB_SUP"):
        got = np_of(yates.transform(mx.array(x), name, axis=axis))
        assert np.array_equal(got, yates.numpy_transform(x, name, axis=axis))


@pytest.mark.parametrize("size", [3, 5, 6, 12, 0, 100])
def test_non_power_of_two_is_rejected_with_a_clear_error(size):
    x = mx.zeros((size,), dtype=mx.uint32)
    with pytest.raises(ValueError, match="not a power of two"):
        yates.transform(x, "WHT")


def test_n_equals_zero_is_the_identity(rng):
    x = rand_uint(rng, (5, 1), np.uint32)
    for name in ALL_VARIANTS:
        assert np.array_equal(np_of(yates.transform(mx.array(x), name)), x)


def test_n_equals_one_is_the_generator_matrix(rng):
    x = rand_uint(rng, (100, 2), np.uint32)
    for name in ALL_VARIANTS:
        got = np_of(yates.transform(mx.array(x), name))
        assert np.array_equal(got, yates.numpy_transform(x, name))


@pytest.mark.parametrize("dtype", [mx.int32, mx.int64, mx.float16, mx.bfloat16, mx.int8])
def test_signed_and_narrow_dtypes_are_rejected(dtype):
    """Signed overflow is UB in MSL: exact paths must never use int/long."""
    with pytest.raises(TypeError, match="unsupported element type"):
        yates.transform(mx.zeros((8,), dtype=dtype), "WHT")


def test_normalize_is_rejected_for_ring_dtypes():
    for dt in (mx.uint32, mx.uint64):
        with pytest.raises(TypeError, match="not a ring element"):
            yates.transform(mx.zeros((8,), dtype=dt), "WHT", normalize=True)


def test_custom_generator_matrix(rng):
    """The kernel is templated on the matrix, not on the five named variants."""
    m = ((1, 2), (3, -1))
    x = rand_uint(rng, (4, 1 << 9), np.uint32)
    assert np.array_equal(np_of(yates.transform(mx.array(x), m)),
                          yates.numpy_transform(x, m))


def test_unknown_variant_name_is_rejected():
    with pytest.raises(ValueError, match="unknown variant"):
        yates.transform(mx.zeros((8,), dtype=mx.uint32), "HADAMARD")
