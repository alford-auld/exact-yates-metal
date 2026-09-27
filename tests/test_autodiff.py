"""Autodiff: every VJP is the same kernel with a transposed generator.

The transpose identity itself is checked in test_group6_transpose_adjoint.py;
this file covers the plumbing, composition, and the MLX limitations that shape
the public API.
"""

import inspect
import re
import warnings

import mlx.core as mx
import numpy as np
import pytest

import yates
from yates import autodiff
from conftest import ALL_VARIANTS, np_of


def test_there_is_no_backward_kernel():
    """One transform kernel body exists; the backward pass reuses it.

    kernel.metal holds exactly one transform kernel ("pass"), one helper header
    and one benchmark copy kernel.  autodiff.py compiles no Metal of its own and
    reaches the GPU only through the shared forward entry point.
    """
    assert set(yates.kernel._sections()) == {"header", "pass", "copy"}
    assert "@section" not in inspect.getsource(autodiff)

    src = inspect.getsource(autodiff)
    assert "metal_kernel" not in src, "autodiff must not compile its own kernel"
    assert "kernel.metal" not in src
    # The only GPU entry point used by the backward pass is the forward one.
    assert re.findall(r"\btransform\(", src)
    assert not re.findall(r"\b_run_pass\b|\bmx\.fast\b", src)


@pytest.mark.parametrize("name", ALL_VARIANTS)
def test_grad_uses_the_transposed_generator(name, rng):
    n = 7
    g = mx.array(rng.standard_normal(1 << n).astype(np.float32))
    x = mx.array(rng.standard_normal(1 << n).astype(np.float32))
    grad = np_of(mx.grad(lambda a: mx.sum(yates.yates_transform(a, name) * g))(x))
    want = yates.kron_power(name, n).T @ np_of(g)
    assert np.allclose(grad, want, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("name", ALL_VARIANTS)
def test_value_and_grad_and_batching(name, rng):
    n, b = 6, 4
    x = mx.array(rng.standard_normal((b, 1 << n)).astype(np.float32))
    fn = mx.value_and_grad(lambda a: mx.sum(yates.yates_transform(a, name) ** 2))
    val, grad = fn(x)
    y = yates.transform(x, name)
    assert np.isclose(float(val), float(mx.sum(y ** 2)), rtol=1e-5)
    want = np_of(yates.transform(2.0 * y, yates.TRANSPOSE[name]))
    assert np.allclose(np_of(grad), want, rtol=1e-4, atol=1e-3 * max(1.0, float(np.abs(want).max())))


def test_gradient_through_a_zeta_mobius_composition_is_the_identity(rng):
    x = mx.array(rng.standard_normal((3, 64)).astype(np.float32))
    g = mx.array(rng.standard_normal((3, 64)).astype(np.float32))

    def loss(a):
        return mx.sum(yates.autodiff.mobius_sub(yates.autodiff.zeta_sub(a)) * g)

    assert np.allclose(np_of(mx.grad(loss)(x)), np_of(g), atol=1e-5)


def test_normalized_wht_gradient_is_its_own_adjoint(rng):
    n = 8
    g = mx.array(rng.standard_normal(1 << n).astype(np.float32))
    x = mx.array(rng.standard_normal(1 << n).astype(np.float32))
    grad = np_of(mx.grad(lambda a: mx.sum(yates.autodiff.wht(a, normalize=True) * g))(x))
    want = np_of(yates.transform(g, "WHT", normalize=True))
    assert np.allclose(grad, want, rtol=1e-5, atol=1e-5)


def test_explicit_vjp_and_jvp_helpers(rng):
    n = 5
    v = mx.array(rng.standard_normal(1 << n).astype(np.float32))
    for name in ALL_VARIANTS:
        assert np.allclose(np_of(autodiff.vjp(v, name)),
                           yates.kron_power(name, n).T @ np_of(v), atol=1e-4)
        assert np.allclose(np_of(autodiff.jvp(v, name)),
                           yates.kron_power(name, n) @ np_of(v), atol=1e-4)


def test_differentiable_functions_are_cached():
    a = autodiff.differentiable("WHT", -1, False)
    assert a is autodiff.differentiable("WHT", -1, False)
    assert a is not autodiff.differentiable("WHT", -1, True)


def test_mlx_vmap_and_jvp_limitation_is_still_present():
    """MLX 0.32.2 bypasses custom_function .vmap/.jvp rules for CustomKernel bodies.

    Warns if a newer MLX has fixed it, so those rules can be added back.
    """
    f = autodiff.differentiable("WHT", -1, False)
    x = mx.zeros((4, 16))
    fixed = []
    for label, call in [
        ("vmap", lambda: mx.eval(mx.vmap(f)(x))),
        ("jvp", lambda: mx.eval(mx.jvp(f, [mx.zeros((16,))], [mx.ones((16,))])[1])),
    ]:
        try:
            call()
            fixed.append(label)
        except ValueError as exc:
            assert "Not implemented for CustomKernel" in str(exc), str(exc)
    if fixed:
        warnings.warn(
            f"MLX {mx.__version__} now supports {fixed} through CustomKernel; "
            "yates/autodiff.py can register those rules again.", stacklevel=1)

    # Whatever MLX does, native batching and the linear-map JVP always work.
    assert np_of(yates.transform(x, "WHT")).shape == (4, 16)
