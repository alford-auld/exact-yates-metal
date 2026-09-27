"""Autodiff for the Yates transform, via the matrix transpose identity.

``(M^(x)n)^T == (M^T)^(x)n``.  The transform is linear, so its VJP is its
adjoint, which is *the same kernel* run with a transposed generator.  There is
no backward kernel in this package -- :data:`yates.variants.TRANSPOSE` is the
whole of the backward pass:

===========  =============
forward      backward
===========  =============
WHT          WHT   (self-adjoint)
ZETA_SUB     ZETA_SUP
ZETA_SUP     ZETA_SUB
MOB_SUB      MOB_SUP
MOB_SUP      MOB_SUB
===========  =============

Gradients require a floating-point element type; MLX does not differentiate
integer arrays.

Scope, measured on MLX 0.32.2
-----------------------------
Reverse mode (``mx.grad``, ``mx.value_and_grad``, ``mx.vjp``) works, including
through repeated application.  Forward mode (``mx.jvp``) and ``mx.vmap`` do
*not*: when a ``custom_function`` body contains a ``CustomKernel`` primitive,
MLX consults the registered ``.vjp`` rule but bypasses ``.jvp`` and ``.vmap``,
descending into the kernel and raising ``[Primitive::jvp] Not implemented for
CustomKernel``.  Registering those rules here would be dead code, so they are
deliberately absent and :func:`jvp` below is offered instead -- the transform is
linear, so its directional derivative is just the transform of the tangent.
Batching needs no vmap either: :func:`yates.transform` already handles every
leading axis in a single launch.  ``tests/test_autodiff.py`` pins this
behaviour so a future MLX release that fixes it is noticed.
"""

from __future__ import annotations

import functools
from typing import Callable, Tuple

import mlx.core as mx

from .kernel import transform
from .variants import get, transpose_of


@functools.lru_cache(maxsize=None)
def differentiable(
    variant_name: str, axis: int = -1, normalize: bool = False
) -> Callable[[mx.array], mx.array]:
    """A :class:`mx.custom_function` computing ``M^(x)n x`` with an exact VJP.

    Cached per ``(variant, axis, normalize)``.
    """
    fwd_v = get(variant_name)
    bwd_v = transpose_of(fwd_v)

    @mx.custom_function
    def yates_fn(x: mx.array) -> mx.array:
        return transform(x, fwd_v, axis=axis, normalize=normalize)

    @yates_fn.vjp
    def _vjp(primals, cotangent, output):
        # d/dx <A x, g> = A^T g, and A^T = (M^T)^(x)n: the same kernel with a
        # transposed generator.  The isometry scale factor is its own adjoint.
        del primals, output
        return transform(cotangent, bwd_v, axis=axis, normalize=normalize)

    return yates_fn


def yates_transform(
    x: mx.array, variant, axis: int = -1, normalize: bool = False
) -> mx.array:
    """Differentiable Yates transform.  Drop-in for :func:`yates.transform`."""
    return differentiable(get(variant).name, axis, normalize)(x)


def adjoint_of(variant) -> str:
    """Name of the generator used for the backward pass of ``variant``."""
    return transpose_of(variant).name


def vjp(cotangent: mx.array, variant, axis: int = -1, normalize: bool = False):
    """The adjoint applied directly: ``(M^(x)n)^T g == (M^T)^(x)n g``."""
    return transform(cotangent, transpose_of(variant), axis=axis, normalize=normalize)


def jvp(tangent: mx.array, variant, axis: int = -1, normalize: bool = False):
    """Directional derivative.  A linear map is its own JVP.

    Use this instead of ``mx.jvp``; see the module docstring for why.
    """
    return transform(tangent, variant, axis=axis, normalize=normalize)


# named differentiable entry points ----------------------------------------

def wht(x, axis: int = -1, normalize: bool = False):
    """Differentiable Walsh-Hadamard transform (self-adjoint)."""
    return yates_transform(x, "WHT", axis, normalize)


def zeta_sub(x, axis: int = -1):
    """Differentiable subset-sum transform; adjoint is ZETA_SUP."""
    return yates_transform(x, "ZETA_SUB", axis)


def zeta_sup(x, axis: int = -1):
    """Differentiable superset-sum transform; adjoint is ZETA_SUB."""
    return yates_transform(x, "ZETA_SUP", axis)


def mobius_sub(x, axis: int = -1):
    """Differentiable subset Mobius inversion; adjoint is MOB_SUP."""
    return yates_transform(x, "MOB_SUB", axis)


def mobius_sup(x, axis: int = -1):
    """Differentiable superset Mobius inversion; adjoint is MOB_SUB."""
    return yates_transform(x, "MOB_SUP", axis)
