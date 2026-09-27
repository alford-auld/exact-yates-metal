"""Yates-algorithm butterfly kernel in Metal, exposed through MLX.

A single templated kernel computes the n-fold Kronecker power of any 2x2
generator matrix: Walsh-Hadamard, subset/superset zeta and Mobius, and the
polar-code kernel all differ only in four butterfly constants.
"""

from .variants import (
    VARIANTS, TRANSPOSE, INVERSE, POLAR_KERNEL, Variant,
    get, transpose_of, is_unipotent, bit_loss, guaranteed_bits,
    smith_normal_form_2x2, elementary_divisor_multiplicities,
)
from .reference import (
    reference_transform, numpy_transform, dense_transform,
    kron_power, naive_convolution, ring_bits, SUPPORTED_DTYPES,
)
from .kernel import (
    transform, wht, zeta_sub, mobius_sub, zeta_sup, mobius_sup,
    device_copy, plan_passes, naive_passes, describe_plan, choose_tile_bytes, Pass,
)
from .device import limits, host_info, memory_ceiling_bytes
from . import autodiff
from .autodiff import yates_transform, differentiable, adjoint_of

__all__ = [
    "VARIANTS", "TRANSPOSE", "INVERSE", "POLAR_KERNEL", "Variant",
    "get", "transpose_of", "is_unipotent", "bit_loss", "guaranteed_bits",
    "smith_normal_form_2x2", "elementary_divisor_multiplicities",
    "reference_transform", "numpy_transform", "dense_transform",
    "kron_power", "naive_convolution", "ring_bits", "SUPPORTED_DTYPES",
    "transform", "wht", "zeta_sub", "mobius_sub", "zeta_sup", "mobius_sup",
    "device_copy", "plan_passes", "naive_passes", "describe_plan", "choose_tile_bytes", "Pass",
    "limits", "host_info", "memory_ceiling_bytes",
    "autodiff", "yates_transform", "differentiable", "adjoint_of",
]
