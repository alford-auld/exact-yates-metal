"""Transform variants: 2x2 generator matrices and the exactness contract.

Every transform in this package is the n-fold Kronecker power ``M^(x)n`` of a
2x2 matrix ``M`` over a commutative ring, applied to a vector of length
``N = 2**n``.  The five named variants differ only in the four entries of M.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Dict, Tuple

Matrix = Tuple[Tuple[int, int], Tuple[int, int]]


@dataclass(frozen=True)
class Variant:
    """A named 2x2 generator matrix."""

    name: str
    m: Matrix
    semantics: str

    @property
    def entries(self) -> Tuple[int, int, int, int]:
        (m00, m01), (m10, m11) = self.m
        return m00, m01, m10, m11

    @property
    def matrix_T(self) -> Matrix:
        (m00, m01), (m10, m11) = self.m
        return ((m00, m10), (m01, m11))

    @property
    def det(self) -> int:
        (m00, m01), (m10, m11) = self.m
        return m00 * m11 - m01 * m10


VARIANTS: Dict[str, Variant] = {
    "WHT": Variant(
        "WHT",
        ((1, 1), (1, -1)),
        "Fourier transform on (Z/2)^n; diagonalizes XOR-convolution",
    ),
    "ZETA_SUB": Variant(
        "ZETA_SUB",
        ((1, 0), (1, 1)),
        "zeta over subsets: f(S) = sum_{T subset S} f(T); diagonalizes OR-convolution",
    ),
    "MOB_SUB": Variant(
        "MOB_SUB",
        ((1, 0), (-1, 1)),
        "Mobius inversion over subsets; inverse of ZETA_SUB",
    ),
    "ZETA_SUP": Variant(
        "ZETA_SUP",
        ((1, 1), (0, 1)),
        "zeta over supersets: f(S) = sum_{T superset S} f(T); diagonalizes AND-convolution",
    ),
    "MOB_SUP": Variant(
        "MOB_SUP",
        ((1, -1), (0, 1)),
        "Mobius inversion over supersets; inverse of ZETA_SUP",
    ),
}

#: ``(M^(x)n)^T == (M^T)^(x)n``, so every adjoint is the same kernel with a
#: transposed generator.  This table is the whole of the backward pass.
TRANSPOSE: Dict[str, str] = {
    "WHT": "WHT",          # H is symmetric: the WHT is self-adjoint
    "ZETA_SUB": "ZETA_SUP",
    "ZETA_SUP": "ZETA_SUB",
    "MOB_SUB": "MOB_SUP",
    "MOB_SUP": "MOB_SUB",
}

#: Ring-exact inverses.  WHT is its own inverse only up to the factor ``N``.
INVERSE: Dict[str, str] = {
    "WHT": "WHT",          # WHT(WHT(x)) == N * x
    "ZETA_SUB": "MOB_SUB",
    "MOB_SUB": "ZETA_SUB",
    "ZETA_SUP": "MOB_SUP",
    "MOB_SUP": "ZETA_SUP",
}

#: ZETA_SUB is the polar-code kernel F = [[1,0],[1,1]]; ZETA_SUP is its transpose.
POLAR_KERNEL = "ZETA_SUB"


def get(variant) -> Variant:
    """Resolve a name, a :class:`Variant`, or a raw 2x2 matrix to a Variant."""
    if isinstance(variant, Variant):
        return variant
    if isinstance(variant, str):
        try:
            return VARIANTS[variant.upper()]
        except KeyError:
            raise ValueError(
                f"unknown variant {variant!r}; known: {sorted(VARIANTS)}"
            ) from None
    m = tuple(tuple(int(v) for v in row) for row in variant)
    if len(m) != 2 or any(len(r) != 2 for r in m):
        raise ValueError(f"generator matrix must be 2x2, got {variant!r}")
    return Variant("CUSTOM", m, "user-supplied generator")


def transpose_of(variant) -> Variant:
    """The adjoint generator.  ``(M^(x)n)^T = (M^T)^(x)n``."""
    v = get(variant)
    if v.name in TRANSPOSE:
        return VARIANTS[TRANSPOSE[v.name]]
    return Variant(v.name + "_T", v.matrix_T, "transpose of " + v.semantics)


# --------------------------------------------------------------------------
# Smith normal form and the exactness contract over Z/2^k
# --------------------------------------------------------------------------


def smith_normal_form_2x2(m: Matrix) -> Tuple[int, int]:
    """Elementary divisors ``(d1, d2)`` of an integral 2x2 matrix, ``d1 | d2``.

    For a 2x2 integer matrix ``d1 = gcd(all entries)`` and ``d1 * d2 = |det|``.
    """
    (m00, m01), (m10, m11) = m
    d1 = gcd(gcd(abs(m00), abs(m01)), gcd(abs(m10), abs(m11)))
    det = abs(m00 * m11 - m01 * m10)
    if d1 == 0:
        return 0, 0
    if det == 0:
        return d1, 0
    return d1, det // d1


def _v2(x: int) -> int:
    """2-adic valuation."""
    if x == 0:
        raise ValueError("v2(0) is undefined")
    k = 0
    while x % 2 == 0:
        x //= 2
        k += 1
    return k


def is_unipotent(variant) -> bool:
    """True when SNF(M) is the identity, i.e. M is invertible over Z."""
    return smith_normal_form_2x2(get(variant).m) == (1, 1)


def elementary_divisor_multiplicities(variant, n: int) -> Dict[int, int]:
    """Elementary divisors of ``M^(x)n`` with multiplicities.

    ``SNF(A (x) B)`` has elementary divisor multiset ``{d_i * e_j}``, so the
    n-fold Kronecker power of ``diag(d1, d2)`` gives ``d1^(n-j) d2^j`` with
    multiplicity ``binomial(n, j)``.
    """
    from math import comb

    d1, d2 = smith_normal_form_2x2(get(variant).m)
    out: Dict[int, int] = {}
    for j in range(n + 1):
        d = d1 ** (n - j) * d2**j
        out[d] = out.get(d, 0) + comb(n, j)
    return out


def bit_loss(variant, n: int) -> int:
    """Number of low-order bits lost by a forward/inverse round trip mod 2^k.

    The largest elementary divisor of ``M^(x)n`` is ``d2^n``; inverting the
    transform over ``Z/2^k`` requires dividing by it, so the round trip only
    determines the input modulo ``2**(k - v2(d2) * n)``.

    * unipotent generators (all four zeta/Mobius variants): ``d2 = 1`` -> 0 bits
    * Hadamard: ``SNF(H_2) = diag(1, 2)`` -> exactly ``n`` bits
    """
    d1, d2 = smith_normal_form_2x2(get(variant).m)
    if d1 != 1:
        raise ValueError(
            f"generator {get(variant).m} is not primitive (gcd of entries {d1} != 1); "
            "it is not injective over Z/2^k"
        )
    if d2 == 0:
        raise ValueError("singular generator: transform is not invertible")
    if d2 % 2 != 0:
        return 0
    return _v2(d2) * n


def guaranteed_bits(variant, n: int, bits: int) -> int:
    """How many low bits of ``inverse(forward(x))`` are guaranteed to equal x.

    ``bits`` is the width of the unsigned ring, 32 or 64.  A return value equal
    to ``bits`` means the round trip is bit-for-bit exact.
    """
    lost = bit_loss(variant, n)
    return max(0, bits - lost)
