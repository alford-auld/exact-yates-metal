"""The AES S-box, derived from its definition rather than pasted as a table."""

import numpy as np

AES_MODULUS = 0x11B


def gf_mul(a: int, b: int) -> int:
    """Multiplication in GF(2^8) with the AES modulus x^8+x^4+x^3+x+1."""
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        if a & 0x100:
            a ^= AES_MODULUS
    return r


def gf_inv(a: int) -> int:
    """Multiplicative inverse in GF(2^8); 0 maps to 0 by AES convention."""
    if a == 0:
        return 0
    for b in range(1, 256):
        if gf_mul(a, b) == 1:
            return b
    raise AssertionError("unreachable: GF(2^8) is a field")


def _affine(a: int) -> int:
    """b_i = a_i ^ a_{i+4} ^ a_{i+5} ^ a_{i+6} ^ a_{i+7} ^ c_i, c = 0x63."""
    out = 0
    for i in range(8):
        bit = (
            (a >> i)
            ^ (a >> ((i + 4) % 8))
            ^ (a >> ((i + 5) % 8))
            ^ (a >> ((i + 6) % 8))
            ^ (a >> ((i + 7) % 8))
            ^ (0x63 >> i)
        ) & 1
        out |= bit << i
    return out


def sbox() -> np.ndarray:
    return np.array([_affine(gf_inv(x)) for x in range(256)], dtype=np.uint16)
