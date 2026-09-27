// Pointwise k-th power, optionally modular, optionally sign-folded.
//
// This is the only place the algorithm needs modular *multiplication*.  The
// zeta and Mobius butterflies need addition and subtraction only, so they come
// straight from yates/ unmodified -- see apps/chromatic/count.py for why the
// modulus is kept below 2^31 to make that work.

// ===== @section powmod =====
// inp : uint32  i(S), the number of independent subsets of S
// par : uint64  [p, k, n]
// out : uint64  i(S)^k  (mod p if MODULAR, else mod 2^64), times (-1)^(n-|S|)
//               if SIGNED
  const uint s = thread_position_in_grid.x;
  const ulong p = par[0];
  const uint k = uint(par[1]);
  const uint n = uint(par[2]);

  ulong b = ulong(inp[s]);
  ulong r;
  uint e = k;

  if (MODULAR != 0) {
    // p < 2^31, so every product below is < 2^62 and cannot overflow a ulong.
    b %= p;
    r = 1ul % p;
    while (e != 0u) {
      if ((e & 1u) != 0u) {
        r = (r * b) % p;
      }
      b = (b * b) % p;
      e >>= 1;
    }
  } else {
    // Unsigned wraparound is defined in MSL, so this is exact mod 2^64.
    r = 1ul;
    while (e != 0u) {
      if ((e & 1u) != 0u) {
        r = r * b;
      }
      b = b * b;
      e >>= 1;
    }
  }

  if (SIGNED != 0) {
    // Fold the inclusion-exclusion sign (-1)^(n-|S|) in here so the alternating
    // reduction is a plain sum.  Negation is 0 - r: exact in two's complement.
    if (((n - uint(metal::popcount(s))) & 1u) != 0u) {
      r = ulong(0) - r;
    }
  }
  out[s] = r;
