// Pointwise k-th power, optionally modular, optionally sign-folded.
//
// This is the only place the algorithm needs modular *multiplication*.  The
// zeta and Mobius butterflies need addition and subtraction only, so they come
// straight from yates/ unmodified -- see apps/chromatic/count.py for why the
// modulus is kept below 2^31 to make that work.
//
// MODMODE selects the arithmetic:
//   0  no modulus: wrapping uint64, i.e. exact mod 2^64
//   1  generic:    a 64-bit `%` per multiply.  Correct for any modulus, and
//                  measurably slow -- 64-bit integer division is not cheap on
//                  this GPU (26 GB/s effective vs ~97 GB/s for the same kernel
//                  without it).  Kept for even moduli and as a cross-check.
//   2  Montgomery: for odd p < 2^31.  No division at all; the pass returns to
//                  being bandwidth-bound.

// ===== @section powmod =====
// inp : uint32  i(S), the number of independent subsets of S
// par : uint64  [p, k, n, pinv, r2]
//                 pinv = -p^-1 mod 2^32,  r2 = 2^64 mod p
// out : uint64  i(S)^k  (mod p, or mod 2^64), times (-1)^(n-|S|) if SIGNED
  const uint s = thread_position_in_grid.x;
  const ulong p = par[0];
  const uint k = uint(par[1]);
  const uint n = uint(par[2]);
  const uint pinv = uint(par[3]);
  const ulong r2 = par[4];

  ulong r;
  uint e = k;

  if (MODMODE == 2) {
    // Montgomery form with R = 2^32.  For a,b < p < 2^31:
    //   a*b < 2^62,  m*p < 2^63,  so a*b + m*p < 2^64 and nothing overflows.
    ulong b = ulong(inp[s]) % p;
    b = yates_montmul(b, r2, p, pinv);          // into Montgomery form
    r = yates_montmul(1ul, r2, p, pinv);        // == R mod p, the Montgomery 1
    while (e != 0u) {
      if ((e & 1u) != 0u) {
        r = yates_montmul(r, b, p, pinv);
      }
      b = yates_montmul(b, b, p, pinv);
      e >>= 1;
    }
    r = yates_montmul(r, 1ul, p, pinv);         // back out of Montgomery form
  } else if (MODMODE == 1) {
    // p < 2^31, so every product below is < 2^62 and cannot overflow a ulong.
    ulong b = ulong(inp[s]) % p;
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
    ulong b = ulong(inp[s]);
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

// ===== @section header =====
#ifndef YATES_CHROMATIC_HEADER
#define YATES_CHROMATIC_HEADER

// Montgomery multiplication modulo an odd p < 2^31, with R = 2^32.
//
//   REDC(T) = (T + ((T mod R) * (-p^-1 mod R) mod R) * p) / R
//
// The added multiple of p is chosen so the low 32 bits cancel, making the
// division by R a shift.  With a,b < p the result is < 2p, so one conditional
// subtraction normalises it.
inline ulong yates_montmul(ulong a, ulong b, ulong p, uint pinv) {
  const ulong t = a * b;                      // < p^2 < 2^62
  const uint m = uint(t) * pinv;              // wraps mod 2^32, as intended
  const ulong u = (t + ulong(m) * p) >> 32;   // t + m*p < 2^64; exact shift
  return (u >= p) ? (u - p) : u;
}

#endif  // YATES_CHROMATIC_HEADER
