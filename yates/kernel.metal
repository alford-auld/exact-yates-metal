// Yates-algorithm butterfly kernel, templated on a 2x2 generator matrix.
//
// This file is consumed by yates/kernel.py, which splits it on the @section
// markers and hands the pieces to mx.fast.metal_kernel (the `header` string and
// the `source` body respectively -- MLX generates the function signature).
//
// EXACTNESS: every integer path uses uint / ulong.  Unsigned overflow in MSL is
// defined wraparound, so the arithmetic is exact reduction mod 2^32 / 2^64.
// Signed overflow is undefined behaviour, so int / long are never used to hold
// transform data -- only for compile-time template parameters and loop counters.
//
// Matrix entries are passed as (sign, magnitude) pairs of NON-NEGATIVE template
// integers.  MLX interpolates template integers directly into the generated C++
// identifier, and a negative value produces an invalid identifier, so -1 must be
// encoded rather than passed literally.

// ===== @section header =====
#ifndef YATES_HEADER
#define YATES_HEADER

#define YATES_MPARAMS \
  int S00, int A00, int S01, int A01, int S10, int A10, int S11, int A11
#define YATES_MARGS S00, A00, S01, A01, S10, A10, S11, A11

// One matrix entry applied to a ring element.  SGN is 0 (+) or 1 (-), MAG is
// |entry|.  All three branches fold at compile time.
template <typename T, int SGN, int MAG>
inline T yates_cmul(T x) {
  if (MAG == 0) {
    return T(0);
  }
  if (MAG == 1) {
    return (SGN != 0) ? T(T(0) - x) : x;
  }
  T v = x * T(MAG);
  return (SGN != 0) ? T(T(0) - v) : v;
}

// metal::simd_shuffle_xor is a per-32-bit-register operation on Apple GPUs: on
// this hardware it silently returns garbage for 64-bit operands (measured, see
// tests/test_simd_shuffle.py).  64-bit ring elements are therefore shuffled as
// two 32-bit halves.
template <typename T>
struct yates_shuffle {
  static inline T xor_lane(T v, ushort mask) {
    return metal::simd_shuffle_xor(v, mask);
  }
};

template <>
struct yates_shuffle<ulong> {
  static inline ulong xor_lane(ulong v, ushort mask) {
    uint lo = metal::simd_shuffle_xor(uint(v & 0xffffffffUL), mask);
    uint hi = metal::simd_shuffle_xor(uint(v >> 32), mask);
    return (ulong(hi) << 32) | ulong(lo);
  }
};

// Full butterfly, both halves owned by this thread.
//   a' = m00*a + m01*b
//   b' = m10*a + m11*b
template <typename T, YATES_MPARAMS>
inline void yates_pair(thread T &a, thread T &b) {
  T na = yates_cmul<T, S00, A00>(a) + yates_cmul<T, S01, A01>(b);
  T nb = yates_cmul<T, S10, A10>(a) + yates_cmul<T, S11, A11>(b);
  a = na;
  b = nb;
}

// Half butterfly: this thread owns one element and has fetched its partner.
// is_hi is false when this thread owns i0, true when it owns i1.
template <typename T, YATES_MPARAMS>
inline T yates_half(T mine, T other, bool is_hi) {
  if (is_hi) {
    return yates_cmul<T, S10, A10>(other) + yates_cmul<T, S11, A11>(mine);
  }
  return yates_cmul<T, S00, A00>(mine) + yates_cmul<T, S01, A01>(other);
}

#endif  // YATES_HEADER

// ===== @section pass =====
// One pass over the array, applying transform stages for index bits
// [SOFF, SOFF + P).  Stages commute (they act on distinct tensor factors), so a
// pass may take any contiguous run of bits.
//
// Tile geometry.  A tile is 2^P "rows" (the bits this pass transforms, stride
// 2^SOFF apart) by C = 2^LOGC contiguous "columns" (low bits, untouched here and
// present purely so that neighbouring threads touch neighbouring addresses).
// G = 2^LOGG independent tiles share a threadgroup so that small transforms
// still fill one.
//
//   threads per group  TG = C * R * G          (R = 2^LOGR rows resident at once)
//   registers/thread   E  = 2^P / R
//
// Three tiers, by the distance between butterfly partners:
//   bits [LOGR, P)        partner lives in this thread's own registers -- free
//   bits [0, NB_SIMD)     partner is another lane of this simdgroup -- shuffle
//   bits [NB_SIMD, LOGR)  partner is another thread -- threadgroup memory
// NB_SIMD is derived from LOGW, the *measured* log2 of this GPU's SIMD width.
  constexpr int C     = 1 << LOGC;
  constexpr int R     = 1 << LOGR;
  constexpr int G     = 1 << LOGG;
  constexpr int LOGTG = LOGC + LOGR + LOGG;
  constexpr int E     = 1 << (P - LOGR);

  constexpr int SIMD_RAW = LOGW - LOGC;
  constexpr int NB_SIMD =
      SIMD_RAW < 0 ? 0 : (SIMD_RAW > LOGR ? LOGR : SIMD_RAW);
  constexpr int TGBUF = (LOGR > NB_SIMD) ? (1 << (P + LOGC + LOGG)) : 1;

  threadgroup T tile[TGBUF];

  const uint tid = thread_position_in_threadgroup.x;
  const uint col = tid & uint(C - 1);
  const uint rp  = (tid >> LOGC) & uint(R - 1);
  const uint sub = tid >> (LOGC + LOGR);

  // Flat index = batch*N + high*2^(SOFF+P) + row*2^SOFF + lowhi*C + col.
  // (batch, high) and lowhi together enumerate tiles.
  const uint tile_id = threadgroup_position_in_grid.x * uint(G) + sub;
  const uint low_hi  = tile_id & ((1u << (SOFF - LOGC)) - 1u);
  const uint bh      = tile_id >> (SOFF - LOGC);
  const ulong base   = (ulong(bh) << (SOFF + P)) + (ulong(low_hi) << LOGC) + col;

  ulong addr[E];
  T x[E];
#pragma clang loop unroll(full)
  for (int m = 0; m < E; ++m) {
    addr[m] = base + (ulong(rp + (uint(m) << LOGR)) << SOFF);
    x[m] = inp[addr[m]];
  }

  // ---- register tier: partner is a different m of the same thread ----
#pragma clang loop unroll(full)
  for (int b = LOGR; b < P; ++b) {
    const int bit = 1 << (b - LOGR);
#pragma clang loop unroll(full)
    for (int m = 0; m < E; ++m) {
      if ((m & bit) == 0) {
        yates_pair<T, YATES_MARGS>(x[m], x[m | bit]);
      }
    }
  }

  // ---- register tier: partner is another lane, reached by simd shuffle ----
#pragma clang loop unroll(full)
  for (int b = 0; b < NB_SIMD; ++b) {
    const ushort mask = ushort(uint(C) << b);
    const bool is_hi = ((rp >> b) & 1u) != 0u;
#pragma clang loop unroll(full)
    for (int m = 0; m < E; ++m) {
      T other = yates_shuffle<T>::xor_lane(x[m], mask);
      x[m] = yates_half<T, YATES_MARGS>(x[m], other, is_hi);
    }
  }

  // ---- threadgroup tier: partner is another thread of this threadgroup ----
#pragma clang loop unroll(full)
  for (int b = NB_SIMD; b < LOGR; ++b) {
    const uint partner = tid ^ (uint(C) << b);
    const bool is_hi = ((rp >> b) & 1u) != 0u;
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    for (int m = 0; m < E; ++m) {
      tile[(m << LOGTG) + tid] = x[m];
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    for (int m = 0; m < E; ++m) {
      x[m] = yates_half<T, YATES_MARGS>(x[m], tile[(m << LOGTG) + partner], is_hi);
    }
  }

  // ---- optional isometry scaling, folded into the final pass ----
  if constexpr (NRM != 0) {
    T s = T(1) / T(1u << NRME);
    if (NRMH != 0) {
      s = s * T(0.70710678118654752440);
    }
#pragma clang loop unroll(full)
    for (int m = 0; m < E; ++m) {
      x[m] = x[m] * s;
    }
  }

#pragma clang loop unroll(full)
  for (int m = 0; m < E; ++m) {
    out[addr[m]] = x[m];
  }

// ===== @section copy =====
// Pure device-to-device copy, VEC elements per thread.  This is the denominator
// for every bandwidth number reported by bench/.
  const ulong i = ulong(thread_position_in_grid.x) * VEC;
#pragma clang loop unroll(full)
  for (int k = 0; k < VEC; ++k) {
    out[i + k] = inp[i + k];
  }
