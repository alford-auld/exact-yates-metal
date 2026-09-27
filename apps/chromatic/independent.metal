// Independence indicator: a(S) = 1 iff the vertex set S spans no edge.
//
// Split on "// ===== @section NAME =====" by independent.py, same convention as
// yates/kernel.metal.

// ===== @section direct =====
// One thread per subset S, iterating over the set bits of S.  O(2^n * n) work
// but no dependencies at all, so it is a single launch with perfect occupancy.
//
// The per-thread loop runs popcount(S) times, and the *average* popcount over
// the cube is n/2, so the mean cost is n/2 AND-tests.  The early exit below
// cuts it further: most subsets fail on one of their first few vertices.
  const uint s = thread_position_in_grid.x;
  uint rest = s;
  uint bad = 0;
#pragma clang loop unroll(disable)
  while (rest != 0u) {
    const uint v = uint(metal::ctz(rest));
    rest &= rest - 1u;                  // clear lowest set bit
    bad |= (adj[v] & s);
    if (bad != 0u) {
      break;
    }
  }
  out[s] = (bad == 0u) ? T(1) : T(0);

// ===== @section branchless =====
// Same function without the early exit: always popcount(S) iterations, no
// divergent break.  Kept so the benchmark can show which one this GPU prefers.
  const uint s = thread_position_in_grid.x;
  uint rest = s;
  uint bad = 0;
  while (rest != 0u) {
    const uint v = uint(metal::ctz(rest));
    rest &= rest - 1u;
    bad |= (adj[v] & s);
  }
  out[s] = (bad == 0u) ? T(1) : T(0);
