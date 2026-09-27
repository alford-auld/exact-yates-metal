# Exact chromatic number by inclusion–exclusion

Built on the Yates butterfly kernel in [`yates/`](../../README.md); the transform
is reused unmodified.

For `G = (V, E)` with `|V| = n`, let `a(S) = [S is independent]`. The subset zeta
transform of `a` counts independent subsets, and inclusion–exclusion
(Björklund–Husfeldt–Koivisto) turns that into a count of covers:

    i(S) = (ZETA_SUB a)(S)                  # independent sets contained in S
    c_k  = sum_S (-1)^(n-|S|) i(S)^k        # ordered k-tuples of independent
                                            # sets whose union is V
    chi(G) = min { k : c_k > 0 }

`c_k` is exactly the subset Möbius transform of `S -> i(S)^k` read at `S = V`.
Both paths are implemented and cross-checked bitwise, though only the top index
is needed, so the production path folds the sign into the pointwise power and
does a single reduction.

**One subset-zeta over the whole cube, for the entire computation.** It does not
depend on `k` or on the modulus, so it is computed once and shared. Each
candidate `k` then costs one pointwise power and one reduction. `O*(2^n)` time
and `O(2^n)` space for every graph, with no search and no dependence on instance
hardness.

## The exactness contract

**This is the part that determines what an answer means. Read it before using
the number.**

All arithmetic is in unsigned rings, where overflow is defined wraparound.
`i(S) <= 2^n` is exact outright. The `k`-th power and the alternating sum are
computed modulo something, deliberately — and that yields a **one-sided**
guarantee:

| observation | conclusion |
|---|---|
| `c_k mod m != 0` | `c_k != 0`, so `c_k > 0`, so **G is k-colourable**. Sound certificate, for any `m`. |
| `c_k mod m == 0` | either `c_k = 0`, **or** `c_k` is a nonzero multiple of `m`. **False negative possible.** |

The asymmetry is the whole story: `c_k` is a *count*, so a single nonzero
residue proves colourability outright, while a zero residue proves nothing on
its own. Consequently `min { k : c_k mod 2^64 != 0 }` is an **upper bound** on
`chi(G)`, not `chi(G)`. It is always *achievable* — the result object carries a
colouring that realises it — but it is not by itself a proof of minimality.
The single-modulus result is never described as exact anywhere in this code, and
`ChromaticResult.guarantee` states which case applies for every run.

### This is not hypothetical

`tests/chromatic/test_modular.py` pins a constructed instance where the
`mod 2^64` test really does fail. `c_k` is multiplicative over disjoint unions,
and `v2(c_30(K_4)) = 10`, so **seven disjoint copies of K₄ — 28 vertices, inside
the memory ceiling — have `v2(c_30) = 70`**. `c_30` is a 149-digit number and
`c_30 mod 2^64 == 0`: the single-modulus test reports "not 30-colourable" for a
graph that is obviously 30-colourable. An eight-vertex analogue against a `2^16`
modulus runs in the fast suite.

What could *not* be constructed here is an instance where a false negative
corrupts the reported `chi`. That needs the false negative to land at `k = chi`
itself rather than at some `k` the binary search never visits, i.e.
`2^64 | c_chi`. Over the families searched (complete, Turán, cycles, and 280
random graphs on 2–8 vertices) the best ratio `v2(c_chi)/n` is `7/8`, attained
by K₈ — which would need `n >= 74` vertices to reach valuation 64, far past the
~29-vertex memory ceiling. That ratio is pinned by a test, but it is a search
result, not a theorem: it is evidence that the mod-2^64 path is safe in this
range, not a proof that it always is.

### Closing the gap

**Multi-modular (probabilistic).** Recompute `c_k` modulo `r` primes drawn
uniformly at random from `[2^30, 2^31)`. If `c_k != 0`, it has at most
`D = floor(log(c_k) / log(2^30))` distinct prime divisors in that range, and by
Rosser–Schoenfeld the pool size is

    M = pi(2^31) - pi(2^30) > 2^31/ln(2^31) - 1.25506 * 2^30/ln(2^30) > 3.5e7

so one draw hides a nonzero `c_k` with probability at most `D/M`, and `r`
independent draws with probability at most `(D/M)^r`. **The assumption is that
the primes are drawn independently of the instance** — true here, since the seed
does not depend on the graph, but it would not hold against an adversary who
knows the seed.

**CRT (unconditional).** `c_k` counts ordered `k`-tuples of independent sets, so

    0 <= c_k <= i(V)^k

and `i(V)` is already computed — it is the top entry of the zeta. Once the
product of the moduli exceeds `i(V)^k`, CRT reconstructs `c_k` as an exact
integer and no probability remains. This is much cheaper than "small n" suggests,
because `i(V)` is usually far below `2^n`: the Chvátal graph has 127 independent
sets on 12 vertices, so `c_4 <= 127^4 < 2^28` and **a single 31-bit prime already
makes the answer unconditional**. `mode="exact"` is therefore the default, and
every ground-truth test asserts `result.certain`.

`ChromaticResult.guarantee` is derived from the evidence actually obtained, not
from the mode requested: asking for `multimodular` and happening to get enough
primes for CRT reports an unconditional answer, and says so.

## Honest positioning

This is `O*(2^n)` for **every** graph. That cuts both ways:

- **Where it wins:** small, dense, hard instances — the ones where
  branch-and-bound blows up. Runtime depends only on `n`, never on how hard the
  instance is. Triangle-free graphs with large chromatic number (the
  Mycielskians) are trivial for it and adversarial for every clique-based bound.
- **Where it is useless:** the large sparse DIMACS benchmarks. Specialised
  branch-and-bound colouring solvers handle hundreds of vertices there; this
  cannot handle 40, ever, on any machine. `2^40` uint32 words is 4 TiB.

**This is not competitive with colouring solvers in general and is not offered
as such.** The claim made and supported here is narrower: *exact chromatic
number for arbitrary graphs up to roughly 28 vertices, in time that depends only
on `n`, on a passively cooled laptop.*

The memory ceiling is detected at runtime, not hardcoded, and oversized inputs
are refused with a message rather than thrashing.

## Usage

```sh
python -m apps.chromatic --generator mycielskian:5
python -m apps.chromatic --generator kneser:8,2 --mode multimodular --primes 6
python -m apps.chromatic --dimacs graph.col --json
```

```python
import apps.chromatic as ch

g = ch.mycielskian(5)                      # 23 vertices, chi = 5, clique number 2
r = ch.chromatic_number(g, mode="exact")   # "exact" | "multimodular" | "mod2_64"
print(r.chromatic_number, r.certain)
print(r.guarantee)                         # always states what the number means
```

Every run prints the guarantee. `--cross-check` additionally runs the full
Möbius transform and asserts bitwise agreement with the reduction path.

## Implementation notes

### The modulus, and why the kernel needed no changes

The task brief suggested running the kernel mod `p` "with a trivially different
generator, or with a post-pass reduction". Neither holds, and
`tests/chromatic/test_modulus_design.py` demonstrates both:

- **Not a generator change.** A generator is a 2×2 matrix over the element ring.
  Reduction mod `p` is a property of the *ring*, not of the matrix, and is not a
  function of the residue mod `2^64`, so no choice of four constants produces it.
- **Not a post-pass reduction**, at 62-bit primes. The dispatcher fuses up to 13
  stages into one device pass and Möbius values can double per stage, so an
  intermediate reaches `2^75` well before the pass boundary where a reduction
  would happen.

What works instead, with **no kernel change at all**: keep the primes small
enough that the entire transform is exact over `Z`, and reduce once at the end.
With `p < 2^31`:

- `i(S) <= 2^n` — the zeta runs in `uint32`, exact, and is shared across every
  modulus and every `k`.
- `i(S)^k mod p` needs products `< 2^62`: one `ulong` multiply, no 128-bit path.
  This is the only modular multiplication in the algorithm, and it is a pointwise
  op outside the butterfly.
- The alternating sum of `2^n` terms each `< p` has magnitude `< 2^62 < 2^63`, so
  the unmodified `uint64` kernel returns the **exact integer** in two's
  complement. The bound is necessary as well as sufficient — a test violates it
  deliberately and shows the reading break.

31 bits is a smaller modulus than the brief proposed, which weakens the
per-prime failure bound; the bound above accounts for that, and it remains
negligible at `r >= 2`. In exchange the kernel is reused verbatim, so all of its
own measured numbers stay valid.

### The independence indicator

`a(S) = 1` iff `adj[v] & S == 0` for every `v` in `S`. Three implementations,
all cross-checked against the definition:

- **`indicator_gpu`** — one Metal thread per subset, `O(2^n * n)` work, no
  dependencies, one launch. The production path.
- **`indicator_numpy_dp`** — the lowbit recurrence
  `a(S) = a(S \ {v}) and (adj[v] & S == 0)` for `v` the lowest set bit,
  which is `O(2^n)` *total* — asymptotically better.
- **`indicator_numpy_direct`** — the obvious `O(2^n * n)` form, used as the oracle.

The direct form is preferred on the GPU despite doing more work, because the DP
recurrence is sequential in `popcount(S)` and `mx.fast.metal_kernel` is
out-of-place: a GPU DP would need one full-array pass per level, spending
`O(2^n * n)` of *memory traffic* to save `O(2^n * n)* of *ALU* on a kernel that
is already memory-bound. Both are timed in the benchmark.

<!-- BENCH -->

## Tests

`.venv/bin/python -m pytest tests/chromatic` (add `-m slow` for the 28-vertex
false-negative regression).

- **Ground truth**, every family with a published value: `K_n`, complete
  bipartite, even and odd cycles, paths, Petersen (two independent
  constructions, checked isomorphic), Chvátal, Grötzsch, the Mycielskian chain
  `M_2..M_5`, Kneser `K(n,2)` against Lovász, Turán graphs. All three modes must
  agree, and `mode="exact"` must report `certain`.
- **Brute force**: random `G(n,p)` up to `n = 12` against an exhaustive
  backtracking search, which is itself validated against unpruned `k^n`
  enumeration for `n <= 8`.
- **The identity**: `c_k` against direct enumeration of covers; against closed
  forms for empty graphs `((2^k-1)^n)` and complete graphs (via Stirling
  numbers, derived independently of the inclusion–exclusion formula);
  multiplicativity over disjoint unions; and `c_k = 0` exactly for `k < chi`.
- **Both evaluation paths agree bitwise**, mod `2^64` and mod prime, for every
  `k` and every test graph.
- **Multi-modular**: all `r` prime residues simultaneously zero or simultaneously
  nonzero on every instance; CRT reconstruction equals the exact integer; the
  false-negative regressions above.
- **The modulus design claims**, each demonstrated rather than asserted.

## Limitations

- `n <= 32` by the bitmask representation, and ~29 by memory on this machine —
  the latter binds first and is detected at runtime.
- The clique lower bound is deliberately weak (it returns 2 on every
  Mycielskian). The Lovász theta bound would be much stronger there, but needs
  an SDP solve — not worth the dependency for a bound that only seeds a binary
  search over at most `n` values.
- `c_k` counts covers by independent sets, not proper colourings; the two are
  zero together but are different numbers. The API and tests keep them distinct.
