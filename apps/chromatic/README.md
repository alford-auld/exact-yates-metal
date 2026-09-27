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
`2^64 | c_chi`. Over 287 graphs on 2–9 vertices (complete, Turán, cycles, paths,
empty, and random at varying density) the best ratio `v2(c_chi)/n` is `7/8`,
attained by K₈ — which would need `n >= 74` vertices to reach valuation 64, far
past the ~29-vertex memory ceiling.

Both searches are reproducible:

    .venv/bin/python bench/chromatic/false_negative_search.py

That ratio is pinned by a test, but it is a search result, not a theorem: it is
evidence that the mod-2^64 path is safe in this range, not a proof that it
always is. Nothing here argues you should rely on it — `mode="exact"` is the
default precisely because it does not need this argument at all.

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

The direct form is the default despite doing more work. Two reasons, one
structural and one measured:

- A GPU DP is unattractive structurally: the recurrence is sequential in
  `popcount(S)` and `mx.fast.metal_kernel` is out-of-place, so it would need one
  full-array pass per level — spending `O(2^n · n)` of *memory traffic* to save
  `O(2^n · n)` of *ALU* on a kernel that is already memory-bound.
- Measured, the CPU DP is actually **faster below n ≈ 15**, where the GPU is
  entirely launch-overhead-bound, and loses by 36x by n = 22. See the benchmark
  section; the crossover is real and the "obviously parallel wins" intuition is
  wrong at the small end.

## Measured performance

Every number here was measured on this machine and is reproducible with a command
in `bench/chromatic/`. The headline: **an arbitrary 29-vertex graph
(6 GiB of subset state, the detected memory ceiling) takes
1531 ms end to end**, and a 28-vertex one takes 544 ms — regardless of how hard
the instance is.

- Machine: MacBook Air (Mac16,13), Apple M4, 16 GB unified memory, macOS 26.3
- GPU: Apple M4 (applegpu_g16g), MLX 0.32.2
- Memory ceiling detected at runtime: n=29 (12 B per subset, i(S) as uint32 plus the uint64 power terms)
- Phases are measured as nested prefixes of the real computation (indicator; indicator+zeta; end-to-end) and the inner ones subtracted, so the three shares sum to the measured total by construction
- Protocol: 1.5s steady-state window per phase (median of its second half), 0.4s warmup, 2.0s idle cooldown, at most 0 iterations per window
- Instances: random G(n, 0.5), seed 7
- Run 2026-09-27T14:41:21-0300 to 2026-09-27T14:45:49-0300

### Where the time actually goes

The butterfly is **not** the bottleneck, and that is the honest and slightly
deflating point of this benchmark. Across n = 10..29 the single subset-zeta is
3–24% of end-to-end time, while the k-search is 69–88%.

The reason is structural, not a tuning failure. The algorithm runs the zeta
**once**, over a `uint32` array. The k-search then runs, for each candidate `k`,
one pointwise power plus reduction *per CRT prime* (plus the mod-2^64 residue) —
at n = 29 that is 2 values of k times 3 primes plus one, over `uint64` arrays twice the
width. More passes over more bytes, so it dominates. Making the transform
faster would barely move the total; making the modular power faster did.

| n | edges | chi | i(V) | memory | indicator | zeta | k-search | end-to-end | indicator / zeta / k-search share | k tested |
|--:|------:|----:|-----:|-------:|----------:|-----:|---------:|-----------:|:--|--:|
| 10 | 22 | 4 | 57 | 0 MiB | 0.18 ms | 0.02 ms | 0.5 ms | 0.7 ms | 27% / 3% / 70% | 1 |
| 11 | 27 | 4 | 64 | 0 MiB | 0.18 ms | 0.02 ms | 0.5 ms | 0.7 ms | 27% / 3% / 70% | 1 |
| 12 | 32 | 5 | 83 | 0 MiB | 0.18 ms | 0.03 ms | 0.7 ms | 0.9 ms | 20% / 3% / 77% | 1 |
| 13 | 37 | 4 | 118 | 0 MiB | 0.18 ms | 0.03 ms | 0.5 ms | 0.7 ms | 26% / 5% / 69% | 1 |
| 14 | 44 | 5 | 121 | 0 MiB | 0.18 ms | 0.04 ms | 0.7 ms | 0.9 ms | 19% / 5% / 76% | 1 |
| 15 | 51 | 5 | 152 | 0 MiB | 0.18 ms | 0.03 ms | 0.8 ms | 1.0 ms | 17% / 3% / 80% | 1 |
| 16 | 57 | 5 | 194 | 1 MiB | 0.18 ms | 0.08 ms | 1.5 ms | 1.8 ms | 10% / 5% / 85% | 2 |
| 17 | 66 | 5 | 242 | 2 MiB | 0.19 ms | 0.09 ms | 1.7 ms | 1.9 ms | 10% / 5% / 86% | 2 |
| 18 | 71 | 6 | 290 | 3 MiB | 0.20 ms | 0.11 ms | 1.9 ms | 2.2 ms | 9% / 5% / 86% | 2 |
| 19 | 80 | 5 | 352 | 6 MiB | 0.23 ms | 0.11 ms | 2.3 ms | 2.7 ms | 9% / 4% / 87% | 2 |
| 20 | 89 | 5 | 372 | 12 MiB | 0.27 ms | 0.21 ms | 1.7 ms | 2.2 ms | 12% / 9% / 79% | 1 |
| 21 | 97 | 6 | 490 | 24 MiB | 0.30 ms | 0.51 ms | 5.4 ms | 6.2 ms | 5% / 8% / 87% | 2 |
| 22 | 105 | 6 | 667 | 48 MiB | 0.37 ms | 1.16 ms | 8.3 ms | 9.8 ms | 4% / 12% / 84% | 2 |
| 23 | 117 | 6 | 660 | 96 MiB | 0.53 ms | 2.18 ms | 7.1 ms | 9.9 ms | 5% / 22% / 72% | 1 |
| 24 | 132 | 6 | 784 | 192 MiB | 0.93 ms | 4.03 ms | 25.7 ms | 30.6 ms | 3% / 13% / 84% | 2 |
| 25 | 140 | 6 | 830 | 384 MiB | 1.53 ms | 8.02 ms | 24.7 ms | 34.2 ms | 4% / 23% / 72% | 1 |
| 26 | 150 | 6 | 969 | 768 MiB | 2.76 ms | 15.96 ms | 96.4 ms | 115.1 ms | 2% / 14% / 84% | 2 |
| 27 | 167 | 7 | 1126 | 1536 MiB | 5.23 ms | 42.28 ms | 130.3 ms | 177.8 ms | 3% / 24% / 73% | 1 |
| 28 | 181 | 6 | 1066 | 3072 MiB | 10.28 ms | 85.50 ms | 448.5 ms | 544.3 ms | 2% / 16% / 82% | 2 |
| 29 | 193 | 7 | 1417 | 6144 MiB | 20.33 ms | 170.59 ms | 1340.3 ms | 1531.2 ms | 1% / 11% / 88% | 2 |

### The modular multiply, and what Montgomery bought

The pointwise k-th power is the only modular multiplication in the algorithm.
With a 64-bit `%` in the square-and-multiply loop it is compute-bound — 64-bit
integer division is expensive on this GPU. Montgomery reduction with R = 2^32
removes the division entirely. Measured at n = 26, k = 6
(`bench/chromatic/modmul.py`), 768 MiB moved per pass:

| pointwise power variant | time | effective bandwidth |
|---|--:|--:|
| mod 2^64 (wrapping, no reduction) | 7.88 ms | 102.2 GB/s |
| mod p, generic 64-bit % | 30.67 ms | 26.3 GB/s |
| mod p, Montgomery | 12.83 ms | 62.8 GB/s |
| *the subset-zeta at the same n, for scale* | 16.15 ms | 99.7 GB/s |

Montgomery is **2.39x** faster than the generic `%` in that controlled
comparison (same run, same conditions). End to end the effect is nearly as
large, because the k-search dominates: n = 29 went from 2778 ms to 1531 ms
(1.81x) across two runs of `bench/chromatic/bench.py` with identical settings.
The "before" run is kept as `bench/results/chromatic_pre_montgomery.json` so
the comparison is checkable rather than remembered:

| n | before Montgomery | after | speedup |
|--:|--:|--:|--:|
| 26 | 169 ms | 115 ms | 1.47x |
| 27 | 342 ms | 178 ms | 1.92x |
| 28 | 860 ms | 544 ms | 1.58x |
| 29 | 2778 ms | 1531 ms | 1.81x |

It is still short of the
102 GB/s bandwidth floor that the reduction-free `mod 2^64` path reaches,
so the power kernel remains partly compute-bound; the remaining gap is the
Montgomery multiply chain itself, not division.

Montgomery needs an odd modulus, so even moduli fall back to the generic path
automatically — which is why the 2^16 false-negative regression still works.

### Instances where the clique bound does not reach chi

On these the binary search actually has to run; on a random `G(n, 1/2)` the
greedy clique often already equals chi, and the search does almost nothing.

| instance | n | chi | bounds | k tested | indicator | zeta | k-search | end-to-end | zeta share |
|---|--:|--:|:--|--:|--:|--:|--:|--:|--:|
| M_5 (Mycielskian) | 23 | 5 | [2, 5] | 3 | 0.54 ms | 2.19 ms | 22.8 ms | 25.5 ms | 8.6% |
| Kneser(7,2) | 21 | 5 | [3, 5] | 2 | 0.29 ms | 0.50 ms | 5.4 ms | 6.2 ms | 8.2% |
| Kneser(8,2) | 28 | 6 | [4, 6] | 2 | 10.33 ms | 84.60 ms | 448.7 ms | 543.6 ms | 15.6% |

`M_5` is the case worth looking at: 23 vertices, chi = 5, clique number 2, so
every clique-based bound is useless and a branch-and-bound solver has to search.
Here it is 25.5 ms, and it would be 25.5 ms for *any* 23-vertex graph.

### Indicator variants

Two results here, both against my expectation:

**The asymptotically better algorithm loses, but only eventually.**
`indicator_numpy_dp` is the O(2^n) lowbit dynamic program; `indicator_gpu` is the
O(2^n · n) direct form. At n = 11..14 the *DP is faster* — the GPU is entirely
launch-overhead-bound there, pinned at 0.18 ms regardless of size. They cross
over around n = 15, and from there the gap widens fast: 1.7x at n = 16, 11x at
n = 20, 36x at n = 22. So the direct form is the right default, but not for the
reason "more parallel" alone — below the crossover it is not faster at all. (The
n = 10 DP figure is a first-call NumPy warmup artefact, not a real cost.)

**The early exit matters much more than the branch divergence costs.** The
`direct` kernel breaks out of the vertex loop as soon as one neighbour is found
inside `S`; the `branchless` variant always runs `popcount(S)` iterations. The
branchless version is *3.4x slower* at n = 29 (69.85 ms vs 20.33 ms) and the gap
grows with n. Most subsets are non-independent and fail on one of their first
few vertices, so the early exit cuts the average iteration count far more than
divergence costs — worth measuring rather than assuming, since the usual GPU
instinct is the opposite.

| n | GPU direct | GPU branchless | NumPy lowbit DP (CPU) |
|--:|-----------:|---------------:|----------------------:|
| 10 | 0.18 ms | 0.18 ms | 1.4 ms |
| 11 | 0.18 ms | 0.18 ms | 0.1 ms |
| 12 | 0.18 ms | 0.18 ms | 0.1 ms |
| 13 | 0.18 ms | 0.18 ms | 0.1 ms |
| 14 | 0.18 ms | 0.18 ms | 0.1 ms |
| 15 | 0.18 ms | 0.19 ms | 0.2 ms |
| 16 | 0.18 ms | 0.19 ms | 0.3 ms |
| 17 | 0.19 ms | 0.24 ms | 1.9 ms |
| 18 | 0.20 ms | 0.26 ms | 0.8 ms |
| 19 | 0.23 ms | 0.29 ms | 2.7 ms |
| 20 | 0.27 ms | 0.36 ms | 3.0 ms |
| 21 | 0.30 ms | 0.47 ms | 6.1 ms |
| 22 | 0.37 ms | 0.76 ms | 13.4 ms |
| 23 | 0.53 ms | 1.24 ms | - |
| 24 | 0.93 ms | 2.19 ms | - |
| 25 | 1.53 ms | 4.19 ms | - |
| 26 | 2.76 ms | 8.31 ms | - |
| 27 | 5.23 ms | 16.76 ms | - |
| 28 | 10.28 ms | 34.14 ms | - |
| 29 | 20.33 ms | 69.85 ms | - |

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
