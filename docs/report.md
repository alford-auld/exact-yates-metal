# A templated Yates butterfly kernel in Metal, and exact chromatic number on top of it

**Repository:** `exact-yates-metal` · **Date:** 2026-09-27 · **Hardware:** MacBook Air
(Mac16,13), Apple M4, 16 GB unified memory, macOS 26.3 · **Software:** MLX 0.32.2

Two deliverables, built in sequence. First, a single templated Metal kernel that
computes the n-fold Kronecker power of any 2×2 generator matrix by the Yates
algorithm — one kernel covering the Walsh–Hadamard transform, the subset and
superset zeta and Möbius transforms, and the polar-code kernel. Second, an
application that uses it to compute **exact chromatic numbers** by
Björklund–Husfeldt–Koivisto inclusion–exclusion.

Everything below was measured on the machine named above. Nothing is a vendor
figure, an estimate, or a remembered number; every value traces to a JSON
artifact in `data/` and a re-runnable command.

---

## 1. Summary

| | Result |
|---|---|
| **Kernel throughput** | 91.3–100.6% of a device-to-device copy measured on the same machine, across `uint32`/`uint64`/`float32` and n = 10…29. Median ≈ 96%. |
| **Largest transform** | n = 29 (2 GiB per array), 2 GiB in / 2 GiB out, in 174 ms |
| **Exactness** | zeta/Möbius round trips bit-exact over Z/2³² and Z/2⁶⁴; Hadamard bit-loss bound proved by Smith normal form, and *verified tight at every tested n* by an explicit two-sided witness |
| **Chromatic number** | exact χ for an arbitrary 29-vertex graph in **1.53 s** (`mode="exact"`, `G(29,0.5)`, 3 CRT primes); 28 vertices in 544 ms; runtime depends only on n |
| **Tests** | 1522 passed + 1 skipped in the fast suite, 11 more marked slow, all passing; all six required kernel test groups cover all three element types |
| **Code** | 1.4k lines kernel, 1.9k application, 3.0k tests, 2.3k benchmarks |

The three results most worth a reader's time are in §3.2 (a tiling
policy change worth 1.75× at the largest sizes), §4.3 (a divisibility lemma
that settles by arithmetic what a 287-graph search could not, bounding how far a
mod-2⁶⁴ false negative can reach), and §4.4 (the butterfly turning out not to be
the bottleneck of its own application).

---

## 2. The kernel

### 2.1 What it is

For a 2×2 matrix `M` over a commutative ring, the transform is `M^⊗n` applied to
a vector of length `N = 2ⁿ`, computed in `n·2^(n-1)` butterflies. Stage `j`
pairs each index `i0` whose bit `j` is zero with `i1 = i0 | (1<<j)`:

```
new[i0] = m00·x[i0] + m01·x[i1]
new[i1] = m10·x[i0] + m11·x[i1]
```

The five named variants differ only in those four constants. Stages act on
distinct tensor factors, so they commute and may be grouped freely — which is
what makes the tiling below legal.

Source: [`yates/kernel.metal`](../yates/kernel.metal), [`yates/kernel.py`](../yates/kernel.py),
[`yates/variants.py`](../yates/variants.py).

### 2.2 Three tiers in one kernel

A pass transforms index bits `[s, s+P)`. Its tile is `2^P` rows (stride `2^s`
apart) × `C` contiguous columns, with `G` independent tiles per threadgroup. The
tier of a stage is decided purely by how far apart its butterfly partners are:

With `R = 2^LOGR` rows resident per thread and `C = 2^LOGC` columns, the local
bit `b` of a stage selects the tier:

| local bit range | partner lives in | mechanism | cost |
|---|---|---|---|
| `[LOGR, P)` | this thread's own registers | array indexing | free, no traffic |
| `[0, NB_SIMD)` | another lane of the simdgroup | `simd_shuffle_xor(x, C<<b)` | no memory traffic |
| `[NB_SIMD, LOGR)` | another thread | threadgroup memory | two barriers per stage |

`NB_SIMD = clamp(log2(simd_width) − LOGC, 0, LOGR)`, derived from the *measured*
SIMD width rather than an assumed 32. The three counts always sum to `n`; they
are the `tiers simd/tg/reg` column of Appendix A.

Nothing about the GPU is hardcoded. The SIMD width is read out of a live kernel
(`threads_per_simdgroup`), the threadgroup memory limit and maximum threadgroup
size come from the Metal device, and the memory ceiling is derived from the
detected recommended working set. All are printed in test and benchmark output.
[`yates/device.py`](../yates/device.py)

Autodiff needs no backward kernel at all: `(M^⊗n)ᵀ = (Mᵀ)^⊗n`, so every VJP is
the same kernel with a transposed generator. [`yates/autodiff.py`](../yates/autodiff.py)

---

## 3. Kernel results

![Kernel bandwidth](figures/kernel-bandwidth.svg)

**Figure 1. The kernel is bandwidth-bound across the whole range, once
high-stride passes use long enough contiguous runs.** (a) Achieved bandwidth as
a fraction of a pure device-to-device copy kernel measured on the same machine
at the same working-set size, for n = 10…29. Traffic is
`2·passes·N·sizeof(elem)` with the pass count taken from the plan actually
dispatched. Points slightly above 100% are inter-pass reuse in the system cache.
(b) The tiling-policy change behind that: the pre-fix policy minimised pass
count and left 128-byte runs on passes whose rows are megabytes apart; the
shipped policy scales the required run length with the stride, at the cost of a
fourth device pass. Bar labels are % of copy. Single run per configuration;
each value is the median of the second half of a steady-state measurement
window, with idle cooldown between configurations.
Data: [`bench/results/bench.json`](../bench/results/bench.json), [`bench/results/coalescing_tuning.json`](../bench/results/coalescing_tuning.json).

### 3.1 Bandwidth

| element type | n | transform GB/s | fraction of measured copy |
|---|---|---|---|
| `uint32` | 10–29 | 89.9–98.9 | 91.3%–100.2%, median 95.9% |
| `uint64` | 10–28 | 91.6–99.3 | 92.9%–100.6%, median 95.8% |
| `float32` | 10–29 | 90.6–99.5 | 91.6%–100.0%, median 96.7% |

The denominator deserves emphasis: it is re-measured at **every** working-set
size, because achievable copy bandwidth on this machine depends strongly on the
footprint — **48.0–51.9 GB/s at 16 MiB**
([`bench_small_footprint.json`](../bench/results/bench_small_footprint.json))
against **94.3–100.4 GB/s across the 256 MiB–2 GiB footprints used here**
([`bench.json`](../bench/results/bench.json)). A single global denominator would
have been meaningless, and quoting the 16 MiB figure as "peak" would have made
the kernel look like it exceeded hardware limits.

### 3.2 The finding worth keeping: run length must scale with stride

A pass at bit offset `s` reads rows `2^s` elements apart. When that stride is
large, every row of a tile lands on a different DRAM page and TLB entry, and
short runs stop amortising the page activation. Measured in one controlled run:

| configuration | n=28, 1 GiB | n=29, 2 GiB |
|---|--:|--:|
| minimise passes, 128 B runs (pre-fix) | 61.0% | 57.5% |
| 4 passes, 512 B runs | 97.7% | 102.1% |
| 4 passes, 1 KiB runs | 95.8% | 102.4% |
| 4 passes, 2 KiB runs | 92.6% | **103.6%** |
| shipped policy (scaled to stride) | **102.1%** | 100.5% |

Spending an extra full pass over 2 GiB to lengthen the runs is a large net win,
and the optimum run length *grows* with n — 512 B at n=28, 2 KiB at n=29. The
shipped policy tracks that automatically and lands within a percent of the best
hand-tuned plan at both sizes. This was the single largest performance change in
the project and it came from measuring, not from reasoning about the kernel.

### 3.3 Thermal behaviour

![Thermal soak](figures/thermal-soak.svg)

**Figure 2. Passive cooling costs 6.8% after seven minutes, and macOS reports
nothing.** Sustained throughput during 7 minutes of continuous `WHT` at n = 24
(256 MiB working set, 1.5 GiB of traffic per iteration, 23 834 iterations) with
no cooldown at all, in 15-second
buckets. Each point is the median of its bucket. The grey line is the first
bucket, the reference the slowdown is measured against. **Note the y-axis does
not start at zero** — the claim is the shape of the decline, which spans
86.5–92.4 GB/s. Single run. `pmset -g therm` reported no thermal or performance
warning before or after, so measured throughput is the only usable signal.
Data: [`bench/results/thermal.json`](../bench/results/thermal.json).

| elapsed | median ms | GB/s | vs. first bucket |
|--:|--:|--:|--:|
| 15 s | 17.44 | 92.4 | 1.000× |
| 30 s | 17.51 | 92.0 | 1.004× |
| 45 s | 17.54 | 91.8 | 1.006× |
| 60 s | 17.55 | 91.8 | 1.006× |
| 120 s | 17.57 | 91.7 | 1.007× |
| 180 s | 17.66 | 91.2 | 1.013× |
| 240 s | 18.04 | 89.3 | 1.034× |
| 300 s | 18.36 | 87.7 | 1.053× |
| 360 s | 18.36 | 87.7 | 1.053× |
| 420 s | 18.63 | 86.5 | 1.068× |

Throughput is flat for roughly three minutes — the first movement outside noise
is 1.013× at 180 s, and the break is 1.034× at 240 s — then declines steadily,
ending 6.8% down and still falling. This is why the sweep in Figure 1 uses cooldowns
between configurations and reports the median of a window rather than a
best-of-N peak: within a configuration those numbers are steady state, but they
are *not* what the machine sustains over tens of minutes.

### 3.4 The exactness contract

The exactness argument is the core of the kernel, and it is stated in full in
[`README.md`](../README.md). In summary:

- Unsigned overflow in MSL is defined wraparound, so `uint`/`ulong` arithmetic
  is *exact reduction* mod 2³²/2⁶⁴. Signed overflow is undefined behaviour.
  **Every exact path uses `uint` or `ulong`.** A source lint enforces that
  signed types appear only in compile-time positions, and a behavioural test
  runs inputs with the top bit set — where a signed path would be UB — against
  arbitrary-precision Python.
- The four zeta/Möbius generators are unipotent (SNF = identity), hence exact
  automorphisms of Z/2^k: `MOB_SUB(ZETA_SUB(x)) == x` bit for bit, no tolerance.
- `SNF(H₂) = diag(1,2)`, so `H_(2ⁿ)` has elementary divisors `2^j` with
  multiplicity `C(n,j)`, and a forward-then-inverse WHT recovers the input only
  **mod 2^(k-n)**. `bit_loss()` computes this from the Smith normal form rather
  than hardcoding it per variant, so it is correct for a user-supplied matrix.
- The bound is proved **tight in both directions** by exhibiting, for each n, two
  inputs differing only in bit `k-n` that are indistinguishable after
  `WHT∘WHT`, while a flip at bit `k-n-1` *is* visible.

### 3.5 Platform findings

Three MLX 0.32.2 / `applegpu_g16g` behaviours shaped the implementation. Each
has a test that pins it and *warns* if a future release fixes it.

1. **`metal::simd_shuffle_xor` silently corrupts 64-bit operands** — it operates
   per 32-bit register. This caused a real `uint64` miscompare during
   development; `ulong` is now shuffled as two halves.
2. **MLX pastes template integers into the generated Metal identifier**, so a
   negative value emits an invalid C++ name and the library fails to build.
   Matrix entries are carried as `(sign, magnitude)` pairs instead.
3. **`custom_function` `.jvp` and `.vmap` rules are bypassed for kernel bodies.**
   MLX consults the registered `.vjp` but descends into the `CustomKernel` for
   the other two. Reverse mode works; forward mode
   and `vmap` are documented as unsupported rather than papered over with dead
   rules.

A fourth thing, relevant to anyone benchmarking MLX, and **not a defect**:
`mx.contiguous()` on an already-contiguous prefix slice is often a real device
copy. Measuring inside that copy halved every bandwidth number until I caught
it — but the behaviour is intended. `Contiguous::eval_gpu`/`eval_cpu` alias the
input only when the parent buffer is at most 16 KiB larger than the view, so a
small slice can release a large parent allocation
([MLX PR #1270](https://github.com/ml-explore/mlx/pull/1270)). Measured, the
threshold is exact: a slice of a 256 MiB array with 16 KiB of slack aliases in
~23 µs, and one with 16 KiB + 4 B copies in ~5.4 ms. `base[1:]` aliases, so the
rule is about buffer size rather than offset. Two consequences: materialise
benchmark inputs outside the timed region, and use MLX's row-contiguous flag —
which `mx.fast.metal_kernel(ensure_row_contiguous=True)` applies internally — as
the defensive check before a kernel, not a blanket `mx.contiguous()`. The
threshold is now pinned by a test that warns if it moves. I originally recorded
this as a fourth platform *bug*; it is not one, and the only fair criticism is
that the docstring says "Copy if necessary" without mentioning the size rule.

---

## 4. Exact chromatic number

With `a(S) = [S independent]`:

```
i(S) = (ZETA_SUB a)(S)                 # independent subsets of S
c_k  = Σ_S (-1)^(n-|S|) · i(S)^k       # ordered k-tuples of independent
                                       # sets whose union is V
χ(G) = min { k : c_k > 0 }
```

One subset-zeta over the whole cube, shared by every `k` and every modulus; then
a pointwise power and a reduction per candidate `k`, with a binary search between
a greedy-clique lower bound and a DSATUR upper bound. `O*(2ⁿ)` for every graph,
no search over colourings. [`apps/chromatic/README.md`](../apps/chromatic/README.md)

### 4.1 Correctness

Every family with a published chromatic number is reproduced: `K_n`, complete
bipartite, even/odd cycles, paths, Petersen (two independent constructions,
checked isomorphic), Chvátal, Grötzsch, the Mycielskian chain M₂–M₅, Kneser
`K(n,2)` against Lovász, Turán graphs. Random `G(n,p)` up to n = 12 is checked
against an exhaustive backtracking oracle, which is *itself* validated against
unpruned `kⁿ` enumeration for n ≤ 8.

The Mycielskians are the family that matters: χ(M_k) = k while the clique number
stays at 2, so every clique-based bound is useless and the method has to do the
work. M₅ (23 vertices, χ = 5) resolves in 25.5 ms.

### 4.2 The modulus design question

Two approaches suggest themselves: a "different generator", or a reduction
after each device pass. **Both are wrong**, and
[`tests/chromatic/test_modulus_design.py`](../tests/chromatic/test_modulus_design.py) demonstrates each:

- A generator is a matrix over the element ring; reduction mod p is a property
  of the *ring* and is not a function of the residue mod 2⁶⁴, so no choice of
  four constants produces it.
- A post-pass reduction is too late at 62-bit primes: the dispatcher fuses up to
  13 stages per device pass and Möbius values can double per stage, so an
  intermediate reaches 2⁷⁵ before the first pass boundary.

What works, with **no kernel change at all**, is to keep primes under 2³¹ so the
entire transform is exact over ℤ and reduce once at the end. The zeta runs in
`uint32` (since `i(S) ≤ 2ⁿ`) and is exact outright; the alternating sum of `2ⁿ`
terms each `< 2³¹` stays below 2⁶³, so the unmodified `uint64` kernel returns the
exact integer in two's complement. A test violates that bound deliberately and
shows the reading break, so the condition is established as necessary, not just
sufficient.

This is a smaller modulus than the ~62-bit primes one might reach for, which
weakens the per-prime failure bound; §4.3 accounts for that. In exchange the kernel is reused verbatim
and all of its own measured numbers stay valid.

### 4.3 The one-sided guarantee, and the divisibility lemma that bounds its failure

`c_k` is a count, so the two directions are not symmetric:

- `c_k mod m ≠ 0` ⟹ `c_k ≠ 0` ⟹ G is k-colourable. **Sound for any m.**
- `c_k mod m = 0` ⟹ either `c_k = 0` *or* `c_k` is a nonzero multiple of `m`.
  **False negative possible.**

So `min{k : c_k mod 2⁶⁴ ≠ 0}` is an **upper bound** on χ, not χ. It is always
achievable — a colouring realising it is attached — but not a proof of
minimality. Nothing in the code calls the single-modulus result exact.

**This is not hypothetical.**
`c_k` is multiplicative over disjoint unions and `v₂(c₃₀(K₄)) = 10`, so **seven
disjoint copies of K₄ — 28 vertices, inside the memory ceiling — give
`v₂(c₃₀) = 70`**. `c₃₀` is a 149-digit number and `c₃₀ mod 2⁶⁴ = 0`: the
single-modulus test reports "not 30-colourable" for a graph that obviously is.
Pinned as a test, with an 8-vertex analogue against a 2¹⁶ modulus for the fast
suite. The knapsack that finds it is `bench/chromatic/false_negative_search.py`;
[`bench/results/chromatic_false_negative.json`](../bench/results/chromatic_false_negative.json).

**Can a false negative corrupt the reported χ?** That needs the failure at
`k = χ` itself, i.e. `2⁶⁴ | c_χ`. A search answers this badly — the ratio
`v₂(c_χ)/n` is not constant, so extrapolating it from a sample proves nothing.
There is an exact answer.

**Lemma.** `χ! | c_χ(G)` for every graph G. *Proof.* If a covering
`(S₁,…,S_χ)` had `Sᵢ = S_j` for `i ≠ j`, dropping `S_j` would still cover V,
giving a covering by `χ−1` independent sets and hence `(χ−1)`-colourability —
contradicting minimality. So the components are pairwise distinct, `S_χ` acts
**freely** on the coverings by permuting coordinates, every orbit has size `χ!`,
and `χ!` divides their number. ∎

The lemma is **tight on cliques** (`c_n(K_n) = n!`, since every slot must hold a
distinct singleton), so by Legendre `v₂(n!) = n − popcount(n)` the first clique
with `2⁶⁴ | c_χ` is exactly **`K₆₆`**: `v₂(66!) = 66 − 2 = 64`, while
`v₂(65!) = v₂(64!) = 63`. `K₆₆` is a counterexample at `k = χ` by arithmetic,
not by search.

This makes the practical claim **stronger** — but it has to be stated
**additively**, because `c_k` is multiplicative over connected components and so
`v₂` is additive. The lemma applies to a component only at `k = χ(component)`:

    forced(G) = #{components with χᵢ = χ(G)} · v₂(χ!)

Components below χ are evaluated at `k > χᵢ`, where the action is not free, so
they force nothing. The connected formula applied to a disconnected graph
understates badly: **seven disjoint K₄** has `c₄ = (4!)⁷`, `v₂ = 21` — all
forced, each component's cofactor being 1 — where `v₂(χ!) = 3` alone would
account for three.

A component attaining χ needs ≥ χ vertices, so `m·χ ≤ n` and the maximum forced
part over all graphs on ≤ n vertices is `max_{m·χ≤n} m·v₂(χ!)`. **At n = 29 that
is 25**, attained by one connected component with χ = 28 or 29.

Allowing disconnected graphs does not raise it — but that step is load-bearing
for the number 25, and the runners-up are closer than the claim sounds, so it is
enumerated rather than asserted. The search space is small enough to exhaust:

| m | best χ | forced part `m·v₂(χ!)` |
|--:|--:|--:|
| 1 | 28 or 29 | **25** |
| 2 | 14 | 22 |
| 7 | 4 | 21 |
| any m > 1 | — | < 25 |

`test_disconnected_graphs_never_raise_the_forced_bound` checks every `(m, χ)`
with `m·χ ≤ n` for each n ≤ 29 and confirms the maximum is always attained at
m = 1; `test_forced_bound_runners_up_at_the_ceiling` pins the table above. The
extra multiplicity never buys back what the smaller χ gives up. (The value 25
was right; the formula behind it was not, and would have been wrong on a
disconnected input.)

So ≥ **39 of the 64 bits** must come from the unforced part. The survey
measuring it is stratified, because two families have provably odd cofactors —
`K_a` × m disjoint has `c_χ = (a!)^m` and cofactor 1; `K_a + E_b` has
`c_χ = a!(2^a − 1)^b` and cofactor `(2^a − 1)^b`, odd for every a and b — so
every graph from them contributes `v₂ = 0` by theorem, and counting them as
evidence would be circular. They are measured as a cross-check instead: 55 of
55 match the closed form exactly.

The evidence is the other stratum, 139 graphs with no closed-form cofactor:

| `v₂` of the unforced part | 0 | 1 | 2 | 3 | 4 | 5 | 8 |
|---|--:|--:|--:|--:|--:|--:|--:|
| graphs | 116 | 10 | 5 | 3 | 2 | 2 | 1 |
| share | 83.5% | 7.2% | 3.6% | 2.2% | 1.4% | 1.4% | 0.7% |
| a "random" integer, `2^-(j+1)` | 50% | 25% | 12.5% | 6.25% | 3.13% | 1.56% | 0.20% |

**Among graphs with no closed-form cofactor the cofactor is odd 83.5% of the
time against a 50% baseline, over 139 graphs.** The largest unforced valuation
seen is 8, on one `G(9,0.3)`, equal to the ceiling the test asserts — and it
grew from 4 to 8 when the survey grew, so the unforced part is small relative
to the 39 bits needed rather than bounded by a constant.
Data: [`bench/results/chromatic_unforced_survey.json`](../bench/results/chromatic_unforced_survey.json)
(`bench/chromatic/unforced_survey.py`).

The bias toward oddness, not merely the absence of large values, is the
evidence. Two things are recorded as unproved: that `|X| ≡ |X^P| (mod 2)` for a
Sylow 2-subgroup of `Aut(G)` acting on the unordered covers *explains* that bias
(the congruence itself is a standard theorem, verified here on 15 graphs by
brute-force automorphism enumeration, but it implies nothing about how often
`|X^P|` is odd).

**The second conjecture recorded here, `v₂(c_χ) ≤ n`, has since been refuted,
and replaced by the exact function.**
Its published witnesses were K₆₆ (64 of 66) and seven disjoint K₄ (21 of 28),
both closed-form clique families. Extending the survey produced a connected
8-vertex counterexample with χ = 4 and `c_χ = 1536 = 2⁹·3`, so `v₂ = 9 > 8`.
The boundary is exact: the bound holds over **every** labeled graph on n ≤ 7
(all 2 097 152 at n = 7, attained with equality, never exceeded) and fails at
n = 8. Nothing relied on it, and the residual-risk argument is unaffected — it
needs 39 unforced bits and this reaches 9 of 64. Data:
[`bench/results/chromatic_v2_conjecture.json`](../bench/results/chromatic_v2_conjecture.json)
(`bench/chromatic/v2_conjecture_search.py`).

The conjecture was the only route to an *unconditional* statement here —
`forced ≤ 25` is a theorem, and `v₂(c_χ) ≤ n ≤ 29` would have capped the total
at 29 against the 64 needed. What replaces it is the function itself.
`f(n) = max_{|V|=n} v₂(c_χ(G))` is exactly computable: `v₂(c_χ)` is
isomorphism-invariant, so the space is *unlabeled* graphs (261 080 connected on
9 vertices, not 2³⁶ labeled), and `c_k` is multiplicative over disjoint unions,
so the disconnected case is a knapsack over the connected table — one needing
`v₂(c_k(Gᵢ))` at every `k ≥ χᵢ`, since a component sits at `k = max χᵢ`.

| n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `f(n)` | 0 | 1 | 1 | 3 | 3 | 5 | 7 | 10 | 12 | 17 |
| `f(n) − n` | −1 | −1 | −2 | −1 | −2 | −1 | **0** | **+2** | **+3** | **+7** |

`f(7) = 7` reproduces the exhaustive labeled result. The published 8-vertex
counterexample has `v₂ = 9` but is not extremal: `f(8) = 10`.

**The gap has widened at every step since n = 7 — +2, +3, then +7 at n = 10.**
Least squares gives 2.69 bits per vertex over n = 5…10 and 3.50 over n = 8…10,
extrapolating to **67–83 at n = 29** against the 64 a corrupted χ requires.
Every fit window crosses 64 before n = 29, and the slope rises as points are
added. The conjecture would have promised `v₂(c_χ) ≤ 29`; the measured function
extrapolates past the threshold instead, so the single-modulus mode's safety at
n ≤ 29 is not merely unproved — the only exactly computable evidence trends
against it.

Three things keep that from being alarming. It is an extrapolation across 19
vertices from five points with no derivation behind the slope. `f` is a worst
case over *engineered* graphs — `f(10) = 17` is one particular connected
10-vertex graph with χ = 6 — while the survey's evidence stratum tops out at an
unforced valuation of 8. And nothing in the solver depends on it:
`mode="exact"` is the default, is CRT-reconstructed, and carries no probability
at all. The consequence is a sharpened recommendation rather than a defect —
`mode="mod2_64"` is a fast upper bound on χ and should be read as one. Data:
[`bench/results/chromatic_max_v2.json`](../bench/results/chromatic_max_v2.json)
(`bench/chromatic/max_v2_table.py`, needs `geng` from nauty).

Two ways to close the gap, both implemented:

- **Multi-modular.** r random primes from `[2³⁰, 2³¹)`; failure probability
  ≤ `(D/M)^r` with `D ≤ log(c_k)/log(2³⁰)` divisors and, by Rosser–Schoenfeld,
  `M = π(2³¹) − π(2³⁰) > 3.5·10⁷`. The assumption is that primes are drawn
  independently of the instance.
- **CRT, unconditional.** `0 ≤ c_k ≤ i(V)^k` and `i(V)` is already computed (it
  is the top entry of the zeta), so once the moduli product exceeds `i(V)^k` the
  reconstruction is exact and no probability remains. This is far cheaper than
  "small n" suggests: the Chvátal graph has 127 independent sets, so
  `c₄ ≤ 127⁴ < 2²⁸` and **one 31-bit prime already suffices**. Hence
  `mode="exact"` is the default and every ground-truth test asserts
  `result.certain`.

### 4.4 Where the time actually goes

![Chromatic phases](figures/chromatic-phases.svg)

**Figure 3. The butterfly is not the bottleneck of its own application.**
(a) Share of end-to-end time by phase, n = 10…29 on random `G(n, 0.5)`. The
subset-zeta — the entire point of the kernel — is 3–24% of runtime; the k-search
is 69–88%. Phases are measured as nested prefixes of the real computation
(indicator; indicator+zeta; end-to-end) and the inner ones subtracted, so the
three shares sum to the measured total by construction. (b) End-to-end time,
log scale, before and after replacing the generic 64-bit `%` in the modular
power with Montgomery reduction. Single run per point, median of a steady-state
window, idle cooldown between configurations.
Data: [`bench/results/chromatic.json`](../bench/results/chromatic.json), [`bench/results/chromatic_pre_montgomery.json`](../bench/results/chromatic_pre_montgomery.json).

The reason is structural: the zeta runs **once**, over a `uint32` array, while
the k-search runs one power-plus-reduction *per CRT prime per candidate k* over
`uint64` arrays twice the width. More passes over more bytes.

Chasing that led somewhere useful. The 64-bit `%` in the square-and-multiply
loop was compute-bound, and Montgomery reduction with R = 2³² removes the
division entirely:

| pointwise power variant (n=26, k=6, 768 MiB/pass) | time | effective GB/s |
|---|--:|--:|
| mod 2⁶⁴, no reduction (bandwidth floor) | 7.88 ms | 102.2 |
| mod p, generic 64-bit `%` | 30.67 ms | 26.3 |
| mod p, Montgomery | 12.83 ms | 62.8 |
| *the subset-zeta at the same n, for scale* | 16.15 ms | 99.7 |

**2.39×** in the controlled comparison. End to end the effect is nearly as
large, because the k-search dominates — two runs of `bench/chromatic/bench.py`
with identical settings, the "before" run kept as
[`chromatic_pre_montgomery.json`](../bench/results/chromatic_pre_montgomery.json)
so the comparison is checkable rather than remembered:

| n | before Montgomery | after | speedup |
|--:|--:|--:|--:|
| 26 | 169 ms | 115 ms | 1.47× |
| 27 | 342 ms | 178 ms | 1.92× |
| 28 | 860 ms | 544 ms | 1.58× |
| 29 | 2778 ms | 1531 ms | 1.81× |

It remains short of the bandwidth floor, so the power kernel is still
partly compute-bound — the Montgomery multiply chain itself, not division.
Data: [`bench/results/chromatic_modmul.json`](../bench/results/chromatic_modmul.json).

### 4.5 Cost against density

The n-sweep holds density at 0.5, which is the wrong axis for the cost of exact
mode: that scales with the CRT prime count, hence with `k·log₂ i(V)`. The two
factors pull against each other — sparse graphs have many independent sets but
small χ, dense graphs the reverse — so which end is worst is empirical.
Measured at n = 26, single run per row, idle cooldown between rows.
Data: [`bench/results/chromatic_density.json`](../bench/results/chromatic_density.json)
(`bench/chromatic/density.py`).

| density `p` | edges | i(V) | log₂ i(V) | chi | k tested | CRT primes at chi | `mode="exact"` | `mode="mod2_64"` |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.05 | 18 | 2,017,280 | 20 | 2 | 1 | 2 | 64 ms | 34 ms |
| 0.10 | 31 | 283,518 | 18 | 3 | 2 | 2 | 107 ms | 46 ms |
| 0.15 | 47 | 81,717 | 16 | 3 | 1 | 2 | 63 ms | 33 ms |
| 0.20 | 67 | 21,447 | 14 | 4 | 2 | 2 | 112 ms | 47 ms |
| 0.30 | 91 | 6,692 | 12 | 5 | 1 | 3 | 87 ms | 33 ms |
| 0.50 | 150 | 969 | 9 | 6 | 2 | 2 | 117 ms | 46 ms |
| 0.70 | 222 | 236 | 7 | 9 | 2 | 3 | 170 ms | 47 ms |
| 0.90 | 295 | 61 | 5 | 15 | 1 | 3 | 96 ms | 34 ms |

**Within `G(n,p)` the prime count never leaves 2–3 and the range spans 2.7× in
wall clock.** `χ·log₂ i(V)` stays in 40–75 bits, self-limiting *for random
graphs*: `α ~ 2 log_b n` gives `χ ~ n/(2 log_b n)` and `log₂ i(V) = Θ(log² n)`,
so the product is `Θ(n log n)` rather than `Θ(n²)`.

**That does not extend to all graphs.** The driver is heterogeneity, not
density. For `K_a + E_b` (clique beside an independent set, `n = a+b`), `χ = a`
and `i(V) = (a+1)2^b`, so the product is `Θ(n²)` near `a ≈ n/2` — and `G(26,0.5)`
reaches such a graph with probability zero:

| `a` (so `b = 26 − a`) | i(V) | log₂ i(V) | chi | bound bits | CRT primes | `mode="exact"` |
|--:|--:|--:|--:|--:|--:|--:|
| 2 | 50,331,648 | 25 | 2 | 41 | 2 | 72 ms |
| 4 | 20,971,520 | 24 | 4 | 97 | 4 | 109 ms |
| 8 | 2,359,296 | 21 | 8 | 169 | 6 | 158 ms |
| 12 | 212,992 | 17 | 12 | 212 | 8 | 203 ms |
| 13 | 114,688 | 16 | 13 | 218 | 8 | 205 ms |
| 16 | 17,408 | 14 | 16 | 225 | 8 | 227 ms |
| 20 | 1,344 | 10 | 20 | 207 | 7 | 202 ms |
| 24 | 100 | 6 | 24 | 159 | 6 | 179 ms |

**Eight primes against two or three**, and the slot bound does not help here
(338 bits at a=13, worse than the 218 it replaces). Still only 227 ms, so no
practical harm, but the headline must name the family. Two details worth
keeping apart: the planner sizes from a *bound*, so at a=13 it is the 218-bit
bound rather than the 201.5-bit true `c₁₃ = 13!(2¹³−1)¹³` that buys the eighth
prime; and the worst bound (a=12) and worst true value (a=16) do not coincide.
The family doubles as exact ground truth — `c_k(K_a + E_b)` is closed-form for
every k, derived in the test rather than quoted.

### 4.6 Honest positioning

`O*(2ⁿ)` for every graph cuts both ways.

- **Where it wins:** small, dense, hard instances, and any instance where
  branch-and-bound blows up. Runtime depends only on n, never on instance
  hardness. Triangle-free graphs with large χ are trivial for it.
- **Where it is useless:** the large sparse DIMACS benchmarks. Specialised
  branch-and-bound colouring solvers handle hundreds of vertices there; this
  cannot handle 40 on any machine — `2⁴⁰` uint32 words is 4 TiB.

**It is not competitive with colouring solvers in general and is not offered as
such.** The supportable claim is narrower: *exact chromatic number for arbitrary
graphs up to ~28 vertices, in time depending only on n, on a passively cooled
laptop.*

---

## 5. Limitations, and how to read the numbers

- **Every number here is a single run.** Each is the median of the second half
  of a steady-state measurement window, not a mean over seeds, and there is no
  seed-variance estimate anywhere. Run-to-run spread of a few percent is visible
  in the data — the same configuration measured 82.2% and 67.4% of copy in two
  early runs — so differences of a few percent should not be read as resolved.
  Where a comparison matters (Montgomery, the coalescing policy) it is made
  *within* one run, and that is said explicitly at the point of use.
- **The memory ceiling binds everything**, and it is detected at runtime rather
  than assumed: n = 29 for the kernel, ~29 vertices for the chromatic
  application, on this 16 GB machine. Both are `O*(2ⁿ)` in space by
  construction; there is no streaming variant to fall back on, and oversized
  inputs are refused rather than swapped.
- **Small-n batching is not optimised.** Transforms with n < 5 and a large batch
  dispatch fewer than one full simdgroup per tile — correct, not
  bandwidth-optimal, and outside everything benchmarked.
- **One statement in §4.3 is a conjecture, not a theorem**: that the Sylow
  congruence explains the cofactor's bias toward oddness. It is not relied on
  by the default mode. A second conjecture, `v₂(c_χ) ≤ n`, was published here
  and is now **refuted** — see §4.3 for the 8-vertex witness and the exact
  n ≤ 7 boundary.
- **This is not a general-purpose colouring solver** — see §4.6 for where it
  wins and where it is useless.

### Intuitions that did not survive measurement

Three results worth stating plainly, because the natural guess is wrong in each
case and the measurement is cheap to repeat:

1. **The asymptotically better indicator algorithm loses — but only above
   n ≈ 15.** The `O(2ⁿ)` NumPy lowbit DP beats the `O(2ⁿ·n)` GPU kernel below
   that, where the GPU is entirely launch-overhead-bound at a flat 0.18 ms. It
   then loses by 36× by n = 22.
2. **The early exit beats branchless by 3.4× at n = 29**, despite the
   divergence. Most subsets are not independent and fail on one of their first
   few vertices, so the exit cuts the average iteration count far more than
   divergence costs.
3. **The butterfly is 3–24% of the runtime of its own application.** Optimising
   the transform further would have moved almost nothing; the 64-bit modulo in
   the pointwise power was the real cost, and removing it was worth 2.39×.
4. **The natural bound on `v₂(c_χ)` is not just false, it is badly false.**
   `v₂(c_χ) ≤ n` holds exhaustively through n = 7, fails at n = 8, and the
   exact maximum `f(n)` then pulls away fast: f(8) = 10, f(9) = 12, f(10) = 17
   against n. Extrapolated, it crosses the 64-bit threshold well before n = 29
   — the opposite of the reassurance the conjecture was published to give.
   §4.3.
5. **Sustained copy bandwidth does not decay with working-set size on this
   machine** — it is flat at 99–103 GB/s from 1 GiB to 10 GiB of live data.
   The expected explanation for the n = 29 chromatic row being ~40% above its
   own scaling was that the memory system slows down at large footprints. It
   does not. What does happen, isolated by holding a ballast array live, is
   that allocation pressure switches on between 8 and 10 GiB of peak and costs
   1.16× — on the *same* n = 28 instance, so it has nothing to do with n. That
   accounts for under half the excess; ~1.20× is still unexplained. §4.4.

### Errors caught, and how

The three above are findings about the hardware; a reader could infer most of
them from the tables. These are mistakes in this project's own code and
documents, and they are the half that cannot be inferred from anything here.
Each is listed with what caught it, because that is the part that transfers.

- **A published number that traced to nothing — in the project whose stated
  standard is that every value traces to a JSON artifact and a re-runnable
  command.** §4.3 reported the unforced-valuation distribution over "a
  156-graph survey, 57 of them disconnected" with shares of 92.3 / 4.5 / 1.9 /
  0.6 / 0.6%. Both this report and the application README carried it, it was
  the evidential basis for the residual-risk argument, and it survived review
  twice. Nothing in the repository produced it: the only survey tracked here
  was 27 graphs in `tests/chromatic/test_modular.py` with a visibly different
  distribution, and `chromatic_false_negative.json` records a component search,
  not a distribution. The number was real once and then outlived its source.

  **What caught it was arithmetic a reader can do in their head:** 92.3% of 156
  is exactly 144.0, and 4.5% is exactly 7.0 — percentages that clean are
  computed from integer counts, so the counts existed and were not reported.
  Publishing counts beside shares would have made the loss obvious immediately.

  That the discipline was stated but not *enforced* is the instructive part,
  and it is mechanisable, so it is now mechanised:
  `tests/test_docs_claims.py` walks the tracked Markdown, and (a) every
  published percentage distribution must carry integer counts that imply those
  percentages, and (b) the survey figures quoted in prose must match the keys
  in `chromatic_unforced_survey.json`. The replacement survey is a tracked
  script with tracked output, and it is stratified — see §4.3 — because the
  first version of it repeated a subtler form of the same error, inflating the
  headline with families whose cofactors are odd by theorem.

- **The same mistake twice, and it is worth stating as one lesson:
  *witnesses from families with closed forms are witnesses about those
  families.*** Two separate claims in §4.3 were built on evidence drawn from
  `K_a` repeated and `K_a + E_b`, and both came out too good.

  The unforced-valuation survey was "weighted toward the hard cases", which
  meant weighted toward those two families — and both have provably odd
  cofactors, `(a!)^m` giving cofactor 1 and `a!(2^a−1)^b` giving `(2^a−1)^b`.
  Every such graph contributed a `v₂ = 0` *by theorem*, so the oddness rate was
  partly a measure of how many cliques were in the survey. Fixed by
  stratifying, with the two populations named in the script: among graphs with
  no closed-form cofactor the rate is 83.5%, not 92.3%.

  The conjecture `v₂(c_χ) ≤ n` was published as "verified over the survey and
  never violated", with K₆₆ (64 of 66) and seven disjoint K₄ (21 of 28) as its
  tight witnesses — again both closed-form clique families, and again the
  number was too good. It is false; the smallest counterexample is an ordinary
  connected 8-vertex graph. A survey whose extremal cases all come from
  families with closed forms cannot see the extremal cases that do not.

  What caught both: extending the survey to graphs with no closed form. What
  fixes it going forward: the stratification is a specification in code rather
  than an adjective in prose, and `f(n)` is now computed exactly rather than
  conjectured.

- **A correctness bug in the code whose job was to detect correctness failures.**
  The modular driver folded the mod-2⁶⁴ residue into the "all residues agree"
  consistency check. A *detected* false negative therefore looked like an
  inconsistency, and the solver would have aborted on the very instance that
  demonstrates the failure mode — seven disjoint K₄ at k = 30, where
  `v₂(c₃₀) = 70`. Caught by the slow test that constructs exactly that instance
  (`test_false_negative_against_the_real_2_64_modulus`). Prime agreement and
  false-negative detection are now separate concepts, carried as separate fields
  — the test asserts `consistent` *and* `mod_2_64_is_false_negative` at once,
  which the old code could not have represented.
- **Two claims in the application README contradicted by my own data.** I wrote
  that the GPU indicator always beats the NumPy `O(2ⁿ)` DP; the DP is faster
  below n ≈ 15, where the GPU is launch-overhead-bound at a flat 0.18 ms. And I
  wrote that the early-exit and branchless indicator kernels were within noise;
  the early exit is worth 3.4× at n = 29. Caught by reading the benchmark JSON
  against the prose rather than trusting the prose — which is why the numbers
  now come from the tracked JSON instead of being retyped.
- **A benchmark artifact that inverted a table.** A 5-sample cap starved the
  sub-millisecond small-n phases, and the derived phase shares summed to more
  than 100%. Caught by that impossible total. Sampling is now governed by the
  time budget, and rows whose phase prefixes fail to nest are flagged rather
  than silently reported.
- **A traceability gap I had to go back and close.** The coalescing before/after
  table in §3.2 originally cited a script that, after the policy change, no
  longer produced the "before" number — the comparison was real when made but
  had become unreproducible. Caught by trying to re-run it. The pre-fix plan is
  now an explicit candidate the script rebuilds, so both rows come from one run.
- **I reached for a search where a theorem was available.** The `2⁶⁴ | c_χ`
  question in §4.3 is settled exactly by the free-action lemma. I instead
  surveyed 287 graphs and extrapolated a ratio that is not constant. The
  conclusion happened to be right and conservative, but the argument was weak.
  Caught in review; the search is now a theorem plus a bounded residual.
- **Three claims corrected in review, all of them mine, all the same error.**
  The `(2^k−1)ⁿ` tightening is a closed form, not an empirical 10–20%; the
  forced valuation is per-component and additive, not connected; "no sparse
  worst case" was true for `G(n,p)` and false in general. In each case I had
  generalised from the sample I happened to measure. §4.5 exists for the same
  reason — the density axis was added after it was pointed out that a sweep at
  fixed `p = 0.5` describes one slice.

---

## 6. Reproducing this

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest                    # 1523 fast tests
.venv/bin/python -m pytest -m slow            # 11 multi-GiB tests

# kernel
.venv/bin/python bench/bench.py            --out bench/results/bench.json
.venv/bin/python bench/thermal.py          --minutes 7 --out bench/results/thermal.json
.venv/bin/python bench/tune_coalescing.py  --out bench/results/coalescing_tuning.json

# application
.venv/bin/python bench/chromatic/bench.py                 --out bench/results/chromatic.json
.venv/bin/python bench/chromatic/modmul.py                --out bench/results/chromatic_modmul.json
.venv/bin/python bench/chromatic/false_negative_search.py
.venv/bin/python bench/chromatic/unforced_survey.py      --out bench/results/chromatic_unforced_survey.json
.venv/bin/python bench/chromatic/v2_conjecture_search.py --out bench/results/chromatic_v2_conjecture.json
.venv/bin/python bench/chromatic/memory_pressure.py      --out bench/results/chromatic_memory_pressure.json
brew install nauty   # geng, for the f(n) table only
.venv/bin/python bench/chromatic/max_v2_table.py --max-n 10 --out bench/results/chromatic_max_v2.json  # ~45 min at n=10
.venv/bin/python bench/chromatic/density.py --out bench/results/chromatic_density.json

# figures in this report (isolated env; they read only the JSON in data/)
cd docs/figures && uv run --no-project --with matplotlib --with numpy python kernel-bandwidth.py
```

The figure scripts read the tracked benchmark output in `bench/results/`
directly, so there is one copy of each result in the repository rather than two
that can drift. Everything this report cites — the JSON, the figure scripts and
the figures themselves — is tracked here; there is nothing to request.

**Provenance.** All measurements in this report were taken at `c1d4437`.
No executable code has changed since: `git diff c1d4437..HEAD -- yates/ apps/
bench/` touches documentation only, so the numbers describe the current tree.
The test count above is current and therefore higher than it was at `c1d4437`.
MLX 0.32.2, pinned in
[`requirements.txt`](../requirements.txt). Apple M4 GPU (`applegpu_g16g`), SIMD width 32
measured in-kernel, 32768 B threadgroup memory, 11.84 GiB recommended working
set. No thermal or performance warning was recorded by macOS during any run.

---

## Appendix A: Full per-n results

The three tables below are the complete sweep summarised in §3.1, one row per
`(element type, n)`. Regenerate them from the tracked JSON with:

```sh
.venv/bin/python bench/report.py bench/results/bench.json
```

**Column legend.**

| column | meaning |
|---|---|
| `n` | transform size; the array has `N = 2^n` elements per row |
| `batch` | rows transformed per launch, chosen to reach the minimum working set |
| `working set` | `batch × N × sizeof(elem)`, the footprint the copy denominator is measured at |
| `passes` | device passes the planner dispatched; traffic is `2 × passes × N × sizeof(elem)` |
| `tile` | threadgroup tile size in **bytes** (`choose_tile_bytes`) |
| `tiers simd/tg/reg` | how the `n` stages were resolved, summed over all passes: `simd_shuffle_xor` lanes / threadgroup memory / this thread's own registers. The three always sum to `n`. |
| `copy GB/s` | pure device-to-device copy kernel, re-measured at this working-set size in the same session |
| `transform ms` | median of the second half of the steady-state window |
| `transform GB/s` | `2 × passes × N × sizeof(elem)` divided by that time |
| `% of copy` | transform GB/s over copy GB/s. Above 100% is inter-pass reuse in the system cache. |
| `drift` | median of the window's second half over its first half. `1.000` means no slowdown while the configuration ran; `> 1` means it got slower. |

Protocol: 2.0 s steady-state window per configuration, 0.5 s warmup, 3.0 s idle
cooldown between configurations, working set at least 67 108 864 elements.
MLX 0.32.2 on Apple M4 (`applegpu_g16g`), 16 GB unified memory, macOS 26.3.
Run 2026-09-27T11:48:55-0300 to 11:57:46-0300; macOS recorded no thermal
warning at any point.

### A.1 uint32

- Fraction of measured copy bandwidth: 91.3% (n=25) to 100.2% (n=20); median 95.9%
- Transform bandwidth 89.9–98.9 GB/s
- Thermal re-check at n=10: 5.73 ms at the start of the sweep, 5.70 ms at the end (0.996x)

| n | batch | working set | passes | tile | tiers simd/tg/reg | copy GB/s | transform ms | transform GB/s | % of copy | drift |
|--:|------:|------------:|-------:|-----:|:------------------|----------:|-------------:|---------------:|----------:|------:|
| 10 | 65536 | 256 MiB | 1 | 4096 | 5/5/0 | 97.8 | 5.73 | 93.8 | 95.9% | 1.003 |
| 11 | 32768 | 256 MiB | 1 | 8192 | 5/5/1 | 98.2 | 5.66 | 94.8 | 96.5% | 1.005 |
| 12 | 16384 | 256 MiB | 1 | 16384 | 5/5/2 | 96.8 | 5.71 | 94.1 | 97.3% | 1.005 |
| 13 | 8192 | 256 MiB | 1 | 32768 | 5/5/3 | 97.8 | 5.65 | 95.0 | 97.1% | 1.001 |
| 14 | 4096 | 256 MiB | 2 | 4096 | 5/9/0 | 98.2 | 11.30 | 95.0 | 96.8% | 1.006 |
| 15 | 2048 | 256 MiB | 2 | 4096 | 5/10/0 | 98.1 | 11.40 | 94.2 | 96.0% | 1.007 |
| 16 | 1024 | 256 MiB | 2 | 8192 | 5/9/2 | 97.4 | 11.49 | 93.4 | 95.9% | 0.943 |
| 17 | 512 | 256 MiB | 2 | 8192 | 5/10/2 | 98.3 | 11.22 | 95.7 | 97.4% | 1.001 |
| 18 | 256 | 256 MiB | 2 | 16384 | 5/9/4 | 97.9 | 11.63 | 92.3 | 94.2% | 1.035 |
| 19 | 128 | 256 MiB | 2 | 16384 | 5/10/4 | 95.8 | 11.25 | 95.5 | 99.6% | 0.996 |
| 20 | 64 | 256 MiB | 2 | 32768 | 5/9/6 | 94.3 | 11.37 | 94.5 | 100.2% | 1.007 |
| 21 | 32 | 256 MiB | 2 | 32768 | 5/10/6 | 94.3 | 11.88 | 90.4 | 95.9% | 1.008 |
| 22 | 16 | 256 MiB | 3 | 8192 | 5/14/3 | 98.5 | 17.53 | 91.9 | 93.3% | 1.000 |
| 23 | 8 | 256 MiB | 3 | 8192 | 5/15/3 | 98.3 | 17.83 | 90.3 | 91.9% | 1.002 |
| 24 | 4 | 256 MiB | 3 | 16384 | 5/13/6 | 98.2 | 17.50 | 92.0 | 93.7% | 0.999 |
| 25 | 2 | 256 MiB | 3 | 16384 | 5/14/6 | 98.5 | 17.91 | 89.9 | 91.3% | 1.003 |
| 26 | 1 | 256 MiB | 3 | 32768 | 5/12/9 | 98.4 | 17.64 | 91.3 | 92.8% | 1.011 |
| 27 | 1 | 512 MiB | 4 | 16384 | 5/15/7 | 99.7 | 46.02 | 93.3 | 93.6% | 1.007 |
| 28 | 1 | 1024 MiB | 4 | 16384 | 5/15/8 | 98.1 | 92.21 | 93.2 | 94.9% | 0.998 |
| 29 | 1 | 2048 MiB | 4 | 32768 | 5/13/11 | 99.6 | 173.77 | 98.9 | 99.3% | 0.971 |

### A.2 uint64

- Fraction of measured copy bandwidth: 92.9% (n=26) to 100.6% (n=28); median 95.8%
- Transform bandwidth 91.6–99.3 GB/s
- Thermal re-check at n=10: 11.16 ms at the start of the sweep, 11.13 ms at the end (0.998x)

| n | batch | working set | passes | tile | tiers simd/tg/reg | copy GB/s | transform ms | transform GB/s | % of copy | drift |
|--:|------:|------------:|-------:|-----:|:------------------|----------:|-------------:|---------------:|----------:|------:|
| 10 | 65536 | 512 MiB | 1 | 8192 | 5/5/0 | 98.4 | 11.16 | 96.2 | 97.8% | 0.998 |
| 11 | 32768 | 512 MiB | 1 | 16384 | 5/5/1 | 98.4 | 11.12 | 96.6 | 98.2% | 0.999 |
| 12 | 16384 | 512 MiB | 1 | 32768 | 5/5/2 | 98.7 | 11.15 | 96.3 | 97.5% | 1.000 |
| 13 | 8192 | 512 MiB | 2 | 4096 | 5/8/0 | 98.4 | 22.62 | 94.9 | 96.5% | 0.995 |
| 14 | 4096 | 512 MiB | 2 | 4096 | 6/8/0 | 98.9 | 22.67 | 94.7 | 95.8% | 1.001 |
| 15 | 2048 | 512 MiB | 2 | 8192 | 5/10/0 | 98.3 | 22.83 | 94.1 | 95.7% | 1.001 |
| 16 | 1024 | 512 MiB | 2 | 8192 | 6/10/0 | 98.4 | 23.44 | 91.6 | 93.1% | 0.985 |
| 17 | 512 | 512 MiB | 2 | 16384 | 5/10/2 | 98.9 | 22.79 | 94.2 | 95.3% | 1.002 |
| 18 | 256 | 512 MiB | 2 | 16384 | 6/10/2 | 98.0 | 22.84 | 94.0 | 96.0% | 1.001 |
| 19 | 128 | 512 MiB | 2 | 32768 | 5/10/4 | 99.1 | 22.88 | 93.9 | 94.7% | 1.002 |
| 20 | 64 | 512 MiB | 2 | 32768 | 6/10/4 | 98.5 | 22.92 | 93.7 | 95.1% | 1.001 |
| 21 | 32 | 512 MiB | 3 | 8192 | 6/15/0 | 98.4 | 34.15 | 94.3 | 95.8% | 1.001 |
| 22 | 16 | 512 MiB | 3 | 8192 | 7/15/0 | 98.8 | 34.58 | 93.2 | 94.3% | 1.003 |
| 23 | 8 | 512 MiB | 3 | 16384 | 6/14/3 | 96.8 | 34.00 | 94.8 | 97.9% | 1.005 |
| 24 | 4 | 512 MiB | 3 | 16384 | 6/15/3 | 99.1 | 34.88 | 92.4 | 93.2% | 1.007 |
| 25 | 2 | 512 MiB | 3 | 32768 | 6/13/6 | 98.9 | 34.15 | 94.3 | 95.4% | 1.001 |
| 26 | 1 | 512 MiB | 4 | 16384 | 6/17/3 | 99.7 | 46.38 | 92.6 | 92.9% | 1.008 |
| 27 | 1 | 1024 MiB | 4 | 16384 | 6/17/4 | 96.2 | 91.61 | 93.8 | 97.4% | 1.003 |
| 28 | 1 | 2048 MiB | 4 | 32768 | 6/15/7 | 98.7 | 173.01 | 99.3 | 100.6% | 0.999 |

### A.3 float32

- Fraction of measured copy bandwidth: 91.6% (n=25) to 100.0% (n=29); median 96.7%
- Transform bandwidth 90.6–99.5 GB/s
- Thermal re-check at n=10: 5.67 ms at the start of the sweep, 5.97 ms at the end (1.054x)

| n | batch | working set | passes | tile | tiers simd/tg/reg | copy GB/s | transform ms | transform GB/s | % of copy | drift |
|--:|------:|------------:|-------:|-----:|:------------------|----------:|-------------:|---------------:|----------:|------:|
| 10 | 65536 | 256 MiB | 1 | 4096 | 5/5/0 | 98.8 | 5.67 | 94.7 | 95.9% | 1.000 |
| 11 | 32768 | 256 MiB | 1 | 8192 | 5/5/1 | 98.3 | 5.64 | 95.1 | 96.8% | 1.002 |
| 12 | 16384 | 256 MiB | 1 | 16384 | 5/5/2 | 98.8 | 5.64 | 95.2 | 96.3% | 0.997 |
| 13 | 8192 | 256 MiB | 1 | 32768 | 5/5/3 | 98.4 | 5.65 | 95.0 | 96.6% | 1.003 |
| 14 | 4096 | 256 MiB | 2 | 4096 | 5/9/0 | 98.7 | 11.24 | 95.5 | 96.7% | 0.996 |
| 15 | 2048 | 256 MiB | 2 | 4096 | 5/10/0 | 98.6 | 11.22 | 95.7 | 97.1% | 1.002 |
| 16 | 1024 | 256 MiB | 2 | 8192 | 5/9/2 | 98.7 | 11.16 | 96.2 | 97.5% | 1.003 |
| 17 | 512 | 256 MiB | 2 | 8192 | 5/10/2 | 97.1 | 11.09 | 96.8 | 99.7% | 0.909 |
| 18 | 256 | 256 MiB | 2 | 16384 | 5/9/4 | 97.3 | 11.12 | 96.6 | 99.2% | 0.999 |
| 19 | 128 | 256 MiB | 2 | 16384 | 5/10/4 | 98.9 | 11.18 | 96.1 | 97.1% | 0.998 |
| 20 | 64 | 256 MiB | 2 | 32768 | 5/9/6 | 98.9 | 11.16 | 96.2 | 97.3% | 0.993 |
| 21 | 32 | 256 MiB | 2 | 32768 | 5/10/6 | 98.7 | 11.21 | 95.8 | 97.0% | 1.002 |
| 22 | 16 | 256 MiB | 3 | 8192 | 5/14/3 | 98.8 | 17.50 | 92.0 | 93.1% | 1.032 |
| 23 | 8 | 256 MiB | 3 | 8192 | 5/15/3 | 98.6 | 17.77 | 90.6 | 91.9% | 1.001 |
| 24 | 4 | 256 MiB | 3 | 16384 | 5/13/6 | 98.9 | 17.13 | 94.0 | 95.1% | 0.981 |
| 25 | 2 | 256 MiB | 3 | 16384 | 5/14/6 | 98.9 | 17.78 | 90.6 | 91.6% | 1.010 |
| 26 | 1 | 256 MiB | 3 | 32768 | 5/12/9 | 98.7 | 17.13 | 94.0 | 95.3% | 0.977 |
| 27 | 1 | 512 MiB | 4 | 16384 | 5/15/7 | 100.4 | 45.39 | 94.6 | 94.2% | 0.998 |
| 28 | 1 | 1024 MiB | 4 | 16384 | 5/15/8 | 98.5 | 91.25 | 94.1 | 95.6% | 1.002 |
| 29 | 1 | 2048 MiB | 4 | 32768 | 5/13/11 | 99.5 | 172.72 | 99.5 | 100.0% | 1.001 |
