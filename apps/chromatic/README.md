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
| `c_k mod m != 0` | `c_k != 0`; and `c_k` is a *count*, hence non-negative, so `c_k > 0` and **G is k-colourable**. Sound certificate, for any `m`. |
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
the memory ceiling — have `v2(c_30) = 70`**. `c_30` is a 149-digit number — the
test computes `c_30(K_4)^7` exactly in Python integers, so the digit count and
the valuation are verified rather than estimated — and `c_30 mod 2^64 == 0`: the single-modulus test reports "not 30-colourable" for a
graph that is obviously 30-colourable. An eight-vertex analogue against a `2^16`
modulus runs in the fast suite.

### Can a false negative corrupt the reported chi?

A false negative only matters if it lands at `k = chi` itself, rather than at
some `k` the binary search never visits — i.e. if `2^64 | c_chi`. That question
has an exact answer, not a heuristic one.

**Lemma.** `chi! | c_chi(G)` for every graph `G`.

*Proof.* Suppose a covering `(S_1, …, S_chi)` of `V` by independent sets has
`S_i = S_j` for some `i != j`. Dropping `S_j` still covers `V`, so `V` is
covered by `chi − 1` independent sets — and a covering by `m` independent sets
implies `m`-colourability (give each vertex the least index covering it; every
colour class is a subset of an independent set). That contradicts minimality of
`chi`. So at `k = chi` the components are pairwise distinct, the
coordinate-permutation action of `S_chi` on the coverings is **free**, every
orbit has size `chi!`, and `chi!` divides their number. ∎

The lemma is **tight on cliques**: at `k = n` every slot of `K_n` must hold a
distinct singleton, so `c_n(K_n) = n!`. By Legendre `v2(n!) = n − popcount(n)`,
so the first clique with `2^64 | c_chi` is exactly

    n = 66  =  1000010₂,   popcount 2,   v2(66!) = 64
    (n = 65 and n = 64 both give 63)

**`K_66` is therefore an exact counterexample at `k = chi`**: `c_66 = 66!`,
`v2 = 64`, so a single-modulus solver declares `K_66` not 66-colourable. That is
arithmetic, not a search.

And this makes the practical conclusion *stronger*, not weaker — but the
statement has to be made **additively**, because `c_k` is multiplicative over
connected components and so `v2` is additive. The lemma applies to a component
only at `k = chi(component)`, giving

    forced(G) = #{components with chi_i = chi(G)} · v2(chi!)

Components with `chi_i < chi` are evaluated at `k > chi_i`, where the action is
not free and the divisibility genuinely fails, so they force nothing. Applying
the connected formula to a disconnected graph badly understates it: **seven
disjoint `K_4`** has `c_4 = (4!)^7`, hence `v2 = 21` — all of it forced, since
each component's unordered cofactor is 1 — while `v2(chi!) = 3` alone would
account for three.

A component attaining `chi` needs at least `chi` vertices, so `m·chi <= n`, and

    max forced over all graphs on <= n vertices = max_{m·chi <= n} m·v2(chi!)

At `n = 29` that is **25**, attained by a single connected component with
`chi = 28` or `29`. Allowing disconnected graphs does *not* raise it: the extra
multiplicity never buys back what the smaller `chi` gives up. That step is
load-bearing for the number 25, and by Legendre the problem is the one-line
optimisation `max_{m*chi <= n} m*(chi - s_2(chi))`, whose top entries at
`n = 29` are close enough that it is worth writing out:

| m | chi | `m*v2(chi!)` |
|--:|--:|--:|
| 1 | 29 | 29 − 4 = **25** |
| 1 | 28 | 28 − 3 = **25** |
| 2 | 14 | 2·11 = 22 |
| 3 | 8 | 3·7 = 21 |
| 7 | 4 | 7·3 = 21 |
| 4 | 7 | 4·4 = 16 |

The maximiser is not monotone in either variable, so the search space is
enumerated over every `(m, chi)` with `m*chi <= n`, for every `n <= 29`, rather
than asserted:
`test_disconnected_graphs_never_raise_the_forced_bound` and
`test_forced_bound_runners_up_at_the_ceiling` in `tests/chromatic/test_modular.py`.
The full leaderboard is in [the project report, §4.3](../../docs/report.md#43-the-one-sided-guarantee-and-the-divisibility-lemma-that-bounds-its-failure).
(So the value 25 was right; the *formula* behind it was not, and would have been
wrong on a disconnected input.)

At least **39 of the 64 bits** would therefore have to come from the *unforced*
part — the per-component unordered cofactors `c_chi(G_i)/chi!`, plus the
components below `chi`. The survey measuring that is **stratified**, because
two of the families in it have provably odd cofactors and would otherwise
inflate the result:

| family | `c_chi` | forced | cofactor |
|---|---|---|---|
| `K_a` × m disjoint | `(a!)^m` | `m·v2(a!)` | `1` |
| `K_a + E_b` | `a!·(2^a − 1)^b` | `v2(a!)` | `(2^a − 1)^b`, **odd for all a, b** |

Every graph from those families contributes `v2 = 0` *by theorem, not by
observation*, so counting them as evidence for the oddness bias would be
circular. They are measured anyway, for a different reason: the script checks
both closed forms against the solver, and **55 of 55 agree exactly**, which is
a real cross-check of the `c_k` path. But they are reported separately.

**The evidence is the other stratum** — 139 graphs with no closed-form
cofactor (cycles, paths, Turán, complete bipartite, Petersen, Grötzsch,
Chvátal, Mycielskians, mixed disjoint unions, and 80 random `G(n,p)` over
n = 5..9 and p = 0.2..0.9):

| `v2` of the unforced part | 0 | 1 | 2 | 3 | 4 | 5 | 8 |
|---|--:|--:|--:|--:|--:|--:|--:|
| graphs | 116 | 10 | 5 | 3 | 2 | 2 | 1 |
| share | 83.5% | 7.2% | 3.6% | 2.2% | 1.4% | 1.4% | 0.7% |
| a "random" integer, `2^-(j+1)` | 50% | 25% | 12.5% | 6.25% | 3.13% | 1.56% | 0.20% |

Counts are given beside the shares because the tail cells are one or two graphs
each, where a percentage alone overstates how much is being measured.
(`bench/chromatic/unforced_survey.py`,
[`bench/results/chromatic_unforced_survey.json`](../../bench/results/chromatic_unforced_survey.json).)

So the defensible claim is: **among graphs with no closed-form cofactor,
the cofactor is odd 83.5% of the time against a 50% baseline, over 139
graphs.** The bias toward oddness — not merely the absence of large values —
is the evidence.

Two caveats that the stratification makes visible. The headline figure is
partly a statement about survey composition: add more cliques and it rises,
which is exactly why the two populations are now named in the script rather
than described as "weighted toward the hard cases". And **the largest unforced
valuation observed is 8** — on a single `G(9, 0.3)` instance — which is the
same value the test asserts as a ceiling. That ceiling was chosen to be
generous and is now tight, and the observed maximum grew from 4 to 8 when the
survey grew. Read that as evidence the unforced part is small relative to the
39 bits needed, not as evidence it is bounded by a constant.

**Why the cofactor tends to be odd (partial).** `c_chi/chi!` counts unordered
minimal covers, a set `X` on which `Aut(G)` acts. For any **2-group** `P`
acting on a finite set, `|X| ≡ |X^P| (mod 2)`, since every non-fixed `P`-orbit
has size a power of 2 greater than 1. Taking `P` to be a Sylow 2-subgroup of
`Aut(G)` is the useful specialisation. That congruence is a standard theorem, and is
verified here on 15 small graphs by brute-force automorphism enumeration
(`|X|` is also checked to equal the cofactor). What is **conjectural** is that
it explains the observed bias: that would need `|X^P|` to be odd unusually
often, for which there is no argument here. Consistent with `K_n` (`X` is the
single partition into singletons, fixed by everything, cofactor 1) and with
Chvátal (`|Aut| = 8`, cofactor `v2 = 3`).

**A separate conjecture, `v2(c_chi) <= n`, is FALSE.** It was recorded here as
"verified over the survey and never violated", with `K_66` (64 of 66) and seven
disjoint `K_4` (21 of 28) as the tight witnesses — both closed-form clique
families. Extending the survey produced a counterexample, and it is small
enough to check by hand:

    G on 8 vertices, connected, 14 edges
    (0,4) (0,5) (1,2) (1,5) (1,7) (2,3) (2,5) (2,6) (2,7) (3,5) (3,6) (4,6) (5,7) (6,7)

    chi = 4,  c_4 = 1536 = 2^9 * 3,  so v2(c_chi) = 9 > 8 = n

The boundary is exact: **the bound holds for every labeled graph on `n <= 7`
vertices — all 2,097,152 of them at `n = 7`, checked exhaustively, where it is
attained with equality but never exceeded — and fails at `n = 8`.**
(`bench/chromatic/v2_conjecture_search.py`,
[`bench/results/chromatic_v2_conjecture.json`](../../bench/results/chromatic_v2_conjecture.json);
`c_4` is confirmed by a union-mask DP that uses no inclusion–exclusion.)

Nothing relied on the conjecture — the default mode is CRT and unconditional —
and the residual-risk argument is unaffected, since it needs 39 unforced bits
and this reaches 9 of 64 on 8 vertices. But the refutation is worth stating,
because the conjecture was published on the strength of a survey containing
only clique families at the boundary, which is the same composition problem
that the stratification above exists to fix.

### What the falsification costs, and what replaces it

The conjecture was the only route to an **unconditional** safety statement
here. `forced <= 25` is a theorem, and `v2(c_chi) <= n <= 29` would have capped
the total at 29 against the 64 bits a false negative needs. That route is gone.

The replacement is not another conjecture but the function itself. Write
`f(n) = max over graphs on n vertices of v2(c_chi(G))`. It is exactly
computable for small `n`, because `v2(c_chi)` is isomorphism-invariant — so the
search space is *unlabeled* graphs, 11,117 connected on 8 vertices, 261,080 on 9
and 11,716,571 on 10, against `2^45` labeled at n = 10 — and because `c_k` is multiplicative over disjoint
unions, so the disconnected case is a knapsack over the connected table. (That
knapsack needs `v2(c_k(G_i))` for every `k >= chi_i`, not only at `chi_i`, since
a component sits at `k = max_i chi_i`. Padding with isolated vertices never
changes `v2`, so max over `|V| = n` and max over `|V| <= n` coincide.)

Computed exactly with `geng` from nauty
(`bench/chromatic/max_v2_table.py`,
[`bench/results/chromatic_max_v2.json`](../../bench/results/chromatic_max_v2.json)):

| n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `f(n)` | 0 | 1 | 1 | 3 | 3 | 5 | 7 | 10 | 12 | 17 |
| `f(n) − n` | −1 | −1 | −2 | −1 | −2 | −1 | **0** | **+2** | **+3** | **+7** |

`f(7) = 7` reproduces the exhaustive labeled result exactly. Note also that the
8-vertex graph published above as the counterexample has `v2 = 9`, but it is not
extremal: `f(8) = 10`.

**This is the part worth knowing, and it is worse than the conjecture
suggested.** The bound was tight at n = 7 and the gap has widened at every step
since — +2, +3, then **+7 at n = 10**, where `f(10) = 17`. Growth is not one
bit per vertex; over the last five points it is roughly three:

| fit window | slope (bits/vertex) | extrapolated `f(29)` |
|---|--:|--:|
| n = 5..10 | 2.69 | 67 |
| n = 6..10 | 2.90 | 71 |
| n = 7..10 | 3.20 | 77 |
| n = 8..10 | 3.50 | 83 |

**Every window crosses 64 before n = 29**, and the slope rises as points are
added. The conjecture would have promised `v2(c_chi) <= 29`; the measured
function instead extrapolates *past* the 64 bits a false negative at
`k = chi` needs. The single-modulus mode's safety at `n <= 29` is therefore
not merely unproved — the trend in the only exactly computable evidence
points against it.

Three caveats. The first two cut against the alarming reading; the third is
why none of it changes what the solver does:

- It is an extrapolation across 19 vertices from five points, with no
  derivation behind the slope. It is not a bound in either direction, and the
  true `f` could bend either way.
- `f(n)` is a **worst case over all graphs**, and the extremal ones are
  engineered: `f(10) = 17` is attained by a particular connected 10-vertex
  graph with `chi = 6`, not by anything a user is likely to hold. The survey's
  evidence stratum — ordinary graphs with no closed-form cofactor — still tops
  out at an unforced valuation of 8.
- **Nothing in the solver depends on any of this.** `mode="exact"` is the
  default, reconstructs `c_k` by CRT, and carries no probability at all. This
  section is entirely about what the single-modulus `mode="mod2_64"` would
  risk, and that mode is never described as exact anywhere.

The practical consequence is a sharpened recommendation, not a bug:
`mode="mod2_64"` is a fast **upper bound** on `chi`, and the reassurance that a
corrupted `chi` is out of reach at `n <= 29` rested on a conjecture that is now
false and a trend that now points the other way. Use the default.

The honest summary is therefore not "not found across a search" but: *the
forced part is bounded by 25 of the 64 bits by a theorem; the unforced part is
bounded by nothing proved, and the exactly computed `f(n)` above extrapolates
past 64 well before n = 29.* Still not a proof for arbitrary `n <= 29`,
and the default mode does not rely on it.

### Closing the gap

**Multi-modular (probabilistic).** Recompute `c_k` modulo `r` primes drawn
uniformly at random from `[2^30, 2^31)`. If `c_k != 0`, it has at most
`D = floor(log(c_k) / log(2^30))` distinct prime divisors in that range, and by
the Rosser–Schoenfeld bounds `x/ln x < pi(x) < 1.25506 x/ln x` for `x >= 17`
[[Rosser & Schoenfeld 1962, *Illinois J. Math.* 6(1), Thm. 1]][rs62] the pool
size is

    M = pi(2^31) - pi(2^30) > 2^31/ln(2^31) - 1.25506 * 2^30/ln(2^30) > 3.5e7

so one draw hides a nonzero `c_k` with probability at most `D/M`, and `r`
independent draws with probability at most `(D/M)^r`. **The assumption is that
the primes are drawn independently of the instance** — true here, since the seed
does not depend on the graph, but it would not hold against an adversary who
knows the seed.

**CRT (unconditional).** `c_k` counts ordered `k`-tuples of independent sets, so

    0 <= c_k <= i(V)^k

and `i(V)` is already computed — it is the top entry of the zeta. A second bound
is free: a covering is determined by which *nonempty* subset of the `k` slots
holds each vertex, so `c_k <= (2^k − 1)^n`, and the implementation uses
`min(i(V)^k, (2^k − 1)^n)`. (The slot bound is an *equality* on the edgeless
graph, where every subset is independent.)

How much that buys is a closed form, not a measurement. Since `log2 i(V) <= n`
always, the ratio of bit-counts is minimised at `i(V) = 2^n`. Note what that
case is: it is the edgeless graph, where the slot bound is exact and `i(V)^k`
is at its loosest — so this is the bound on how little the slot bound can help,
not a pessimistic instance in any other sense:

    log2 B₂ / log2 B₁ = n·log2(2^k − 1) / (k·log2 i(V))
                      ≥ log2(2^k − 1)/k = 1 + log2(1 − 2^-k)/k

which is 0.9358 at k=3, 0.99621 at k=6 and 0.999859 at k=10 — the saving decays
as `Θ(2^-k/k)`. Across the feasible grid it therefore changes the prime count in
exactly 42 `(n,k)` combinations and never by two: a consequence of the formula,
not an observation about it. Worth taking because it is free; not a lever.
Once the product of the moduli exceeds that bound, CRT reconstructs `c_k` as an exact integer and no probability remains. This is much cheaper than "small n" suggests,
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

Two approaches suggest themselves for running the kernel mod `p`: a "different
generator", or a reduction after each device pass. Neither works, and
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

31 bits is a smaller modulus than the ~62-bit primes one might reach for, which
weakens the per-prime failure bound; the bound above accounts for that, and it
remains negligible at `r >= 2`. In exchange the kernel is reused verbatim, so all of its
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
(6 GiB of subset state, the detected memory ceiling) takes 1531 ms end to end**,
and a 28-vertex one takes 544 ms — regardless of how hard the instance is.

Both figures are `mode="exact"` — unconditional, CRT-reconstructed — on a
`G(n, 0.5)` instance, which at n = 29 needs **3 CRT primes** per candidate `k`.
Naming the family matters, not just the density: the cost of exact mode scales
with `k · log2 i(V)`, and the hardest instances are heterogeneous rather than
dense. At n = 26 a `G(26, 0.5)` takes 117 ms with 2 primes, while
`K_13 + E_13` — a clique beside an independent set — takes 205 ms with 8, and
`K_16 + E_10` takes 227 ms. See the two sweeps below.

**Two measured results that went against expectation**, both detailed below and
worth stating up front because the natural guess is wrong in each case:

- **The worst case for exact mode is not the densest graph, and is not reachable
  by `G(n,p)` at all.** Cost scales with `chi · log2 i(V)`, and within `G(n,p)`
  that product is self-limiting — the whole density sweep stays at 2–3 primes
  and spans only 2.7× in wall clock. The extremum is *heterogeneous*:
  `K_a + E_b` drives the product to `Θ(n²)` near `a ≈ n/2` and needs 8 primes
  where `G(n,0.5)` needs 2. [Cost against density](#cost-against-density-at-fixed-n)
- **The early exit beats the branchless kernel by 3.4× at n = 29**, despite the
  divergence — the usual GPU instinct is the opposite. Most subsets are not
  independent and fail on one of their first few vertices, so the exit cuts the
  average iteration count far more than divergence costs.
  [Indicator variants](#indicator-variants)

- Machine: MacBook Air (Mac16,13), Apple M4, 16 GB unified memory, macOS 26.3
- GPU: Apple M4 (applegpu_g16g), MLX 0.32.2
- Memory ceiling detected at runtime: n=29 (12 B per subset, i(S) as uint32 plus the uint64 power terms). The binding constraint is 60% of the detected recommended working set (11.84 GiB, so a 7.10 GiB budget): 2^29 x 12 B = 6 GiB fits and 2^30 would need 12 GiB. Max buffer length (8.88 GiB) does not bind: no single array here exceeds it
- Phases are measured as nested prefixes of the real computation (indicator; indicator+zeta; end-to-end) and the inner ones subtracted, so the three shares sum to the measured total by construction
- Protocol: 1.5s steady-state window per phase (median of its second half), 0.4s warmup, 2.0s idle cooldown, no cap on iterations per window
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

**Reading the table.** `i(V)` is the number of independent sets in `G` — the
top entry of the zeta, and the quantity the CRT bound is taken against. The
k-search's cost is proportional to `power units x 2^n`, where a **power unit**
is one pointwise power plus reduction: the search evaluates `c_k` at each
candidate `k`, once per CRT prime needed *at that k*. Both factors vary between
rows for instance-dependent reasons — `k` depends on how tight the greedy
bounds happened to be on this seed — so the raw `k-search` and `end-to-end`
columns are **not comparable between consecutive rows**. `k-search / unit` is
the one to read for scaling.

A caution that cost a wrong column here: the unit count is **not** `k` times
the prime count at `chi`. Each `k` needs its own number of primes. At n = 28
the search runs k = 6 with 3 primes and k = 5 with 2, so 5 units; at n = 29 it
runs k = 6 and k = 7 with 3 each, so 6. Normalising by `k x primes_at_chi`
made both rows look like 6 units and inflated the apparent anomaly below.

| n | edges | chi | i(V) | memory | indicator | zeta | k-search | k | power units | k-search / unit | end-to-end | indicator / zeta / k-search share |
|--:|------:|----:|-----:|-------:|----------:|-----:|---------:|--:|-----------:|----------------:|-----------:|:--|
| 10 | 22 | 4 | 57 | 0 MiB | 0.18 ms | 0.02 ms | 0.5 ms | 1 | 1 | 0.47 ms | 0.7 ms | 27% / 3% / 70% |
| 11 | 27 | 4 | 64 | 0 MiB | 0.18 ms | 0.02 ms | 0.5 ms | 1 | 1 | 0.47 ms | 0.7 ms | 27% / 3% / 70% |
| 12 | 32 | 5 | 83 | 0 MiB | 0.18 ms | 0.03 ms | 0.7 ms | 1 | 2 | 0.35 ms | 0.9 ms | 20% / 3% / 77% |
| 13 | 37 | 4 | 118 | 0 MiB | 0.18 ms | 0.03 ms | 0.5 ms | 1 | 1 | 0.48 ms | 0.7 ms | 26% / 5% / 69% |
| 14 | 44 | 5 | 121 | 0 MiB | 0.18 ms | 0.04 ms | 0.7 ms | 1 | 2 | 0.36 ms | 0.9 ms | 19% / 5% / 76% |
| 15 | 51 | 5 | 152 | 0 MiB | 0.18 ms | 0.03 ms | 0.8 ms | 1 | 2 | 0.41 ms | 1.0 ms | 17% / 3% / 80% |
| 16 | 57 | 5 | 194 | 1 MiB | 0.18 ms | 0.08 ms | 1.5 ms | 2 | 4 | 0.37 ms | 1.8 ms | 10% / 5% / 85% |
| 17 | 66 | 5 | 242 | 2 MiB | 0.19 ms | 0.09 ms | 1.7 ms | 2 | 4 | 0.42 ms | 1.9 ms | 10% / 5% / 86% |
| 18 | 71 | 6 | 290 | 3 MiB | 0.20 ms | 0.11 ms | 1.9 ms | 2 | 4 | 0.48 ms | 2.2 ms | 9% / 5% / 86% |
| 19 | 80 | 5 | 352 | 6 MiB | 0.23 ms | 0.11 ms | 2.3 ms | 2 | 4 | 0.58 ms | 2.7 ms | 9% / 4% / 87% |
| 20 | 89 | 5 | 372 | 12 MiB | 0.27 ms | 0.21 ms | 1.7 ms | 1 | 2 | 0.87 ms | 2.2 ms | 12% / 9% / 79% |
| 21 | 97 | 6 | 490 | 24 MiB | 0.30 ms | 0.51 ms | 5.4 ms | 2 | 4 | 1.35 ms | 6.2 ms | 5% / 8% / 87% |
| 22 | 105 | 6 | 667 | 48 MiB | 0.37 ms | 1.16 ms | 8.3 ms | 2 | 4 | 2.07 ms | 9.8 ms | 4% / 12% / 84% |
| 23 | 117 | 6 | 660 | 96 MiB | 0.53 ms | 2.18 ms | 7.1 ms | 1 | 2 | 3.57 ms | 9.9 ms | 5% / 22% / 72% |
| 24 | 132 | 6 | 784 | 192 MiB | 0.93 ms | 4.03 ms | 25.7 ms | 2 | 4 | 6.41 ms | 30.6 ms | 3% / 13% / 84% |
| 25 | 140 | 6 | 830 | 384 MiB | 1.53 ms | 8.02 ms | 24.7 ms | 1 | 2 | 12.34 ms | 34.2 ms | 4% / 23% / 72% |
| 26 | 150 | 6 | 969 | 768 MiB | 2.76 ms | 15.96 ms | 96.4 ms | 2 | 4 | 24.09 ms | 115.1 ms | 2% / 14% / 84% |
| 27 | 167 | 7 | 1126 | 1536 MiB | 5.23 ms | 42.28 ms | 130.3 ms | 1 | 3 | 43.43 ms | 177.8 ms | 3% / 24% / 73% |
| 28 | 181 | 6 | 1066 | 3072 MiB | 10.28 ms | 85.50 ms | 448.5 ms | 2 | 5 | 89.70 ms | 544.3 ms | 2% / 16% / 82% |
| 29 | 193 | 7 | 1417 | 6144 MiB | 20.33 ms | 170.59 ms | 1340.3 ms | 2 | 6 | 223.38 ms | 1531.2 ms | 1% / 11% / 88% |

#### The n = 29 row costs more than its parts predict

Per unit, **n = 29 is 2.49x n = 28**, where 2.0x is expected. That is the
headline number's row, so it was measured rather than guessed
(`bench/chromatic/memory_pressure.py`,
[`bench/results/chromatic_memory_pressure.json`](../../bench/results/chromatic_memory_pressure.json)).

**1. It is not the memory system.** The kernel benchmark re-measures copy
bandwidth at every working-set size because it varies — but it stops at
2048 MiB, and n = 29 runs at 6144 MiB. Extending that measurement, copy
bandwidth is **flat at 99–103 GB/s from 1 GiB to 10 GiB of live data**: no
sustained degradation at large footprints on this machine. (12 GiB live cannot
be measured at all — an out-of-place copy of a 6144 MiB array needs twice that,
against an 11.84 GiB budget.)

**2. Part of it is work the old column hid.** n = 29 runs 6 power units to
n = 28's 5. Timed directly, one unit scales cleanly — 69.8 ms to 140.6 ms, a factor of
**2.015x** — and barely depends on `k` (69.5 ms at k = 5 against
69.7 ms at k = 6). The kernel scales exactly as it should; 6/5 units on 2x data
predicts **2.42x** of the measured 2.99x k-search ratio.

**3. Part is allocation pressure, isolated by ballast.** Re-running the *same*
n = 28 instance — same `k`, same primes, same plan, only a live dummy array
raising the peak:

| run | peak | median | vs. baseline |
|---|--:|--:|--:|
| n = 28, no ballast | 5.00 GiB | 565.9 ms | 1.000x |
| n = 28, +3 GiB | 8.00 GiB | 560.3 ms | 0.990x |
| n = 28, +4 GiB | 9.00 GiB | 562.9 ms | 0.995x |
| n = 28, +5 GiB | 10.00 GiB | 626.2 ms | **1.106x** |
| n = 28, +6 GiB | 11.00 GiB | 659.2 ms | 1.165x |
| n = 28, +7 GiB | 12.00 GiB | 660.2 ms | 1.166x |
| n = 29, no ballast | 10.00 GiB | 1511.3 ms | — |

The effect has nothing to do with `n`, switches on **between 9 and 10 GiB of
peak** rather than gradually, and then flattens (1.17x at 11 GiB, 1.17x at
12 GiB) — it does not steepen past 10 GiB, which is where n = 29 natively sits.

**What is left.** 2.43x of work times
1.11x of pressure is 2.69x against a
measured 2.99x: about **1.11x
unexplained**, down from the 1.20x reported before any of this was measured. One hypothesis died on the way. Under ballast
the slowdown is intermittent — median 660.8 ms but individual iterations to
914 ms — whereas n = 29's own spread is 1490.8–1522.5 ms, under 3%.
Whatever the last 11% is, it is uniform rather than bursty, so
allocator churn is not it.

The headline 1531 ms stands — it is what the machine does — and is now within
about 11% of what its own parts predict.

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

### Cost against density, at fixed n

The n-sweep above holds density at 0.5. That is the wrong axis for the cost of
`mode="exact"`, which scales with the CRT prime count and hence with
`k · log2 i(V)`. Those two factors pull against each other: sparse graphs have
many independent sets but small chi, dense graphs the reverse. Whether the worst
case is sparse, dense or in between is empirical, so here it is, at n = 26
(`bench/chromatic/density.py`):

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

**Within `G(n, p)` the prime count never leaves 2–3, and the whole density range
spans only 2.7x in wall clock.** The product `chi · log2 i(V)` stays between 40
and 75 bits, and it is self-limiting *for random graphs*: `alpha ~ 2 log_b n`
gives `chi ~ n / (2 log_b n)` and `log2 i(V) = Theta(log^2 n)`, so the product is
`Theta(n log n)` rather than the naive `Theta(n^2)`.

**That reasoning does not extend to all graphs, and the extremum is outside
`G(n,p)`.** What drives the product up is *heterogeneity*, not density. For a
clique beside an independent set, `G = K_a + E_b` with `n = a + b`, we have
`chi = a` and `i(V) = (a+1)·2^b`, so `chi · log2 i(V) ≈ a(log2(a+1) + b)` is
`Theta(n^2)` near `a ≈ n/2`. `G(26, 0.5)` produces such a graph with probability
zero for practical purposes, so the sweep above cannot reach it:

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

**Eight primes, against two or three for every `G(n,p)`** — and the slot bound
does not rescue it (338 bits at `a = 13`, worse than the 218 it replaces). The
cost is still only 227 ms, so there is no practical harm, but the headline has
to name the family and not just the density.

Two details worth separating, since they are easy to conflate. The planner sizes
from a *bound*, not from `c_chi` itself: at `a = 13` the true `c_13 = 13!·(2^13−1)^13`
is 201.5 bits while the bound `i(V)^13` is 218.5, and it is the bound that buys
the eighth prime. And the worst *bound* (`a = 12`) and the worst *true value*
(`a = 16`) do not coincide.

This family is also exact ground truth: `c_k(K_a + E_b)` has a closed form for
every `k`, derived and checked against the solver in
`tests/chromatic/test_modular.py`.

Single run per configuration, median of a steady-state window, idle cooldown
between configurations.

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

[rs62]: https://projecteuclid.org/euclid.ijm/1255631807
