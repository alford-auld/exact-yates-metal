# exact-yates-metal

**Exact subset-lattice transforms on Apple Silicon, at copy bandwidth.**

Subset-lattice algorithms — inclusion–exclusion, Möbius and Harsanyi
decompositions, Walsh spectra — spend nearly all of their time in one
primitive: the `2^n`-point zeta / Möbius / Hadamard transform over the Boolean
lattice. This is that primitive as a single templated Metal kernel: **exact over
`Z/2^32` and `Z/2^64`** — no floats, no tolerance, no headroom — and
**bandwidth-bound**. On a 16 GB MacBook Air it reaches `n = 29` (a 2 GiB array),
transforms it in **174 ms**, and sustains **~96% of the same machine's measured
device-to-device copy bandwidth** across `n = 10…29` for `uint32`, `uint64` and
`float32` alike.

**Where this shows up.** The subset zeta transform and its Möbius inverse are
the workhorse of exact exponential algorithms: set cover, chromatic number,
Steiner tree and the rest of the Björklund–Husfeldt–Koivisto family reduce to
`O*(2^n)` inclusion–exclusion over the subset lattice. The same transform is
the Harsanyi-dividend step behind Shapley values and interaction indices in
cooperative game theory and in feature attribution. The Walsh–Hadamard case is
the Fourier transform on `(Z/2)^n` — Boolean function analysis, S-box
nonlinearity, XOR-convolution. And `ZETA_SUB` is the polar-code kernel `F`.
One kernel, four butterfly constants.

**Three MLX / Metal platform bugs** were found and pinned in the process,
including `metal::simd_shuffle_xor` silently corrupting 64-bit operands on
`applegpu_g16g`. See [*Platform findings*](#platform-findings) — each is
pinned by a test that warns if a future release fixes it, so the workaround can
be deleted when it stops being needed.

## The generator matrices

One templated kernel computes the n-fold Kronecker power `M^(x)n` of any 2x2
generator matrix over a commutative ring, by the Yates algorithm in
`n * 2^(n-1)` butterflies. The named variants differ only in `M`:

| name       | M                | semantics                                                        | inverse    |
|------------|------------------|------------------------------------------------------------------|------------|
| `WHT`      | `[[1,1],[1,-1]]` | `(Hf)(S) = Σ_T (-1)^{\|S ∩ T\|} f(T)`; Fourier on `(Z/2)^n`, diagonalizes XOR-convolution | itself / N |
| `ZETA_SUB` | `[[1,0],[1,1]]`  | `(ζf)(S) = Σ_{T ⊆ S} f(T)`; diagonalizes OR-convolution           | `MOB_SUB`  |
| `MOB_SUB`  | `[[1,0],[-1,1]]` | `(μg)(S) = Σ_{T ⊆ S} (-1)^{\|S \ T\|} g(T)`; Möbius inversion over subsets | `ZETA_SUB` |
| `ZETA_SUP` | `[[1,1],[0,1]]`  | `(ζf)(S) = Σ_{T ⊇ S} f(T)`; diagonalizes AND-convolution          | `MOB_SUP`  |
| `MOB_SUP`  | `[[1,-1],[0,1]]` | `(μg)(S) = Σ_{T ⊇ S} (-1)^{\|T \ S\|} g(T)`; Möbius inversion over supersets | `ZETA_SUP` |

`ZETA_SUB` is the polar-code kernel `F`; `ZETA_SUP` is its transpose.

Any 2x2 **integer** matrix works too — pass one in place of a name. Entries in
`{-1, 0, 1}` compile to adds and negations; entries of larger magnitude emit a
real multiply per butterfly, which is correct and still exact, just not free.
Non-integer entries are not supported (the matrix is coerced with `int()`).
`bit_loss` and `guaranteed_bits` additionally require the matrix to be
primitive (gcd of entries 1) and non-singular, and raise otherwise.

Stage `j` of the algorithm, for every index `i0` whose bit `j` is zero, with
`i1 = i0 | (1 << j)`:

    new[i0] = m00 * x[i0] + m01 * x[i1]
    new[i1] = m10 * x[i0] + m11 * x[i1]

Stages act on distinct tensor factors, so they commute and may be grouped freely.
The implementation exploits this to fuse stages into threadgroup tiles.

## The exactness contract

This is the part to read before using the kernel for anything.

### Unsigned only, and why

Unsigned integer overflow in the Metal Shading Language is defined wraparound,
so `uint`/`ulong` arithmetic is *exact reduction mod 2^32 / 2^64*. Signed
overflow is undefined behaviour. **Every exact-arithmetic path in this kernel
uses `uint` or `ulong` and never `int` or `long`.** Signed integers appear in
`kernel.metal` only as template parameters, `constexpr` constants and unrolled
loop counters — never as a container for transform data. Two tests enforce this:
a source lint (`tests/test_platform_contracts.py::test_no_signed_integer_carries_transform_data`)
and a behavioural check that inputs with the top bit set — where a signed path
would be UB — still match arbitrary-precision Python arithmetic.

Supported element types are `uint32`, `uint64` (exact, mod 2^k) and `float32`
(for the `H/sqrt(N)` isometry). `int32`, `int64` and the narrow floats are
rejected with a `TypeError`.

### Zeta and Möbius are exact automorphisms

The four zeta/Möbius generators are unipotent: determinant 1, Smith normal form
`diag(1,1)`. Over `Z/2^32` and `Z/2^64` they are exact automorphisms, so

    MOB_SUB(ZETA_SUB(x)) == x        bit for bit, no tolerance, no headroom

and likewise for the superset pair and for either composition order.
`yates.guaranteed_bits(variant, n, bits) == bits` for all of them, at every `n`.

### The Hadamard matrix is not, and loses exactly n bits

`SNF(H_2) = diag(1, 2)`. The elementary divisors of `H_(2^n)` are therefore
`2^j` with multiplicity `binomial(n, j)`, and `det H_(2^n) = ±2^(n·2^(n-1))`.
Since `WHT ∘ WHT = 2^n · I`, inverting a WHT over `Z/2^k` means dividing by
`2^n`, which the ring cannot do, since 2 is not a unit in `Z/2^k`. A
forward-then-inverse WHT therefore recovers the input **only modulo
`2^(k-n)`**:

```python
yates.bit_loss("WHT", n)                 # == n
yates.guaranteed_bits("WHT", n, 32)      # == 32 - n
yates.guaranteed_bits("ZETA_SUB", n, 32) # == 32
```

`bit_loss` is computed from the Smith normal form of the generator, not
hardcoded per variant, so it is correct for a user-supplied matrix too.

The bound is tight in both directions, and
`tests/test_group2_hadamard_bits.py::test_bound_is_tight_exhibit_input_losing_exactly_n_bits`
exhibits a witness for each `n`: two inputs that differ only in bit `k-n` are
*indistinguishable* after `WHT ∘ WHT` (so at least `n` bits are lost), while
flipping bit `k-n-1` does change the result (so not one bit more than `n` is
lost). With 64-bit headroom the same composition is the exact integer identity
`WHT(WHT(x)) == N·x`.

`float32` has no exactness contract at all; `yates.ring_bits(mx.float32) == 0`.

## Quickstart

```sh
python3 -m venv .venv          # or: uv venv --python 3.12 .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest     # fast suite; add -m slow for the large-n tests
```

`requirements.txt` pins the exact versions every number below was measured with.
`pyobjc-framework-Metal` is optional: it exposes the threadgroup limits directly,
and without it `yates/device.py` probes them instead (`DeviceLimits.source`
records which happened). Metal availability is checked before anything runs.

```python
import mlx.core as mx, numpy as np, yates

# exact subset-sum transform over Z/2^32, and its exact inverse
x = mx.array(np.random.default_rng(0).integers(0, 2**32, 1 << 20, dtype=np.uint32))
y = yates.zeta_sub(x)
assert bool(mx.all(yates.mobius_sub(y) == x).item())          # bit for bit

# batched orthonormal Hadamard isometry, H/sqrt(N), along the last axis
h = yates.wht(mx.random.normal((8, 1 << 14)), normalize=True)

# differentiable: the VJP is the same kernel with the transposed generator
loss = lambda a: mx.sum(yates.yates_transform(a, "ZETA_SUB") ** 2)
g = mx.grad(loss)(mx.random.normal((1 << 10,)))

# what the dispatcher will do, without running it
yates.describe_plan(n=24, total_elems=1 << 24, dtype=mx.uint32)
```

## Platform findings

Three behaviours of MLX 0.32.2 / `applegpu_g16g` shaped the implementation. Each
has a test that pins it and *warns* if a future release fixes it.

1. **`metal::simd_shuffle_xor` silently corrupts 64-bit operands.** Shuffling a
   `ulong` returns garbage — it operates per 32-bit register. `kernel.metal`
   specialises `yates_shuffle<ulong>` to shuffle the two halves separately. This
   was the cause of a real `uint64` miscompare during development, not a
   theoretical concern.
2. **MLX pastes template integers into the generated Metal identifier.** A
   negative value emits `custom_kernel_..._-1_...`, which is not a valid C++
   name, and the library fails to build. Matrix entries are therefore carried as
   `(sign, magnitude)` pairs of non-negative template integers.
3. **`custom_function` `.jvp` and `.vmap` rules are bypassed for kernel bodies.**
   When the wrapped function contains a `CustomKernel` primitive, MLX consults
   the registered `.vjp` rule but descends into the kernel for the other two,
   raising `[Primitive::jvp] Not implemented for CustomKernel`. Registering
   those rules would be dead code, so `autodiff.py` omits them and offers
   `autodiff.jvp()` instead (a linear map is its own directional derivative).
   Batching needs no `vmap`: the transform already handles every leading axis in
   one launch.

A fourth thing worth knowing, **not a bug**: `mx.contiguous()` on an
already-contiguous prefix slice is often a real device copy. That is deliberate.
`Contiguous::eval_gpu`/`eval_cpu` let the output alias the input only when the
parent buffer is at most **16 KiB** larger than the view, so that a small slice
can release a large parent allocation ([MLX PR #1270][pr1270]). The rule is
about buffer *size*, not offset:

| view of a 256 MiB array | parent slack | result |
|---|--:|---|
| `base[:N-4096]` | 16 KiB | aliased, ~23 µs |
| `base[:N-4097]` | 16 KiB + 4 B | full copy, ~5.4 ms |
| `base[1:]` | 4 B | aliased |

The practical consequences: the benchmark materialises its inputs outside the
timed region, because charging that copy to the kernel under test halves every
reported bandwidth; and the right defensive check before a kernel is MLX's
row-contiguous flag — which `mx.fast.metal_kernel(ensure_row_contiguous=True)`
already applies — rather than a blanket `mx.contiguous()`. The threshold is
pinned by `tests/test_platform_contracts.py::test_contiguous_copies_a_contiguous_slice_past_16_kib_of_slack`,
which warns if it moves. The only fair criticism here is of the docstring,
which says "Copy if necessary" and never mentions the buffer-size rule.

[pr1270]: https://github.com/ml-explore/mlx/pull/1270

## Implementation

```
yates/
  variants.py     generator matrices, Smith normal form, the exactness contract
  reference.py    three independent oracles (MLX, NumPy, dense Kronecker power)
  device.py       runtime capability discovery -- nothing about the GPU is hardcoded
  kernel.metal    the templated MSL: helper header, transform pass, copy kernel
  kernel.py       mx.fast.metal_kernel wrapper, dispatch, tiering policy
  autodiff.py     mx.custom_function VJPs via the matrix transpose
```

### Nothing about the GPU is assumed

The SIMD width is read out of a live kernel (`threads_per_simdgroup`), not
assumed to be 32; the per-threadgroup memory limit and maximum threadgroup size
come from the Metal device; the memory ceiling for the largest test case is
derived from the detected recommended working set. All of these are reported in
test and benchmark output.

### Three tiers, in one kernel

A pass transforms index bits `[s, s+P)`. Its tile is `2^P` rows (stride `2^s`
apart) by `C = 2^LOGC` contiguous columns; `G` independent tiles share a
threadgroup so small transforms still fill one. With `R = 2^LOGR` rows resident
and `E = 2^P / R` elements per thread, the tier of a stage is decided purely by
how far apart its butterfly partners are:

| local bit range   | partner lives in            | cost |
|-------------------|-----------------------------|------|
| `[LOGR, P)`       | this thread's own registers | free, no traffic |
| `[0, NB_SIMD)`    | another lane of the simdgroup, via `simd_shuffle_xor(x, C << b)` | no memory traffic |
| `[NB_SIMD, LOGR)` | another thread, via threadgroup memory | two barriers per stage |

`NB_SIMD = clamp(log2(simd_width) - LOGC, 0, LOGR)` is derived from the
*measured* SIMD width. Each lane derives its role from `(rp >> b) & 1`. The
threadgroup buffer is sized `2^(P + LOGC + LOGG)` elements from `constexpr`
template arithmetic and elided entirely when no stage needs it.

Remaining stages become further device passes, each fusing as many stages as the
tile holds at that pass's required coalescing width. `yates.describe_plan(n, total, dtype)` reports the whole decision,
and `bench/plan.py` prints it for a range of `n`.

### Tiering policy

Two decisions, both measured rather than assumed (see *Why the coalescing policy
exists* below).

**Contiguous run length scales with stride.** The first pass uses `LOGC = 0`:
its tiles are contiguous and coalesce naturally. A later pass at offset `s` reads
rows `2^s` elements apart, so when that stride is large every row of a tile lands
on a different DRAM page. The required run length therefore grows with the
stride — roughly `stride_bytes / 8192`, clamped to `[128, 2048]` bytes — which
shrinks `P = log2(tile_elems) - LOGC` and can cost an extra device pass. On this
machine that trade is strongly worth it at large `n`.

**Tile size.** `choose_tile_bytes` then picks the smallest threadgroup tile that
achieves the minimum achievable pass count, since larger tiles fuse more stages
but cost occupancy.

### Device traffic, and why pass count is the whole game

The kernel moves `2 × passes × N × sizeof(elem)` bytes and runs at copy
bandwidth, so the pass count is the only free variable. Three ways to organise
the same `n` stages, with `uint32` at `n = 24`:

| strategy | passes |
|---|--:|
| one device pass per stage | `n` = 24 |
| fuse `t` stages into one tile pass, then one stage per pass for the rest | `1 + (n - t)` = 13 |
| fuse **every** pass, each taking as many stages as its own tile holds | **3** |

At `n = 24` the planner picks a 16 KiB tile (4096 `uint32`), which at `LOGC = 0`
fuses `t = 12` stages in the first pass. The naive follow-up — one stage per
pass for the remaining 12 — costs 13 passes. Instead the remaining 12 stages are
cleared by two more tile passes. Those fuse fewer stages each, because a
high-stride pass needs `LOGC > 0` for coalescing and `P = log2(tile_elems) - LOGC`
shrinks accordingly — but two passes still beat twelve. 3 against 13 is a
**4.3× reduction in device traffic**.

### Autodiff: no backward kernel

`(M^(x)n)^T == (M^T)^(x)n`, so every VJP is the same kernel with a transposed
generator. `autodiff.py` compiles no Metal of its own; the entire backward pass
is this table:

| forward | backward |
|---------|----------|
| `WHT` | `WHT` (self-adjoint) |
| `ZETA_SUB` | `ZETA_SUP` |
| `ZETA_SUP` | `ZETA_SUB` |
| `MOB_SUB` | `MOB_SUP` |
| `MOB_SUP` | `MOB_SUB` |

Reverse mode (`mx.grad`, `mx.value_and_grad`, `mx.vjp`) works, including through
compositions. `mx.jvp` and `mx.vmap` do not — see *Platform findings* above.

## Measured performance

Every number below was measured on this machine with the versions pinned in
`requirements.txt`, and every one is reproducible with a command in `bench/`.
Nothing here is a vendor figure or an estimate.

- Machine: MacBook Air (Mac16,13), Apple M4, 10 (4 Performance and 6 Efficiency) CPU cores, 16 GB unified memory, macOS 26.3
- GPU: Apple M4 (applegpu_g16g), SIMD width 32 (measured in-kernel), max 1024 threads/threadgroup, 32768 B threadgroup memory (via MTLDevice)
- Recommended working set 11.84 GiB, max buffer 8.88 GiB
- MLX 0.32.2
- Protocol: 2.0s steady-state window per configuration (reported value is the median of its second half), 0.5s warmup, 3.0s idle cooldown between configurations, working set at least 67108864 elements
- Run started 2026-09-27T11:48:55-0300, finished 2026-09-27T11:57:46-0300
- macOS recorded no thermal warning at any point during the run.

The denominator is a pure device-to-device copy kernel (`yates/kernel.metal`,
section `copy`), re-measured at every working-set size in the same session.
That matters: achievable copy bandwidth on this machine is
94-100 GB/s at the 256 MiB-2 GiB footprints used below, but only
48-52 GB/s at 16 MiB
(`bench/bench.py --min-elems 4194304 --min-n 10 --max-n 13 --dtypes uint32`,
saved as `bench/results/bench_small_footprint.json`), so a single global
denominator would be meaningless.  Small `n` are batched up to a fixed working
set for the same reason: otherwise the measurement is launch overhead.

Regenerate everything with:

    .venv/bin/python bench/bench.py    --out bench/results/bench.json
    .venv/bin/python bench/thermal.py  --minutes 7 --out bench/results/thermal.json
    .venv/bin/python bench/report.py   bench/results/bench.json

### Summary

| element type | n | transform GB/s | fraction of measured copy bandwidth |
|---|---|---|---|
| `uint32` | 10-29 | 89.9-98.9 | 91.3%-100.2%, median 95.9% |
| `uint64` | 10-28 | 91.6-99.3 | 92.9%-100.6%, median 95.8% |
| `float32` | 10-29 | 90.6-99.5 | 91.6%-100.0%, median 96.7% |

The transform is bandwidth-bound across the whole range: it moves
`2 x passes x N x sizeof(elem)` bytes, and moves them at essentially the rate the
GPU can copy the same volume. The handful of fractions slightly above 100% are
inter-pass reuse in the system cache, not a violation of anything.

Per-`n` tables for all three element types — pass count, tile size, tier split,
copy and transform bandwidth, drift — are in
[`docs/report.md`, Appendix A](docs/report.md#appendix-a-full-per-n-results),
and the underlying measurements in
[`bench/results/bench.json`](bench/results/bench.json). Regenerate the tables
with `bench/report.py bench/results/bench.json`.

### Thermal behaviour under continuous load

`bench/bench.py` idles between configurations, so its numbers are steady state
*within* a configuration but say nothing about minutes of unbroken load.
`bench/thermal.py` measures that regime: one configuration, back to back, no
cooldown at all.

7 minutes of continuous `WHT` over 2^24-element rows (1536 MiB of device traffic per iteration, 23834 iterations total):

| elapsed | median ms | GB/s | vs. first bucket |
|--:|--:|--:|--:|
| 15 s | 17.44 | 92.4 | 1.000x |
| 30 s | 17.51 | 92.0 | 1.004x |
| 45 s | 17.54 | 91.8 | 1.006x |
| 60 s | 17.55 | 91.8 | 1.006x |
| 120 s | 17.57 | 91.7 | 1.007x |
| 180 s | 17.66 | 91.2 | 1.013x |
| 240 s | 18.04 | 89.3 | 1.034x |
| 300 s | 18.36 | 87.7 | 1.053x |
| 360 s | 18.36 | 87.7 | 1.053x |
| 420 s | 18.63 | 86.5 | 1.068x |

Throughput is flat for roughly the first three minutes, then declines steadily,
ending 6.8% slower than the first bucket and still falling. That is
real throttling on a passively cooled MacBook Air, and it is why the sweep uses
cooldowns and reports the median of a measurement window rather than a
best-of-N peak. macOS recorded **no** thermal warning at any point during the
soak -- `pmset -g therm` reads identically before and after -- so the only way
to observe this is to measure throughput directly.

### Why the coalescing policy exists

The stride-scaled coalescing rule in `yates/kernel.py` is not a guess. Before
it, the planner simply minimised pass count, which left the high-stride passes
reading 128-byte runs from rows megabytes apart: a single threadgroup touching
128 distinct DRAM pages. Measured with `bench/tune_coalescing.py`:

All rows below are from one run of that script, so they are directly
comparable. The pre-fix policy is rebuilt explicitly by the script (as
`old greedy 3p/128B`) because `plan_passes()` no longer emits it:

| configuration | n=28, 1 GiB | n=29, 2 GiB |
|---|--:|--:|
| minimise passes, 128 B runs (pre-fix policy) | 61.0% of copy | 57.5% of copy |
| 4 passes, 512 B runs | 97.7% | 102.1% |
| 4 passes, 1 KiB runs | 95.8% | 102.4% |
| 4 passes, 2 KiB runs | 92.6% | **103.6%** |
| **shipped policy** (run length scaled to stride) | **102.1%** | 100.5% |

Paying an extra full pass over 2 GiB in exchange for longer contiguous runs is a
large net win — and the required run length grows with the stride, which is why
512 B is best at n=28 and 2 KiB at n=29. The shipped policy tracks that
automatically and lands within a percent of the best hand-tuned plan at both
sizes.

One caveat on that table: the script's smallest-tile candidate at n=29
(`tile8k`, 6 passes) recorded 161.8 s, ~600x more than its traffic can account
for. It keeps a reference copy of the input alive for the correctness check, so
at 2 GiB with six live pass buffers the machine is paging — a harness artifact,
not a property of the kernel. It is left in the JSON rather than deleted, and
excluded from the table.

## Applications

- **[`apps/chromatic/`](apps/chromatic/README.md)** — exact chromatic number by
  Björklund–Husfeldt–Koivisto inclusion–exclusion. One subset-zeta over the
  whole cube, then a pointwise power and a reduction per candidate `k`:
  `O*(2^n)` for every graph, no search. Reuses this kernel unmodified. Reads its
  own README first — the single-modulus result carries a **one-sided** guarantee,
  and the default mode upgrades it to unconditional via CRT.

## Tests

Six groups, all passing on `uint32`, `uint64` and `float32`:

1. **Exact round trip** — `MOB_SUB(ZETA_SUB(x)) == x` bitwise for random
   full-range `uint32` and `uint64`, every `n <= 20`, both orders, plus the
   superset pair, ring extremes, and agreement with arbitrary-precision Python.
2. **Hadamard bit loss** — `WHT(WHT(x)) == N·x` over `Z` with 64-bit headroom;
   recovery mod `2^(k-n)` over `Z/2^32` and `Z/2^64`; a tightness witness losing
   exactly `n` bits; the Smith normal form and elementary-divisor multiplicities.
3. **Convolution theorems** — XOR by `WHT`, OR by `ZETA_SUB`, AND by `ZETA_SUP`,
   against naive `O(4^n)` references for `n <= 10`, exactly in both rings and to
   tolerance in float32; plus direct enumeration of the subset/superset sums.
4. **Parseval** — `sum |Hx|^2 == 2^n sum |x|^2`, exactly mod `2^32` and `2^64`
   and to tolerance in float32; the normalized transform is a norm-preserving
   isometry.
5. **Published ground truth** — the inner-product bent function on `n = 2m`
   variables has `|f_hat(a)| = 2^m` everywhere; the AES S-box has maximum Walsh
   coefficient 32 and nonlinearity 112. The S-box is derived from the GF(2^8)
   definition rather than pasted, and checked against published entries. A
   control test confirms the flatness assertion can fail.
6. **Transpose identity** — `<Tx, y> == <x, T^T y>` for every variant, exactly
   in both rings and to tolerance in float32, which validates the VJP at the
   same time.

Beyond the six groups: tier-isolation tests that exercise the register, SIMD and
threadgroup tiers one at a time; a forced-naive dispatch path
(`yates.transform(..., naive=True)`, one unfused device pass per stage with every
tier disabled) used as an independent on-GPU reference for the tiered path;
tile-size invariance; batching, axis selection and empty batches; and rejection
of non-power-of-two lengths and unsupported dtypes.

`tests/test_large_n.py` is marked `slow` and excluded from the default run. It
covers `n = 24..29`, where the wide-column high-stride templates first appear,
using checks that need no host-side oracle of the same size: the zeta/Möbius
round trip must be bitwise identity, and the tiered result must equal the
forced-naive one.

    .venv/bin/python -m pytest             # fast suite
    .venv/bin/python -m pytest -m slow     # multi-GiB large-n tests

## Limitations

- Transforms with `n < 5` and a large batch dispatch fewer than one full
  simdgroup per tile when `G` cannot be raised, so they are correct but not
  bandwidth-optimal. Everything the benchmark covers (`n >= 10`) fills a
  1024-thread threadgroup.
- `normalize=True` is float-only by construction: `2^(-n/2)` is not a ring
  element of `Z/2^k`.
- Generator entries must be integers. `{-1, 0, 1}` compile to adds and
  negations; larger magnitudes are supported but cost a real multiply per
  butterfly, so `WHT` and the zeta/Möbius family are the fast path.
- Metal only. There is no CPU or CUDA backend.

## Versioning and vendoring

Releases are tagged; `v0.1.0` is the first. If you are vendoring this, pin the
tag rather than a commit hash — the repository has already been renamed once,
and tags survive that:

```sh
git clone --depth 1 --branch v0.1.0 https://github.com/alford-auld/exact-yates-metal.git
```

**Stable within `0.x`**, and changed only with a version bump and a note:

- the public names in `yates/__init__.py` — `transform`, `yates_transform`,
  `wht`, `zeta_sub`, `mobius_sub`, `zeta_sup`, `mobius_sup`, `bit_loss`,
  `guaranteed_bits`, `ring_bits`, `describe_plan`, `VARIANTS` and the five
  variant names;
- the exactness contract above: which variants are exact over which rings, and
  the `bit_loss` values;
- the accepted dtypes, and rejection with `TypeError` for the rest.

**Not stable**, and expected to change between tags:

- the tiering policy — tile sizes, pass counts, tier assignment. These are
  performance decisions, re-tuned against measurement; the numerical result is
  unchanged bit for bit.
- the contents of `describe_plan()`'s dictionary beyond `passes`;
- `kernel.metal` internals and the private helpers in `yates/kernel.py`;
- every measured number in this README, which is re-measured per release.

`0.x` means the API is young. Nothing here is deprecated silently.

## Project report

[`docs/report.md`](docs/report.md) is the write-up of both deliverables: what
was built, what was measured, the exactness arguments, and the limitations that
qualify the numbers. The section worth reading even if you skip the rest is
[**Intuitions that did not survive measurement**](docs/report.md#intuitions-that-did-not-survive-measurement)
— the three places where the obvious guess was wrong, including the one where
optimising this kernel further would have bought almost nothing. Its figures
regenerate from the tracked benchmark output in `bench/results/`.

## Out of scope

Number-theoretic transforms and anything needing modular multiplication;
additive (Cantor / Gao–Mateer) FFT over `GF(2^k)`; ranked subset convolution;
non-Metal backends. Applications other than the chromatic-number one above
(Shapley values, S-box search) are separate tasks.

## License

MIT — see [`LICENSE`](LICENSE). MLX, NumPy, pytest and pyobjc are all
MIT/BSD-licensed, so there is nothing to inherit.
