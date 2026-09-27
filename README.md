# Yates butterfly kernel — MLX + Metal

One templated Metal kernel computes the n-fold Kronecker power `M^(x)n` of any
2x2 generator matrix over a commutative ring, by the Yates algorithm in
`n * 2^(n-1)` butterflies. The Walsh–Hadamard transform, the subset and superset
zeta and Möbius transforms, and the polar-code kernel are all the same kernel
with four different butterfly constants.

| name       | M                | semantics                                            | inverse  |
|------------|------------------|------------------------------------------------------|----------|
| `WHT`      | `[[1,1],[1,-1]]` | Fourier on `(Z/2)^n`; diagonalizes XOR-convolution     | itself/N |
| `ZETA_SUB` | `[[1,0],[1,1]]`  | `f(S) = sum_{T subset S} f(T)`; diagonalizes OR-conv   | `MOB_SUB`|
| `MOB_SUB`  | `[[1,0],[-1,1]]` | Möbius inversion over subsets                          | `ZETA_SUB`|
| `ZETA_SUP` | `[[1,1],[0,1]]`  | `f(S) = sum_{T superset S} f(T)`; diagonalizes AND-conv | `MOB_SUP`|
| `MOB_SUP`  | `[[1,-1],[0,1]]` | Möbius inversion over supersets                        | `ZETA_SUP`|

`ZETA_SUB` is the polar-code kernel `F`; `ZETA_SUP` is its transpose. Arbitrary
integer 2x2 matrices work too — pass one instead of a name.

Stage `j` of the algorithm, for every index `i0` whose bit `j` is zero, with
`i1 = i0 | (1 << j)`:

    new[i0] = m00 * x[i0] + m01 * x[i1]
    new[i1] = m10 * x[i0] + m11 * x[i1]

Stages act on distinct tensor factors, so they commute and may be grouped freely.
The implementation exploits this to fuse stages into threadgroup tiles.

## Setup

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
`2^n`, which the ring cannot do. A forward-then-inverse WHT therefore recovers
the input **only modulo `2^(k-n)`**:

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

Total device traffic is `2 * passes * N * sizeof(elem)`. This is better than the
`2 * (n - t) * 2^n * sizeof(elem)` figure in the task brief, which assumes only
the first tile pass fuses stages and the remaining `n - t` run one stage each;
fusing the later passes too turns `n - t + 1` passes into `1 + ceil((n - t)/t')`.
For `uint32` at `n = 24` that is 13 passes under the unfused model against the 3
actually dispatched — a 4.3x reduction in device traffic. (The brief states both
"fusing as many stages per pass as the tile size allows" and the unfused traffic
figure; the two are inconsistent, and this implementation follows the former.)

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
compositions. `mx.jvp` and `mx.vmap` do not — see *Platform findings* below.

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
- Thermal warnings recorded during the run: False

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

| configuration | n=28, 1 GiB | n=29, 2 GiB |
|---|--:|--:|
| 3 passes, 128 B runs (old policy) | 82.2% of copy | 56.9% of copy |
| 4 passes, 1 KiB runs | **96.9%** | 93.2% |
| 4 passes, 2 KiB runs | 92.7% | **95.5%** |

Paying an extra full pass over 2 GiB in exchange for longer contiguous runs is a
large net win. The shipped policy scales the required run length with the
stride; in the sweep above `uint32` n=29 reaches 99.3% of copy where the old
policy reached 59.9%.

### Full results


#### uint32

- Fraction of measured copy bandwidth: 91.3% (n=25) to 100.2% (n=20); median 95.9%
- Transform bandwidth 89.9-98.9 GB/s
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

#### uint64

- Fraction of measured copy bandwidth: 92.9% (n=26) to 100.6% (n=28); median 95.8%
- Transform bandwidth 91.6-99.3 GB/s
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

#### float32

- Fraction of measured copy bandwidth: 91.6% (n=25) to 100.0% (n=29); median 96.7%
- Transform bandwidth 90.6-99.5 GB/s
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

A fourth, relevant to benchmarking: `mx.contiguous()` on a prefix slice is a
real device copy in MLX 0.32.2, not a no-op. The benchmark materialises its
inputs outside the timed region; doing otherwise halves every reported bandwidth.

## Limitations

- Transforms with `n < 5` and a large batch dispatch fewer than one full
  simdgroup per tile when `G` cannot be raised, so they are correct but not
  bandwidth-optimal. Everything the benchmark covers (`n >= 10`) fills a
  1024-thread threadgroup.
- `normalize=True` is float-only by construction: `2^(-n/2)` is not a ring
  element of `Z/2^k`.
- Only integer generator matrices are supported. General ring constants would
  need a real multiply per butterfly; entries in `{-1, 0, 1}` compile to adds
  and negations.

## Applications

- **[`apps/chromatic/`](apps/chromatic/README.md)** — exact chromatic number by
  Björklund–Husfeldt–Koivisto inclusion–exclusion. One subset-zeta over the
  whole cube, then a pointwise power and a reduction per candidate `k`:
  `O*(2^n)` for every graph, no search. Reuses this kernel unmodified. Reads its
  own README first — the single-modulus result carries a **one-sided** guarantee,
  and the default mode upgrades it to unconditional via CRT.

## Out of scope

Number-theoretic transforms and anything needing modular multiplication;
additive (Cantor / Gao–Mateer) FFT over `GF(2^k)`; ranked subset convolution;
non-Metal backends. Applications other than the chromatic-number one above
(Shapley values, S-box search) are separate tasks.
