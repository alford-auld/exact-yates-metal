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
.venv/bin/python -m pytest     # 600+ tests
```

`requirements.txt` pins the exact versions every number below was measured with.
`pyobjc-framework-Metal` is optional: it exposes the threadgroup limits directly,
and without it `yates/device.py` probes them instead (`DeviceLimits.source`
records which happened). Metal availability is checked before anything runs.

```python
import mlx.core as mx, numpy as np, yates

x = mx.array(np.random.randint(0, 2**32, 1 << 20, dtype=np.uint32))
y = yates.zeta_sub(x)                       # subset-sum transform
assert (np.array(yates.mobius_sub(y)) == np.array(x)).all()   # bit-for-bit

h = yates.wht(mx.random.normal((8, 1 << 14)), normalize=True)  # batched isometry
g = mx.grad(lambda a: mx.sum(yates.yates_transform(a, "ZETA_SUB")))(...)
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
tile holds. `yates.describe_plan(n, total, dtype)` reports the whole decision,
and `bench/plan.py` prints it for a range of `n`.

### Tiering policy

`choose_tile_bytes` picks the *smallest* threadgroup tile that still achieves the
minimum achievable number of passes — larger tiles fuse more stages but cost
occupancy, so there is no reason to pay for more tile than the pass count needs.
The first pass uses `LOGC = 0` (contiguous tiles, naturally coalesced); later
passes use `C >= 32` contiguous columns so that strided row access still
coalesces, widening `C` further when a short final pass has tile memory to spare.

Total device traffic is `2 * passes * N * sizeof(elem)`. Note this is better than
the `2 * (n - t) * 2^n * sizeof(elem)` figure in the task brief, which assumes
only the first tile pass fuses stages and the rest run one stage per pass; fusing
the later passes too reduces `n - t + 1` passes to `1 + ceil((n - t) / t')`.
For `uint32`, `n = 24`: 13 passes under the unfused model, 3 as implemented.

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

<!-- BENCH -->

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
tile-size invariance; batching and axis selection; and rejection of
non-power-of-two lengths and unsupported dtypes.

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

## Out of scope

Number-theoretic transforms and anything needing modular multiplication;
additive (Cantor / Gao–Mateer) FFT over `GF(2^k)`; ranked subset convolution;
non-Metal backends. Applications built on the kernel (Shapley values, chromatic
number, S-box search) are a separate task.
