"""Pins for the platform behaviours the kernel works around, and the uint-only rule.

Each "workaround" test asserts that our path is correct, and warns (rather than
fails) if the underlying platform behaviour has been fixed, so the workaround
can be deleted when it stops being needed.
"""

import re
import warnings

import mlx.core as mx
import numpy as np
import pytest

import yates
from yates import kernel as K
from kernel_helpers import np_of


# --------------------------------------------------------------------------
# HARD RULE: exact-arithmetic paths use uint/ulong, never int/long
# --------------------------------------------------------------------------

#: Signed integers are permitted only in compile-time positions: template
#: parameter lists, constexpr constants, and unrolled loop counters.  They must
#: never carry transform data, because signed overflow is UB in MSL.
_ALLOWED_SIGNED_CONTEXTS = [
    re.compile(r"^\s*template\s*<"),                       # template <typename T, int ...>
    re.compile(r"^\s*#define\s"),                          # macro forwarding template params
    re.compile(r"^\s*(?:int|long)\s+\w+,?\s*(?:int|long)"),  # continued template param list
    re.compile(r"^\s*constexpr\s+int\s+\w+\s*="),          # compile-time constants
    re.compile(r"^\s*const\s+int\s+\w+\s*=\s*1\s*<<"),     # compile-time shift amount
    re.compile(r"^\s*for\s*\(int\s+\w+\s*="),              # unrolled loop counters
    re.compile(r"^\s*//"),                                 # comments
]

_SIGNED_TOKEN = re.compile(r"\b(?:signed\s+)?(?:int|long|short|char)\b")
_DATA_EXPR = re.compile(r"\b(?:inp|out|tile|x|addr)\s*\[")


def _metal_code_lines():
    src = K._sections()
    for section in ("header", "pass", "copy"):
        for lineno, line in enumerate(src[section].splitlines(), 1):
            code = line.split("//", 1)[0]
            yield section, lineno, line, code


def test_no_signed_integer_carries_transform_data():
    offenders = []
    for section, lineno, line, code in _metal_code_lines():
        if not _SIGNED_TOKEN.search(code):
            continue
        if any(p.search(line) for p in _ALLOWED_SIGNED_CONTEXTS):
            continue
        offenders.append(f"{section}:{lineno}: {line.strip()}")
    assert not offenders, (
        "signed integer types outside a compile-time context -- signed overflow "
        "is undefined behaviour in MSL and would break exactness:\n"
        + "\n".join(offenders)
    )


def test_data_expressions_never_appear_on_a_signed_declaration_line():
    offenders = [
        f"{s}:{ln}: {line.strip()}"
        for s, ln, line, code in _metal_code_lines()
        if _DATA_EXPR.search(code)
        and re.search(r"\b(?:signed\s+)?(?:int|long|short|char)\s+\w+\s*=", code)
        and not re.search(r"^\s*(?:const\s+)?int\s+\w+\s*=\s*1\s*<<", code)
    ]
    assert not offenders, "\n".join(offenders)


def test_metal_uses_only_unsigned_storage_types():
    """The only scalar storage types in the kernel body are T and unsigned ones."""
    body = "\n".join(
        line.split("//", 1)[0] for line in K._sections()["pass"].splitlines()
    )
    declared = set(re.findall(r"\b(?:const\s+|threadgroup\s+)*(\w+)\s+\w+\s*(?:=|\[)", body))
    allowed = {"T", "uint", "ulong", "ushort", "bool", "constexpr", "int",
               "threadgroup", "const", "for"}
    assert declared <= allowed, declared - allowed
    # ...and the ones that are signed are the compile-time ones only.
    assert "long" not in re.findall(r"\blong\b", body.replace("ulong", ""))


@pytest.mark.parametrize("mx_dt,np_dt,bits", [(mx.uint32, np.uint32, 32),
                                              (mx.uint64, np.uint64, 64)])
def test_top_of_ring_values_are_exact(mx_dt, np_dt, bits, rng):
    """Values above 2^(k-1) would be negative if the kernel used int/long.

    Every input here has its top bit set, so a signed path would hit UB.
    """
    n = 10
    top = np_dt(1) << np_dt(bits - 1)
    lo = rng.integers(0, 1 << 31, size=1 << n, dtype=np.uint64).astype(np_dt)
    x = (lo | top).astype(np_dt)
    for name in yates.VARIANTS:
        got = np_of(yates.transform(mx.array(x), name))
        assert np.array_equal(got, yates.numpy_transform(x, name)), name


# --------------------------------------------------------------------------
# Platform workaround 1: 64-bit simd_shuffle_xor
# --------------------------------------------------------------------------


def _raw_shuffle_is_correct_for_64bit() -> bool:
    src = ("uint t = thread_position_in_grid.x; T x = inp[t]; "
           "out[t] = metal::simd_shuffle_xor(x, (ushort)1);")
    k = mx.fast.metal_kernel(name="pin_sx64", input_names=["inp"],
                             output_names=["out"], source=src)
    a = np.array([0x1122334455667788 + i for i in range(64)], dtype=np.uint64)
    (o,) = k(inputs=[mx.array(a)], template=[("T", mx.uint64)], grid=(64, 1, 1),
             threadgroup=(64, 1, 1), output_shapes=[(64,)], output_dtypes=[mx.uint64])
    mx.eval(o)
    return bool(np.array_equal(np.array(o), a.reshape(-1, 2)[:, ::-1].ravel()))


def test_uint64_simd_tier_is_correct_despite_the_platform():
    """Our ulong shuffle (two 32-bit halves) is exact even though the raw one is not."""
    logw = (yates.limits().simd_width).bit_length() - 1
    ps = [K.Pass(s=0, p=logw, logc=0, logr=logw, logg=0)]   # pure simd tier
    assert ps[0].tiers(logw)["simd_shuffle"] == logw
    x = np.array([0x1122334455667788 + i * 0x0101010101010101
                  for i in range(1 << logw)], dtype=np.uint64)
    flat = mx.array(x)
    got = np_of(K._run_pass(flat, ps[0], K.encode_matrix("WHT"), logw, (0, 0, 0)))
    assert np.array_equal(got, yates.numpy_transform(x, "WHT"))

    if _raw_shuffle_is_correct_for_64bit():
        warnings.warn(
            "metal::simd_shuffle_xor now handles 64-bit operands correctly on this "
            "platform; the yates_shuffle<ulong> specialisation in kernel.metal can "
            "be removed.", stacklevel=1)


# --------------------------------------------------------------------------
# Platform workaround 2: negative template integers
# --------------------------------------------------------------------------


def test_matrix_entries_are_encoded_as_non_negative_template_integers():
    for name in yates.VARIANTS:
        codes = dict(K.encode_matrix(name))
        assert all(v >= 0 for v in codes.values()), codes
        for pos, entry in zip(("00", "01", "10", "11"), yates.get(name).entries):
            assert codes[f"S{pos}"] == (1 if entry < 0 else 0)
            assert codes[f"A{pos}"] == abs(entry)


def test_negative_template_integers_still_break_mlx_codegen():
    """Why encode_matrix exists: MLX pastes the value into a C++ identifier."""
    k = mx.fast.metal_kernel(name="pin_negtmpl", input_names=["inp"],
                             output_names=["out"],
                             source="out[thread_position_in_grid.x] = T(V);")
    try:
        (o,) = k(inputs=[mx.zeros((4,), dtype=mx.int32)], template=[("T", mx.int32), ("V", -1)],
                 grid=(4, 1, 1), threadgroup=(4, 1, 1),
                 output_shapes=[(4,)], output_dtypes=[mx.int32])
        mx.eval(o)
    except RuntimeError as exc:
        assert "Unable to build metal library" in str(exc)
        return
    warnings.warn(
        "MLX now accepts negative template integers; yates.kernel.encode_matrix "
        "could pass matrix entries directly.", stacklevel=1)


# --------------------------------------------------------------------------
# NOT A BUG: mx.contiguous copies an already-contiguous slice by design
# --------------------------------------------------------------------------

#: ``Contiguous::eval_gpu``/``eval_cpu`` let the output share the input buffer
#: only when the parent is at most this many bytes larger than the slice, so a
#: small view can release a large parent allocation (MLX PR #1270).  A
#: contiguous slice with more slack than this is copied on purpose.
CONTIGUOUS_SLACK_BYTES = 16384


def _contiguous_allocates(make_view):
    """True if mx.contiguous(view) allocated a new buffer rather than aliasing.

    Measured by active-memory delta rather than by timing: an alias costs
    nothing, a copy costs the view's full size.  ``nbytes`` cannot be used --
    it reports the logical view size either way.
    """
    n = (16 << 20) // 4                      # 16 MiB of uint32
    base = mx.zeros((n,), dtype=mx.uint32)
    mx.eval(base)
    view = make_view(base, n)
    mx.eval(view)
    before = mx.get_active_memory()
    out = mx.contiguous(view)
    mx.eval(out)
    return (mx.get_active_memory() - before) > view.nbytes // 2


@pytest.mark.parametrize("slack_elems, expect_copy", [(4096, False), (4097, True)])
def test_contiguous_copies_a_contiguous_slice_past_16_kib_of_slack(
        slack_elems, expect_copy):
    """The benchmark's materialisation step depends on this exact threshold.

    `bench/bench.py` materialises its inputs *outside* the timed region because
    `mx.contiguous()` on a prefix slice is a real device copy whenever the
    parent buffer has more than 16 KiB of slack.  That is intended MLX
    behaviour, not a defect: it lets a small slice release a large parent
    buffer.  But it means a view that looks free is not, so if the threshold
    ever moves, the benchmark would silently start charging a copy to the
    kernel under test and halve every reported bandwidth.
    """
    slack_bytes = slack_elems * 4
    assert (slack_bytes > CONTIGUOUS_SLACK_BYTES) == expect_copy, \
        "test parametrisation disagrees with the documented threshold"

    copied = _contiguous_allocates(lambda b, n: b[: n - slack_elems])
    if copied != expect_copy:
        warnings.warn(
            f"mx.contiguous slack threshold moved: a slice with {slack_bytes} B "
            f"of slack {'copied' if copied else 'aliased'}, expected "
            f"{'copy' if expect_copy else 'alias'}. Re-check "
            f"CONTIGUOUS_SLACK_BYTES and bench/bench.py's materialisation.",
            stacklevel=2,
        )


def test_contiguous_is_free_for_an_offset_slice_with_little_slack():
    """The rule is about buffer *size*, not about the offset being zero.

    `base[1:]` is offset by one element and still aliases, because the parent
    is only 4 bytes larger than the view.  This is why the right defensive
    check before a kernel is MLX's own row-contiguous flag — which
    `mx.fast.metal_kernel(ensure_row_contiguous=True)` applies internally — and
    not a blanket `mx.contiguous()` call.
    """
    assert not _contiguous_allocates(lambda b, n: b[1:])
