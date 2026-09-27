"""Runtime GPU capability discovery.

Nothing about the target GPU is hardcoded.  The SIMD width is read out of an
actual running Metal kernel (``threads_per_simdgroup``), and the threadgroup
limits come from the Metal device object.  Values are cached per process.
"""

from __future__ import annotations

import functools
import subprocess
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import mlx.core as mx


@dataclass(frozen=True)
class DeviceLimits:
    device_name: str
    architecture: str
    simd_width: int
    max_threads_per_threadgroup: int
    max_threadgroup_memory: int
    memory_size: int
    max_recommended_working_set_size: int
    max_buffer_length: int
    mlx_version: str
    source: str  # how the threadgroup limits were obtained

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def require_metal() -> None:
    """Abort early with a clear message if there is no Metal GPU."""
    if not mx.metal.is_available():
        raise RuntimeError(
            "Metal is not available in this MLX build; the Yates kernel needs "
            "an Apple Silicon GPU."
        )


_PROBE_SOURCE = r"""
  if (thread_position_in_grid.x == 0) {
    out[0] = (uint)threads_per_simdgroup;
    out[1] = (uint)threads_per_threadgroup.x;
    out[2] = (uint)simdgroups_per_threadgroup;
  }
"""


@functools.lru_cache(maxsize=1)
def query_simd_width() -> int:
    """Read ``threads_per_simdgroup`` out of a live kernel on this GPU."""
    require_metal()
    kernel = mx.fast.metal_kernel(
        name="yates_probe_simd",
        input_names=["inp"],
        output_names=["out"],
        source=_PROBE_SOURCE,
    )
    (out,) = kernel(
        inputs=[mx.zeros((4,), dtype=mx.uint32)],
        grid=(64, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(4,)],
        output_dtypes=[mx.uint32],
    )
    mx.eval(out)
    width = int(out[0].item())
    if width <= 0 or (width & (width - 1)) != 0:
        raise RuntimeError(f"implausible SIMD width reported by GPU: {width}")
    return width


def _metal_device_object():
    try:
        import Metal  # type: ignore
    except ImportError:
        return None
    return Metal.MTLCreateSystemDefaultDevice()


@functools.lru_cache(maxsize=1)
def _threadgroup_limits():
    """(max_threads_per_threadgroup, max_threadgroup_memory, source)."""
    dev = _metal_device_object()
    if dev is not None:
        return (
            int(dev.maxThreadsPerThreadgroup().width),
            int(dev.maxThreadgroupMemoryLength()),
            "MTLDevice",
        )
    # Fallback: bisect the largest threadgroup MLX will actually dispatch, and
    # use the Metal-guaranteed 32 KiB threadgroup memory floor for Apple GPUs.
    return _probe_max_threads(), 32768, "probed"


def _probe_max_threads() -> int:
    src = "out[thread_position_in_grid.x] = inp[thread_position_in_grid.x];"
    kernel = mx.fast.metal_kernel(
        name="yates_probe_tg", input_names=["inp"], output_names=["out"], source=src
    )
    best = 32
    t = 32
    while t <= 2048:
        try:
            (o,) = kernel(
                inputs=[mx.zeros((t,), dtype=mx.uint32)],
                grid=(t, 1, 1),
                threadgroup=(t, 1, 1),
                output_shapes=[(t,)],
                output_dtypes=[mx.uint32],
            )
            mx.eval(o)
            best = t
        except Exception:
            break
        t *= 2
    return best


@functools.lru_cache(maxsize=1)
def limits() -> DeviceLimits:
    """Everything the tiering policy is allowed to depend on."""
    require_metal()
    info = mx.device_info()
    max_threads, tg_mem, source = _threadgroup_limits()
    return DeviceLimits(
        device_name=str(info.get("device_name", "unknown")),
        architecture=str(info.get("architecture", "unknown")),
        simd_width=query_simd_width(),
        max_threads_per_threadgroup=max_threads,
        max_threadgroup_memory=tg_mem,
        memory_size=int(info.get("memory_size", 0)),
        max_recommended_working_set_size=int(
            info.get("max_recommended_working_set_size", 0)
        ),
        max_buffer_length=int(info.get("max_buffer_length", 0)),
        mlx_version=mx.__version__,
        source=source,
    )


def host_info() -> Dict[str, Any]:
    """Machine description for benchmark provenance, from system_profiler."""
    out: Dict[str, Any] = {}
    try:
        text = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True, text=True, timeout=60,
        ).stdout
        for line in text.splitlines():
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key, val = key.strip(), val.strip()
            if key in ("Model Name", "Model Identifier", "Chip",
                       "Total Number of Cores", "Memory"):
                out[key] = val
    except Exception as exc:  # pragma: no cover - provenance is best-effort
        out["error"] = str(exc)
    try:
        out["macOS"] = subprocess.run(
            ["sw_vers", "-productVersion"], capture_output=True, text=True, timeout=30
        ).stdout.strip()
    except Exception:
        pass
    return out


def memory_ceiling_bytes(fraction: float = 0.5) -> int:
    """Usable bytes for one live array, given unified memory and headroom.

    The transform is out-of-place, so two arrays of this size must coexist;
    ``fraction`` therefore defaults to half of the recommended working set.
    """
    lim = limits()
    budget = lim.max_recommended_working_set_size or lim.memory_size
    return int(budget * fraction)
