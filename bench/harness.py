"""Steady-state timing primitives.

Passive-cooled hardware throttles under sustained GPU load, so every number
this module produces comes from a measurement window long enough to reach
steady state, and configurations are separated by an idle cooldown.  The
headline metric is the median of the *second half* of the window, never a
best-of-N peak -- the peak is reported alongside it only as a throttle
indicator.
"""

from __future__ import annotations

import gc
import json
import statistics
import subprocess
import time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional

import mlx.core as mx


@dataclass
class Timing:
    samples: int
    sustained_s: float      # median of the second half of the window
    first_half_s: float     # median of the first half
    peak_s: float           # fastest single iteration
    p90_s: float
    window_s: float

    @property
    def throttle_ratio(self) -> float:
        """sustained / peak.  1.0 means no observed slowdown under load."""
        return self.sustained_s / self.peak_s if self.peak_s else float("nan")

    @property
    def drift_ratio(self) -> float:
        """second-half / first-half median: >1 means it got slower while running."""
        return self.sustained_s / self.first_half_s if self.first_half_s else float("nan")

    def as_dict(self) -> Dict:
        d = asdict(self)
        d["throttle_ratio"] = self.throttle_ratio
        d["drift_ratio"] = self.drift_ratio
        return d


def time_sustained(
    fn: Callable[[], mx.array],
    duration_s: float = 3.0,
    warmup_s: float = 0.7,
    min_samples: int = 8,
) -> Timing:
    """Run ``fn`` back to back for ``duration_s`` and report steady-state timing."""
    deadline = time.perf_counter() + warmup_s
    while time.perf_counter() < deadline:
        mx.eval(fn())
    mx.synchronize()

    times: List[float] = []
    start = time.perf_counter()
    deadline = start + duration_s
    while time.perf_counter() < deadline or len(times) < min_samples:
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        mx.synchronize()
        times.append(time.perf_counter() - t0)
        del out
    window = time.perf_counter() - start

    half = max(1, len(times) // 2)
    return Timing(
        samples=len(times),
        sustained_s=statistics.median(times[half:]),
        first_half_s=statistics.median(times[:half]),
        peak_s=min(times),
        p90_s=sorted(times)[min(len(times) - 1, int(0.9 * len(times)))],
        window_s=window,
    )


def cooldown(seconds: float, label: str = "") -> None:
    """Idle the GPU so the next configuration starts from a comparable state."""
    if seconds <= 0:
        return
    mx.clear_cache()
    gc.collect()
    print(f"    cooling {seconds:.0f}s {label}", flush=True)
    time.sleep(seconds)


def thermal_snapshot() -> Dict[str, str]:
    """Whatever thermal state macOS will report without elevated privileges.

    ``pmset -g therm`` is the only unprivileged source; it reports nothing at
    all until the system records a thermal or performance warning, so an empty
    reading means "no throttling event recorded", not "no data".  The
    load-induced slowdown actually observed is reported separately as
    ``drift_ratio`` and ``throttle_ratio``.
    """
    out: Dict[str, str] = {}
    try:
        text = subprocess.run(["pmset", "-g", "therm"], capture_output=True,
                              text=True, timeout=30).stdout.strip()
        out["pmset_therm"] = text or "(empty)"
        out["thermal_warning_recorded"] = str(
            "No thermal warning level has been recorded" not in text
        )
    except Exception as exc:  # pragma: no cover
        out["error"] = str(exc)
    out["uptime"] = subprocess.run(["uptime"], capture_output=True,
                                   text=True).stdout.strip()
    return out


def write_json(path: str, payload: Dict) -> None:
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\nwrote {path}", flush=True)
