#!/usr/bin/env python3
"""Render bench/results/bench.json as the markdown tables used in README.md.

    .venv/bin/python bench/report.py bench/results/bench.json

Prints only what was measured.  If a field is absent from the JSON it does not
appear in the table.
"""

from __future__ import annotations

import argparse
import json
import sys


def provenance(d: dict) -> str:
    dev, host, st = d["device"], d["host"], d["settings"]
    return "\n".join([
        f"- Machine: {host.get('Model Name','?')} ({host.get('Model Identifier','?')}), "
        f"{host.get('Chip','?')}, {host.get('Total Number of Cores','?')} CPU cores, "
        f"{host.get('Memory','?')} unified memory, macOS {host.get('macOS','?')}",
        f"- GPU: {dev['device_name']} ({dev['architecture']}), SIMD width "
        f"{dev['simd_width']} (measured in-kernel), max {dev['max_threads_per_threadgroup']} "
        f"threads/threadgroup, {dev['max_threadgroup_memory']} B threadgroup memory "
        f"(via {dev['source']})",
        f"- Recommended working set {dev['max_recommended_working_set_size']/2**30:.2f} GiB, "
        f"max buffer {dev['max_buffer_length']/2**30:.2f} GiB",
        f"- MLX {d['mlx_version']}",
        f"- Protocol: {st['duration']}s steady-state window per configuration "
        f"(reported value is the median of its second half), {st['warmup']}s warmup, "
        f"{st['cooldown']}s idle cooldown between configurations, "
        f"working set at least {st['min_elems']} elements",
        f"- Run started {d['started']}, finished {d.get('finished','?')}",
        f"- Thermal warnings recorded during the run: "
        f"{d.get('thermal_after',{}).get('thermal_warning_recorded','?')}",
    ])


def table(entry: dict) -> str:
    rows = [
        "| n | batch | working set | passes | tile | tiers simd/tg/reg | copy GB/s | "
        "transform ms | transform GB/s | % of copy | drift |",
        "|--:|------:|------------:|-------:|-----:|:------------------|----------:|"
        "-------------:|---------------:|----------:|------:|",
    ]
    for r in entry["sizes"]:
        t = r["stages_by_tier"]
        rows.append(
            f"| {r['n']} | {r['batch']} | {r['array_bytes']/2**20:.0f} MiB | "
            f"{r['passes']} | {r['tile_bytes']} | "
            f"{t['simd_shuffle']}/{t['threadgroup']}/{t['register']} | "
            f"{r['copy']['gb_per_s']:.1f} | {r['timing']['sustained_s']*1e3:.2f} | "
            f"{r['gb_per_s']:.1f} | {r['fraction_of_copy']*100:.1f}% | "
            f"{r['timing']['drift_ratio']:.3f} |"
        )
    return "\n".join(rows)


def summary(entry: dict) -> str:
    fr = [r["fraction_of_copy"] for r in entry["sizes"]]
    gb = [r["gb_per_s"] for r in entry["sizes"]]
    lo = min(entry["sizes"], key=lambda r: r["fraction_of_copy"])
    hi = max(entry["sizes"], key=lambda r: r["fraction_of_copy"])
    out = [
        f"- Fraction of measured copy bandwidth: {min(fr)*100:.1f}% (n={lo['n']}) "
        f"to {max(fr)*100:.1f}% (n={hi['n']}); median {sorted(fr)[len(fr)//2]*100:.1f}%",
        f"- Transform bandwidth {min(gb):.1f}-{max(gb):.1f} GB/s",
    ]
    if "thermal_recheck" in entry:
        tr = entry["thermal_recheck"]
        out.append(
            f"- Thermal re-check at n={tr['n']}: {tr['first_s']*1e3:.2f} ms at the start "
            f"of the sweep, {tr['last_s']*1e3:.2f} ms at the end ({tr['slowdown']:.3f}x)"
        )
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="bench/results/bench.json")
    args = ap.parse_args()
    with open(args.path) as fh:
        d = json.load(fh)

    print("### Measurement provenance\n")
    print(provenance(d))
    print(f"\nRegenerate with:\n\n    {d['command']}\n")
    for name, entry in d["dtypes"].items():
        print(f"\n### {name}\n")
        print(summary(entry))
        print()
        print(table(entry))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
