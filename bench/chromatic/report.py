#!/usr/bin/env python3
"""Render bench/results/chromatic.json as the tables in apps/chromatic/README.md.

    .venv/bin/python bench/chromatic/report.py bench/results/chromatic.json
"""

from __future__ import annotations

import argparse
import json


def provenance(d: dict) -> str:
    dev, host, st = d["device"], d["host"], d["settings"]
    return "\n".join([
        f"- Machine: {host.get('Model Name','?')} ({host.get('Model Identifier','?')}), "
        f"{host.get('Chip','?')}, {host.get('Memory','?')} unified memory, "
        f"macOS {host.get('macOS','?')}",
        f"- GPU: {dev['device_name']} ({dev['architecture']}), MLX {d['mlx_version']}",
        f"- Memory ceiling detected at runtime: n={d['memory_ceiling_n']} "
        f"({d['bytes_per_subset']} B per subset, i(S) as uint32 plus the uint64 "
        "power terms)",
        f"- Phases are measured as nested prefixes of the real computation "
        "(indicator; indicator+zeta; end-to-end) and the inner ones subtracted, so "
        "the three shares sum to the measured total by construction",
        f"- Protocol: {st['duration']}s steady-state window per phase (median of its "
        f"second half), {st['warmup']}s warmup, {st['cooldown']}s idle cooldown, "
        f"at most {st['max_samples']} iterations per window",
        f"- Instances: {st['family']} G(n, 0.5), seed {st['seed']}",
        f"- Run {d['started']} to {d.get('finished','?')}",
    ])


def scaling_table(d: dict) -> str:
    rows = ["| n | edges | chi | i(V) | memory | indicator | zeta | k-search | "
            "end-to-end | indicator / zeta / k-search share | k tested |",
            "|--:|------:|----:|-----:|-------:|----------:|-----:|---------:|"
            "-----------:|:--|--:|"]
    for r in d["sizes"]:
        flag = "" if r.get("phase_consistent", True) else " ⚠"
        rows.append(
            f"| {r['n']}{flag} | {r['num_edges']} | {r['chi']} | {r['i_of_V']} | "
            f"{r['bytes']/2**20:.0f} MiB | {r['indicator_direct_s']*1e3:.2f} ms | "
            f"{r['zeta_s']*1e3:.2f} ms | {r['k_search_s']*1e3:.1f} ms | "
            f"{r['total_s']*1e3:.1f} ms | "
            f"{r['indicator_fraction']*100:.0f}% / {r['zeta_fraction']*100:.0f}% / "
            f"{r['k_search_fraction']*100:.0f}% | {r['k_tested']} |")
    return "\n".join(rows)


def indicator_table(d: dict) -> str:
    rows = ["| n | GPU direct | GPU branchless | NumPy lowbit DP (CPU) |",
            "|--:|-----------:|---------------:|----------------------:|"]
    for r in d["sizes"]:
        dp = r.get("indicator_numpy_dp_s")
        rows.append(
            f"| {r['n']} | {r['indicator_direct_s']*1e3:.2f} ms | "
            f"{r['indicator_branchless_s']*1e3:.2f} ms | "
            f"{dp*1e3:.1f} ms |" if dp else
            f"| {r['n']} | {r['indicator_direct_s']*1e3:.2f} ms | "
            f"{r['indicator_branchless_s']*1e3:.2f} ms | - |")
    return "\n".join(rows)


def named_table(d: dict) -> str:
    if not d.get("named"):
        return ""
    rows = ["| instance | n | chi | bounds | k tested | indicator | zeta | "
            "k-search | end-to-end | zeta share |",
            "|---|--:|--:|:--|--:|--:|--:|--:|--:|--:|"]
    for r in d["named"]:
        rows.append(
            f"| {r['label']} | {r['n']} | {r['chi']} | "
            f"[{r['lower']}, {r['upper']}] | {r['k_tested']} | "
            f"{r['indicator_s']*1e3:.2f} ms | {r['zeta_s']*1e3:.2f} ms | "
            f"{r['k_search_s']*1e3:.1f} ms | {r['total_s']*1e3:.1f} ms | "
            f"{r['zeta_fraction']*100:.1f}% |")
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="bench/results/chromatic.json")
    args = ap.parse_args()
    with open(args.path) as fh:
        d = json.load(fh)
    print("### Measurement provenance\n")
    print(provenance(d))
    print(f"\nRegenerate with:\n\n    {d['command']}\n")
    print("\n### Scaling\n")
    print(scaling_table(d))
    print("\n### Indicator variants\n")
    print(indicator_table(d))
    if d.get("named"):
        print("\n### Instances where the clique bound does not reach chi\n")
        print(named_table(d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
