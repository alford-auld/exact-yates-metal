#!/usr/bin/env python3
"""Distribution of the *unforced* 2-adic valuation of c_chi, stratified.

    .venv/bin/python bench/chromatic/unforced_survey.py \
        --out bench/results/chromatic_unforced_survey.json

``v2(c_chi)`` splits as forced + unforced.  The forced part is bounded exactly
(25 of 64 bits within the memory ceiling; see apps/chromatic/README.md).  The
unforced part -- the per-component unordered cofactors plus the components
below chi -- has no theorem bounding it, and the residual-risk argument rests
on it being odd far more often than a random integer would be.

**The survey is stratified, because two families have provably odd cofactors
and would otherwise inflate that claim:**

    disjoint K_a x m      c_a = (a!)^m            cofactor 1
    K_a + E_b             c_a = a! (2^a - 1)^b    cofactor (2^a - 1)^b, odd

Every graph from those families contributes v2 = 0 *by theorem*, not by
observation.  They are still worth measuring -- the script checks the closed
forms against the solver, which is a real cross-check -- but they are reported
separately and the evidence for the oddness bias is the OTHER stratum.

Naming the two populations explicitly is also what keeps the number stable:
"weighted toward the hard cases" is not a specification, and the next person to
extend the survey would otherwise move the headline without noticing.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from math import factorial

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import apps.chromatic as ch


def _v2(x: int) -> int:
    k = 0
    while x and x % 2 == 0:
        x //= 2
        k += 1
    return k


def closed_form_stratum():
    """Graphs whose cofactor is odd by a closed form, with that closed form.

    Yields ``(graph, predicted_c_chi, description)``.  The prediction is
    checked against the solver, so this stratum doubles as a correctness test
    of the two formulas the README derives.
    """
    for a in range(1, 8):
        for m in range(1, 5):
            if 2 <= a * m <= 10 and (a > 1 or m > 1):
                yield (ch.disjoint_copies(ch.complete_graph(a), m),
                       factorial(a) ** m, f"K_{a} x {m}")
    for a in range(1, 8):
        for b in range(1, 8):
            if 2 <= a + b <= 10:
                yield (ch.clique_plus_independent(a, b),
                       factorial(a) * (2 ** a - 1) ** b, f"K_{a} + E_{b}")


def evidence_stratum():
    """Everything with no closed-form cofactor.  This is the actual evidence."""
    gs = []
    for n in range(3, 10):
        gs.append(ch.cycle(n))
    for n in range(2, 10):
        gs.append(ch.path(n))
    for n, r in [(6, 2), (6, 3), (7, 3), (8, 3), (8, 4), (9, 3), (9, 4), (10, 4)]:
        gs.append(ch.turan(n, r))
    for a in range(1, 6):
        for b in range(a, 7):
            if a + b <= 10:
                gs.append(ch.complete_bipartite(a, b))
    gs += [ch.petersen(), ch.grotzsch(), ch.chvatal(),
           ch.mycielskian(3), ch.mycielskian(4)]
    # unions that mix a clique with something that is not a clique: the
    # cofactor is not a closed form even though one component is
    for a in range(2, 6):
        gs.append(ch.disjoint_union(ch.cycle(5), ch.complete_graph(a)))
        gs.append(ch.disjoint_union(ch.path(4), ch.complete_graph(a)))
        gs.append(ch.disjoint_union(ch.cycle(4), ch.cycle(a + 2)))
    for n in (5, 6, 7, 8, 9):
        for p in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            for seed in (900 + n, 1700 + n):
                gs.append(ch.random_graph(n, p, seed))
    return gs


def _measure(g):
    chi = ch.chromatic_number(g, mode="exact").chromatic_number
    c = ch.c_k_exact_reference(g, chi)
    forced = ch.forced_two_adic_valuation(g)
    unforced = _v2(c) - forced
    assert unforced >= 0, f"{g.name}: forced part is not a lower bound"
    return chi, c, forced, unforced


def _table(dist, total, label):
    ks = sorted(dist)
    return "\n".join([
        f"| `v2` of the unforced part ({label}) | " + " | ".join(str(j) for j in ks) + " |",
        "|---|" + "--:|" * len(ks),
        "| graphs | " + " | ".join(str(dist[j]) for j in ks) + " |",
        "| share | " + " | ".join(f"{100*dist[j]/total:.1f}%" for j in ks) + " |",
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    closed_dist, ev_dist = collections.Counter(), collections.Counter()
    rows, mismatches = [], []

    for g, predicted, desc in closed_form_stratum():
        chi, c, forced, unforced = _measure(g)
        if c != predicted:
            mismatches.append({"name": desc, "predicted": predicted, "actual": c})
        closed_dist[unforced] += 1
        rows.append({"stratum": "closed_form", "family": desc, "n": g.n, "chi": chi,
                     "v2_c_chi": _v2(c), "forced": forced, "unforced": unforced,
                     "closed_form_matches": c == predicted})

    for g in evidence_stratum():
        chi, c, forced, unforced = _measure(g)
        ev_dist[unforced] += 1
        rows.append({"stratum": "evidence", "family": g.name, "n": g.n, "chi": chi,
                     "v2_c_chi": _v2(c), "forced": forced, "unforced": unforced})

    n_closed, n_ev = sum(closed_dist.values()), sum(ev_dist.values())
    odd_rate = ev_dist[0] / n_ev
    out = {
        "schema": 2,
        "command": "python bench/chromatic/unforced_survey.py",
        "closed_form": {"graphs": n_closed,
                        "distribution": {str(j): closed_dist[j] for j in sorted(closed_dist)},
                        "all_cofactors_odd": set(closed_dist) == {0},
                        "closed_form_mismatches": mismatches},
        "evidence": {"graphs": n_ev,
                     "distribution": {str(j): ev_dist[j] for j in sorted(ev_dist)},
                     "odd_rate": odd_rate,
                     "max_unforced": max(ev_dist)},
        "rows": rows,
    }

    print(f"closed-form stratum: {n_closed} graphs, "
          f"all cofactors odd: {set(closed_dist) == {0}}, "
          f"formula mismatches: {len(mismatches)}")
    print(_table(closed_dist, n_closed, "closed form"))
    print()
    print(f"evidence stratum: {n_ev} graphs, odd {ev_dist[0]}/{n_ev} = "
          f"{100*odd_rate:.1f}% against a 50% baseline, "
          f"max unforced {max(ev_dist)}")
    print(_table(ev_dist, n_ev, "evidence"))
    if mismatches:
        print("\nCLOSED-FORM MISMATCHES:", json.dumps(mismatches, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
