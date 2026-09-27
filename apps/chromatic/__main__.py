#!/usr/bin/env python3
"""CLI: chi(G) for a DIMACS .col file or a named generator.

    python -m apps.chromatic --dimacs graph.col
    python -m apps.chromatic --generator mycielskian:5
    python -m apps.chromatic --generator kneser:7,2 --mode multimodular --primes 6

Every run prints the guarantee attached to its answer.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import chromatic as C
from . import graph as G

GENERATORS = {
    "complete": lambda *a: G.complete_graph(int(a[0])),
    "cycle": lambda *a: G.cycle(int(a[0])),
    "path": lambda *a: G.path(int(a[0])),
    "empty": lambda *a: G.empty_graph(int(a[0])),
    "bipartite": lambda *a: G.complete_bipartite(int(a[0]), int(a[1])),
    "mycielskian": lambda *a: G.mycielskian(int(a[0])),
    "grotzsch": lambda *a: G.grotzsch(),
    "petersen": lambda *a: G.petersen(),
    "chvatal": lambda *a: G.chvatal(),
    "kneser": lambda *a: G.kneser(int(a[0]), int(a[1])),
    "turan": lambda *a: G.turan(int(a[0]), int(a[1])),
    "random": lambda *a: G.random_graph(int(a[0]), float(a[1]) if len(a) > 1 else 0.5,
                                        int(a[2]) if len(a) > 2 else 0),
}


def build(spec: str):
    name, _, rest = spec.partition(":")
    if name not in GENERATORS:
        raise SystemExit(f"unknown generator {name!r}; known: {sorted(GENERATORS)}")
    args = [a for a in rest.split(",") if a]
    return GENERATORS[name](*args)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dimacs", help="path to a DIMACS .col file")
    src.add_argument("--generator", help="e.g. mycielskian:5, kneser:7,2, random:14,0.5,3")
    ap.add_argument("--mode", default="exact",
                    choices=["exact", "multimodular", "mod2_64"])
    ap.add_argument("--primes", type=int, default=4,
                    help="number of random primes when --mode multimodular")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cross-check", action="store_true",
                    help="also run the full Mobius path and assert agreement")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    g = G.read_dimacs(args.dimacs) if args.dimacs else build(args.generator)
    limit = C.max_feasible_n(with_mobius=args.cross_check)
    if g.n > limit:
        print(f"error: n={g.n} exceeds this machine's limit of n={limit}",
              file=sys.stderr)
        return 2

    r = C.chromatic_number(g, mode=args.mode, num_primes=args.primes,
                           seed=args.seed, cross_check_mobius=args.cross_check)
    if args.json:
        print(json.dumps({
            "graph": r.graph, "n": r.n, "edges": r.num_edges,
            "chromatic_number": r.chromatic_number,
            "lower_bound": r.lower_bound, "upper_bound": r.upper_bound,
            "num_independent_sets": r.num_independent_sets,
            "mode": r.mode, "certain": r.certain,
            "unproved_zero_tests": r.unproved_zero_tests,
            "guarantee": r.guarantee,
            "colouring": r.colouring, "clique": r.clique,
            "timings_s": r.timings,
        }, indent=2))
    else:
        print(r.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
