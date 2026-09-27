"""Graphs as vertex bitmasks, DIMACS .col I/O, and the test-family generators.

A graph on ``n <= 32`` vertices is carried as ``n`` adjacency bitmasks, one per
vertex: ``adj[v]`` has bit ``u`` set iff ``uv`` is an edge.  That is exactly the
form the independence-indicator kernel wants, and it makes
"``S`` contains no edge" a single AND per vertex.

32 vertices is not a limitation in practice: the algorithm is O*(2^n) in both
time and memory, so the memory ceiling (see :mod:`apps.chromatic.chromatic`)
binds far below 32 on any real machine.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence, Tuple

import numpy as np

MAX_VERTICES = 32


@dataclass(frozen=True)
class Graph:
    """Simple undirected graph as ``n`` adjacency bitmasks."""

    n: int
    adj: Tuple[int, ...]
    name: str = ""

    def __post_init__(self) -> None:
        if self.n < 0:
            raise ValueError(f"vertex count must be non-negative, got {self.n}")
        if self.n > MAX_VERTICES:
            raise ValueError(
                f"{self.n} vertices exceeds the {MAX_VERTICES}-vertex bitmask limit; "
                "note the algorithm needs O(2^n) memory and so is infeasible far "
                "below that anyway"
            )
        if len(self.adj) != self.n:
            raise ValueError(f"expected {self.n} adjacency masks, got {len(self.adj)}")
        full = (1 << self.n) - 1
        for v, m in enumerate(self.adj):
            if m & ~full:
                raise ValueError(f"vertex {v} has a neighbour outside [0,{self.n})")
            if m >> v & 1:
                raise ValueError(f"vertex {v} has a self-loop")
        for v, m in enumerate(self.adj):
            for u in bits(m):
                if not (self.adj[u] >> v) & 1:
                    raise ValueError(f"adjacency is not symmetric at ({u},{v})")

    # -- basic properties ---------------------------------------------------

    @property
    def edges(self) -> List[Tuple[int, int]]:
        return [(u, v) for u in range(self.n) for v in bits(self.adj[u]) if u < v]

    @property
    def num_edges(self) -> int:
        return sum(bin(m).count("1") for m in self.adj) // 2

    def degree(self, v: int) -> int:
        return bin(self.adj[v]).count("1")

    @property
    def max_degree(self) -> int:
        return max((self.degree(v) for v in range(self.n)), default=0)

    def adj_array(self) -> np.ndarray:
        """Adjacency masks as ``uint32``, ready for the device."""
        return np.array(self.adj, dtype=np.uint32)

    def is_independent(self, s: int) -> bool:
        """True iff the vertex set with bitmask ``s`` spans no edge."""
        for v in bits(s):
            if self.adj[v] & s:
                return False
        return True

    def complement(self) -> "Graph":
        full = (1 << self.n) - 1
        return Graph(self.n,
                     tuple(full & ~self.adj[v] & ~(1 << v) for v in range(self.n)),
                     f"complement({self.name})")

    def relabel(self, perm: Sequence[int]) -> "Graph":
        """Relabel vertices: new vertex ``i`` is old vertex ``perm[i]``."""
        inv = [0] * self.n
        for i, p in enumerate(perm):
            inv[p] = i
        adj = []
        for i in range(self.n):
            m = 0
            for u in bits(self.adj[perm[i]]):
                m |= 1 << inv[u]
            adj.append(m)
        return Graph(self.n, tuple(adj), self.name)

    def __str__(self) -> str:
        return (f"Graph({self.name or 'unnamed'}, n={self.n}, "
                f"m={self.num_edges}, maxdeg={self.max_degree})")


def bits(mask: int) -> Iterator[int]:
    """Indices of the set bits of ``mask``, ascending."""
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def from_edges(n: int, edges: Iterable[Tuple[int, int]], name: str = "") -> Graph:
    adj = [0] * n
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge ({u},{v}) outside [0,{n})")
        if u == v:
            raise ValueError(f"self-loop at {u}; simple graphs only")
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return Graph(n, tuple(adj), name)


def from_matrix(matrix, name: str = "") -> Graph:
    """Build from a dense symmetric 0/1 adjacency matrix."""
    a = np.asarray(matrix)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"adjacency matrix must be square, got shape {a.shape}")
    if not np.array_equal(a, a.T):
        raise ValueError("adjacency matrix must be symmetric")
    if np.any(np.diag(a)):
        raise ValueError("adjacency matrix must have zero diagonal (no self-loops)")
    n = a.shape[0]
    return from_edges(n, ((u, v) for u in range(n) for v in range(u + 1, n) if a[u, v]),
                      name)


def to_matrix(g: Graph) -> np.ndarray:
    a = np.zeros((g.n, g.n), dtype=np.uint8)
    for u, v in g.edges:
        a[u, v] = a[v, u] = 1
    return a


# --------------------------------------------------------------------------
# DIMACS .col
# --------------------------------------------------------------------------


def parse_dimacs(text: str, name: str = "") -> Graph:
    """Read the DIMACS ``.col`` edge format (1-indexed ``e u v`` lines)."""
    n = None
    declared_edges = None
    edges: List[Tuple[int, int]] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line[0] in "c\0":
            continue
        parts = line.split()
        tag = parts[0]
        if tag == "p":
            # p edge <n> <m>   (some files say "col" or "edges" instead of "edge")
            if len(parts) < 4:
                raise ValueError(f"line {lineno}: malformed problem line {line!r}")
            n, declared_edges = int(parts[2]), int(parts[3])
        elif tag == "e":
            if n is None:
                raise ValueError(f"line {lineno}: edge before the 'p' problem line")
            if len(parts) < 3:
                raise ValueError(f"line {lineno}: malformed edge line {line!r}")
            u, v = int(parts[1]) - 1, int(parts[2]) - 1
            if u == v:
                raise ValueError(f"line {lineno}: self-loop at vertex {u + 1}")
            if not (0 <= u < n and 0 <= v < n):
                raise ValueError(
                    f"line {lineno}: edge ({u + 1},{v + 1}) outside 1..{n}")
            edges.append((u, v))
        elif tag in ("n", "d", "v", "x"):
            continue          # vertex weights / descriptors: not used here
        else:
            raise ValueError(f"line {lineno}: unrecognised DIMACS line {line!r}")
    if n is None:
        raise ValueError("no 'p' problem line found; is this a DIMACS .col file?")
    dedup = {(min(u, v), max(u, v)) for u, v in edges}
    g = from_edges(n, sorted(dedup), name)
    if declared_edges is not None and declared_edges != len(dedup):
        g = Graph(g.n, g.adj, (name + " ") .strip() +
                  f"[header declared {declared_edges} edges, {len(dedup)} distinct]")
    return g


def read_dimacs(path: str) -> Graph:
    import os
    with open(path) as fh:
        return parse_dimacs(fh.read(), name=os.path.basename(path))


def to_dimacs(g: Graph) -> str:
    out = [f"c {g.name}"] if g.name else []
    out.append(f"p edge {g.n} {g.num_edges}")
    out += [f"e {u + 1} {v + 1}" for u, v in g.edges]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# generators with published chromatic numbers
# --------------------------------------------------------------------------


def empty_graph(n: int) -> Graph:
    """No edges.  chi = 1 for n >= 1, chi = 0 for n = 0."""
    return Graph(n, tuple([0] * n), f"empty_{n}")


def complete_graph(n: int) -> Graph:
    """K_n.  chi = n."""
    full = (1 << n) - 1
    return Graph(n, tuple(full & ~(1 << v) for v in range(n)), f"K_{n}")


def complete_bipartite(a: int, b: int) -> Graph:
    """K_{a,b}.  chi = 2 when a,b >= 1."""
    left = (1 << a) - 1
    right = ((1 << (a + b)) - 1) ^ left
    return Graph(a + b, tuple([right] * a + [left] * b), f"K_{a},{b}")


def cycle(n: int) -> Graph:
    """C_n.  chi = 2 if n even, 3 if n odd (n >= 3)."""
    if n < 3:
        raise ValueError(f"a cycle needs at least 3 vertices, got {n}")
    return from_edges(n, [(i, (i + 1) % n) for i in range(n)], f"C_{n}")


def path(n: int) -> Graph:
    """P_n.  chi = 2 for n >= 2."""
    return from_edges(n, [(i, i + 1) for i in range(n - 1)], f"P_{n}")


def mycielski_step(g: Graph) -> Graph:
    """One Mycielski step: 2n+1 vertices, chi goes up by exactly one.

    Vertices ``0..n-1`` are the original, ``n..2n-1`` are the shadows ``u_i``,
    and ``2n`` is the apex ``w``.  ``u_i`` is joined to the *neighbours* of
    ``v_i`` (not to ``v_i``), and ``w`` to every ``u_i``.  Shadows form an
    independent set, so the construction preserves triangle-freeness.
    """
    n = g.n
    edges = list(g.edges)
    for i in range(n):
        for j in bits(g.adj[i]):
            edges.append((n + i, j))          # u_i ~ v_j for every edge v_i v_j
    edges += [(n + i, 2 * n) for i in range(n)]
    dedup = sorted({(min(u, v), max(u, v)) for u, v in edges})
    return from_edges(2 * n + 1, dedup, f"M({g.name or 'G'})")


def mycielskian(k: int) -> Graph:
    """``M_k`` with chi(M_k) = k: M_2 = K_2, M_3 = C_5, M_4 = Grotzsch, ...

    Sizes are 2, 5, 11, 23, 47, ... (``|M_{k+1}| = 2|M_k| + 1``).
    """
    if k < 2:
        raise ValueError(f"the Mycielski chain starts at k=2 (K_2), got {k}")
    g = Graph(2, (0b10, 0b01), "M_2")
    for j in range(3, k + 1):
        g = mycielski_step(g)
        g = Graph(g.n, g.adj, f"M_{j}")
    return g


def grotzsch() -> Graph:
    """The Grotzsch graph: 11 vertices, triangle-free, chi = 4.  Same as M_4."""
    g = mycielskian(4)
    return Graph(g.n, g.adj, "Grotzsch")


def petersen() -> Graph:
    """The Petersen graph: 10 vertices, chi = 3.  Built as the Kneser graph K(5,2)."""
    g = kneser(5, 2)
    return Graph(g.n, g.adj, "Petersen")


def petersen_standard() -> Graph:
    """Petersen from the textbook outer-cycle / inner-pentagram construction.

    Kept as an independent witness that :func:`petersen` really is Petersen.
    """
    edges = []
    for i in range(5):
        edges.append((i, (i + 1) % 5))            # outer C_5
        edges.append((5 + i, 5 + (i + 2) % 5))    # inner pentagram
        edges.append((i, 5 + i))                  # spokes
    return from_edges(10, edges, "Petersen(standard)")


def chvatal() -> Graph:
    """The Chvatal graph: 12 vertices, 4-regular, triangle-free, chi = 4."""
    edges = [
        (0, 1), (0, 4), (0, 6), (0, 9),
        (1, 2), (1, 5), (1, 7),
        (2, 3), (2, 6), (2, 8),
        (3, 4), (3, 7), (3, 9),
        (4, 5), (4, 8),
        (5, 10), (5, 11),
        (6, 10), (6, 11),
        (7, 8), (7, 11),
        (8, 10),
        (9, 10), (9, 11),
    ]
    return from_edges(12, edges, "Chvatal")


def kneser(n: int, k: int) -> Graph:
    """Kneser graph K(n,k): k-subsets of [n], adjacent iff disjoint.

    chi(K(n,k)) = n - 2k + 2 for n >= 2k (Lovasz).  For n < 2k the graph has no
    edges at all and chi = 1.
    """
    if k < 1 or n < k:
        raise ValueError(f"K({n},{k}) needs 1 <= k <= n")
    verts = [frozenset(c) for c in itertools.combinations(range(n), k)]
    edges = [(i, j) for i in range(len(verts)) for j in range(i + 1, len(verts))
             if not (verts[i] & verts[j])]
    return from_edges(len(verts), edges, f"Kneser({n},{k})")


def kneser_chromatic_number(n: int, k: int) -> int:
    """The Lovasz value, for use as published ground truth."""
    if n < 2 * k:
        return 1 if len(list(itertools.combinations(range(n), k))) else 0
    return n - 2 * k + 2


def random_graph(n: int, p: float = 0.5, seed: int = 0) -> Graph:
    """G(n, p) with an explicit seed."""
    rng = np.random.default_rng(seed)
    edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    return from_edges(n, edges, f"G({n},{p})#{seed}")


def turan(n: int, r: int) -> Graph:
    """Turan graph T(n,r): complete r-partite, balanced.  chi = min(r, n)."""
    part = [v % r for v in range(n)]
    edges = [(u, v) for u in range(n) for v in range(u + 1, n) if part[u] != part[v]]
    return from_edges(n, edges, f"Turan({n},{r})")


def disjoint_union(a: Graph, b: Graph) -> Graph:
    """chi(A + B) = max(chi(A), chi(B)); useful for disconnected test cases."""
    adj = list(a.adj) + [m << a.n for m in b.adj]
    return Graph(a.n + b.n, tuple(adj), f"{a.name}+{b.name}")
