# Next applications

The kernel and [chromatic-number application](../apps/chromatic/README.md) are
implemented here. The companion
[exact-padic-metal](https://github.com/alford-auld/exact-padic-metal) project
uses the same subset-zeta primitive inside Mihara's digit dynamic program for
polynomial congruences at `p = 2`.

The following are **planned experiments**, not implemented features or measured
speedups. The [detailed application roadmap](https://github.com/alford-auld/exact-padic-metal/blob/main/docs/roadmap.md)
specifies the inputs, unknowns, outputs, data and checks for each.

| Application | What the digit search solves for | First independent check |
|---|---|---|
| Matrix square roots, involutions, idempotents | entries of a matrix satisfying polynomial constraints modulo `2^E` | exhaustive `2 × 2` cases and direct multiplication |
| Periodic orbits | state tuples forming cycles of a known polynomial map | enumerate the small finite state graph |
| Modular matrix completion | factor entries fitted to observed matrix entries | planted factors, held-out entries, exhaustive tiny cases |
| Local Diophantine obstructions | common modular zeros, or the first certified empty level | brute force and known obstructions modulo 8 |
| Valuation distributions | exact finite counts of polynomial zeros | closed forms, exhaustive counts and probability mass checks |

At one lifting level, each candidate has a Boolean vector of possible next
digits. A subset-zeta transform evaluates its polynomial constraints at all
`2^L` digit vectors. Above level 0, affine elimination over `F_2` can replace
that transform; explicit survivor enumeration may still dominate the cost.
The companion project's default structural path currently uses the CPU even
at level 0, so integrating a GPU first level is part of the plan.

Work relevant to this repository:

- Benchmark `ZETA_SUB` on actual coefficient-table batches from these problems,
  including table construction and transfers. The current headline bandwidth
  sweep measures `WHT`; it is not a benchmark of these applications.
- Compare a hybrid GPU/CPU solve with the existing CPU structural and dense
  paths, using identical inputs, caps and output guarantees.
- Keep transform time, effective traffic bandwidth, survivor enumeration and
  complete application time separate in reports. Store the source commit and
  run protocol alongside new measurement artifacts.
- Compare against a simple specialised solver for each tiny problem before
  proposing a larger demonstration. Exactness and an appealing algebraic
  reduction do not establish a performance advantage.

The first targets are matrix constraints and local obstructions because their
outputs are easy to check independently. Matrix completion and the optional
learning of polynomial maps add a separate question: does fitting the observed
data produce correct predictions on held-out data?
