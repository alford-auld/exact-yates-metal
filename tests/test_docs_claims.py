"""Enforce the documentation standard the project states but did not check.

The README claims every number traces to a tracked artifact and a re-runnable
command.  That was true by care rather than by construction, and it failed
silently once: a "156-graph survey" with shares of 92.3 / 4.5 / 1.9 / 0.6 /
0.6% appeared in two published documents, was cited as the evidential basis for
the residual-risk argument, survived two reviews, and traced to nothing in the
repository.

Two checks here, from weakest to strongest:

1. Any percentage distribution published in a tracked Markdown table must carry
   integer counts in the same table, and those counts must imply the
   percentages.  Publishing counts beside shares is what makes a lost source
   obvious -- 92.3% of 156 being exactly 144.0 is the tell.
2. The survey figures quoted in prose must equal the corresponding keys in
   bench/results/chromatic_unforced_survey.json.  This is the specific claim
   that broke, pinned to its artifact.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ["README.md", "docs/report.md", "apps/chromatic/README.md"]

#: Rows whose label marks them as a baseline/model rather than a measurement.
#: "a random integer, 2^-(j+1)" is a theoretical column, not a count.
_MODEL_ROW = re.compile(r"random|baseline|expected|theoret|model|2\^-", re.I)
#: Rows whose label marks them as counts of things.
_COUNT_ROW = re.compile(r"^\s*\|?\s*(graphs|count|n|instances|samples)\b", re.I)

_PCT = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*%\s*$")


def _tables(text: str):
    """Yield Markdown tables as lists of cell-lists."""
    rows, table = [], []
    for line in text.splitlines():
        if line.lstrip().startswith("|") and line.rstrip().endswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):      # separator row
                continue
            table.append(cells)
        else:
            if len(table) >= 2:
                rows.append(table)
            table = []
    if len(table) >= 2:
        rows.append(table)
    return rows


def _as_pcts(cells):
    out = []
    for c in cells:
        m = _PCT.match(c)
        if not m:
            return None
        out.append(float(m.group(1)))
    return out


def _as_ints(cells):
    out = []
    for c in cells:
        if not re.fullmatch(r"\d+", c):
            return None
        out.append(int(c))
    return out


@pytest.mark.parametrize("doc", DOCS)
def test_published_percentage_distributions_carry_the_counts_behind_them(doc):
    """A share without its count is how a survey outlives its source."""
    text = (ROOT / doc).read_text()
    problems = []

    for table in _tables(text):
        pct_rows, count_rows = [], []
        for cells in table:
            label, values = cells[0], cells[1:]
            if not values or _MODEL_ROW.search(label):
                continue
            if (p := _as_pcts(values)) is not None:
                pct_rows.append((label, p))
            elif (i := _as_ints(values)) is not None and _COUNT_ROW.match(label):
                count_rows.append((label, i))

        for label, pcts in pct_rows:
            # a distribution sums to ~100%; a column of unrelated percentages
            # (e.g. "% of copy" per row) does not, and is out of scope here
            if not (95.0 <= sum(pcts) <= 105.0) or len(pcts) < 2:
                continue
            if not count_rows:
                problems.append(
                    f"{doc}: percentage distribution {label!r} = {pcts} has no "
                    f"integer count row in its table. Publish the counts: a "
                    f"share whose source is gone still looks correct.")
                continue
            for clabel, counts in count_rows:
                if len(counts) != len(pcts):
                    continue
                total = sum(counts)
                for got, c in zip(pcts, counts):
                    want = 100.0 * c / total
                    if abs(got - want) > 0.05 + 0.5 * 10 ** -_decimals(got):
                        problems.append(
                            f"{doc}: {label!r} reports {got}% where {clabel!r} "
                            f"gives {c}/{total} = {want:.4f}%")
    assert not problems, "\n".join(problems)


def _decimals(x: float) -> int:
    s = repr(x)
    return len(s.split(".")[1]) if "." in s else 0


def test_unforced_survey_prose_matches_its_json():
    """The specific claim that broke, pinned to the artifact that backs it."""
    data = json.loads(
        (ROOT / "bench/results/chromatic_unforced_survey.json").read_text())
    ev, cf = data["evidence"], data["closed_form"]
    rate = f"{100 * ev['odd_rate']:.1f}%"

    counts = " | ".join(str(ev["distribution"][k])
                        for k in sorted(ev["distribution"], key=int))
    for doc in ["docs/report.md", "apps/chromatic/README.md"]:
        # prose wraps, so compare against whitespace-normalised text
        text = " ".join((ROOT / doc).read_text().split())
        assert f"{ev['graphs']} graphs" in text, \
            f"{doc} does not state the evidence-stratum size {ev['graphs']}"
        assert rate in text, f"{doc} does not state the odd rate {rate}"
        assert f"{cf['graphs']} of {cf['graphs']}" in text, \
            f"{doc} does not state the closed-form cross-check " \
            f"({cf['graphs']} of {cf['graphs']})"
        assert counts in text, \
            f"{doc}: the published count row does not match the JSON " \
            f"distribution ({counts})"


def test_closed_form_stratum_is_excluded_from_the_evidence_claim():
    """The stratification is the claim; a merged survey would be circular."""
    data = json.loads(
        (ROOT / "bench/results/chromatic_unforced_survey.json").read_text())
    assert data["closed_form"]["all_cofactors_odd"], \
        "a closed-form family produced an even cofactor: the derivation is wrong"
    assert not data["closed_form"]["closed_form_mismatches"], \
        "a closed form disagreed with the solver"
    assert data["evidence"]["graphs"] >= 100, \
        "the evidence stratum has shrunk; the odd-rate claim rests on it"
    strata = {r["stratum"] for r in data["rows"]}
    assert strata == {"closed_form", "evidence"}, \
        f"unexpected strata {strata}: the two populations must stay named"
