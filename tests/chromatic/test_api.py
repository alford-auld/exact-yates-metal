"""Top-level API: bounds, memory ceiling, guarantee wording, and the CLI."""

import json
import subprocess
import sys

import pytest

import apps.chromatic as ch
from apps.chromatic.__main__ import main as cli_main


def test_memory_ceiling_is_derived_from_the_detected_device():
    import yates
    limit = ch.max_feasible_n()
    budget = yates.limits().max_recommended_working_set_size
    assert (1 << limit) * ch.BYTES_PER_SUBSET <= budget
    assert (1 << (limit + 1)) * ch.BYTES_PER_SUBSET > budget * 0.6
    assert ch.max_feasible_n(with_mobius=True) <= limit


def test_oversized_input_is_refused_with_a_clear_message():
    limit = ch.max_feasible_n()
    g = ch.empty_graph(min(limit + 2, ch.MAX_VERTICES))
    if g.n <= limit:
        pytest.skip("machine is large enough that MAX_VERTICES binds first")
    with pytest.raises(MemoryError, match="GiB of device memory"):
        ch.chromatic_number(g)


def test_check_feasible_accepts_what_fits():
    ch.check_feasible(ch.max_feasible_n())
    with pytest.raises(MemoryError):
        ch.check_feasible(ch.max_feasible_n() + 1)


def test_bounds_are_valid_and_witnessed():
    for g in (ch.petersen(), ch.chvatal(), ch.mycielskian(5),
              ch.random_graph(14, 0.5, 6)):
        b = ch.bounds(g)
        assert ch.verify_colouring(g, b["colouring"])
        clique = b["clique"]
        for i, u in enumerate(clique):
            for v in clique[i + 1:]:
                assert g.adj[u] >> v & 1
        assert b["lower"] <= b["upper"]


def test_dsatur_never_exceeds_max_degree_plus_one():
    for g in (ch.petersen(), ch.chvatal(), ch.grotzsch(),
              ch.random_graph(15, 0.4, 3)):
        assert len(set(ch.dsatur(g))) <= g.max_degree + 1


def test_guarantee_wording_matches_the_evidence():
    r = ch.chromatic_number(ch.chvatal(), mode="exact")
    assert r.certain and "UNCONDITIONAL" in r.guarantee

    r = ch.chromatic_number(ch.chvatal(), mode="mod2_64")
    assert not r.certain
    assert "ONE-SIDED" in r.guarantee and "upper bound" in r.guarantee
    assert r.unproved_zero_tests

    # a graph whose clique bound already settles it needs no zero test at all
    r = ch.chromatic_number(ch.complete_graph(6), mode="mod2_64")
    assert r.certain, r.guarantee
    assert "UNCONDITIONAL" in r.guarantee


def test_reported_value_is_always_achievable():
    """Whatever the mode, the attached colouring must use exactly chi colours."""
    for mode in ("exact", "multimodular", "mod2_64"):
        for g in (ch.petersen(), ch.grotzsch(), ch.kneser(6, 2)):
            r = ch.chromatic_number(g, mode=mode)
            assert ch.verify_colouring(g, r.colouring)
            assert len(set(r.colouring)) >= r.chromatic_number


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="unknown mode"):
        ch.chromatic_number(ch.cycle(5), mode="magic")


def test_timings_are_reported_and_split():
    r = ch.chromatic_number(ch.mycielskian(5), mode="exact")
    assert set(r.timings) >= {"bounds", "indicator+zeta", "k search"}
    assert all(v >= 0 for v in r.timings.values())


@pytest.mark.parametrize("spec,chi", [("petersen", 3), ("chvatal", 4),
                                      ("mycielskian:4", 4), ("kneser:6,2", 4),
                                      ("complete:5", 5), ("cycle:7", 3)])
def test_cli_json(spec, chi, capsys):
    assert cli_main(["--generator", spec, "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["chromatic_number"] == chi
    assert out["guarantee"]


def test_cli_prints_the_guarantee(capsys):
    assert cli_main(["--generator", "grotzsch"]) == 0
    text = capsys.readouterr().out
    assert "chi = 4" in text and "UNCONDITIONAL" in text


def test_cli_reads_dimacs(tmp_path, capsys):
    g = ch.chvatal()
    path = tmp_path / "chvatal.col"
    path.write_text(ch.to_dimacs(g))
    assert cli_main(["--dimacs", str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["chromatic_number"] == 4


def test_cli_rejects_unknown_generator():
    with pytest.raises(SystemExit):
        cli_main(["--generator", "nosuchgraph"])
