"""Unit tests for the Stage-2/3 tail guard (rebalance/tail_guard.py).

The guard exists because `meaninv` weights by 1/N_c**alpha, which amplifies
exactly the link noise that was inert under `max`. These tests pin the two
failure modes it catches and the invariants that keep it from over-reaching.

    python concept_rebalancing/tests/test_tail_guard.py
"""

import os
import sys

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from rebalance.tail_guard import (
    CONFIRMED_MISLINKS,
    REVIEWED_CORRECT,
    REVIEWED_MIXED,
    audit,
    excluded_nodes,
    split_surface_forms,
)


def test_split_form_detected():
    """The real case: `building` exists at N=10 and at N=3.5M."""
    counts = {"head": 3_509_499, "twin": 10}
    names = {"head": "building", "twin": "building"}
    got = split_surface_forms(counts, names)
    assert [r[0] for r in got] == ["twin"], got
    assert got[0][2] == 10 and got[0][3] == 3_509_499


def test_parenthetical_qualifier_shares_base():
    """`pitcher` and `pitcher (container)` must be recognised as one surface form."""
    counts = {"ball": 254, "jug": 40_000}
    names = {"ball": "pitcher", "jug": "pitcher (container)"}
    assert [r[0] for r in split_surface_forms(counts, names)] == ["ball"]


def test_head_twin_never_excluded():
    """Only the rare twin is dropped — never the concept itself."""
    counts = {"head": 3_000_000, "twin": 10}
    names = {"head": "building", "twin": "building"}
    excl = excluded_nodes(counts, names)
    assert "twin" in excl and "head" not in excl


def test_all_rare_group_is_left_alone():
    """Two rare nodes sharing a name are a genuinely rare concept, not a split
    head — excluding them would delete real long-tail supervision."""
    counts = {"x": 20, "y": 30}
    names = {"x": "garum", "y": "garum"}
    assert split_surface_forms(counts, names) == []
    assert not (excluded_nodes(counts, names) & {"x", "y"})


def test_unique_rare_node_is_kept():
    """A rare concept with no common namesake is exactly what we want to boost."""
    counts = {"rare": 5, "other": 900_000}
    names = {"rare": "chiavari chair", "other": "chair"}
    assert split_surface_forms(counts, names) == []


def test_confirmed_mislinks_always_excluded():
    excl = excluded_nodes({}, {}, drop_split_forms=False)
    assert excl == set(CONFIRMED_MISLINKS)
    assert "n02076196" in excl          # seal -> wax sealing


def test_reviewed_sets_are_disjoint():
    """A node cannot be both confirmed-wrong and reviewed-correct."""
    assert not (set(CONFIRMED_MISLINKS) & set(REVIEWED_CORRECT))
    assert not (set(CONFIRMED_MISLINKS) & set(REVIEWED_MIXED))
    assert not (set(REVIEWED_CORRECT) & set(REVIEWED_MIXED))


def test_mixed_and_correct_are_not_excluded():
    """`pool` is the right sense and `mint`/`jersey` carry real instances —
    dropping them would remove genuine supervision."""
    excl = excluded_nodes({}, {}, drop_split_forms=False)
    for nid in list(REVIEWED_CORRECT) + list(REVIEWED_MIXED):
        assert nid not in excl, nid


def test_audit_reports_without_mutating():
    counts = {"head": 3_000_000, "twin": 10, "seal_ish": 500}
    names = {"head": "building", "twin": "building", "seal_ish": "seal"}
    rep = audit(counts, names, suspect_words=["seal"])
    assert rep["n_split_surface_forms"] == 1
    assert rep["split_link_mass"] == 10
    assert rep["polysemous_tail_flagged"] == [("seal_ish", "seal", 500)]
    assert counts == {"head": 3_000_000, "twin": 10, "seal_ish": 500}


def test_n_low_boundary():
    """A node at exactly n_low is mid-band, not tail: not excluded."""
    counts = {"head": 3_000_000, "edge": 1000}
    names = {"head": "building", "edge": "building"}
    assert split_surface_forms(counts, names, n_low=1000.0) == []
    assert split_surface_forms(counts, names, n_low=1001.0)


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print(f"ok: {fn.__name__}")
    print("\nAll tail-guard tests passed.")
