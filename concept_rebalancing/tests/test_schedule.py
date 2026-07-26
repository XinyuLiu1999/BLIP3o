"""Unit tests for the per-concept schedule + rarest-wins (rebalance/schedule.py).

Verifies the piecewise formula, boundary continuity, the caps/floor, and that a
sample's multiplicity is the max over its concepts (rarest-wins), including the
no-concept -> 1.0 rule.

    python concept_rebalancing/tests/test_schedule.py
"""

import math
import os
import sys

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from rebalance.schedule import (
    NO_CONCEPT_MULTIPLICITY,
    ScheduleConfig,
    concept_multiplicity,
    concept_multiplicities,
    sample_multiplicity,
    sample_multiplicity_geomean,
    sample_weight_meaninv,
)

CFG = ScheduleConfig()  # plan defaults


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_mid_band_is_retain():
    for n in [CFG.n_low, 50_000, CFG.n_high - 1]:
        assert concept_multiplicity(n, CFG) == 1.0
    print("ok: mid band retains (m=1)")


def test_head_downsamples_and_continuous():
    # At exactly n_high, m == 1 (continuity with the mid band): a_head default
    # is chosen so 2*a_head/log10(n_high) == 1.
    assert approx(concept_multiplicity(CFG.n_high, CFG), 1.0)
    # Larger head -> smaller m (< 1).
    m_1e6 = concept_multiplicity(1_000_000, CFG)
    m_1e7 = concept_multiplicity(10_000_000, CFG)
    assert m_1e7 < m_1e6 < 1.0
    # plan-quoted value: m_c(1e7) ~= 0.71
    assert approx(m_1e7, 5.0 / 7.0, tol=1e-3)
    print("ok: head downsamples, continuous at n_high")


def test_head_floor():
    cfg = ScheduleConfig(r_min=0.5)
    # a gigantic head would push m below 0.5 without the floor
    assert concept_multiplicity(10 ** 30, cfg) == 0.5
    print("ok: head floor r_min")


def test_tail_oversamples_and_continuous():
    # At exactly n_low, tail formula gives (n_low/n_low)^gamma = 1 (continuity).
    assert approx(concept_multiplicity(CFG.n_low, CFG), 1.0)
    # plan example: kayak N=8000 -> min(4, (20000/8000)^0.5) = 1.5811
    m = concept_multiplicity(8000, CFG)
    assert approx(m, math.sqrt(20000 / 8000), tol=1e-4)
    print("ok: tail oversamples, continuous at n_low")


def test_tail_cap():
    # a singleton would blow up without the cap
    assert concept_multiplicity(1, CFG) == CFG.m_max
    print("ok: tail cap M_max")


def test_rarest_wins_max():
    # river 0.77, helmet 1.0, kayak 1.58 -> sample governed by kayak
    assert sample_multiplicity([0.77, 1.0, 1.58]) == 1.58
    assert sample_multiplicity([0.5, 0.9]) == 0.9
    print("ok: rarest-wins = max")


def test_no_concept_retained():
    assert sample_multiplicity([]) == NO_CONCEPT_MULTIPLICITY == 1.0
    print("ok: no-concept sample retained")


def test_geomean_no_midband_veto():
    # The failure the geomean exists to fix: under max, a single m=1 concept
    # cancels the head's 0.76. Under geomean the head still pulls the sample down.
    assert sample_multiplicity([0.76, 1.0, 1.0, 1.0]) == 1.0     # max: vetoed
    assert sample_multiplicity_geomean([0.76, 1.0, 1.0, 1.0]) < 1.0
    # mid-band concepts abstain (weight 0), so the result is the head value itself
    assert approx(sample_multiplicity_geomean([0.76, 1.0, 1.0, 1.0]), 0.76)
    print("ok: geomean has no mid-band veto")


def test_geomean_all_midband_is_one():
    assert sample_multiplicity_geomean([1.0, 1.0, 1.0]) == 1.0
    print("ok: geomean all-mid-band -> 1.0")


def test_geomean_rare_still_dominates():
    # rarest-wins spirit: a strong tail concept outweighs a mild head one.
    m = sample_multiplicity_geomean([0.9, 3.0])
    assert m > 1.0, m
    # ...but unlike max it is not the raw 3.0 — the head genuinely tempers it.
    assert m < 3.0, m
    print("ok: geomean keeps rare-dominates without ignoring the head")


def test_geomean_caps_and_edges():
    assert sample_multiplicity_geomean([]) == NO_CONCEPT_MULTIPLICITY
    assert sample_multiplicity_geomean([0.0, 2.0]) == 0.0        # drop is absorbing
    assert sample_multiplicity_geomean([100.0, 100.0], m_max=4.0) == 4.0
    print("ok: geomean caps + edge cases")


def test_geomean_symmetry():
    # equal-and-opposite log pulls cancel to 1.0
    assert approx(sample_multiplicity_geomean([0.5, 2.0]), 1.0)
    print("ok: geomean symmetric in log space")


def test_vectorised_map():
    counts = {"a": 8000, "b": 50_000, "c": 10_000_000}
    m = concept_multiplicities(counts, CFG)
    assert m["a"] > 1.0 and m["b"] == 1.0 and m["c"] < 1.0
    print("ok: vectorised map")


def test_config_validation():
    for bad in [
        ScheduleConfig(n_low=0),
        ScheduleConfig(n_low=200_000, n_high=100_000),
        ScheduleConfig(r_min=1.5),
        ScheduleConfig(m_max=0.5),
        ScheduleConfig(gamma=-1),
    ]:
        try:
            bad.validate()
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")
    print("ok: config validation")


def test_meaninv_no_concept_retained():
    assert sample_weight_meaninv([]) == NO_CONCEPT_MULTIPLICITY
    print("ok: meaninv no-concept -> retain")


def test_meaninv_rarer_sample_scores_higher():
    """The property max/geomean fail: a sample carrying rare concepts must
    outrank an all-common one even though both also carry a head concept."""
    common = sample_weight_meaninv([3_500_000, 2_000_000, 900_000], alpha=0.5)
    rare = sample_weight_meaninv([3_500_000, 2_000_000, 1_270], alpha=0.5)
    assert rare > common, (rare, common)
    print("ok: meaninv ranks the rare-bearing sample higher")


def test_meaninv_no_single_concept_veto():
    """Adding one mid-band concept must not collapse the weight the way `max`
    does — the whole profile still moves the score."""
    base = sample_weight_meaninv([1_000, 1_000, 1_000], alpha=0.5)
    plus_mid = sample_weight_meaninv([1_000, 1_000, 1_000, 50_000], alpha=0.5)
    assert plus_mid < base                       # it dilutes ...
    assert plus_mid > sample_weight_meaninv([50_000] * 4, alpha=0.5)   # ... but does not veto
    print("ok: meaninv has no single-concept veto")


def test_meaninv_bounded_by_extremes():
    """A mean lies between the per-concept inverse frequencies — the property
    that keeps one ultra-rare concept from blowing the weight up (unlike
    1/min(N_c), which produced 2.5M-fold duplication of a single sample)."""
    ns = [1, 10_000, 3_500_000]
    w = sample_weight_meaninv(ns, alpha=0.5)
    invs = [1.0 / (n ** 0.5) for n in ns]
    assert min(invs) < w < max(invs)
    print("ok: meaninv bounded strictly between per-concept extremes")


def test_meaninv_alpha_monotone():
    """Larger alpha must widen the gap between a rare-bearing and a common
    sample — this is the knob the calibration sweeps."""
    ratios = []
    for a in (0.25, 0.5, 1.0):
        rare = sample_weight_meaninv([3_500_000, 1_270], alpha=a)
        common = sample_weight_meaninv([3_500_000, 900_000], alpha=a)
        ratios.append(rare / common)
    assert ratios[0] < ratios[1] < ratios[2], ratios
    print("ok: meaninv separation increases with alpha")


if __name__ == "__main__":
    test_mid_band_is_retain()
    test_head_downsamples_and_continuous()
    test_head_floor()
    test_tail_oversamples_and_continuous()
    test_tail_cap()
    test_rarest_wins_max()
    test_no_concept_retained()
    test_geomean_no_midband_veto()
    test_geomean_all_midband_is_one()
    test_geomean_rare_still_dominates()
    test_geomean_caps_and_edges()
    test_geomean_symmetry()
    test_vectorised_map()
    test_config_validation()
    test_meaninv_no_concept_retained()
    test_meaninv_rarer_sample_scores_higher()
    test_meaninv_no_single_concept_veto()
    test_meaninv_bounded_by_extremes()
    test_meaninv_alpha_monotone()
    print("\nAll schedule tests passed.")
