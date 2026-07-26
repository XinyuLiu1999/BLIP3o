"""Unit tests for stratified per-concept sampling (rebalance/stratified.py).

Covers the keep-rate schedule, quota computation with segments/boosts, and the
ascending-frequency + running-dedup selection, plus parity between the scalar
reference and the vectorised Stage-4 path in build_stratified.py.

    python concept_rebalancing/tests/test_stratified.py
"""

import math
import os
import sys
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from rebalance.stratified import (  # noqa: E402
    StratifiedConfig,
    concept_keep_rate,
    concept_target,
    concept_targets,
    stratified_select,
)
from build_stratified import select_stratified  # noqa: E402

CFG = StratifiedConfig()


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_tail_fully_retained():
    for n in [1, 100, 50_000, CFG.n_head - 1]:
        assert concept_keep_rate(n, CFG) == 1.0, n
    print("ok: tail (< n_head) fully retained")


def test_head_downsampled_and_continuous():
    # a_head=2.5 with log10 => rate(1e5) == 1.0, continuous with the tail
    assert approx(concept_keep_rate(CFG.n_head, CFG), 1.0)
    r6, r7 = concept_keep_rate(1e6, CFG), concept_keep_rate(1e7, CFG)
    assert r7 < r6 < 1.0
    assert approx(r7, 5.0 / 7.0, tol=1e-3)
    print("ok: head downsampled via 2/log(Count), continuous at n_head")


def test_head_floor():
    cfg = StratifiedConfig(r_min=0.5)
    assert concept_keep_rate(10 ** 30, cfg) == 0.5
    print("ok: head floor r_min")


def test_rate_never_exceeds_one():
    # ln base makes 2*a_head/ln(n) > 1 well past n_head; must still clamp to 1.
    cfg = StratifiedConfig(log_base="ln")
    for n in [1e5, 1e6, 1e7]:
        assert concept_keep_rate(n, cfg) <= 1.0, n
    print("ok: keep rate clamped to <= 1")


def test_target_bounds():
    assert concept_target(0, "x", CFG) == 0
    assert concept_target(10, "x", CFG) == 10           # tail: all of it
    assert concept_target(10_000_000, "x", CFG) < 10_000_000
    # never below 1 for a non-empty concept
    cfg = StratifiedConfig(r_min=0.0)
    assert concept_target(10 ** 12, "x", cfg) >= 1
    print("ok: target bounds (<= N_c, >= 1)")


def test_boost():
    # Needs a concept whose base rate leaves headroom for +50%, else the boost
    # is clipped by the N_target <= N_c bound (rate 0.83 * 1.5 > 1 at N_c=1e6).
    cfg = StratifiedConfig(boosts={"weak": 0.5})
    n = 10 ** 12                       # rate ~ 0.417, so 1.5x still fits under 1.0
    base = concept_target(n, "other", cfg)
    boosted = concept_target(n, "weak", cfg)
    assert approx(boosted / base, 1.5, tol=0.01), (base, boosted)

    # and where the boost would exceed the concept's own size, it clamps to N_c
    assert concept_target(1_000_000, "weak", cfg) == 1_000_000
    print("ok: weak-capability boost (+50%, clamped at N_c)")


def test_segments():
    cfg = StratifiedConfig(segments={(1e5, 1e6): 0.5})
    assert concept_target(200_000, "x", cfg) < concept_target(200_000, "x", CFG)
    # outside the segment: unaffected
    assert concept_target(5_000_000, "x", cfg) == concept_target(5_000_000, "x", CFG)
    print("ok: segment-specific base rates")


def test_dedup_avoids_double_counting():
    # 'rare' (N=2) and 'common' (N=4) share both of rare's samples.
    # Ascending order: rare takes s1,s2; common's quota is then already met by
    # those two, so it draws fewer than it would standalone.
    concept_samples = {"rare": ["s1", "s2"], "common": ["s1", "s2", "s3", "s4"]}
    counts = {"rare": 2, "common": 4}
    cfg = StratifiedConfig(n_head=3.0, a_head=0.75, r_min=0.0, log_base="log10")
    # common: rate = 0.75*2/log10(4) = 2.49 -> clamped to 1.0 -> target 4
    sel = stratified_select(concept_samples, counts, cfg)
    assert sel == {"s1", "s2", "s3", "s4"}, sel

    # now force common's target to 2: dedup means it adds nothing new
    cfg2 = StratifiedConfig(n_head=3.0, segments={(3.0, 1e9): 0.5},
                            a_head=0.75, r_min=0.0)
    sel2 = stratified_select(concept_samples, counts, cfg2)
    assert sel2 == {"s1", "s2"}, sel2
    print("ok: running dedup avoids double-counting")


def test_ascending_order_matters():
    # The rare concept must claim its samples before the head is processed.
    concept_samples = {"rare": ["s1"], "big": ["s1", "s2", "s3", "s4", "s5", "s6"]}
    counts = {"rare": 1, "big": 6}
    cfg = StratifiedConfig(n_head=2.0, segments={(2.0, 1e9): 0.5}, a_head=1.0, r_min=0.0)
    sel = stratified_select(concept_samples, counts, cfg)
    assert "s1" in sel, sel          # rare's only sample always survives
    print("ok: ascending frequency preserves rare concepts")


def test_every_concept_survives():
    # No concept may be wiped out: each keeps >= 1 sample.
    rng = np.random.default_rng(3)
    nodes = [f"n{i}" for i in range(30)]
    concept_samples, counts = {}, {}
    for i, n in enumerate(nodes):
        k = int(rng.integers(1, 40))
        concept_samples[n] = [f"s{rng.integers(0, 200)}" for _ in range(k)]
        counts[n] = len(set(concept_samples[n]))
    sel = stratified_select(concept_samples, counts, CFG)
    for n in nodes:
        assert any(s in sel for s in concept_samples[n]), n
    print("ok: every concept retains >= 1 sample")


def test_vectorised_matches_reference():
    """The numpy Stage-4 path must equal the scalar reference exactly."""
    rng = np.random.default_rng(11)
    nodes = [f"n{i}" for i in range(60)]
    keys, nids = [], []
    concept_samples = {n: [] for n in nodes}
    for s in range(1500):
        k = f"s{s:05d}"
        for c in rng.choice(nodes, size=int(rng.integers(1, 10)), replace=False):
            keys.append(k); nids.append(c); concept_samples[c].append(k)
    counts = {n: len(set(v)) for n, v in concept_samples.items()}
    # priorities: the deterministic within-concept order both paths must use
    prio = {k: int(rng.integers(0, 2 ** 62)) for k in set(keys)}

    # force a real head/tail split on this small corpus
    cfg = StratifiedConfig(n_head=100.0, a_head=1.0, r_min=0.05)

    tmp = tempfile.mkdtemp()
    lp = os.path.join(tmp, "links.parquet")
    pq.write_table(pa.table({"sample_key": pa.array(keys),
                             "node_id": pa.array(nids)}), lp)

    mask, sample_keys = select_stratified(lp, counts, cfg,
                                          priority_by_key=prio, verbose=False)
    got = {k for k, m in zip(sample_keys.to_pylist(), mask) if m}
    want = stratified_select(concept_samples, counts, cfg, priority_of=prio)

    assert got == want, (
        f"vectorised != reference: only_vec={len(got-want)} only_ref={len(want-got)}")
    print(f"ok: vectorised == scalar reference ({len(got):,} samples)")


def test_determinism():
    rng = np.random.default_rng(5)
    nodes = [f"n{i}" for i in range(20)]
    concept_samples = {n: [f"s{rng.integers(0,100)}" for _ in range(int(rng.integers(1,20)))]
                       for n in nodes}
    counts = {n: len(set(v)) for n, v in concept_samples.items()}
    prio = {f"s{i}": int(rng.integers(0, 2**62)) for i in range(100)}
    a = stratified_select(concept_samples, counts, CFG, priority_of=prio)
    b = stratified_select(concept_samples, counts, CFG, priority_of=prio)
    assert a == b
    print("ok: selection deterministic")


def test_config_validation():
    for bad in [StratifiedConfig(n_head=0), StratifiedConfig(r_min=1.5),
                StratifiedConfig(log_base="log2"), StratifiedConfig(boosts={"a": -1})]:
        try:
            bad.validate()
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")
    print("ok: config validation")


def test_vectorised_map():
    t = concept_targets({"a": 50, "b": 10_000_000}, CFG)
    assert t["a"] == 50 and t["b"] < 10_000_000
    print("ok: vectorised target map")


if __name__ == "__main__":
    test_tail_fully_retained()
    test_head_downsampled_and_continuous()
    test_head_floor()
    test_rate_never_exceeds_one()
    test_target_bounds()
    test_boost()
    test_segments()
    test_dedup_avoids_double_counting()
    test_ascending_order_matters()
    test_every_concept_survives()
    test_vectorised_matches_reference()
    test_determinism()
    test_config_validation()
    test_vectorised_map()
    print("\nAll stratified tests passed.")
