"""Unit tests for the core multiplicity math (rebalance/multiplicity.py).

Verifies the hash rule, its exactness at boundaries, determinism, agreement with
index_tars.compute_priority, and the Monte-Carlo unbiasedness E[count] = m.

    python concept_rebalancing/tests/test_multiplicity.py
"""

import hashlib
import os
import sys

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from rebalance.multiplicity import (
    PRIORITY_MOD,
    compute_priority,
    hash01,
    priority_to_hash01,
    realize_count,
    realize_count_for_key,
)


def test_priority_matches_index_tars():
    # Byte-identical to scripts/data_pipeline/index_tars.compute_priority.
    for key in ["abc", "cd3cd09d12de4c48a72a24a8de952a5e", ""]:
        expected = int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2 ** 63)
        assert compute_priority(key) == expected
    print("ok: priority matches index_tars")


def test_hash01_range_and_determinism():
    for key in ["a", "bb", "ccc", "kayak-sample-42"]:
        h = hash01(key)
        assert 0.0 <= h < 1.0
        assert hash01(key) == h  # deterministic
        assert priority_to_hash01(compute_priority(key)) == h
    print("ok: hash01 range + determinism")


def test_realize_boundaries():
    assert realize_count(0.0, 0.5) == 0          # drop
    assert realize_count(1.0, 0.0) == 1          # retain (frac 0 -> no extra)
    assert realize_count(1.0, 0.999999) == 1
    assert realize_count(2.0, 0.0) == 2          # integer m -> exactly m
    assert realize_count(2.0, 0.999999) == 2
    # downsample: count in {0,1}, threshold at frac
    assert realize_count(0.3, 0.29) == 1
    assert realize_count(0.3, 0.30) == 0
    assert realize_count(0.3, 0.31) == 0
    # oversample: base + probabilistic
    assert realize_count(1.58, 0.57) == 2
    assert realize_count(1.58, 0.58) == 1
    assert realize_count(1.58, 0.59) == 1
    print("ok: realize boundaries")


def test_realize_negative_and_zero():
    assert realize_count(-1.0, 0.5) == 0
    assert realize_count(0.0, 0.0) == 0
    print("ok: non-positive m -> 0")


def test_key_realization_deterministic():
    key = "cd3cd09d12de4c48a72a24a8de952a5e"
    c1 = realize_count_for_key(1.58, key)
    c2 = realize_count_for_key(1.58, key)
    assert c1 == c2
    # equals the explicit two-step computation
    assert c1 == realize_count(1.58, hash01(key))
    print("ok: key realization deterministic")


def test_expectation_unbiased():
    # Over many distinct keys, mean realized count ~= m for several m values.
    N = 40000
    for m in [0.25, 0.77, 1.0, 1.58, 3.4]:
        total = sum(realize_count(m, i / N) for i in range(N))  # uniform grid draws
        mean = total / N
        assert abs(mean - m) < 0.01, f"m={m} mean={mean}"
    print("ok: E[count] == m (uniform-grid)")


def test_priority_mod_value():
    assert PRIORITY_MOD == 2 ** 63
    print("ok: PRIORITY_MOD")


if __name__ == "__main__":
    test_priority_matches_index_tars()
    test_hash01_range_and_determinism()
    test_realize_boundaries()
    test_realize_negative_and_zero()
    test_key_realization_deterministic()
    test_expectation_unbiased()
    test_priority_mod_value()
    print("\nAll multiplicity tests passed.")
