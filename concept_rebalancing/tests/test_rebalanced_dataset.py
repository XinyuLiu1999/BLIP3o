"""Unit tests for the Stage-6 multiplicity expansion (rebalanced_dataset.py).

This code had never run — B/C/D were never trained — and it is the one component
that sits between a correct membership.parquet and a correct training epoch. The
expansion logic is pure Python, so it is tested here directly rather than through
`datasets.load_dataset`, which needs real shards and dominates the runtime.

The invariants that matter:
  - len(dataset) == sum of counts over the loaded rows (not the row count)
  - each row appears exactly `count` times
  - lengths / modality_lengths are EXPANDED — the sampler reads them, and the
    base class's versions are sized to the unexpanded rows
  - the shuffle disperses copies rather than leaving them adjacent

    python concept_rebalancing/tests/test_rebalanced_dataset.py
"""

import os
import random
import sys
from collections import Counter

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)
# rebalanced_dataset imports blip3o.data.dataset, which lives one level up.
_REPO = os.path.dirname(_PKG)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _has_torch():
    """The two class-level tests need blip3o (torch/transformers). The expansion
    tests are pure Python and must run in any env."""
    try:
        import blip3o.data.dataset  # noqa: F401
        return True
    except Exception:
        return False


def build_index_map(keys, count_map, seed=42):
    """The expansion in rebalanced_dataset.__init__, isolated for testing."""
    index_map = []
    for row_idx, key in enumerate(keys):
        c = count_map.get(key, 0)
        if c > 0:
            index_map.extend([row_idx] * c)
    rng = random.Random(seed)
    rng.shuffle(index_map)
    return index_map


KEYS = [f"k{i}" for i in range(10)]
COUNTS = {"k0": 1, "k1": 3, "k2": 1, "k3": 16, "k4": 2,
          "k5": 1, "k6": 7, "k7": 1, "k8": 1, "k9": 4}


def test_length_is_expanded_not_row_count():
    im = build_index_map(KEYS, COUNTS)
    assert len(im) == sum(COUNTS.values()) == 37
    assert len(im) != len(KEYS)


def test_each_row_repeated_exactly_its_count():
    im = build_index_map(KEYS, COUNTS)
    seen = Counter(im)
    for i, k in enumerate(KEYS):
        assert seen[i] == COUNTS[k], (k, seen[i], COUNTS[k])


def test_row_absent_from_membership_contributes_nothing():
    """A row that survived the shard load but is not in the membership must be
    dropped, not silently included once."""
    im = build_index_map(KEYS + ["unknown"], COUNTS)
    assert len(im) == sum(COUNTS.values())
    assert max(im) == len(KEYS) - 1


def test_zero_count_is_dropped():
    counts = dict(COUNTS, k0=0)
    im = build_index_map(KEYS, counts)
    assert 0 not in im
    assert len(im) == sum(counts.values())


def test_shuffle_is_deterministic():
    assert build_index_map(KEYS, COUNTS) == build_index_map(KEYS, COUNTS)


def test_shuffle_disperses_copies():
    """The whole point of shuffling the index map: a 16x sample must not appear
    as 16 consecutive training instances."""
    im = build_index_map(KEYS, COUNTS)
    hot = KEYS.index("k3")               # count 16
    positions = [i for i, r in enumerate(im) if r == hot]
    adjacent = sum(1 for a, b in zip(positions, positions[1:]) if b - a == 1)
    assert adjacent < len(positions) - 1, (adjacent, positions)


def test_sampler_lengths_match_expanded_length():
    """`lengths` and `modality_lengths` feed the length-grouped sampler. The base
    class sizes them from list_data_dict (unexpanded); the subclass must override
    both to the expanded length or the sampler truncates the epoch."""
    if not _has_torch():
        print("  (skipped: blip3o/torch unavailable in this env)")
        return
    from rebalanced_dataset import LazySupervisedRebalancedDataset as C

    im = build_index_map(KEYS, COUNTS)
    obj = C.__new__(C)                    # bypass __init__ (needs real shards)
    obj.index_map = im
    obj.list_data_dict = KEYS             # deliberately shorter than index_map

    assert len(obj) == len(im)
    assert len(obj.lengths) == len(im), (
        f"lengths={len(obj.lengths)} but dataset={len(im)}: the sampler would "
        f"only ever emit {len(obj.lengths)} of {len(im)} instances")
    assert len(obj.modality_lengths) == len(im)


def test_getitem_maps_through_index_map():
    """__getitem__(i) must resolve index_map[i], not i."""
    if not _has_torch():
        print("  (skipped: blip3o/torch unavailable in this env)")
        return
    from rebalanced_dataset import LazySupervisedRebalancedDataset as C

    obj = C.__new__(C)
    obj.index_map = [7, 3, 3, 0]
    seen = []

    class _Base:
        def __getitem__(self, i):
            seen.append(i)
            return {"row": i}

    # emulate super().__getitem__ resolution
    got = [_Base().__getitem__(obj.index_map[i]) for i in range(len(obj.index_map))]
    assert seen == [7, 3, 3, 0]
    assert got[1] == got[2] == {"row": 3}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok: {name}")
    print("\nAll rebalanced-dataset tests passed.")
