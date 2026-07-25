"""The per-sample multiplicity abstraction (plan §4).

Every sample gets a real-valued target multiplicity ``m >= 0`` that is realized
to an integer copy ``count`` deterministically from the sample's own hash::

    h      = hash01(sample_key)          # uniform [0,1), = priority / 2**63
    base   = floor(m)
    frac   = m - base
    count  = base + (1 if h < frac else 0)

    m = 0      -> dropped
    0 < m < 1  -> probabilistic keep (downsample):  count in {0,1}, E[count]=m
    m = 1      -> retained
    m > 1      -> oversampled: base guaranteed copies + 1 probabilistic

Determinism comes entirely from ``sample_key``: ``compute_priority`` is the
*same* sha256 primitive used by ``scripts/data_pipeline/index_tars.py`` (so the
priority column in ``index.parquet`` is exactly ``hash01 * 2**63``), which is
what makes any materialized membership re-derivable from the schedule alone.

This module is intentionally free of heavy dependencies so it can be imported
by the trainer, the offline scorer, and the unit tests alike.
"""

from __future__ import annotations

import hashlib
import math

# The 63-bit modulus shared with index_tars.compute_priority. hash01 divides by
# this so a priority read straight out of index.parquet gives the identical draw.
PRIORITY_MOD = 2 ** 63


def compute_priority(sample_key: str) -> int:
    """Deterministic 63-bit priority for a sample key.

    Byte-identical to ``scripts/data_pipeline/index_tars.compute_priority`` — the
    resampling draw and the catalog priority are the *same* number, so a
    membership can be reproduced from the schedule without re-reading the tars.
    """
    return int(hashlib.sha256(sample_key.encode()).hexdigest(), 16) % PRIORITY_MOD


def hash01(sample_key: str) -> float:
    """Uniform draw in [0, 1) for a sample key (= priority / 2**63)."""
    return compute_priority(sample_key) / PRIORITY_MOD


def priority_to_hash01(priority: int) -> float:
    """Convert a stored ``index.parquet`` priority into its [0,1) draw."""
    return priority / PRIORITY_MOD


def realize_count(m: float, h: float) -> int:
    """Realize a real multiplicity ``m`` to an integer copy count using draw ``h``.

    ``base`` guaranteed copies plus one more with probability ``frac = m - base``.
    Returns 0 for ``m <= 0`` (dropped). ``h`` must be in [0, 1).
    """
    if m <= 0.0:
        return 0
    base = math.floor(m)
    frac = m - base
    return int(base) + (1 if h < frac else 0)


def realize_count_for_key(m: float, sample_key: str) -> int:
    """Convenience: realize ``m`` for a sample using its key-derived draw."""
    return realize_count(m, hash01(sample_key))
