"""Per-concept multiplicity schedule (Stage 3) + rarest-wins reduction (Stage 4).

Stage 3 maps each concept's distinct-sample count ``N_c`` to a target
multiplicity ``m_c`` via a piecewise, boundary-continuous schedule::

    head  (N_c >= N_high)        :  m_c = max(r_min, a_head * 2 / log10(N_c))   # < 1  downsample
    mid   (N_low <= N_c < N_high):  m_c = 1.0                                   # retain
    tail  (N_c < N_low)          :  m_c = min(M_max, (N_low / N_c) ** gamma)    # > 1  oversample

The head branch is the reference's ``N_sample proportional to 2/log(Count)``;
the tail branch is its symmetric oversampling extension. All constants are
tunable and are meant to be *calibrated* against the Stage-6a realized-vs-intended
audit (see ``audit.py``).

Stage 4 reduces a sample's several concepts to one target multiplicity by
**rarest-wins = max**: the rarest concept (largest ``m_c``) sets the bar, so a
sample depicting a rare subject is preserved/boosted even when it also carries
common ones. A sample with no content concept (all tags STOPTAGS-dropped) gets
``m = 1.0`` (retained).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable


# Multiplicity assigned to a sample that survived to the corpus but carries no
# content concept (every tag was dropped by tags.STOPTAGS). Retained, not boosted.
NO_CONCEPT_MULTIPLICITY = 1.0


@dataclass
class ScheduleConfig:
    """Tunable constants for the per-concept schedule.

    Defaults are the plan's starting values (§3, "calibrate later"):
    ``N_high`` from the reference; ``a_head`` chosen so ``m_c(N_high) == 1`` and
    the schedule is continuous at the mid boundary.
    """

    n_high: float = 100_000.0    # head threshold: N_c >= n_high -> downsample
    n_low: float = 20_000.0      # tail threshold: N_c <  n_low  -> oversample
    a_head: float = 2.5          # head coefficient; log10(n_high)/2 => m_c(n_high)=1
    r_min: float = 0.1           # floor so head concepts never vanish entirely
    gamma: float = 0.5           # tail exponent (sub-linear: singletons not boosted 100x)
    m_max: float = 4.0           # oversampling cap

    def validate(self) -> None:
        if not (0 < self.n_low <= self.n_high):
            raise ValueError(f"require 0 < n_low <= n_high, got {self.n_low}, {self.n_high}")
        if not (0.0 <= self.r_min <= 1.0):
            raise ValueError(f"r_min must be in [0,1], got {self.r_min}")
        if self.m_max < 1.0:
            raise ValueError(f"m_max must be >= 1, got {self.m_max}")
        if self.gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {self.gamma}")


def concept_multiplicity(n_c: float, cfg: ScheduleConfig) -> float:
    """Target multiplicity ``m_c`` for a concept seen in ``n_c`` distinct samples."""
    if n_c <= 0:
        # A concept with no samples cannot govern any sample; return retain.
        return 1.0

    if n_c >= cfg.n_high:
        # log10(n_c) >= log10(n_high) > 1 here (n_high defaults to 1e5), so the
        # division is well-defined; guard the degenerate n_high <= 10 config.
        denom = math.log10(n_c)
        if denom <= 0:
            return 1.0
        return max(cfg.r_min, cfg.a_head * 2.0 / denom)

    if n_c >= cfg.n_low:
        return 1.0

    # tail
    return min(cfg.m_max, (cfg.n_low / n_c) ** cfg.gamma)


def concept_multiplicities(
    counts: Dict[str, int], cfg: ScheduleConfig
) -> Dict[str, float]:
    """Vectorised ``concept_multiplicity`` over a ``{node_id: N_c}`` map."""
    return {nid: concept_multiplicity(n, cfg) for nid, n in counts.items()}


def sample_multiplicity(node_ms: Iterable[float]) -> float:
    """Rarest-wins reduction: the max ``m_c`` over a sample's concepts.

    An empty iterable (no content concept) returns ``NO_CONCEPT_MULTIPLICITY``.
    """
    best = None
    for m in node_ms:
        if best is None or m > best:
            best = m
    return NO_CONCEPT_MULTIPLICITY if best is None else best


def sample_multiplicity_geomean(
    node_ms: Iterable[float], m_max: float = 4.0, m_floor: float = 0.0
) -> float:
    """Rarity-weighted geometric mean reduction (the max's veto-free cousin).

    ``max`` gives a *hard veto*: one mid-band concept (``m_c == 1``) is enough to
    cancel every head concept's ``m_c < 1``, so on a corpus averaging ~11 concepts
    per sample the head is never actually downsampled (Stage-6a audit: head
    concepts intended 0.76 realized 1.07, 100% overridden, 84% of them by ``m=1``).

    The geometric mean lets every concept pull. Weighting by ``|log m_c|`` keeps
    the rarest-wins *spirit* — a concept far from 1.0 in either direction speaks
    loudest, while the mid band (``log 1 == 0``) abstains instead of vetoing::

        w_c   = |log m_c|
        log m = Sum(w_c * log m_c) / Sum(w_c)

    Degenerate cases fall back to a plain (unweighted) geometric mean so an
    all-mid-band sample returns exactly 1.0 rather than dividing by zero.

    ``m_max`` caps the result as the tail branch does; ``m_floor`` optionally
    floors it. Empty -> ``NO_CONCEPT_MULTIPLICITY``.
    """
    logs = []
    for m in node_ms:
        # m <= 0 means "drop"; it is absorbing under a product, so honour it.
        if m <= 0.0:
            return 0.0
        logs.append(math.log(m))
    if not logs:
        return NO_CONCEPT_MULTIPLICITY

    weights = [abs(lg) for lg in logs]
    total_w = sum(weights)
    if total_w <= 0.0:
        # every concept sits exactly at m=1 -> retain
        return 1.0
    log_m = sum(w * lg for w, lg in zip(weights, logs)) / total_w
    return max(m_floor, min(m_max, math.exp(log_m)))
