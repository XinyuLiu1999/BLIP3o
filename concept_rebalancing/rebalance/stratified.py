"""Stratified per-concept sampling with running deduplication (paper method).

An alternative to the per-sample multiplicity path (``schedule.py`` +
``multiplicity.py``). That path assigns every sample one scalar ``m`` reduced
over its concepts, which forces an unwinnable arbitration: with ~11 concepts
per sample, ``max`` lets the tail veto all downsampling (head realized 1.07 vs
intended 0.76) while a geometric mean dilutes the tail (N_c=1 concepts realized
0.33 of intent). Measured on blip3o_pretrain, neither moved Gini more than 0.007.

This module implements the reference method instead::

    tail categories (< n_head)  : fully retained
    head categories (>= n_head) : downsampled, N_target ~ Count * 2/log(Count)
    boost categories            : +20..50% quota (weak-capability targeting)

    "Sampling proceeds from lowest to highest frequency with running
     deduplication to avoid double-counting."

Ascending-frequency order with dedup is what makes the head come down without
any reduction rule. Rare concepts claim their samples first; by the time a head
concept is processed, most of its quota is already filled by samples selected
for rarer concepts, so it needs few — if any — additional draws. The head's
exposure falls out of the overlap rather than being arbitrated per sample.

Output is a *set* (membership count == 1), not a multiplicity: a sample is
either in the corpus or not, so the result is a pure subsample of the original.

Selection is deterministic given the same inputs: within a concept, candidate
samples are ordered by the sample's stored ``priority`` (the shared sha256 draw
from ``multiplicity.compute_priority``), so a rerun reproduces the membership
byte-for-byte without storing per-sample state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class StratifiedConfig:
    """Constants for the stratified schedule.

    Defaults follow the reference description: a 100K head threshold, the
    ``2/log(Count)`` inverse-logarithmic rate, and no boosts.
    """

    n_head: float = 100_000.0     # >= this is head (downsampled); below is fully retained
    a_head: float = 2.5           # head coefficient; with log10, a_head*2/log10(n_head)==1
    r_min: float = 0.1            # floor on the head keep-rate
    log_base: str = "log10"       # "log10" or "ln" — the reference does not specify
    # segment-specific base rates: {(lo, hi): multiplier} applied on top of the
    # inverse-log rate for concepts with lo <= N_c < hi ("segment-specific base
    # rates for different frequency ranges").
    segments: Dict = field(default_factory=dict)
    # {node_id: boost} with boost in [0.2, 0.5] => +20..50% quota.
    boosts: Dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.n_head <= 0:
            raise ValueError(f"n_head must be > 0, got {self.n_head}")
        if not (0.0 <= self.r_min <= 1.0):
            raise ValueError(f"r_min must be in [0,1], got {self.r_min}")
        if self.log_base not in ("log10", "ln"):
            raise ValueError(f"log_base must be 'log10' or 'ln', got {self.log_base!r}")
        for nid, b in self.boosts.items():
            if b < 0.0:
                raise ValueError(f"boost for {nid} must be >= 0, got {b}")


def _log(x: float, base: str) -> float:
    return math.log10(x) if base == "log10" else math.log(x)


def concept_keep_rate(n_c: float, cfg: StratifiedConfig) -> float:
    """Fraction of a concept's samples to keep, before boosts.

    Tail (``n_c < n_head``) is fully retained -> 1.0. Head follows the
    inverse-logarithmic schedule ``2/log(Count)``, scaled by ``a_head`` so the
    rate is continuous at ``n_head`` (with the log10 default), floored at
    ``r_min`` so no concept vanishes.
    """
    if n_c <= 0:
        return 1.0
    if n_c < cfg.n_head:
        return 1.0                      # tail: fully retained
    denom = _log(n_c, cfg.log_base)
    if denom <= 0:
        return 1.0
    return max(cfg.r_min, min(1.0, cfg.a_head * 2.0 / denom))


def concept_target(n_c: int, node_id: str, cfg: StratifiedConfig) -> int:
    """Absolute sample quota ``N_target`` for a concept.

    Applies the keep rate, then the segment multiplier, then any weak-capability
    boost. Never exceeds ``n_c`` (a concept cannot yield more distinct samples
    than it has) and never drops below 1 for a non-empty concept.
    """
    if n_c <= 0:
        return 0
    rate = concept_keep_rate(n_c, cfg)

    for (lo, hi), mult in cfg.segments.items():
        if lo <= n_c < hi:
            rate *= mult
            break

    boost = cfg.boosts.get(node_id, 0.0)
    rate *= (1.0 + boost)

    return max(1, min(n_c, int(round(rate * n_c))))


def concept_targets(
    counts: Dict[str, int], cfg: StratifiedConfig
) -> Dict[str, int]:
    """``concept_target`` over a ``{node_id: N_c}`` map."""
    return {nid: concept_target(n, nid, cfg) for nid, n in counts.items()}


def stratified_select(
    concept_samples: Dict[str, list],
    counts: Dict[str, int],
    cfg: StratifiedConfig,
    priority_of: Optional[Dict[str, int]] = None,
) -> set:
    """Reference implementation of ascending-frequency selection with dedup.

    ``concept_samples`` maps node_id -> list of its sample keys. Concepts are
    processed rarest-first; each takes its quota, counting samples already
    selected for a rarer concept (running dedup) so overlap is never
    double-counted. ``priority_of`` gives the deterministic within-concept order
    (falls back to sorting by key).

    This is the scalar reference the vectorised Stage-4 path is checked against;
    it is not meant to run over a 200M-link corpus.
    """
    selected: set = set()
    # rarest first — the ordering the dedup depends on
    order = sorted(concept_samples.keys(), key=lambda n: (counts.get(n, 0), n))

    for nid in order:
        samples = concept_samples[nid]
        target = concept_target(counts.get(nid, len(samples)), nid, cfg)

        already = sum(1 for s in samples if s in selected)
        need = target - already
        if need <= 0:
            continue    # quota already met by rarer concepts' picks

        if priority_of is not None:
            cands = sorted((s for s in samples if s not in selected),
                           key=lambda s: (priority_of.get(s, 0), s))
        else:
            cands = sorted(s for s in samples if s not in selected)
        selected.update(cands[:need])

    return selected
