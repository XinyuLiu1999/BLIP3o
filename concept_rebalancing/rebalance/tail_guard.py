"""Stage-2/3 guard: catch link noise that `meaninv` would amplify.

The 2026-07-24 Stage-1b audit concluded false-friend mislinks were inert to the
schedule, and under `max` they were: a mislink only mattered if it *decided* a
sample's multiplicity, which head mislinks did 0.0-0.5% of the time.

`meaninv` removes that protection. It weights by ``1/N_c**alpha`` over the raw
counts, so weight concentrates precisely where the audit's guard pointed — the
tail. A node with ``N_c=12`` contributes ~577x the weight of a head concept, so
twelve mislinked images get amplified toward the copy cap instead of being
averaged away. §E.2 of that audit says to re-run the guard whenever the schedule
changes; this module is that re-run, made permanent.

Two independent failure modes, both measured on the real corpus:

1. **Wrong sense in the tail** — a high-frequency polysemous surface form linked
   to a rare wrong-sense node. Confirmed by review: ``seal``(N=12) is entirely
   wax-sealing photos, ``bank``(N=335) is entirely financial institutions.
2. **Split surface forms** — the same word existing as both a head node and a
   near-empty node (``building`` at N=3,509,499 *and* ``building`` at N=10).
   Not a sense error at all, but the tail twin still collects ~592x weight.
   433 such pairs exist, covering 0.0268% of links.

Neither is expensive to exclude: an excluded node simply stops contributing to
its samples' weights (the samples themselves are retained via their other
concepts), and the whole affected set is well under 0.1% of the link table.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Set, Tuple

# Nodes confirmed mislinked by manual review of every sample behind them
# (scratchpad/tail_guard_review.py, 2026-07-25). Excluded from the weight
# computation; see the module docstring for the evidence.
CONFIRMED_MISLINKS: Dict[str, str] = {
    "n02076196": "seal: all 12 samples are wax sealing, not the animal",
    "n09213434": "bank: all sampled sites are financial institutions, not a riverbank",
    "n10435988": "pitcher: 251/254 also link 'pitcher (container)'; baseball sense misfires",
}

# Reviewed and kept — recorded so a future re-run does not re-litigate them.
REVIEWED_CORRECT: Dict[str, str] = {
    "n03982060": "pool: swimming/mineral pools, correct sense (not billiards)",
}

# Mixed senses: real instances exist alongside wrong ones. Left in by default —
# excluding them would drop genuine supervision — but flagged for eval curation.
REVIEWED_MIXED: Dict[str, str] = {
    "n12855042": "mint: true mint candy mixed with US Mint / collectible coins",
    "n03595523": "jersey: garment mixed with Jersey cattle and the island of Jersey",
}


def split_surface_forms(
    counts: Dict[str, int], names: Dict[str, str], n_low: float = 1000.0
) -> List[Tuple[str, str, int, int]]:
    """Tail nodes sharing a base surface form with a much more common node.

    Returns ``(tail_node_id, base_name, tail_N_c, head_N_c)``. The base name
    strips a parenthetical qualifier, so ``pitcher (container)`` and ``pitcher``
    share the base ``pitcher``.

    These are a *taxonomy* artifact rather than a linking error, but under an
    inverse-frequency weight the near-empty twin dominates: ``building`` at
    ``N_c=10`` outweighs ``building`` at ``N_c=3,509,499`` by ~592x.
    """
    by_base: Dict[str, List[str]] = {}
    for nid in counts:
        base = str(names.get(nid, nid)).split(" (")[0].strip().lower()
        by_base.setdefault(base, []).append(nid)

    out: List[Tuple[str, str, int, int]] = []
    for base, group in by_base.items():
        if len(group) < 2:
            continue
        biggest = max(counts.get(n, 0) for n in group)
        if biggest < n_low:
            continue        # the whole group is rare: a genuinely rare concept
        for nid in group:
            n_c = counts.get(nid, 0)
            if 0 < n_c < n_low:
                out.append((nid, base, n_c, biggest))
    return out


def excluded_nodes(
    counts: Dict[str, int],
    names: Dict[str, str],
    n_low: float = 1000.0,
    drop_split_forms: bool = True,
) -> Set[str]:
    """The set of node ids whose links should not contribute weight."""
    excl = set(CONFIRMED_MISLINKS)
    if drop_split_forms:
        excl.update(nid for nid, _, _, _ in split_surface_forms(counts, names, n_low))
    return excl


def audit(
    counts: Dict[str, int],
    names: Dict[str, str],
    n_low: float = 1000.0,
    suspect_words: Iterable[str] = (),
) -> dict:
    """Report what the guard would exclude, without applying it.

    Run this whenever the schedule or its constants change — the failure mode it
    catches is specific to how much weight the tail receives.
    """
    splits = split_surface_forms(counts, names, n_low)
    suspects = set(w.lower() for w in suspect_words)
    flagged = [
        (nid, str(names.get(nid, nid)), counts[nid])
        for nid, n_c in counts.items()
        if 0 < n_c < n_low and str(names.get(nid, nid)).split(" (")[0].lower() in suspects
    ]
    total_links = sum(counts.values()) or 1
    split_links = sum(n for _, _, n, _ in splits)
    return {
        "n_confirmed_mislinks": len(CONFIRMED_MISLINKS),
        "n_split_surface_forms": len(splits),
        "split_link_mass": split_links,
        "split_link_frac": split_links / total_links,
        "polysemous_tail_flagged": flagged,
        "reviewed_correct": list(REVIEWED_CORRECT),
        "reviewed_mixed": list(REVIEWED_MIXED),
    }
