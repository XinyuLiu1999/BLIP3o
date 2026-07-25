"""Stages 2-4 — per-concept counts, per-concept schedule, per-sample multiplicity.

Given ``links.parquet`` (``sample_key, node_id``) this produces the schedule the
materializer consumes:

    Stage 2  counts.parquet             (node_id, N_c)            distinct samples/concept
    Stage 3  node_multiplicity.parquet  (node_id, N_c, m_c)       piecewise schedule
    Stage 4  sample_multiplicity.parquet(sample_key, m)           rarest-wins = max m_c

Stage 4 needs the *full* sample universe to assign ``m = 1.0`` to samples with no
content concept (all tags STOPTAGS-dropped). Pass ``--index index.parquet`` (the
Stage-0 catalog) so those samples are represented; without it only samples that
carry >=1 concept appear.

No GPU / browser dependency — this is pure arithmetic over the link table, so it
runs anywhere and is fast to re-run while calibrating the schedule constants.

    python concept_rebalancing/build_schedule.py \
        --links   concept_rebalancing/runs/blip3o_pretrain/links.parquet \
        --index   concept_rebalancing/runs/blip3o_pretrain/index.parquet \
        --output_dir concept_rebalancing/runs/blip3o_pretrain \
        --n_high 100000 --n_low 20000 --gamma 0.5 --m_max 4.0
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from rebalance.schedule import (  # noqa: E402
    NO_CONCEPT_MULTIPLICITY,
    ScheduleConfig,
    concept_multiplicities,
)


def compute_counts(links_path):
    """Stage 2: distinct-sample count per node (N_c) from the link table.

    Reading the whole table is fine even at tens of millions of rows; the group
    is over two string columns. A link table has one row per (sample, concept),
    so a plain group-count already yields distinct samples (a sample never links
    the same node twice — Stage 1 unions nodes per sample)."""
    table = pq.read_table(links_path, columns=["sample_key", "node_id"])
    grouped = table.group_by("node_id").aggregate([("sample_key", "count")])
    node_ids = grouped.column("node_id").to_pylist()
    counts = grouped.column("sample_key_count").to_pylist()
    return dict(zip(node_ids, counts))


def per_sample_multiplicity(links_path, node_m, all_sample_keys=None,
                            reduction="max", m_max=4.0):
    """Stage 4: rarest-wins max over each sample's concepts.

    Arrow-native reduction: dictionary-encode both link columns to int32 (C++
    side, never materializing 218M Python strings) and take the per-sample max
    ``m`` via a group-by. Samples in ``all_sample_keys`` that never appear in the
    links get NO_CONCEPT_MULTIPLICITY. Memory-bounded to fit a cgroup limit that a
    full ``.to_pylist()`` of the link columns would blow past."""
    table = pq.read_table(links_path, columns=["sample_key", "node_id"])

    # cast->large_string before combine_chunks: a >2GB string column overflows
    # the 32-bit offsets that combine_chunks/dictionary_encode assume otherwise.
    samp_dict = pc.dictionary_encode(
        table.column("sample_key").cast(pa.large_string()).combine_chunks())
    node_dict = pc.dictionary_encode(
        table.column("node_id").cast(pa.large_string()).combine_chunks())
    del table

    samp_idx = samp_dict.indices                      # int32, per link
    sample_key_vals = samp_dict.dictionary            # unique samples-with-concept
    node_link_idx = node_dict.indices.to_numpy(zero_copy_only=False)
    node_dict_vals = node_dict.dictionary.to_pylist()
    del node_dict

    # per-link m: node dictionary position -> its m_c (node_m.get default 0.0
    # matches the original; every linked node has a schedule entry in practice).
    dictpos_m = np.array([node_m.get(n, 0.0) for n in node_dict_vals], dtype=np.float64)
    per_link_m = dictpos_m[node_link_idx]
    del node_link_idx

    # per-sample reduction over its links (arrow group-by, C++).
    if reduction == "max":
        grouped = pa.table({"s": samp_idx, "m": pa.array(per_link_m)}) \
            .group_by("s").aggregate([("m", "max")])
        s_of = grouped.column("s").to_numpy(zero_copy_only=False)
        m_of_arr = grouped.column("m_max").to_numpy(zero_copy_only=False)
    elif reduction == "geomean":
        # Rarity-weighted geometric mean, vectorised: two weighted sums per
        # sample (Sum w*log m and Sum w, with w = |log m|), then exp of the ratio.
        # Mirrors schedule.sample_multiplicity_geomean; kept here in numpy form
        # because the Python version cannot run over 218M links.
        log_m = np.log(np.maximum(per_link_m, 1e-12))
        w = np.abs(log_m)
        del per_link_m
        num = pa.table({"s": samp_idx, "v": pa.array(w * log_m)}) \
            .group_by("s").aggregate([("v", "sum")])
        den = pa.table({"s": samp_idx, "v": pa.array(w)}) \
            .group_by("s").aggregate([("v", "sum")])
        # group_by does not guarantee matching row order between the two calls;
        # align them on the sample index rather than assuming it.
        s_num = num.column("s").to_numpy(zero_copy_only=False)
        s_den = den.column("s").to_numpy(zero_copy_only=False)
        num_by_s = np.zeros(len(sample_key_vals), dtype=np.float64)
        den_by_s = np.zeros(len(sample_key_vals), dtype=np.float64)
        num_by_s[s_num] = num.column("v_sum").to_numpy(zero_copy_only=False)
        den_by_s[s_den] = den.column("v_sum").to_numpy(zero_copy_only=False)
        s_of = np.unique(s_num)
        # den == 0 <=> every concept sits at m=1 exactly -> retain (log m = 0).
        safe = den_by_s[s_of] > 0.0
        ratio = np.zeros(len(s_of), dtype=np.float64)
        ratio[safe] = num_by_s[s_of][safe] / den_by_s[s_of][safe]
        m_of_arr = np.minimum(m_max, np.exp(ratio))
    else:
        raise ValueError(f"unknown reduction {reduction!r}")
    keys_list = sample_key_vals.to_pylist()

    m_of: dict = {}
    for s, m in zip(s_of, m_of_arr):
        m_of[keys_list[s]] = float(m)

    if all_sample_keys is not None:
        for key in all_sample_keys:
            if key not in m_of:
                m_of[key] = NO_CONCEPT_MULTIPLICITY
    return m_of


def main():
    ap = argparse.ArgumentParser(description="Stages 2-4: counts + schedule + per-sample multiplicity.")
    ap.add_argument("--links", required=True, help="links.parquet (sample_key, node_id) from Stage 1.")
    ap.add_argument("--index", default=None,
                    help="index.parquet (Stage 0); supplies no-concept sample keys.")
    ap.add_argument("--counts", default=None,
                    help="Reuse a precomputed counts.parquet instead of recomputing.")
    ap.add_argument("--output_dir", required=True)
    # schedule constants (ScheduleConfig defaults = plan starting values)
    ap.add_argument("--n_high", type=float, default=100_000.0)
    ap.add_argument("--n_low", type=float, default=20_000.0)
    ap.add_argument("--a_head", type=float, default=2.5)
    ap.add_argument("--r_min", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--m_max", type=float, default=4.0)
    ap.add_argument("--reduction", choices=["max", "geomean"], default="max",
                    help="Stage-4 rule: 'max' = rarest-wins (one mid-band concept "
                         "vetoes all downsampling); 'geomean' = rarity-weighted "
                         "geometric mean (head intent survives co-occurrence).")
    args = ap.parse_args()

    cfg = ScheduleConfig(
        n_high=args.n_high, n_low=args.n_low, a_head=args.a_head,
        r_min=args.r_min, gamma=args.gamma, m_max=args.m_max,
    )
    cfg.validate()
    os.makedirs(args.output_dir, exist_ok=True)

    # Stage 2: counts
    if args.counts and os.path.exists(args.counts):
        print(f"[stage2] reusing {args.counts}")
        ct = pq.read_table(args.counts, columns=["node_id", "N_c"])
        counts = dict(zip(ct.column("node_id").to_pylist(), ct.column("N_c").to_pylist()))
    else:
        print("[stage2] computing per-concept counts N_c")
        counts = compute_counts(args.links)
        counts_path = os.path.join(args.output_dir, "counts.parquet")
        pq.write_table(pa.table({
            "node_id": pa.array(list(counts.keys()), type=pa.string()),
            "N_c": pa.array(list(counts.values()), type=pa.int64()),
        }), counts_path)
        print(f"[stage2] wrote {counts_path}: {len(counts)} concepts")

    # Stage 3: per-concept multiplicity
    print(f"[stage3] applying schedule {cfg}")
    node_m = concept_multiplicities(counts, cfg)
    node_mult_path = os.path.join(args.output_dir, "node_multiplicity.parquet")
    node_ids = list(node_m.keys())
    pq.write_table(pa.table({
        "node_id": pa.array(node_ids, type=pa.string()),
        "N_c": pa.array([counts[n] for n in node_ids], type=pa.int64()),
        "m_c": pa.array([node_m[n] for n in node_ids], type=pa.float64()),
    }), node_mult_path)
    n_head = sum(1 for n in node_ids if node_m[n] < 1.0)
    n_tail = sum(1 for n in node_ids if node_m[n] > 1.0)
    n_mid = len(node_ids) - n_head - n_tail
    print(f"[stage3] wrote {node_mult_path}: "
          f"{n_head} downsampled (m<1), {n_mid} retained (m=1), {n_tail} oversampled (m>1)")

    # Stage 4: per-sample multiplicity (rarest-wins)
    all_keys = None
    if args.index:
        print(f"[stage4] reading full sample universe from {args.index}")
        idx = pq.read_table(args.index, columns=["sample_key"])
        all_keys = idx.column("sample_key").to_pylist()
    print(f"[stage4] reducing to per-sample multiplicity (reduction={args.reduction})")
    m_of = per_sample_multiplicity(args.links, node_m, all_sample_keys=all_keys,
                                   reduction=args.reduction, m_max=args.m_max)

    sample_mult_path = os.path.join(args.output_dir, "sample_multiplicity.parquet")
    keys = list(m_of.keys())
    ms = [m_of[k] for k in keys]
    pq.write_table(pa.table({
        "sample_key": pa.array(keys, type=pa.string()),
        "m": pa.array(ms, type=pa.float64()),
    }), sample_mult_path)

    n_drop = sum(1 for m in ms if m <= 0.0)
    n_down = sum(1 for m in ms if 0.0 < m < 1.0)
    n_keep = sum(1 for m in ms if m == 1.0)
    n_over = sum(1 for m in ms if m > 1.0)
    exp_total = sum(ms)  # expected E[Sigma count] before hashing
    print(f"[stage4] wrote {sample_mult_path}: {len(keys)} samples")
    print(f"[stage4]   drop {n_drop}  downsample {n_down}  keep {n_keep}  oversample {n_over}")
    print(f"[stage4]   E[expanded total] = Sum m = {exp_total:,.0f} "
          f"(vs {len(keys):,} samples)")
    print("[done] schedule built. Next: materialize_rebalanced.py")


if __name__ == "__main__":
    main()
