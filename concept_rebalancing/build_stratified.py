"""Stage 4-alt — Stratified per-concept sampling with running deduplication.

The reference method (see ``rebalance/stratified.py``), replacing the per-sample
multiplicity path for the whole of Stages 3-5: it reads the link table and emits
a ``membership.parquet`` directly, so ``materialize_rebalanced.py`` is not used.

    tail (< n_head)  : fully retained
    head (>= n_head) : downsampled at ``2/log(Count)``
    ascending frequency + running dedup

Output is a subsample (``count == 1`` for every kept sample), so the corpus
shrinks rather than expanding.

    python concept_rebalancing/build_stratified.py \
        --links   concept_rebalancing/runs/blip3o_pretrain/links.parquet \
        --index   concept_rebalancing/runs/blip3o_pretrain/index.parquet \
        --counts  concept_rebalancing/runs/blip3o_pretrain/counts.parquet \
        --output_dir experiments/rebalanced_D --name rebalanced_D \
        --n_head 100000
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from rebalance.stratified import StratifiedConfig, concept_target  # noqa: E402


def select_stratified(links_path, counts, cfg, priority_by_key=None, verbose=True):
    """Vectorised ascending-frequency selection with running dedup.

    Mirrors ``stratified.stratified_select`` but over dictionary-encoded index
    arrays so a 200M-link corpus stays in bounded memory. Returns a boolean mask
    over the sample dictionary plus the dictionary itself.
    """
    lt = pq.read_table(links_path, columns=["sample_key", "node_id"])
    samp_dict = pc.dictionary_encode(
        lt.column("sample_key").cast(pa.large_string()).combine_chunks())
    node_dict = pc.dictionary_encode(
        lt.column("node_id").cast(pa.large_string()).combine_chunks())
    del lt

    s_idx = samp_dict.indices.to_numpy(zero_copy_only=False)
    n_idx = node_dict.indices.to_numpy(zero_copy_only=False)
    sample_keys = samp_dict.dictionary
    node_vals = node_dict.dictionary.to_pylist()
    n_samples = len(sample_keys)
    if verbose:
        print(f"[stage4s] links={len(s_idx):,} samples={n_samples:,} "
              f"concepts={len(node_vals):,}")

    # Group links by concept once: sort by node, then slice. Cheaper than a
    # per-concept boolean scan over 200M links (which would be O(concepts*links)).
    order = np.argsort(n_idx, kind="stable")
    n_sorted = n_idx[order]
    s_sorted = s_idx[order]
    starts = np.searchsorted(n_sorted, np.arange(len(node_vals)), side="left")
    ends = np.searchsorted(n_sorted, np.arange(len(node_vals)), side="right")
    del n_sorted, n_idx, order

    # Deterministic within-concept ordering: the sample's stored priority (the
    # same sha256 draw the multiplicity path uses), so reruns reproduce exactly.
    if priority_by_key is not None:
        keys_list = sample_keys.to_pylist()
        prio = np.array([priority_by_key.get(k, 0) for k in keys_list],
                        dtype=np.int64)
        del keys_list
    else:
        prio = np.arange(n_samples, dtype=np.int64)

    selected = np.zeros(n_samples, dtype=bool)

    # Ascending frequency — the ordering the dedup depends on. Concepts tying on
    # N_c must break by node_id to match stratified_select's (N_c, node_id) key:
    # dedup makes the result order-dependent, so a different tiebreak yields a
    # different (still valid, but non-reproducible) membership.
    N_c_arr = np.array([counts.get(n, 0) for n in node_vals], dtype=np.int64)
    concept_order = sorted(range(len(node_vals)),
                           key=lambda i: (int(N_c_arr[i]), node_vals[i]))

    targets = np.array(
        [concept_target(int(N_c_arr[i]), node_vals[i], cfg) for i in range(len(node_vals))],
        dtype=np.int64)

    n_skipped = 0
    for rank, ci in enumerate(concept_order):
        lo, hi = starts[ci], ends[ci]
        if hi <= lo:
            continue
        samps = s_sorted[lo:hi]
        target = targets[ci]

        # running dedup: samples already taken for a rarer concept count toward
        # this concept's quota, so only the shortfall is drawn.
        already_mask = selected[samps]
        need = int(target) - int(already_mask.sum())
        if need <= 0:
            n_skipped += 1
            continue

        cands = samps[~already_mask]
        if need >= len(cands):
            selected[cands] = True
        else:
            # Lowest priority first = deterministic, uniform-in-hash subset.
            # Sort (not argpartition) so samples with equal priority break by
            # dictionary index the same way the reference breaks them by key.
            part = np.argsort(prio[cands], kind="stable")[:need]
            selected[cands[part]] = True

        if verbose and rank % 10000 == 0 and rank:
            print(f"[stage4s]   {rank:,}/{len(node_vals):,} concepts, "
                  f"{int(selected.sum()):,} selected", flush=True)

    if verbose:
        print(f"[stage4s] done: {int(selected.sum()):,} selected; "
              f"{n_skipped:,} concepts needed no extra draw (quota met by rarer)")
    return selected, sample_keys


def main():
    ap = argparse.ArgumentParser(
        description="Stage 4-alt: stratified per-concept sampling + running dedup.")
    ap.add_argument("--links", required=True)
    ap.add_argument("--index", required=True,
                    help="index.parquet — supplies tar_path + the deterministic priority.")
    ap.add_argument("--counts", default=None,
                    help="counts.parquet (node_id, N_c); recomputed from links if absent.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--n_head", type=float, default=100_000.0)
    ap.add_argument("--a_head", type=float, default=2.5)
    ap.add_argument("--r_min", type=float, default=0.1)
    ap.add_argument("--log_base", choices=["log10", "ln"], default="log10")
    ap.add_argument("--boosts", default=None,
                    help="JSON {node_id: boost} for weak-capability targeting (+0.2..0.5).")
    ap.add_argument("--segments", default=None,
                    help='JSON [[lo, hi, mult], ...] segment-specific base rates.')
    args = ap.parse_args()

    segments = {}
    if args.segments:
        for lo, hi, mult in json.loads(args.segments):
            segments[(float(lo), float(hi))] = float(mult)
    boosts = json.loads(open(args.boosts).read()) if args.boosts else {}

    cfg = StratifiedConfig(n_head=args.n_head, a_head=args.a_head, r_min=args.r_min,
                           log_base=args.log_base, segments=segments, boosts=boosts)
    cfg.validate()
    os.makedirs(args.output_dir, exist_ok=True)
    name = args.name or os.path.basename(os.path.normpath(args.output_dir))
    print(f"[stage4s] config: n_head={cfg.n_head:,.0f} a_head={cfg.a_head} "
          f"r_min={cfg.r_min} log={cfg.log_base} "
          f"segments={len(segments)} boosts={len(boosts)}")

    if args.counts and os.path.exists(args.counts):
        ct = pq.read_table(args.counts, columns=["node_id", "N_c"])
        counts = dict(zip(ct.column("node_id").to_pylist(), ct.column("N_c").to_pylist()))
    else:
        t = pq.read_table(args.links, columns=["sample_key", "node_id"])
        g = t.group_by("node_id").aggregate([("sample_key", "count")])
        counts = dict(zip(g.column("node_id").to_pylist(),
                          g.column("sample_key_count").to_pylist()))
        del t
    n_head_c = sum(1 for v in counts.values() if v >= cfg.n_head)
    print(f"[stage4s] {len(counts):,} concepts: {n_head_c:,} head (>= {cfg.n_head:,.0f}), "
          f"{len(counts) - n_head_c:,} tail (fully retained)")

    print(f"[stage4s] reading index: {args.index}")
    idx = pq.read_table(args.index, columns=["sample_key", "tar_path", "priority"])
    idx_keys = idx.column("sample_key").to_pylist()
    priority_by_key = dict(zip(idx_keys, idx.column("priority").to_pylist()))
    tar_by_key = dict(zip(idx_keys, idx.column("tar_path").to_pylist()))
    del idx, idx_keys

    selected, sample_keys = select_stratified(args.links, counts, cfg,
                                              priority_by_key=priority_by_key)

    # Samples carrying no content concept never appear in the links; the
    # multiplicity path retains them (NO_CONCEPT_MULTIPLICITY = 1.0), so do the
    # same here for a like-for-like comparison.
    keys_list = sample_keys.to_pylist()
    kept = [k for k, s in zip(keys_list, selected) if s]
    linked = set(keys_list)
    no_concept = [k for k in tar_by_key if k not in linked]
    if no_concept:
        print(f"[stage4s] + {len(no_concept):,} no-concept samples retained")
        kept.extend(no_concept)

    kept = sorted(k for k in kept if k in tar_by_key)
    tars = sorted({tar_by_key[k] for k in kept})

    membership_path = os.path.join(args.output_dir, "membership.parquet")
    pq.write_table(pa.table({
        "sample_key": pa.array(kept, type=pa.string()),
        "count": pa.array([1] * len(kept), type=pa.int32()),
    }), membership_path)

    with open(os.path.join(args.output_dir, "shardlist.txt"), "w") as f:
        for t in tars:
            f.write(t + "\n")

    h = hashlib.sha256()
    for k in kept:
        h.update(f"{k}\t1\n".encode())
    membership_hash = h.hexdigest()[:16]

    cfg_out = {
        "name": name,
        "mode": "stratified_dedup",
        "dataset_cls": "rebalanced",
        "distinct_samples": len(kept),
        "actual_samples": len(kept),   # pure subsample: no expansion
        "num_shards": len(tars),
        "membership_hash": membership_hash,
        "membership_file": "membership.parquet",
        "schedule": {"n_head": cfg.n_head, "a_head": cfg.a_head, "r_min": cfg.r_min,
                     "log_base": cfg.log_base, "n_boosts": len(cfg.boosts)},
    }
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg_out, f, default_flow_style=False, sort_keys=False)

    print(f"\n[stage4s] wrote {args.output_dir}/")
    print(f"  membership.parquet: {len(kept):,} samples "
          f"({100*len(kept)/max(1,len(tar_by_key)):.1f}% of corpus)")
    print(f"  shardlist.txt: {len(tars)} tars")
    print(f"  membership_hash: {membership_hash}")


if __name__ == "__main__":
    main()
