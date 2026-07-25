"""Stage 5 — Materialize the resampled membership (multiplicity-aware).

The multiplicity generalization of ``scripts/data_pipeline/materialize.py``:
instead of a flat membership *set*, it realizes each sample's real multiplicity
``m`` to an integer copy ``count`` via the shared hash rule (plan §4) and emits a
``(sample_key, count)`` membership. ``count = 0`` samples are dropped.

Inputs
------
- ``sample_multiplicity.parquet`` (``sample_key, m``) from Stage 4.
- ``index.parquet`` (``sample_key, tar_path, priority``) from Stage 0 — supplies
  the deterministic draw ``h = priority / 2**63`` and the tar for each sample.
  If the index was quality-filtered upstream (enrich_index + filter, plan open
  item #3), samples absent from it are naturally excluded here.

Outputs (into ``--output_dir``, an ``experiment_dir`` the trainer consumes)
-------
- ``membership.parquet`` — ``(sample_key, count)`` with ``count >= 1``.
- ``shardlist.txt``      — unique ``tar_path``s over kept samples.
- ``config.yaml``        — schedule constants + ``actual_samples = Sum count``
                           (the *expanded* total, so ``run_experiment.sh``'s
                           ``MAX_STEPS`` budgets the tail copies) + a
                           ``membership_hash``.

    python concept_rebalancing/materialize_rebalanced.py \
        --sample_multiplicity concept_rebalancing/runs/blip3o_pretrain/sample_multiplicity.parquet \
        --index               concept_rebalancing/runs/blip3o_pretrain/index.parquet \
        --output_dir experiments/rebalanced_B --name rebalanced_B
"""

import argparse
import hashlib
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from rebalance.multiplicity import priority_to_hash01, realize_count  # noqa: E402


def load_index(index_path):
    """sample_key -> (tar_path, priority)."""
    table = pq.read_table(index_path, columns=["sample_key", "tar_path", "priority"])
    keys = table.column("sample_key").to_pylist()
    tars = table.column("tar_path").to_pylist()
    prios = table.column("priority").to_pylist()
    return {k: (t, p) for k, t, p in zip(keys, tars, prios)}


def main():
    ap = argparse.ArgumentParser(description="Stage 5: materialize a multiplicity membership.")
    ap.add_argument("--sample_multiplicity", required=True,
                    help="sample_multiplicity.parquet (sample_key, m) from Stage 4.")
    ap.add_argument("--index", required=True,
                    help="index.parquet (sample_key, tar_path, priority) from Stage 0.")
    ap.add_argument("--output_dir", required=True, help="Experiment dir to write into.")
    ap.add_argument("--name", default=None, help="Experiment name (default: basename of output_dir).")
    ap.add_argument("--schedule_config", default=None,
                    help="Optional YAML/JSON of schedule constants to embed in config.yaml.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    name = args.name or os.path.basename(os.path.normpath(args.output_dir))

    print(f"[materialize] loading index: {args.index}")
    index = load_index(args.index)

    sm = pq.read_table(args.sample_multiplicity, columns=["sample_key", "m"])
    sample_keys = sm.column("sample_key").to_pylist()
    ms = sm.column("m").to_pylist()
    print(f"[materialize] {len(sample_keys)} samples in schedule; "
          f"{len(index)} in index")

    kept_keys, kept_counts, kept_tars = [], [], set()
    missing = 0
    total_count = 0
    for key, m in zip(sample_keys, ms):
        entry = index.get(key)
        if entry is None:
            missing += 1
            continue  # filtered out upstream (quality gate) or not in corpus
        tar_path, priority = entry
        h = priority_to_hash01(priority)
        count = realize_count(m, h)
        if count >= 1:
            kept_keys.append(key)
            kept_counts.append(count)
            kept_tars.add(tar_path)
            total_count += count

    # membership.parquet — sorted for a reproducible membership_hash independent
    # of input row order.
    order = sorted(range(len(kept_keys)), key=lambda i: kept_keys[i])
    keys_sorted = [kept_keys[i] for i in order]
    counts_sorted = [kept_counts[i] for i in order]
    membership_path = os.path.join(args.output_dir, "membership.parquet")
    pq.write_table(pa.table({
        "sample_key": pa.array(keys_sorted, type=pa.string()),
        "count": pa.array(counts_sorted, type=pa.int32()),
    }), membership_path)

    # shardlist.txt
    shardlist_path = os.path.join(args.output_dir, "shardlist.txt")
    unique_tars = sorted(kept_tars)
    with open(shardlist_path, "w") as f:
        for tar in unique_tars:
            f.write(tar + "\n")

    # membership_hash over the canonical (sorted) membership
    h = hashlib.sha256()
    for k, c in zip(keys_sorted, counts_sorted):
        h.update(f"{k}\t{c}\n".encode())
    membership_hash = h.hexdigest()[:16]

    # config.yaml — consumed by run_experiment.sh (actual_samples) and the
    # multiplicity dataset (dataset_cls hint).
    cfg_out = {
        "name": name,
        "mode": "rebalanced_multiplicity",
        "dataset_cls": "rebalanced",
        "distinct_samples": len(keys_sorted),   # samples kept (count>=1)
        "actual_samples": total_count,           # EXPANDED total = Sum count
        "num_shards": len(unique_tars),
        "membership_hash": membership_hash,
        "membership_file": "membership.parquet",
    }
    if args.schedule_config and os.path.exists(args.schedule_config):
        with open(args.schedule_config) as f:
            cfg_out["schedule"] = yaml.safe_load(f)
    config_path = os.path.join(args.output_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(cfg_out, f, default_flow_style=False, sort_keys=False)

    print(f"\n[materialize] wrote {args.output_dir}/")
    print(f"  membership.parquet: {len(keys_sorted):,} distinct samples")
    print(f"  actual_samples (expanded): {total_count:,}  "
          f"(x{total_count / max(1, len(keys_sorted)):.2f} vs distinct)")
    print(f"  shardlist.txt: {len(unique_tars)} tars")
    print(f"  membership_hash: {membership_hash}")
    if missing:
        print(f"  note: {missing:,} scheduled samples not in index (filtered/absent) — dropped")


if __name__ == "__main__":
    main()
