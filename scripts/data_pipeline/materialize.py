"""
Materialize an experiment from a YAML config and enriched index.

Reads the config, queries the index, and writes:
  - membership.txt: one sample key per line
  - shardlist.txt:  deduplicated tar paths
  - config.yaml:    copy of config with runtime stats

config.yaml format:
    name: baseline_1M
    filter_expression: "aesthetic_score > 0.6 and is_clean == True"
    K: 1000000
    mode: select   # or scan

Usage:
    python scripts/materialize.py \
        --config experiments/baseline_1M/config.yaml \
        --index index_enriched.parquet \
        --output_dir experiments/baseline_1M/
"""

import argparse
import hashlib
import os

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml


def apply_filter(table: pa.Table, expr: str) -> pa.Table:
    """Apply a pandas-style filter expression using pyarrow compute.

    Supported operators: >, <, >=, <=, ==, !=
    Supported connectors: and, or
    Boolean columns: is_clean == True / is_clean == False
    """
    if not expr or not expr.strip():
        return table

    # Split by 'and' / 'or' and apply sequentially
    # For simplicity, only 'and' conjunctions are supported for now.
    # Complex expressions should use parentheses-free 'and' chains.
    clauses = [c.strip() for c in expr.split(" and ")]
    mask = None

    for clause in clauses:
        clause_mask = _eval_clause(table, clause)
        if mask is None:
            mask = clause_mask
        else:
            mask = pc.and_(mask, clause_mask)

    return table.filter(mask)


def _eval_clause(table: pa.Table, clause: str) -> pa.ChunkedArray:
    """Evaluate a single comparison clause like 'aesthetic_score > 0.6'."""
    import re
    m = re.match(r'(\w+)\s*(>=|<=|!=|==|>|<)\s*(.+)', clause)
    if not m:
        raise ValueError(f"Cannot parse filter clause: '{clause}'")

    col_name, op, val_str = m.group(1), m.group(2), m.group(3).strip()
    col = table.column(col_name)

    # Parse value
    if val_str in ("True", "true"):
        val = True
    elif val_str in ("False", "false"):
        val = False
    else:
        try:
            val = int(val_str)
        except ValueError:
            try:
                val = float(val_str)
            except ValueError:
                val = val_str.strip("'\"")

    ops = {
        ">":  pc.greater,
        "<":  pc.less,
        ">=": pc.greater_equal,
        "<=": pc.less_equal,
        "==": pc.equal,
        "!=": pc.not_equal,
    }
    # Handle nulls: null comparisons -> False
    not_null = pc.is_valid(col)
    comparison = ops[op](col, pa.scalar(val))
    return pc.and_(not_null, comparison)


def main():
    parser = argparse.ArgumentParser(description="Materialize experiment from YAML config.")
    parser.add_argument("--config", type=str, required=True,
                        help="Experiment YAML config file.")
    parser.add_argument("--index", type=str, required=True,
                        help="Enriched parquet index file.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for membership.txt, shardlist.txt, config.yaml.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    name = cfg.get("name", "unnamed")
    filter_expr = cfg.get("filter_expression", "")
    K = cfg["K"]
    mode = cfg.get("mode", "select")

    print(f"Experiment: {name}")
    print(f"  filter: {filter_expr or '(none)'}")
    print(f"  K: {K}, mode: {mode}")

    # Load index (already sorted by priority)
    table = pq.read_table(args.index)
    print(f"  index: {table.num_rows} total samples")

    if mode == "select":
        # Filter first, then take top-K by priority
        if filter_expr:
            table = apply_filter(table, filter_expr)
            print(f"  after filter: {table.num_rows} samples")
        table = table.slice(0, min(K, table.num_rows))
        print(f"  selected: {table.num_rows} samples")

    elif mode == "scan":
        # Take top-K by priority first, then filter
        table = table.slice(0, min(K, table.num_rows))
        if filter_expr:
            table = apply_filter(table, filter_expr)
        print(f"  scanned {K}, survived filter: {table.num_rows} samples")

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Write outputs
    os.makedirs(args.output_dir, exist_ok=True)

    # membership.txt
    membership_path = os.path.join(args.output_dir, "membership.txt")
    keys = table.column("sample_key").to_pylist()
    with open(membership_path, "w") as f:
        for key in keys:
            f.write(key + "\n")

    # shardlist.txt
    shardlist_path = os.path.join(args.output_dir, "shardlist.txt")
    unique_tars = sorted(set(table.column("tar_path").to_pylist()))
    with open(shardlist_path, "w") as f:
        for tar in unique_tars:
            f.write(tar + "\n")

    # config.yaml with stats
    cfg_out = dict(cfg)
    cfg_out["actual_samples"] = table.num_rows
    cfg_out["num_shards"] = len(unique_tars)
    cfg_out["membership_hash"] = hashlib.sha256(
        open(membership_path, "rb").read()
    ).hexdigest()[:16]
    config_out_path = os.path.join(args.output_dir, "config.yaml")
    with open(config_out_path, "w") as f:
        yaml.dump(cfg_out, f, default_flow_style=False)

    print(f"\nMaterialized to {args.output_dir}/")
    print(f"  membership.txt: {len(keys)} keys")
    print(f"  shardlist.txt:  {len(unique_tars)} tars")
    print(f"  config.yaml:    saved")


if __name__ == "__main__":
    main()
