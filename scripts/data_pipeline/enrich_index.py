"""
Enrich the parquet index with per-sample scores or boolean flags.

Supports two modes:
  A) Join columns from a CSV/parquet file by sample_key.
  B) Create a boolean column from a list of sample keys.

Usage:
    # Join aesthetic scores from a CSV
    python scripts/enrich_index.py \
        --index index.parquet \
        --scores aesthetic_scores.csv \
        --key_col sample_key \
        --value_cols aesthetic_score \
        --output index_enriched.parquet

    # Mark a subset of keys with a boolean flag
    python scripts/enrich_index.py \
        --index index.parquet \
        --subset_list clean_keys.txt \
        --subset_col is_clean \
        --output index_enriched.parquet
"""

import argparse
import csv

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description="Enrich parquet index with scores or flags.")
    parser.add_argument("--index", type=str, required=True,
                        help="Input parquet index.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output enriched parquet.")

    # Mode A: join from file
    parser.add_argument("--scores", type=str, default=None,
                        help="CSV or parquet file containing scores to join.")
    parser.add_argument("--key_col", type=str, default="sample_key",
                        help="Key column name in the scores file.")
    parser.add_argument("--value_cols", type=str, nargs="+", default=None,
                        help="Column names to join from the scores file.")

    # Mode B: boolean flag from key list
    parser.add_argument("--subset_list", type=str, default=None,
                        help="Text file with one sample_key per line.")
    parser.add_argument("--subset_col", type=str, default=None,
                        help="Name of the boolean column to create.")
    args = parser.parse_args()

    table = pq.read_table(args.index)
    print(f"Loaded index: {table.num_rows} samples, columns: {table.column_names}")

    if args.scores:
        # Load scores into a lookup dict
        if args.scores.endswith(".parquet"):
            scores_table = pq.read_table(args.scores)
            score_keys = scores_table.column(args.key_col).to_pylist()
            all_cols = scores_table.column_names
            cols_to_join = args.value_cols or [c for c in all_cols if c != args.key_col]
            lookup = {}
            for i, key in enumerate(score_keys):
                lookup[key] = {c: scores_table.column(c)[i].as_py() for c in cols_to_join}
        else:
            # Read CSV
            with open(args.scores, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if not rows:
                print("[WARN] Scores file is empty, skipping.")
                lookup = {}
                cols_to_join = []
            else:
                all_cols = list(rows[0].keys())
                cols_to_join = args.value_cols or [c for c in all_cols if c != args.key_col]
                lookup = {}
                for row in rows:
                    key = row[args.key_col]
                    lookup[key] = {c: row[c] for c in cols_to_join}

        # Join onto the index
        if cols_to_join and lookup:
            index_keys = table.column("sample_key").to_pylist()
            for col in cols_to_join:
                values = []
                for k in index_keys:
                    entry = lookup.get(k)
                    if entry is not None:
                        try:
                            values.append(float(entry[col]))
                        except (ValueError, TypeError):
                            values.append(None)
                    else:
                        values.append(None)
                table = table.append_column(col, pa.array(values, type=pa.float64()))

            matched = sum(1 for v in values if v is not None)
            print(f"Joined {cols_to_join}: {matched}/{table.num_rows} samples matched")

    if args.subset_list and args.subset_col:
        with open(args.subset_list) as f:
            subset_keys = set(l.strip() for l in f if l.strip())

        index_keys = table.column("sample_key").to_pylist()
        flags = [k in subset_keys for k in index_keys]
        table = table.append_column(args.subset_col, pa.array(flags, type=pa.bool_()))
        n_true = sum(flags)
        print(f"Subset column '{args.subset_col}': {n_true}/{table.num_rows} = True")

    # Sort by priority to maintain order
    sort_indices = pc.sort_indices(table, sort_keys=[("priority", "ascending")])
    table = table.take(sort_indices)

    pq.write_table(table, args.output)
    print(f"Enriched index written: {args.output}")


if __name__ == "__main__":
    main()
