"""
Tests for enrich_index.py covering:
  - Partial match: scores CSV has keys not in index and vice versa
  - Multiple value columns joined at once
  - Parquet scores file input
  - Subset list with boolean flag
  - Enrichment preserves priority sort order
  - Null handling for unmatched keys
  - Successive enrichments (add column, then add another)

Usage:
    python scripts/data_pipeline/tests/test_enrich_index.py
"""

import os
import shutil
import subprocess
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq


SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENRICH_SCRIPT = os.path.join(SCRIPTS_DIR, "enrich_index.py")


def _run_enrich(args):
    cmd = [sys.executable, ENRICH_SCRIPT] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"enrich_index.py failed:\n{result.stderr}")
    return result.stdout


def _make_index(tmpdir, keys):
    """Create a minimal parquet index with given keys."""
    import hashlib
    records = []
    for k in keys:
        priority = int(hashlib.sha256(k.encode()).hexdigest(), 16) % (2**63)
        records.append({"sample_key": k, "tar_path": "/fake/data.tar", "priority": priority})
    records.sort(key=lambda r: r["priority"])

    table = pa.table({
        "sample_key": [r["sample_key"] for r in records],
        "tar_path":   [r["tar_path"]   for r in records],
        "priority":   [r["priority"]   for r in records],
    })
    path = os.path.join(tmpdir, "index.parquet")
    pq.write_table(table, path)
    return path


def test_partial_match_csv():
    """Scores CSV has some keys not in index, and index has keys not in CSV."""
    tmpdir = tempfile.mkdtemp()
    try:
        index_keys = [f"{i:04d}" for i in range(10)]
        index_path = _make_index(tmpdir, index_keys)

        # CSV only covers keys 0-6 (misses 7-9)
        csv_path = os.path.join(tmpdir, "scores.csv")
        with open(csv_path, "w") as f:
            f.write("sample_key,score\n")
            for i in range(7):
                f.write(f"{i:04d},{i * 0.1}\n")
            # Also add a key NOT in the index
            f.write("9999,0.99\n")

        out = os.path.join(tmpdir, "enriched.parquet")
        _run_enrich([
            "--index", index_path, "--scores", csv_path,
            "--key_col", "sample_key", "--value_cols", "score",
            "--output", out
        ])

        table = pq.read_table(out)
        assert table.num_rows == 10, f"Row count changed: {table.num_rows}"
        assert "score" in table.column_names

        scores = table.column("score").to_pylist()
        # 7 matched, 3 should be None
        non_null = sum(1 for s in scores if s is not None)
        nulls = sum(1 for s in scores if s is None)
        assert non_null == 7, f"Expected 7 non-null scores, got {non_null}"
        assert nulls == 3, f"Expected 3 null scores, got {nulls}"
        print("[PASS] test_partial_match_csv")
    finally:
        shutil.rmtree(tmpdir)


def test_multiple_value_columns():
    """Join two score columns from a single CSV."""
    tmpdir = tempfile.mkdtemp()
    try:
        index_keys = [f"{i:04d}" for i in range(5)]
        index_path = _make_index(tmpdir, index_keys)

        csv_path = os.path.join(tmpdir, "multi.csv")
        with open(csv_path, "w") as f:
            f.write("sample_key,aesthetic_score,clip_score\n")
            for i in range(5):
                f.write(f"{i:04d},{i * 0.2},{i * 0.1 + 0.5}\n")

        out = os.path.join(tmpdir, "enriched.parquet")
        _run_enrich([
            "--index", index_path, "--scores", csv_path,
            "--key_col", "sample_key",
            "--value_cols", "aesthetic_score", "clip_score",
            "--output", out
        ])

        table = pq.read_table(out)
        assert "aesthetic_score" in table.column_names
        assert "clip_score" in table.column_names
        assert table.num_rows == 5
        print("[PASS] test_multiple_value_columns")
    finally:
        shutil.rmtree(tmpdir)


def test_parquet_scores_input():
    """Enrich from a parquet file instead of CSV."""
    tmpdir = tempfile.mkdtemp()
    try:
        index_keys = [f"{i:04d}" for i in range(5)]
        index_path = _make_index(tmpdir, index_keys)

        # Create scores as parquet
        scores_table = pa.table({
            "sample_key": [f"{i:04d}" for i in range(5)],
            "quality": [0.1, 0.5, 0.8, 0.3, 0.9],
        })
        scores_path = os.path.join(tmpdir, "scores.parquet")
        pq.write_table(scores_table, scores_path)

        out = os.path.join(tmpdir, "enriched.parquet")
        _run_enrich([
            "--index", index_path, "--scores", scores_path,
            "--key_col", "sample_key", "--value_cols", "quality",
            "--output", out
        ])

        table = pq.read_table(out)
        assert "quality" in table.column_names
        # All 5 should match
        non_null = sum(1 for v in table.column("quality").to_pylist() if v is not None)
        assert non_null == 5
        print("[PASS] test_parquet_scores_input")
    finally:
        shutil.rmtree(tmpdir)


def test_subset_flag():
    """Create boolean column from a subset key list."""
    tmpdir = tempfile.mkdtemp()
    try:
        index_keys = [f"{i:04d}" for i in range(10)]
        index_path = _make_index(tmpdir, index_keys)

        subset_list = os.path.join(tmpdir, "subset.txt")
        with open(subset_list, "w") as f:
            for i in range(0, 10, 3):  # 0, 3, 6, 9
                f.write(f"{i:04d}\n")

        out = os.path.join(tmpdir, "enriched.parquet")
        _run_enrich([
            "--index", index_path,
            "--subset_list", subset_list, "--subset_col", "in_subset",
            "--output", out
        ])

        table = pq.read_table(out)
        assert "in_subset" in table.column_names
        flags = table.column("in_subset").to_pylist()
        assert sum(flags) == 4, f"Expected 4 True, got {sum(flags)}"
        print("[PASS] test_subset_flag")
    finally:
        shutil.rmtree(tmpdir)


def test_priority_order_preserved():
    """After enrichment, rows must still be sorted by priority ascending."""
    tmpdir = tempfile.mkdtemp()
    try:
        index_keys = [f"{i:04d}" for i in range(20)]
        index_path = _make_index(tmpdir, index_keys)

        csv_path = os.path.join(tmpdir, "scores.csv")
        with open(csv_path, "w") as f:
            f.write("sample_key,score\n")
            for k in index_keys:
                f.write(f"{k},{hash(k) % 100 / 100.0}\n")

        out = os.path.join(tmpdir, "enriched.parquet")
        _run_enrich([
            "--index", index_path, "--scores", csv_path,
            "--key_col", "sample_key", "--value_cols", "score",
            "--output", out
        ])

        table = pq.read_table(out)
        priorities = table.column("priority").to_pylist()
        assert priorities == sorted(priorities), "Priority sort order not preserved"
        print("[PASS] test_priority_order_preserved")
    finally:
        shutil.rmtree(tmpdir)


def test_successive_enrichments():
    """Apply two enrichments in sequence, verify both columns exist."""
    tmpdir = tempfile.mkdtemp()
    try:
        index_keys = [f"{i:04d}" for i in range(5)]
        index_path = _make_index(tmpdir, index_keys)

        # First enrichment: CSV score
        csv1 = os.path.join(tmpdir, "s1.csv")
        with open(csv1, "w") as f:
            f.write("sample_key,score_a\n")
            for k in index_keys:
                f.write(f"{k},0.5\n")

        mid = os.path.join(tmpdir, "mid.parquet")
        _run_enrich([
            "--index", index_path, "--scores", csv1,
            "--key_col", "sample_key", "--value_cols", "score_a",
            "--output", mid
        ])

        # Second enrichment: boolean flag
        subset = os.path.join(tmpdir, "sub.txt")
        with open(subset, "w") as f:
            f.write("0000\n0002\n")

        out = os.path.join(tmpdir, "final.parquet")
        _run_enrich([
            "--index", mid,
            "--subset_list", subset, "--subset_col", "is_selected",
            "--output", out
        ])

        table = pq.read_table(out)
        assert "score_a" in table.column_names, "First enrichment column lost"
        assert "is_selected" in table.column_names, "Second enrichment column missing"
        assert table.num_rows == 5
        print("[PASS] test_successive_enrichments")
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    tests = [
        test_partial_match_csv,
        test_multiple_value_columns,
        test_parquet_scores_input,
        test_subset_flag,
        test_priority_order_preserved,
        test_successive_enrichments,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
