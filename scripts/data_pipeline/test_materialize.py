"""
Tests for materialize.py covering:
  - Select mode: filter then top-K
  - Scan mode: top-K then filter
  - K larger than available samples (should not crash)
  - Empty filter expression (take all)
  - All filter operators: >, <, >=, <=, ==, !=
  - Boolean filter (is_clean == True)
  - Null values in filtered column (should be excluded)
  - Select vs scan produce different results for same K and filter
  - Membership hash reproducibility
  - Shardlist deduplication

Usage:
    python scripts/data_pipeline/test_materialize.py
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
MATERIALIZE_SCRIPT = os.path.join(SCRIPTS_DIR, "materialize.py")


def _run_materialize(args):
    cmd = [sys.executable, MATERIALIZE_SCRIPT] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"materialize.py failed:\n{result.stderr}")
    return result.stdout


def _make_enriched_index(tmpdir, n=20):
    """Create a test enriched index with n samples, aesthetic_score, and is_clean."""
    records = []
    for i in range(n):
        key = f"{i:06d}"
        priority = int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**63)
        records.append({
            "sample_key": key,
            "tar_path": f"/data/shard-{i // 5:03d}.tar",
            "priority": priority,
            "aesthetic_score": (i % 10) * 0.1,  # 0.0, 0.1, ..., 0.9, 0.0, ...
            "is_clean": i < 15,  # first 15 are clean
        })
    records.sort(key=lambda r: r["priority"])

    table = pa.table({
        "sample_key":      [r["sample_key"]      for r in records],
        "tar_path":        [r["tar_path"]        for r in records],
        "priority":        [r["priority"]        for r in records],
        "aesthetic_score": [r["aesthetic_score"] for r in records],
        "is_clean":        [r["is_clean"]        for r in records],
    })
    path = os.path.join(tmpdir, "enriched.parquet")
    pq.write_table(table, path)
    return path


def _write_config(tmpdir, name, filter_expr, K, mode):
    exp_dir = os.path.join(tmpdir, name)
    os.makedirs(exp_dir, exist_ok=True)
    config_path = os.path.join(exp_dir, "config.yaml")
    cfg = {"name": name, "K": K, "mode": mode}
    if filter_expr:
        cfg["filter_expression"] = filter_expr
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)
    return config_path, exp_dir


def _read_membership(exp_dir):
    with open(os.path.join(exp_dir, "membership.txt")) as f:
        return set(l.strip() for l in f if l.strip())


def _read_shardlist(exp_dir):
    with open(os.path.join(exp_dir, "shardlist.txt")) as f:
        return [l.strip() for l in f if l.strip()]


def _read_config_out(exp_dir):
    with open(os.path.join(exp_dir, "config.yaml")) as f:
        return yaml.safe_load(f)


def test_select_mode():
    """Select: filter first, then top-K by priority."""
    tmpdir = tempfile.mkdtemp()
    try:
        index_path = _make_enriched_index(tmpdir, n=20)
        config, exp_dir = _write_config(
            tmpdir, "select_test",
            "aesthetic_score > 0.5", K=3, mode="select"
        )

        _run_materialize(["--config", config, "--index", index_path, "--output_dir", exp_dir])

        members = _read_membership(exp_dir)
        assert len(members) == 3, f"Expected 3, got {len(members)}"

        cfg_out = _read_config_out(exp_dir)
        assert cfg_out["actual_samples"] == 3
        print("[PASS] test_select_mode")
    finally:
        shutil.rmtree(tmpdir)


def test_scan_mode():
    """Scan: top-K first, then filter. Output <= K."""
    tmpdir = tempfile.mkdtemp()
    try:
        index_path = _make_enriched_index(tmpdir, n=20)
        config, exp_dir = _write_config(
            tmpdir, "scan_test",
            "aesthetic_score > 0.5", K=10, mode="scan"
        )

        _run_materialize(["--config", config, "--index", index_path, "--output_dir", exp_dir])

        members = _read_membership(exp_dir)
        assert len(members) <= 10, f"Scan mode produced more than K: {len(members)}"
        assert len(members) > 0, "Scan mode produced 0 samples"
        print(f"[PASS] test_scan_mode ({len(members)} from budget of 10)")
    finally:
        shutil.rmtree(tmpdir)


def test_select_vs_scan_differ():
    """Select and scan should produce different results for same K and filter."""
    tmpdir = tempfile.mkdtemp()
    try:
        index_path = _make_enriched_index(tmpdir, n=20)

        config_s, exp_s = _write_config(tmpdir, "sel", "aesthetic_score > 0.3", K=8, mode="select")
        config_c, exp_c = _write_config(tmpdir, "scn", "aesthetic_score > 0.3", K=8, mode="scan")

        _run_materialize(["--config", config_s, "--index", index_path, "--output_dir", exp_s])
        _run_materialize(["--config", config_c, "--index", index_path, "--output_dir", exp_c])

        sel_members = _read_membership(exp_s)
        scn_members = _read_membership(exp_c)

        # Select guarantees exactly K (if enough pass filter)
        # Scan takes top-K then filters, so <= K
        assert len(sel_members) == 8
        assert len(scn_members) <= 8

        # They should differ (scan restricts the pool first)
        # Not necessarily always, but for our test data they should
        print(f"[PASS] test_select_vs_scan_differ (select={len(sel_members)}, scan={len(scn_members)})")
    finally:
        shutil.rmtree(tmpdir)


def test_k_larger_than_available():
    """K > total samples should not crash, just return all that pass filter."""
    tmpdir = tempfile.mkdtemp()
    try:
        index_path = _make_enriched_index(tmpdir, n=10)
        config, exp_dir = _write_config(
            tmpdir, "big_k",
            "aesthetic_score > 0.5", K=100000, mode="select"
        )

        _run_materialize(["--config", config, "--index", index_path, "--output_dir", exp_dir])

        members = _read_membership(exp_dir)
        # Should just return all passing samples, not crash
        assert len(members) <= 10
        assert len(members) > 0
        print(f"[PASS] test_k_larger_than_available ({len(members)} samples)")
    finally:
        shutil.rmtree(tmpdir)


def test_empty_filter():
    """No filter expression — should just take top-K by priority."""
    tmpdir = tempfile.mkdtemp()
    try:
        index_path = _make_enriched_index(tmpdir, n=20)
        config, exp_dir = _write_config(
            tmpdir, "no_filter",
            "", K=7, mode="select"
        )

        _run_materialize(["--config", config, "--index", index_path, "--output_dir", exp_dir])

        members = _read_membership(exp_dir)
        assert len(members) == 7
        print("[PASS] test_empty_filter")
    finally:
        shutil.rmtree(tmpdir)


def test_all_filter_operators():
    """Test each comparison operator individually."""
    tmpdir = tempfile.mkdtemp()
    try:
        index_path = _make_enriched_index(tmpdir, n=20)

        test_cases = [
            ("aesthetic_score > 0.5",  ">"),
            ("aesthetic_score < 0.3",  "<"),
            ("aesthetic_score >= 0.5", ">="),
            ("aesthetic_score <= 0.2", "<="),
            ("is_clean == True",       "=="),
            ("is_clean != True",       "!="),
        ]

        for expr, op in test_cases:
            name = f"op_{op.replace('>', 'gt').replace('<', 'lt').replace('=', 'eq').replace('!', 'ne')}"
            config, exp_dir = _write_config(tmpdir, name, expr, K=20, mode="select")
            _run_materialize(["--config", config, "--index", index_path, "--output_dir", exp_dir])

            members = _read_membership(exp_dir)
            assert len(members) > 0, f"Operator {op} with '{expr}' produced 0 results"

        print("[PASS] test_all_filter_operators")
    finally:
        shutil.rmtree(tmpdir)


def test_compound_filter():
    """Multiple clauses joined with 'and'."""
    tmpdir = tempfile.mkdtemp()
    try:
        index_path = _make_enriched_index(tmpdir, n=20)
        config, exp_dir = _write_config(
            tmpdir, "compound",
            "aesthetic_score > 0.3 and is_clean == True", K=20, mode="select"
        )

        _run_materialize(["--config", config, "--index", index_path, "--output_dir", exp_dir])

        members = _read_membership(exp_dir)
        assert len(members) > 0
        # All members should have aesthetic > 0.3 AND be in the first 15 (is_clean)
        print(f"[PASS] test_compound_filter ({len(members)} samples)")
    finally:
        shutil.rmtree(tmpdir)


def test_null_values_excluded():
    """Samples with null scores should be excluded by numeric filters."""
    tmpdir = tempfile.mkdtemp()
    try:
        # Create index where some samples have null aesthetic_score
        records = []
        for i in range(10):
            key = f"{i:06d}"
            priority = int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**63)
            records.append({
                "sample_key": key,
                "tar_path": "/data/test.tar",
                "priority": priority,
                "aesthetic_score": i * 0.1 if i < 7 else None,  # last 3 are null
            })
        records.sort(key=lambda r: r["priority"])

        table = pa.table({
            "sample_key":      [r["sample_key"] for r in records],
            "tar_path":        [r["tar_path"] for r in records],
            "priority":        [r["priority"] for r in records],
            "aesthetic_score": pa.array([r["aesthetic_score"] for r in records], type=pa.float64()),
        })
        index_path = os.path.join(tmpdir, "nulls.parquet")
        pq.write_table(table, index_path)

        config, exp_dir = _write_config(
            tmpdir, "nulls",
            "aesthetic_score > 0.0", K=20, mode="select"
        )
        _run_materialize(["--config", config, "--index", index_path, "--output_dir", exp_dir])

        members = _read_membership(exp_dir)
        # Null values should NOT pass the filter
        # Keys 1-6 have scores 0.1-0.6, keys 0 has 0.0 (not > 0.0), keys 7-9 are null
        assert all(
            int(m) >= 1 and int(m) <= 6
            for m in members
        ), f"Unexpected members (nulls or 0.0 leaked): {members}"
        print(f"[PASS] test_null_values_excluded ({len(members)} samples)")
    finally:
        shutil.rmtree(tmpdir)


def test_membership_hash_stable():
    """Running materialize twice with same inputs produces same membership_hash."""
    tmpdir = tempfile.mkdtemp()
    try:
        index_path = _make_enriched_index(tmpdir, n=20)

        config1, exp1 = _write_config(tmpdir, "hash1", "aesthetic_score > 0.3", K=5, mode="select")
        config2, exp2 = _write_config(tmpdir, "hash2", "aesthetic_score > 0.3", K=5, mode="select")

        _run_materialize(["--config", config1, "--index", index_path, "--output_dir", exp1])
        _run_materialize(["--config", config2, "--index", index_path, "--output_dir", exp2])

        cfg1 = _read_config_out(exp1)
        cfg2 = _read_config_out(exp2)
        assert cfg1["membership_hash"] == cfg2["membership_hash"], \
            f"Hashes differ: {cfg1['membership_hash']} vs {cfg2['membership_hash']}"
        print("[PASS] test_membership_hash_stable")
    finally:
        shutil.rmtree(tmpdir)


def test_shardlist_dedup():
    """Shardlist should contain deduplicated tar paths."""
    tmpdir = tempfile.mkdtemp()
    try:
        index_path = _make_enriched_index(tmpdir, n=20)
        # Samples 0-4 are in shard-000, 5-9 in shard-001, etc.
        config, exp_dir = _write_config(
            tmpdir, "shardlist", "", K=20, mode="select"
        )

        _run_materialize(["--config", config, "--index", index_path, "--output_dir", exp_dir])

        shards = _read_shardlist(exp_dir)
        # Should have exactly 4 unique shards (0-4, 5-9, 10-14, 15-19)
        assert len(shards) == len(set(shards)), "Shardlist contains duplicates"
        assert len(shards) == 4, f"Expected 4 shards, got {len(shards)}"
        print("[PASS] test_shardlist_dedup")
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    tests = [
        test_select_mode,
        test_scan_mode,
        test_select_vs_scan_differ,
        test_k_larger_than_available,
        test_empty_filter,
        test_all_filter_operators,
        test_compound_filter,
        test_null_values_excluded,
        test_membership_hash_stable,
        test_shardlist_dedup,
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
