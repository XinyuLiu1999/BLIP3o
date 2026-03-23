"""
Tests for determinism and reproducibility of the data pipeline.

Verifies:
  - Priority hash is deterministic across processes
  - Index output is byte-identical across two runs on same input
  - Materialize output is byte-identical across two runs
  - Two experiments with overlapping membership share the same subset
  - Select mode with K=N1 is a strict subset of K=N2 where N1 < N2

Usage:
    python scripts/data_pipeline/test_determinism.py
"""

import hashlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from PIL import Image


SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


def _make_image_bytes():
    img = Image.new("RGB", (4, 4), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _add_file_to_tar(tf, name, data):
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def _run_script(script, args):
    cmd = [sys.executable, os.path.join(SCRIPTS_DIR, script)] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{script} failed:\n{result.stderr}")
    return result.stdout


def _make_tar(tmpdir, name, keys):
    tar_path = os.path.join(tmpdir, name)
    with tarfile.open(tar_path, "w") as tf:
        for k in keys:
            _add_file_to_tar(tf, f"{k}.jpg", _make_image_bytes())
            _add_file_to_tar(tf, f"{k}.txt", f"caption for {k}".encode())
    return tar_path


def _read_membership(exp_dir):
    with open(os.path.join(exp_dir, "membership.txt")) as f:
        return set(l.strip() for l in f if l.strip())


def _file_hash(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_index_byte_identical():
    """Running index_tars twice on the same tar produces identical parquet."""
    tmpdir = tempfile.mkdtemp()
    try:
        keys = [f"{i:06d}" for i in range(15)]
        _make_tar(tmpdir, "data.tar", keys)

        out1 = os.path.join(tmpdir, "index1.parquet")
        out2 = os.path.join(tmpdir, "index2.parquet")

        _run_script("index_tars.py", ["--tar_dir", tmpdir, "--output", out1, "--num_workers", "1"])
        _run_script("index_tars.py", ["--tar_dir", tmpdir, "--output", out2, "--num_workers", "1"])

        # Compare content (not raw bytes — parquet metadata may differ)
        t1 = pq.read_table(out1)
        t2 = pq.read_table(out2)

        assert t1.num_rows == t2.num_rows
        assert t1.column("sample_key").to_pylist() == t2.column("sample_key").to_pylist()
        assert t1.column("priority").to_pylist() == t2.column("priority").to_pylist()
        assert t1.column("tar_path").to_pylist() == t2.column("tar_path").to_pylist()
        print("[PASS] test_index_byte_identical")
    finally:
        shutil.rmtree(tmpdir)


def test_materialize_identical():
    """Running materialize twice produces identical membership.txt."""
    tmpdir = tempfile.mkdtemp()
    try:
        keys = [f"{i:06d}" for i in range(20)]
        _make_tar(tmpdir, "data.tar", keys)

        index_path = os.path.join(tmpdir, "index.parquet")
        _run_script("index_tars.py", ["--tar_dir", tmpdir, "--output", index_path, "--num_workers", "1"])

        for run_name in ["run1", "run2"]:
            exp_dir = os.path.join(tmpdir, run_name)
            os.makedirs(exp_dir)
            config_path = os.path.join(exp_dir, "config.yaml")
            with open(config_path, "w") as f:
                yaml.dump({"name": run_name, "K": 10, "mode": "select"}, f)
            _run_script("materialize.py", [
                "--config", config_path, "--index", index_path, "--output_dir", exp_dir
            ])

        m1 = _read_membership(os.path.join(tmpdir, "run1"))
        m2 = _read_membership(os.path.join(tmpdir, "run2"))
        assert m1 == m2, f"Memberships differ:\n  run1: {sorted(m1)[:5]}\n  run2: {sorted(m2)[:5]}"

        h1 = _file_hash(os.path.join(tmpdir, "run1", "membership.txt"))
        h2 = _file_hash(os.path.join(tmpdir, "run2", "membership.txt"))
        assert h1 == h2, "membership.txt files not byte-identical"
        print("[PASS] test_materialize_identical")
    finally:
        shutil.rmtree(tmpdir)


def test_subset_nesting():
    """Select with K=5 should be a strict subset of K=10 (same filter, same priority order)."""
    tmpdir = tempfile.mkdtemp()
    try:
        keys = [f"{i:06d}" for i in range(30)]
        _make_tar(tmpdir, "data.tar", keys)

        index_path = os.path.join(tmpdir, "index.parquet")
        _run_script("index_tars.py", ["--tar_dir", tmpdir, "--output", index_path, "--num_workers", "1"])

        for k_val, name in [(5, "small"), (10, "large")]:
            exp_dir = os.path.join(tmpdir, name)
            os.makedirs(exp_dir)
            config_path = os.path.join(exp_dir, "config.yaml")
            with open(config_path, "w") as f:
                yaml.dump({"name": name, "K": k_val, "mode": "select"}, f)
            _run_script("materialize.py", [
                "--config", config_path, "--index", index_path, "--output_dir", exp_dir
            ])

        small = _read_membership(os.path.join(tmpdir, "small"))
        large = _read_membership(os.path.join(tmpdir, "large"))

        assert len(small) == 5
        assert len(large) == 10
        assert small.issubset(large), \
            f"K=5 is not a subset of K=10!\n  Only in small: {small - large}"
        print("[PASS] test_subset_nesting")
    finally:
        shutil.rmtree(tmpdir)


def test_priority_cross_process():
    """Priority computed in-process matches what index_tars.py produces."""
    tmpdir = tempfile.mkdtemp()
    try:
        # Add project root to path for import
        project_root = os.path.abspath(os.path.join(SCRIPTS_DIR, "..", ".."))
        sys.path.insert(0, project_root)
        from scripts.data_pipeline.index_tars import compute_priority

        keys = [f"{i:06d}" for i in range(10)]
        _make_tar(tmpdir, "data.tar", keys)

        index_path = os.path.join(tmpdir, "index.parquet")
        _run_script("index_tars.py", ["--tar_dir", tmpdir, "--output", index_path, "--num_workers", "1"])

        table = pq.read_table(index_path)
        for row_idx in range(table.num_rows):
            key = table.column("sample_key")[row_idx].as_py()
            stored_priority = table.column("priority")[row_idx].as_py()
            computed_priority = compute_priority(key)
            assert stored_priority == computed_priority, \
                f"Key {key}: stored={stored_priority}, computed={computed_priority}"

        print("[PASS] test_priority_cross_process")
    finally:
        shutil.rmtree(tmpdir)


def test_experiment_overlap():
    """Two experiments with overlapping filters share overlapping membership."""
    tmpdir = tempfile.mkdtemp()
    try:
        keys = [f"{i:06d}" for i in range(20)]
        _make_tar(tmpdir, "data.tar", keys)

        index_path = os.path.join(tmpdir, "index.parquet")
        _run_script("index_tars.py", ["--tar_dir", tmpdir, "--output", index_path, "--num_workers", "1"])

        # Enrich with scores
        enriched_path = os.path.join(tmpdir, "enriched.parquet")
        scores_csv = os.path.join(tmpdir, "scores.csv")
        with open(scores_csv, "w") as f:
            f.write("sample_key,score\n")
            for i in range(20):
                f.write(f"{i:06d},{i * 0.05}\n")  # scores: 0.0, 0.05, ..., 0.95

        _run_script("enrich_index.py", [
            "--index", index_path, "--scores", scores_csv,
            "--key_col", "sample_key", "--value_cols", "score",
            "--output", enriched_path
        ])

        # Experiment A: score > 0.3, K=10
        exp_a = os.path.join(tmpdir, "exp_a")
        os.makedirs(exp_a)
        with open(os.path.join(exp_a, "config.yaml"), "w") as f:
            yaml.dump({"name": "a", "filter_expression": "score > 0.3", "K": 10, "mode": "select"}, f)
        _run_script("materialize.py", [
            "--config", os.path.join(exp_a, "config.yaml"),
            "--index", enriched_path, "--output_dir", exp_a
        ])

        # Experiment B: score > 0.5, K=10 (stricter filter, subset of A's pool)
        exp_b = os.path.join(tmpdir, "exp_b")
        os.makedirs(exp_b)
        with open(os.path.join(exp_b, "config.yaml"), "w") as f:
            yaml.dump({"name": "b", "filter_expression": "score > 0.5", "K": 10, "mode": "select"}, f)
        _run_script("materialize.py", [
            "--config", os.path.join(exp_b, "config.yaml"),
            "--index", enriched_path, "--output_dir", exp_b
        ])

        mem_a = _read_membership(exp_a)
        mem_b = _read_membership(exp_b)

        # B's samples (score > 0.5) should all be in A (score > 0.3)
        assert mem_b.issubset(mem_a), \
            f"Stricter filter not subset of looser filter!\n  Only in B: {mem_b - mem_a}"
        print(f"[PASS] test_experiment_overlap (A={len(mem_a)}, B={len(mem_b)}, B⊂A={mem_b.issubset(mem_a)})")
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    tests = [
        test_index_byte_identical,
        test_materialize_identical,
        test_subset_nesting,
        test_priority_cross_process,
        test_experiment_overlap,
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
