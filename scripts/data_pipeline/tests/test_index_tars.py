"""
Tests for index_tars.py covering:
  - Multiple tar files
  - Nested directory keys (e.g., subdir/000001.jpg)
  - PNG-only tars
  - Mixed extensions within a sample (.jpg, .txt, .json)
  - Deduplication of same key across tars
  - Empty tar
  - tar_list input mode
  - Priority determinism and uniqueness

Usage:
    python scripts/data_pipeline/tests/test_index_tars.py
"""

import hashlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile

import pyarrow.parquet as pq
from PIL import Image


SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_SCRIPT = os.path.join(SCRIPTS_DIR, "index_tars.py")


def _make_image_bytes(fmt="JPEG"):
    img = Image.new("RGB", (4, 4), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _add_file_to_tar(tf, name, data):
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def _run_index(args):
    cmd = [sys.executable, INDEX_SCRIPT] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"index_tars.py failed:\n{result.stderr}")
    return result.stdout


def test_multiple_tars():
    """Index across two separate tar files, verify all samples found."""
    tmpdir = tempfile.mkdtemp()
    try:
        # Tar A: 5 samples
        tar_a = os.path.join(tmpdir, "a.tar")
        with tarfile.open(tar_a, "w") as tf:
            for i in range(5):
                _add_file_to_tar(tf, f"{i:04d}.jpg", _make_image_bytes())
                _add_file_to_tar(tf, f"{i:04d}.txt", b"caption A")

        # Tar B: 3 samples with different key range
        tar_b = os.path.join(tmpdir, "b.tar")
        with tarfile.open(tar_b, "w") as tf:
            for i in range(10, 13):
                _add_file_to_tar(tf, f"{i:04d}.jpg", _make_image_bytes())
                _add_file_to_tar(tf, f"{i:04d}.txt", b"caption B")

        out = os.path.join(tmpdir, "index.parquet")
        _run_index(["--tar_dir", tmpdir, "--output", out, "--num_workers", "1"])

        table = pq.read_table(out)
        assert table.num_rows == 8, f"Expected 8 rows, got {table.num_rows}"

        # Verify tar_path is correctly recorded for each sample
        tar_paths = set(table.column("tar_path").to_pylist())
        assert tar_a in tar_paths, f"tar_a not found in tar_paths"
        assert tar_b in tar_paths, f"tar_b not found in tar_paths"

        keys = set(table.column("sample_key").to_pylist())
        assert len(keys) == 8
        print("[PASS] test_multiple_tars")
    finally:
        shutil.rmtree(tmpdir)


def test_nested_keys():
    """Keys with subdirectory prefixes like 'subdir/000001'."""
    tmpdir = tempfile.mkdtemp()
    try:
        tar_path = os.path.join(tmpdir, "nested.tar")
        with tarfile.open(tar_path, "w") as tf:
            for i in range(3):
                _add_file_to_tar(tf, f"shard-001/{i:06d}.jpg", _make_image_bytes())
                _add_file_to_tar(tf, f"shard-001/{i:06d}.txt", b"nested caption")

        out = os.path.join(tmpdir, "index.parquet")
        _run_index(["--tar_dir", tmpdir, "--output", out, "--num_workers", "1"])

        table = pq.read_table(out)
        assert table.num_rows == 3
        keys = table.column("sample_key").to_pylist()
        # Keys should preserve the subdirectory prefix
        assert all(k.startswith("shard-001/") for k in keys), f"Keys missing prefix: {keys}"
        print("[PASS] test_nested_keys")
    finally:
        shutil.rmtree(tmpdir)


def test_png_tar():
    """Tar containing .png images instead of .jpg."""
    tmpdir = tempfile.mkdtemp()
    try:
        tar_path = os.path.join(tmpdir, "pngs.tar")
        with tarfile.open(tar_path, "w") as tf:
            for i in range(4):
                _add_file_to_tar(tf, f"{i:04d}.png", _make_image_bytes("PNG"))
                _add_file_to_tar(tf, f"{i:04d}.txt", b"png caption")

        out = os.path.join(tmpdir, "index.parquet")
        _run_index(["--tar_dir", tmpdir, "--output", out, "--num_workers", "1"])

        table = pq.read_table(out)
        assert table.num_rows == 4
        print("[PASS] test_png_tar")
    finally:
        shutil.rmtree(tmpdir)


def test_mixed_extensions():
    """Sample with multiple file extensions (.jpg, .txt, .json) should produce one key."""
    tmpdir = tempfile.mkdtemp()
    try:
        tar_path = os.path.join(tmpdir, "mixed.tar")
        with tarfile.open(tar_path, "w") as tf:
            _add_file_to_tar(tf, "sample_001.jpg", _make_image_bytes())
            _add_file_to_tar(tf, "sample_001.txt", b"caption")
            _add_file_to_tar(tf, "sample_001.json", b'{"meta": "data"}')
            _add_file_to_tar(tf, "sample_002.jpg", _make_image_bytes())
            _add_file_to_tar(tf, "sample_002.txt", b"caption 2")

        out = os.path.join(tmpdir, "index.parquet")
        _run_index(["--tar_dir", tmpdir, "--output", out, "--num_workers", "1"])

        table = pq.read_table(out)
        # Should be 2 unique samples, not 5 files
        assert table.num_rows == 2, f"Expected 2 rows, got {table.num_rows}"
        print("[PASS] test_mixed_extensions")
    finally:
        shutil.rmtree(tmpdir)


def test_dedup_across_tars():
    """Same sample_key in two different tars should be deduplicated."""
    tmpdir = tempfile.mkdtemp()
    try:
        # Both tars contain key "000000"
        for name in ["dup_a.tar", "dup_b.tar"]:
            tar_path = os.path.join(tmpdir, name)
            with tarfile.open(tar_path, "w") as tf:
                _add_file_to_tar(tf, "000000.jpg", _make_image_bytes())
                _add_file_to_tar(tf, "000000.txt", b"dup caption")

        out = os.path.join(tmpdir, "index.parquet")
        _run_index(["--tar_dir", tmpdir, "--output", out, "--num_workers", "1"])

        table = pq.read_table(out)
        assert table.num_rows == 1, f"Expected 1 row after dedup, got {table.num_rows}"
        print("[PASS] test_dedup_across_tars")
    finally:
        shutil.rmtree(tmpdir)


def test_empty_tar():
    """An empty tar should not crash and contribute 0 rows."""
    tmpdir = tempfile.mkdtemp()
    try:
        tar_path = os.path.join(tmpdir, "empty.tar")
        with tarfile.open(tar_path, "w") as tf:
            pass  # empty

        # Also add a non-empty tar so we get some output
        tar_path2 = os.path.join(tmpdir, "nonempty.tar")
        with tarfile.open(tar_path2, "w") as tf:
            _add_file_to_tar(tf, "000000.jpg", _make_image_bytes())
            _add_file_to_tar(tf, "000000.txt", b"caption")

        out = os.path.join(tmpdir, "index.parquet")
        _run_index(["--tar_dir", tmpdir, "--output", out, "--num_workers", "1"])

        table = pq.read_table(out)
        assert table.num_rows == 1
        print("[PASS] test_empty_tar")
    finally:
        shutil.rmtree(tmpdir)


def test_tar_list_input():
    """Use --tar_list instead of --tar_dir."""
    tmpdir = tempfile.mkdtemp()
    try:
        tar_path = os.path.join(tmpdir, "data.tar")
        with tarfile.open(tar_path, "w") as tf:
            for i in range(3):
                _add_file_to_tar(tf, f"{i:04d}.jpg", _make_image_bytes())
                _add_file_to_tar(tf, f"{i:04d}.txt", b"caption")

        tar_list_file = os.path.join(tmpdir, "tars.txt")
        with open(tar_list_file, "w") as f:
            f.write(tar_path + "\n")

        out = os.path.join(tmpdir, "index.parquet")
        _run_index(["--tar_list", tar_list_file, "--output", out, "--num_workers", "1"])

        table = pq.read_table(out)
        assert table.num_rows == 3
        print("[PASS] test_tar_list_input")
    finally:
        shutil.rmtree(tmpdir)


def test_priority_determinism():
    """Same key must always produce the same priority, across separate runs."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("index_tars", os.path.join(SCRIPTS_DIR, "index_tars.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    compute_priority = mod.compute_priority

    key = "shard-042/test_sample_12345"
    p1 = compute_priority(key)
    p2 = compute_priority(key)
    assert p1 == p2, f"Priority not deterministic: {p1} != {p2}"

    # Different keys should (almost certainly) produce different priorities
    other = compute_priority("different_key")
    assert p1 != other, "Two different keys have identical priority (astronomically unlikely)"
    print("[PASS] test_priority_determinism")


def test_priority_sorted():
    """Output parquet should be sorted by priority ascending."""
    tmpdir = tempfile.mkdtemp()
    try:
        tar_path = os.path.join(tmpdir, "sort.tar")
        with tarfile.open(tar_path, "w") as tf:
            for i in range(50):
                _add_file_to_tar(tf, f"{i:06d}.jpg", _make_image_bytes())
                _add_file_to_tar(tf, f"{i:06d}.txt", b"caption")

        out = os.path.join(tmpdir, "index.parquet")
        _run_index(["--tar_dir", tmpdir, "--output", out, "--num_workers", "1"])

        table = pq.read_table(out)
        priorities = table.column("priority").to_pylist()
        assert priorities == sorted(priorities), "Output not sorted by priority"
        print("[PASS] test_priority_sorted")
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    tests = [
        test_multiple_tars,
        test_nested_keys,
        test_png_tar,
        test_mixed_extensions,
        test_dedup_across_tars,
        test_empty_tar,
        test_tar_list_input,
        test_priority_determinism,
        test_priority_sorted,
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
