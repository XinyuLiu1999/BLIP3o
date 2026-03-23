"""
Integration test for the experiment_dir loading path in LazySupervisedMixDataset.

Creates a synthetic tar, runs the full pipeline (index -> enrich -> materialize),
then loads the dataset via the same code path that training uses, and verifies:
  - Correct number of samples after filtering
  - __getitem__ returns valid data dicts with expected keys
  - Images are loadable PIL objects
  - Text captions are present
  - The "type" column is correctly set to "T2I"

This test requires HuggingFace datasets to be installed.

Usage:
    python scripts/data_pipeline/test_dataset_integration.py
"""

import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile

import pyarrow.parquet as pq
import yaml
from PIL import Image


SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


def _make_image_bytes():
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
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


def test_dataset_loading_with_experiment_dir():
    """Full integration: create tar -> index -> enrich -> materialize -> load dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[SKIP] HuggingFace datasets not installed")
        return

    tmpdir = tempfile.mkdtemp(prefix="blip3o_ds_test_")
    try:
        # 1. Create synthetic tar with 10 samples
        tar_path = os.path.join(tmpdir, "test_data.tar")
        num_samples = 10
        captions = {}
        with tarfile.open(tar_path, "w") as tf:
            for i in range(num_samples):
                key = f"{i:06d}"
                _add_file_to_tar(tf, f"{key}.jpg", _make_image_bytes())
                caption = f"A beautiful landscape scene number {i}"
                _add_file_to_tar(tf, f"{key}.txt", caption.encode())
                captions[key] = caption

        print(f"  Created tar with {num_samples} samples")

        # 2. Index
        index_path = os.path.join(tmpdir, "index.parquet")
        _run_script("index_tars.py", [
            "--tar_dir", tmpdir, "--output", index_path, "--num_workers", "1"
        ])

        # 3. Enrich with scores
        scores_csv = os.path.join(tmpdir, "scores.csv")
        with open(scores_csv, "w") as f:
            f.write("sample_key,aesthetic_score\n")
            for i in range(num_samples):
                f.write(f"{i:06d},{0.9 if i < 6 else 0.2}\n")

        enriched_path = os.path.join(tmpdir, "enriched.parquet")
        _run_script("enrich_index.py", [
            "--index", index_path, "--scores", scores_csv,
            "--key_col", "sample_key", "--value_cols", "aesthetic_score",
            "--output", enriched_path
        ])

        # 4. Materialize: select top 4 with aesthetic > 0.5
        exp_dir = os.path.join(tmpdir, "experiment")
        os.makedirs(exp_dir)
        config_path = os.path.join(exp_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump({
                "name": "integration_test",
                "filter_expression": "aesthetic_score > 0.5",
                "K": 4,
                "mode": "select",
            }, f)

        _run_script("materialize.py", [
            "--config", config_path, "--index", enriched_path,
            "--output_dir", exp_dir
        ])

        # Read membership for verification
        with open(os.path.join(exp_dir, "membership.txt")) as f:
            membership = set(l.strip() for l in f if l.strip())
        assert len(membership) == 4, f"Expected 4 members, got {len(membership)}"
        print(f"  Materialized {len(membership)} samples")

        # 5. Load via HuggingFace datasets (same path as LazySupervisedMixDataset)
        with open(os.path.join(exp_dir, "shardlist.txt")) as f:
            shards = [l.strip() for l in f if l.strip()]

        ds = load_dataset("webdataset", data_files=shards, split="train")
        print(f"  Loaded dataset: {len(ds)} samples, columns: {ds.column_names}")

        # Check __key__ exists
        assert "__key__" in ds.column_names, \
            f"__key__ not in columns: {ds.column_names}. Cannot filter by membership."

        # Filter
        ds_filtered = ds.filter(lambda s: s["__key__"] in membership)
        assert len(ds_filtered) == 4, \
            f"Expected 4 after filter, got {len(ds_filtered)}"

        # Rename and clean columns (same as dataset.py experiment path)
        if "jpg" in ds_filtered.column_names:
            ds_filtered = ds_filtered.rename_column("jpg", "image")
        elif "png" in ds_filtered.column_names:
            ds_filtered = ds_filtered.rename_column("png", "image")
        ds_filtered = ds_filtered.add_column("type", ["T2I"] * len(ds_filtered))
        ds_filtered = ds_filtered.remove_columns(
            [c for c in ds_filtered.column_names if c not in ("image", "txt", "type")]
        )

        # 6. Verify each sample
        for i in range(len(ds_filtered)):
            sample = ds_filtered[i]

            # Has required fields
            assert "image" in sample, f"Sample {i} missing 'image'"
            assert "txt" in sample, f"Sample {i} missing 'txt'"
            assert "type" in sample, f"Sample {i} missing 'type'"

            # Type is T2I
            assert sample["type"] == "T2I", f"Sample {i} type is {sample['type']}"

            # Image is a PIL Image
            img = sample["image"]
            assert isinstance(img, Image.Image), f"Sample {i} image is {type(img)}"
            assert img.mode == "RGB", f"Sample {i} image mode is {img.mode}"

            # Caption is non-empty string
            assert isinstance(sample["txt"], str), f"Sample {i} txt is {type(sample['txt'])}"
            assert len(sample["txt"]) > 0, f"Sample {i} has empty caption"

        print(f"  All {len(ds_filtered)} samples valid")
        print("[PASS] test_dataset_loading_with_experiment_dir")

    finally:
        shutil.rmtree(tmpdir)


def test_empty_membership():
    """What happens when membership is empty (filter too strict)."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[SKIP] HuggingFace datasets not installed")
        return

    tmpdir = tempfile.mkdtemp(prefix="blip3o_empty_")
    try:
        # Create tar
        tar_path = os.path.join(tmpdir, "data.tar")
        with tarfile.open(tar_path, "w") as tf:
            for i in range(5):
                _add_file_to_tar(tf, f"{i:04d}.jpg", _make_image_bytes())
                _add_file_to_tar(tf, f"{i:04d}.txt", b"caption")

        # Load and filter with an empty membership set
        ds = load_dataset("webdataset", data_files=tar_path, split="train")
        empty_membership = set()
        ds_filtered = ds.filter(lambda s: s["__key__"] in empty_membership)

        assert len(ds_filtered) == 0, f"Expected 0, got {len(ds_filtered)}"
        print("[PASS] test_empty_membership")
    finally:
        shutil.rmtree(tmpdir)


def test_full_membership():
    """All samples in membership — nothing should be filtered out."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[SKIP] HuggingFace datasets not installed")
        return

    tmpdir = tempfile.mkdtemp(prefix="blip3o_full_")
    try:
        tar_path = os.path.join(tmpdir, "data.tar")
        num_samples = 8
        with tarfile.open(tar_path, "w") as tf:
            for i in range(num_samples):
                _add_file_to_tar(tf, f"{i:04d}.jpg", _make_image_bytes())
                _add_file_to_tar(tf, f"{i:04d}.txt", b"caption")

        ds = load_dataset("webdataset", data_files=tar_path, split="train")

        # All keys in membership
        all_keys = set(ds["__key__"])
        ds_filtered = ds.filter(lambda s: s["__key__"] in all_keys)

        assert len(ds_filtered) == num_samples, \
            f"Expected {num_samples}, got {len(ds_filtered)}"
        print("[PASS] test_full_membership")
    finally:
        shutil.rmtree(tmpdir)


def test_png_images():
    """Verify PNG tars work through the full pipeline."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[SKIP] HuggingFace datasets not installed")
        return

    tmpdir = tempfile.mkdtemp(prefix="blip3o_png_")
    try:
        tar_path = os.path.join(tmpdir, "pngs.tar")
        img = Image.new("RGB", (32, 32), color=(50, 100, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        with tarfile.open(tar_path, "w") as tf:
            for i in range(3):
                _add_file_to_tar(tf, f"{i:04d}.png", png_bytes)
                _add_file_to_tar(tf, f"{i:04d}.txt", b"png caption")

        ds = load_dataset("webdataset", data_files=tar_path, split="train")

        # Verify PNG column name
        assert "png" in ds.column_names, f"No 'png' column: {ds.column_names}"

        # Simulate dataset.py logic
        membership = set(ds["__key__"])
        ds_filtered = ds.filter(lambda s: s["__key__"] in membership)
        ds_filtered = ds_filtered.rename_column("png", "image")

        for i in range(len(ds_filtered)):
            img = ds_filtered[i]["image"]
            assert isinstance(img, Image.Image)

        print("[PASS] test_png_images")
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    tests = [
        test_dataset_loading_with_experiment_dir,
        test_empty_membership,
        test_full_membership,
        test_png_images,
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
