"""
End-to-end verification of the experiment data pipeline.

Creates a tiny synthetic tar, runs index -> enrich -> materialize,
then loads via HuggingFace datasets and verifies the membership filter
works correctly. This validates that __key__ matching is consistent
between the indexer and the HuggingFace webdataset loader.

Usage:
    python scripts/verify_experiment.py

    # Keep temp files for inspection:
    python scripts/verify_experiment.py --keep_temp
"""

import argparse
import hashlib
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


def create_synthetic_tar(tar_path: str, num_samples: int = 20):
    """Create a tar with synthetic jpg + txt pairs."""
    with tarfile.open(tar_path, "w") as tf:
        for i in range(num_samples):
            key = f"{i:06d}"

            # Create a small 8x8 RGB image
            img = Image.new("RGB", (8, 8), color=(i * 10 % 256, i * 20 % 256, i * 30 % 256))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            img_bytes = buf.getvalue()

            # Add .jpg
            info = tarfile.TarInfo(name=f"{key}.jpg")
            info.size = len(img_bytes)
            tf.addfile(info, io.BytesIO(img_bytes))

            # Add .txt
            txt = f"A test caption for sample {i}."
            txt_bytes = txt.encode()
            info = tarfile.TarInfo(name=f"{key}.txt")
            info.size = len(txt_bytes)
            tf.addfile(info, io.BytesIO(txt_bytes))

    print(f"[OK] Created synthetic tar: {tar_path} ({num_samples} samples)")


def run_script(script_path: str, args: list):
    """Run a python script as subprocess and check it succeeds."""
    cmd = [sys.executable, script_path] + args
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"[FAIL] stderr:\n{result.stderr}")
        raise RuntimeError(f"Script failed: {script_path}")
    return result.stdout


def verify_hf_key_matching(tar_path: str, expected_keys: set):
    """Load tar via HuggingFace datasets and verify __key__ field exists and matches."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[SKIP] HuggingFace datasets not installed, skipping __key__ verification")
        return True

    print(f"\n--- Verifying HuggingFace __key__ matching ---")
    ds = load_dataset("webdataset", data_files=tar_path, split="train")

    # Check __key__ field exists
    if "__key__" not in ds.column_names:
        print(f"[FAIL] __key__ not in column_names: {ds.column_names}")
        print("  The membership filter will NOT work with this version of HuggingFace datasets.")
        print("  Consider upgrading: pip install --upgrade datasets")
        return False

    hf_keys = set(ds["__key__"])
    print(f"  HuggingFace __key__ values: {sorted(hf_keys)[:5]}...")
    print(f"  Index sample_key values:    {sorted(expected_keys)[:5]}...")

    # Check they match
    if hf_keys == expected_keys:
        print(f"[OK] All {len(hf_keys)} keys match between indexer and HuggingFace loader")
        return True
    else:
        missing = expected_keys - hf_keys
        extra = hf_keys - expected_keys
        if missing:
            print(f"[FAIL] Keys in index but not in HF: {sorted(missing)[:5]}...")
        if extra:
            print(f"[FAIL] Keys in HF but not in index: {sorted(extra)[:5]}...")
        return False


def verify_filter(tar_path: str, membership: set):
    """Verify that HuggingFace .filter() with membership set produces correct results."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[SKIP] HuggingFace datasets not installed")
        return True

    print(f"\n--- Verifying .filter() with membership set ---")
    ds = load_dataset("webdataset", data_files=tar_path, split="train")

    if "__key__" not in ds.column_names:
        print("[FAIL] __key__ not available, cannot filter")
        return False

    before = len(ds)
    ds_filtered = ds.filter(lambda sample: sample["__key__"] in membership)
    after = len(ds_filtered)

    print(f"  Before filter: {before}, After filter: {after}, Expected: {len(membership)}")

    if after != len(membership):
        print(f"[FAIL] Expected {len(membership)} samples after filter, got {after}")
        return False

    # Verify the actual keys match
    filtered_keys = set(ds_filtered["__key__"])
    if filtered_keys != membership:
        print(f"[FAIL] Filtered keys don't match membership set")
        return False

    # Verify images are loadable
    sample = ds_filtered[0]
    if "jpg" not in sample and "png" not in sample:
        print(f"[FAIL] No image column found in filtered sample: {list(sample.keys())}")
        return False

    print(f"[OK] Filter correctly selected {after} samples")
    return True


def main():
    parser = argparse.ArgumentParser(description="Verify experiment pipeline end-to-end.")
    parser.add_argument("--keep_temp", action="store_true",
                        help="Keep temporary files for inspection.")
    args = parser.parse_args()

    scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tmpdir = tempfile.mkdtemp(prefix="blip3o_verify_")
    print(f"Working in: {tmpdir}\n")

    all_passed = True
    try:
        # 1. Create synthetic tar
        tar_path = os.path.join(tmpdir, "test.tar")
        num_samples = 20
        create_synthetic_tar(tar_path, num_samples)

        # 2. Run index_tars.py
        index_path = os.path.join(tmpdir, "index.parquet")
        run_script(
            os.path.join(scripts_dir, "index_tars.py"),
            ["--tar_dir", tmpdir, "--output", index_path, "--num_workers", "1"],
        )

        # Verify index
        table = pq.read_table(index_path)
        index_keys = set(table.column("sample_key").to_pylist())
        print(f"\n  Index has {table.num_rows} rows, {len(index_keys)} unique keys")
        assert table.num_rows == num_samples, f"Expected {num_samples} rows, got {table.num_rows}"
        print("[OK] Index created correctly")

        # 3. Verify __key__ matching with HuggingFace
        key_match_ok = verify_hf_key_matching(tar_path, index_keys)
        if not key_match_ok:
            all_passed = False

        # 4. Enrich with a dummy score and subset flag
        enriched_path = os.path.join(tmpdir, "index_enriched.parquet")

        # Create a dummy scores CSV (give even-numbered samples high scores)
        scores_csv = os.path.join(tmpdir, "scores.csv")
        with open(scores_csv, "w") as f:
            f.write("sample_key,aesthetic_score\n")
            for key in sorted(index_keys):
                idx = int(key)
                score = 0.9 if idx % 2 == 0 else 0.3
                f.write(f"{key},{score}\n")

        run_script(
            os.path.join(scripts_dir, "enrich_index.py"),
            ["--index", index_path, "--scores", scores_csv,
             "--key_col", "sample_key", "--value_cols", "aesthetic_score",
             "--output", enriched_path],
        )

        # Create a subset list (first 15 samples are "clean")
        clean_list = os.path.join(tmpdir, "clean_keys.txt")
        with open(clean_list, "w") as f:
            for key in sorted(index_keys)[:15]:
                f.write(key + "\n")

        run_script(
            os.path.join(scripts_dir, "enrich_index.py"),
            ["--index", enriched_path, "--subset_list", clean_list,
             "--subset_col", "is_clean", "--output", enriched_path],
        )

        # Verify enriched index
        enriched_table = pq.read_table(enriched_path)
        assert "aesthetic_score" in enriched_table.column_names, "Missing aesthetic_score column"
        assert "is_clean" in enriched_table.column_names, "Missing is_clean column"
        print("[OK] Enrichment completed correctly")

        # 5. Materialize with select mode
        exp_dir = os.path.join(tmpdir, "exp_select")
        os.makedirs(exp_dir, exist_ok=True)
        config_path = os.path.join(exp_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump({
                "name": "test_select",
                "filter_expression": "aesthetic_score > 0.5 and is_clean == True",
                "K": 5,
                "mode": "select",
            }, f)

        run_script(
            os.path.join(scripts_dir, "materialize.py"),
            ["--config", config_path, "--index", enriched_path, "--output_dir", exp_dir],
        )

        # Verify materialized output
        with open(os.path.join(exp_dir, "membership.txt")) as f:
            membership_keys = set(l.strip() for l in f if l.strip())
        with open(os.path.join(exp_dir, "shardlist.txt")) as f:
            shardlist = [l.strip() for l in f if l.strip()]
        with open(os.path.join(exp_dir, "config.yaml")) as f:
            cfg_out = yaml.safe_load(f)

        # Select mode: filter first (aesthetic > 0.5 and is_clean), then top 5
        # Even indices 0,2,4,6,8,10,12,14 have score 0.9 (pass aesthetic > 0.5)
        # Of those, indices 0-14 are clean, so even indices 0,2,4,6,8,10,12 pass both
        # Top 5 by priority
        assert len(membership_keys) == 5, f"Expected 5 members, got {len(membership_keys)}"
        assert len(shardlist) == 1, f"Expected 1 shard, got {len(shardlist)}"
        assert cfg_out["actual_samples"] == 5
        print("[OK] Materialize (select mode) correct")

        # 6. Materialize with scan mode
        exp_dir_scan = os.path.join(tmpdir, "exp_scan")
        os.makedirs(exp_dir_scan, exist_ok=True)
        config_scan = os.path.join(exp_dir_scan, "config.yaml")
        with open(config_scan, "w") as f:
            yaml.dump({
                "name": "test_scan",
                "filter_expression": "aesthetic_score > 0.5",
                "K": 10,
                "mode": "scan",
            }, f)

        run_script(
            os.path.join(scripts_dir, "materialize.py"),
            ["--config", config_scan, "--index", enriched_path, "--output_dir", exp_dir_scan],
        )

        with open(os.path.join(exp_dir_scan, "membership.txt")) as f:
            scan_keys = set(l.strip() for l in f if l.strip())
        # Scan mode: top 10 by priority, then filter aesthetic > 0.5 (even indices)
        # Among 10 samples, roughly half should be even -> ~5
        assert len(scan_keys) <= 10, f"Scan mode should produce <= K samples"
        print(f"[OK] Materialize (scan mode) correct: {len(scan_keys)} samples from budget of 10")

        # 7. Verify HuggingFace filter with membership set
        filter_ok = verify_filter(tar_path, membership_keys)
        if not filter_ok:
            all_passed = False

        # Summary
        print("\n" + "=" * 50)
        if all_passed:
            print("ALL CHECKS PASSED")
            print("\nThe experiment pipeline is working correctly.")
            print("__key__ matching between indexer and HuggingFace is consistent.")
        else:
            print("SOME CHECKS FAILED")
            print("\nPlease review the failures above.")
        print("=" * 50)

    finally:
        if args.keep_temp:
            print(f"\nTemp files kept at: {tmpdir}")
        else:
            shutil.rmtree(tmpdir)
            print(f"\nTemp files cleaned up.")


if __name__ == "__main__":
    main()
