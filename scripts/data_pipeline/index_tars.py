"""
Build a parquet index of all samples across webdataset tar files.

For each sample, records sample_key (the shared filename stem within a tar),
tar_path, and a deterministic priority hash. Output is sorted by priority.

Usage:
    python scripts/index_tars.py \
        --tar_dir /fsx/data/blip3o_tars \
        --output index.parquet \
        --num_workers 32

    # Or provide an explicit list of tar paths:
    python scripts/index_tars.py \
        --tar_list tar_paths.txt \
        --output index.parquet
"""

import argparse
import hashlib
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def compute_priority(sample_key: str) -> int:
    """Deterministic hash used to rank samples consistently across experiments."""
    return int(hashlib.sha256(sample_key.encode()).hexdigest(), 16) % (2**63)


def scan_single_tar(tar_path: str) -> list:
    """Extract unique sample keys from a single tar file.

    In webdataset tars, files belonging to the same sample share the same
    stem (e.g. 000042.jpg and 000042.txt both have key "000042").
    """
    keys_seen = set()
    records = []
    try:
        with tarfile.open(tar_path, "r") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                sample_key = os.path.splitext(member.name)[0]
                if sample_key not in keys_seen:
                    keys_seen.add(sample_key)
                    records.append({
                        "sample_key": sample_key,
                        "tar_path": tar_path,
                        "priority": compute_priority(sample_key),
                    })
    except Exception as e:
        print(f"[WARN] Error scanning {tar_path}: {e}")
    return records


def main():
    parser = argparse.ArgumentParser(description="Index webdataset tar files into a parquet catalog.")
    parser.add_argument("--tar_dir", type=str, default=None,
                        help="Directory to recursively search for .tar files.")
    parser.add_argument("--tar_list", type=str, default=None,
                        help="Text file with one tar path per line.")
    parser.add_argument("--output", type=str, default="index.parquet",
                        help="Output parquet path.")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of parallel workers for scanning.")
    args = parser.parse_args()

    if args.tar_list:
        with open(args.tar_list) as f:
            tar_paths = [l.strip() for l in f if l.strip()]
    elif args.tar_dir:
        tar_paths = sorted(str(p) for p in Path(args.tar_dir).rglob("*.tar"))
    else:
        parser.error("Provide either --tar_dir or --tar_list.")

    print(f"Found {len(tar_paths)} tar files")

    all_records = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for i, records in enumerate(executor.map(scan_single_tar, tar_paths)):
            all_records.extend(records)
            if (i + 1) % 100 == 0:
                print(f"  scanned {i + 1}/{len(tar_paths)} tars, "
                      f"{len(all_records)} samples so far")

    # Deduplicate by sample_key (defensive — keys should be unique across tars)
    seen = set()
    deduped = []
    for r in all_records:
        if r["sample_key"] not in seen:
            seen.add(r["sample_key"])
            deduped.append(r)
    if len(deduped) < len(all_records):
        print(f"  deduplicated: {len(all_records)} -> {len(deduped)}")

    # Sort by priority
    deduped.sort(key=lambda r: r["priority"])

    table = pa.table({
        "sample_key": [r["sample_key"] for r in deduped],
        "tar_path":   [r["tar_path"]   for r in deduped],
        "priority":   [r["priority"]   for r in deduped],
    })
    pq.write_table(table, args.output)
    print(f"Index written: {args.output} ({len(deduped)} samples)")


if __name__ == "__main__":
    main()
