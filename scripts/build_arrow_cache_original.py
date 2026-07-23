#!/usr/bin/env python3
# works for original BLIP3o pretrain without extra json fields. 
import argparse
import glob
import json as json_lib
import os

from datasets import load_dataset, concatenate_datasets, Features, Value, Image

# base_features = Features({
#     "__key__": Value("string"),
#     "jpg": Image(),
#     "txt": Value("string"),
#     "json": Value("string"),
# })


# def extract_fields(example):
#     if isinstance(example["json"], str):
#         meta = json_lib.loads(example["json"])
#     elif isinstance(example["json"], dict):
#         meta = example["json"]
#     else:
#         meta = {}
#     example["json"] = json_lib.dumps(meta)  # 统一为字符串
#     example["tagging_caption"] = meta.get("tagging_caption", "")
#     example["short_caption"] = meta.get("short_caption", "")
#     example["medium_caption"] = meta.get("medium_caption", "")
#     example["long_caption"] = meta.get("long_caption", "")
#     example["ocr_text"] = meta.get("ocr_text", "")
#     example["long_text"] = meta.get("long_text", False)
#     example["has_artistic_text"] = meta.get("has_artistic_text", False)
#     example["has_watermark"] = meta.get("has_watermark", False)
#     return example


# def normalize_to_base(ds):
#     """只保留 base 四列，json 强制转为字符串"""
#     # 删除多余列（如 __url__、pkl 展开的列等）
#     cols_to_remove = [c for c in ds.column_names if c not in ("__key__", "jpg", "txt", "json")]
#     if cols_to_remove:
#         ds = ds.remove_columns(cols_to_remove)

#     # 如果 json 被解析成了 dict，转回字符串
#     if not isinstance(ds.features["json"], Value):
#         ds = ds.map(
#             lambda x: {"json": json_lib.dumps(x["json"]) if isinstance(x["json"], dict) else x["json"]},
#             num_proc=16,
#         )
#         ds = ds.cast_column("json", Value("string"))

#     return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--data_cache_dir", required=True)
    parser.add_argument("--map_num_proc", type=int, default=16)
    args = parser.parse_args()

    all_shards = sorted(glob.glob(os.path.join(args.data_dir, "*.tar")))
    assert len(all_shards) > 0, f"No tar files found in {args.data_dir}"

    has_pkl = [s for s in all_shards if os.path.basename(s).startswith(("sa_", "webdataset_shard_"))]
    no_pkl = [s for s in all_shards if os.path.basename(s).startswith("shard-")]

    covered = set(has_pkl + no_pkl)
    missed = [s for s in all_shards if s not in covered]
    if missed:
        print(f"WARNING: {len(missed)} unclassified: {[os.path.basename(f) for f in missed[:10]]}")

    print(f"Found {len(has_pkl)} with pkl, {len(no_pkl)} without pkl, total {len(all_shards)}")

    datasets = []
    for name, files in [("has_pkl", has_pkl), ("no_pkl", no_pkl)]:
        if not files:
            continue
        print(f"\nLoading '{name}' ({len(files)} files)...")
        ds = load_dataset(
            "webdataset",
            data_files=files,
            split="train",
            num_proc=1,
            cache_dir=args.data_cache_dir,
        )
        print(f"  Raw columns: {ds.column_names}")

        # 只保留三列
        cols_to_remove = [c for c in ds.column_names if c not in ("__key__", "jpg", "txt")]
        if cols_to_remove:
            ds = ds.remove_columns(cols_to_remove)

        print(f"  {name}: {len(ds)} samples")
        datasets.append(ds)

    print("\nConcatenating...")
    ds = concatenate_datasets(datasets)

    print(f"\nDone: {len(ds)} samples, columns={ds.column_names}")

    final_dir = os.path.join(args.data_cache_dir, "unified_arrow_cache")
    print(f"Saving to {final_dir}...")
    ds.save_to_disk(final_dir, num_proc=args.map_num_proc)
    print(f"\nUse this for --data_arrow_dir:\n  {final_dir}")


if __name__ == "__main__":
    main()