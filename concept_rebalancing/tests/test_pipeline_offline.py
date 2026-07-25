"""Offline integration test for Stages 2-5 (no GPU / no corpus).

Builds a tiny synthetic ``index.parquet`` + ``links.parquet``, runs
``build_schedule.py`` then ``materialize_rebalanced.py`` as subprocesses, and
checks the outputs against an independent recomputation from the rebalance
primitives:

- N_c counts, per-concept m_c (head / mid / tail branches all exercised),
- rarest-wins per-sample m (incl. the no-concept -> 1.0 sample),
- realized integer counts via the shared hash rule,
- ``actual_samples`` == Sum count and membership determinism.

    python concept_rebalancing/tests/test_pipeline_offline.py
"""

import math
import os
import shutil
import subprocess
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from rebalance.multiplicity import compute_priority, priority_to_hash01, realize_count
from rebalance.schedule import ScheduleConfig, concept_multiplicity, sample_multiplicity

# Tiny thresholds so a handful of samples exercises head/mid/tail.
N_HIGH = 5.0
N_LOW = 2.0
A_HEAD = math.log10(N_HIGH) / 2.0   # keeps continuity: m_c(N_HIGH) == 1
CFG = ScheduleConfig(n_high=N_HIGH, n_low=N_LOW, a_head=A_HEAD)

# sample -> set of concept nodes
SAMPLE_CONCEPTS = {
    "s1": ["H", "M"],
    "s2": ["H", "M"],
    "s3": ["H"],
    "s4": ["H"],
    "s5": ["H"],
    "s6": ["T"],
    "s7": [],          # no content concept -> m = 1.0
}
# Resulting N_c:  H=5 (head, m<1... actually m=1 at N=5, m<1 at N>5),
#                 M=2 (mid, m=1), T=1 (tail, oversample)


def _write_inputs(d):
    # links.parquet
    keys, nodes = [], []
    for s, cs in SAMPLE_CONCEPTS.items():
        for c in cs:
            keys.append(s)
            nodes.append(c)
    pq.write_table(pa.table({
        "sample_key": pa.array(keys, pa.string()),
        "node_id": pa.array(nodes, pa.string()),
    }), os.path.join(d, "links.parquet"))

    # index.parquet with the real priority for each key
    all_keys = list(SAMPLE_CONCEPTS.keys())
    pq.write_table(pa.table({
        "sample_key": pa.array(all_keys, pa.string()),
        "tar_path": pa.array([f"/data/shard-{i%2}.tar" for i in range(len(all_keys))], pa.string()),
        "priority": pa.array([compute_priority(k) for k in all_keys], pa.int64()),
    }), os.path.join(d, "index.parquet"))


def _expected():
    counts = {"H": 5, "M": 2, "T": 1}
    node_m = {n: concept_multiplicity(v, CFG) for n, v in counts.items()}
    sample_m = {}
    for s, cs in SAMPLE_CONCEPTS.items():
        sample_m[s] = sample_multiplicity([node_m[c] for c in cs])
    exp_count = {}
    for s, m in sample_m.items():
        h = priority_to_hash01(compute_priority(s))
        exp_count[s] = realize_count(m, h)
    return counts, node_m, sample_m, exp_count


def main():
    d = tempfile.mkdtemp(prefix="rebal_test_")
    try:
        _write_inputs(d)
        counts, node_m, sample_m, exp_count = _expected()

        # ---- Stage 2-4 ----
        exp_dir = os.path.join(d, "exp")
        subprocess.run([
            sys.executable, os.path.join(_PKG, "build_schedule.py"),
            "--links", os.path.join(d, "links.parquet"),
            "--index", os.path.join(d, "index.parquet"),
            "--output_dir", d,
            "--n_high", str(N_HIGH), "--n_low", str(N_LOW), "--a_head", str(A_HEAD),
        ], check=True)

        # verify counts.parquet
        ct = pq.read_table(os.path.join(d, "counts.parquet"))
        got_counts = dict(zip(ct.column("node_id").to_pylist(), ct.column("N_c").to_pylist()))
        assert got_counts == counts, f"counts {got_counts} != {counts}"
        print("ok: Stage 2 counts")

        # verify node_multiplicity.parquet
        nm = pq.read_table(os.path.join(d, "node_multiplicity.parquet"))
        got_nm = dict(zip(nm.column("node_id").to_pylist(), nm.column("m_c").to_pylist()))
        for n in counts:
            assert abs(got_nm[n] - node_m[n]) < 1e-9, f"m_c[{n}] {got_nm[n]} != {node_m[n]}"
        assert got_nm["M"] == 1.0
        assert got_nm["T"] > 1.0     # tail oversample
        assert got_nm["H"] <= 1.0    # head at/below retain
        print("ok: Stage 3 node multiplicity")

        # verify sample_multiplicity.parquet (incl. s7 no-concept -> 1.0)
        sm = pq.read_table(os.path.join(d, "sample_multiplicity.parquet"))
        got_sm = dict(zip(sm.column("sample_key").to_pylist(), sm.column("m").to_pylist()))
        assert set(got_sm) == set(SAMPLE_CONCEPTS), "missing samples (no-concept not represented?)"
        for s in SAMPLE_CONCEPTS:
            assert abs(got_sm[s] - sample_m[s]) < 1e-9, f"m[{s}] {got_sm[s]} != {sample_m[s]}"
        assert got_sm["s7"] == 1.0
        print("ok: Stage 4 rarest-wins + no-concept")

        # ---- Stage 5 ----
        subprocess.run([
            sys.executable, os.path.join(_PKG, "materialize_rebalanced.py"),
            "--sample_multiplicity", os.path.join(d, "sample_multiplicity.parquet"),
            "--index", os.path.join(d, "index.parquet"),
            "--output_dir", exp_dir, "--name", "test_arm",
        ], check=True)

        mb = pq.read_table(os.path.join(exp_dir, "membership.parquet"))
        got_mc = dict(zip(mb.column("sample_key").to_pylist(), mb.column("count").to_pylist()))
        expected_kept = {s: c for s, c in exp_count.items() if c >= 1}
        assert got_mc == expected_kept, f"membership {got_mc} != {expected_kept}"
        print("ok: Stage 5 realized counts")

        cfg = yaml.safe_load(open(os.path.join(exp_dir, "config.yaml")))
        assert cfg["actual_samples"] == sum(expected_kept.values())
        assert cfg["distinct_samples"] == len(expected_kept)
        assert cfg["dataset_cls"] == "rebalanced"
        print("ok: config.yaml actual_samples / distinct_samples")

        # determinism: a second materialize gives the same membership_hash
        exp_dir2 = os.path.join(d, "exp2")
        subprocess.run([
            sys.executable, os.path.join(_PKG, "materialize_rebalanced.py"),
            "--sample_multiplicity", os.path.join(d, "sample_multiplicity.parquet"),
            "--index", os.path.join(d, "index.parquet"),
            "--output_dir", exp_dir2, "--name", "test_arm",
        ], check=True)
        cfg2 = yaml.safe_load(open(os.path.join(exp_dir2, "config.yaml")))
        assert cfg["membership_hash"] == cfg2["membership_hash"], "membership_hash not reproducible"
        print("ok: membership_hash reproducible")

        print("\nAll offline pipeline tests passed.")
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
