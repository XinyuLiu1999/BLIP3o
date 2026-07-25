"""Stage 6a — Distribution-level audit (cheap, pre-training).

The primary guard against rarest-wins coupling (plan §6a). Given the links, the
per-concept schedule, and a materialized membership, it reports:

- **Before/after concentration + Gini** over per-concept frequency: expect a
  measurable Gini drop and a flatter head.
- **Realized vs intended keep-rate per concept**: realized
  ``Sum count over its samples / N_c`` vs the schedule's ``m_c``. A head concept
  that always co-occurs with tails can be re-inflated by oversampling; this
  surfaces the worst offenders so the caps (``M_max``, ``r_min``) / constants can
  be re-calibrated until realized ~= intended.
- **Budget accounting**: expanded total, effective epochs at fixed compute.

Pure arithmetic (no GPU / browser), so it is cheap to re-run each calibration
iteration.

    python concept_rebalancing/audit.py \
        --links      concept_rebalancing/runs/blip3o_pretrain/links.parquet \
        --node_multiplicity concept_rebalancing/runs/blip3o_pretrain/node_multiplicity.parquet \
        --membership experiments/rebalanced_B/membership.parquet \
        --output concept_rebalancing/runs/blip3o_pretrain/audit_B.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def gini(values):
    """Gini coefficient of a list of non-negative frequencies (same formula the
    browser's composition.get_skew uses)."""
    vals = sorted(v for v in values if v > 0)
    n = len(vals)
    total = sum(vals)
    if n == 0 or total == 0:
        return 0.0
    s = sum((i + 1) * c for i, c in enumerate(vals))
    return (2 * s) / (n * total) - (n + 1) / n


def top_share(values, frac):
    """Share of total mass held by the top ``frac`` of concepts (descending)."""
    vals = sorted(values, reverse=True)
    n = len(vals)
    total = sum(vals)
    if n == 0 or total == 0:
        return 0.0
    k = max(1, int(round(frac * n)))
    return sum(vals[:k]) / total


def main():
    ap = argparse.ArgumentParser(description="Stage 6a: distribution audit (before/after + realized vs intended).")
    ap.add_argument("--links", required=True, help="links.parquet (sample_key, node_id).")
    ap.add_argument("--node_multiplicity", required=True,
                    help="node_multiplicity.parquet (node_id, N_c, m_c) from Stage 3.")
    ap.add_argument("--membership", required=True,
                    help="membership.parquet (sample_key, count) from Stage 5.")
    ap.add_argument("--counts_names", default=None,
                    help="Optional counts.parquet (node_id, node_name) for readable output.")
    ap.add_argument("--output", default=None, help="Write full JSON report here.")
    ap.add_argument("--top_offenders", type=int, default=25)
    args = ap.parse_args()

    # membership counts
    mt = pq.read_table(args.membership, columns=["sample_key", "count"])
    count_of = dict(zip(mt.column("sample_key").to_pylist(), mt.column("count").to_pylist()))

    # schedule
    nm = pq.read_table(args.node_multiplicity, columns=["node_id", "N_c", "m_c"])
    node_ids = nm.column("node_id").to_pylist()
    N_c = dict(zip(node_ids, nm.column("N_c").to_pylist()))
    m_c = dict(zip(node_ids, nm.column("m_c").to_pylist()))

    names = {}
    if args.counts_names and os.path.exists(args.counts_names):
        ct = pq.read_table(args.counts_names)
        if "node_name" in ct.column_names:
            names = dict(zip(ct.column("node_id").to_pylist(),
                             ct.column("node_name").to_pylist()))

    # realized per-concept mass = Sum count over the concept's samples.
    # Arrow-native: dictionary-encode both link columns to int32 (never
    # materializing 218M Python strings — a full .to_pylist() here would blow a
    # tight cgroup memory limit) and sum per-link counts by node via group-by.
    lt = pq.read_table(args.links, columns=["sample_key", "node_id"])
    samp_dict = pc.dictionary_encode(
        lt.column("sample_key").cast(pa.large_string()).combine_chunks())
    node_dict = pc.dictionary_encode(
        lt.column("node_id").cast(pa.large_string()).combine_chunks())
    del lt

    # per-link count = count of that sample (0 if dropped / absent from membership)
    samp_vals = samp_dict.dictionary.to_pylist()
    samp_count = np.array([count_of.get(k, 0) for k in samp_vals], dtype=np.int64)
    per_link_count = samp_count[samp_dict.indices.to_numpy(zero_copy_only=False)]

    node_vals = node_dict.dictionary.to_pylist()
    grouped = pa.table({"n": node_dict.indices, "c": pa.array(per_link_count)}) \
        .group_by("n").aggregate([("c", "sum")])
    realized = defaultdict(int)
    for n_pos, csum in zip(grouped.column("n").to_numpy(zero_copy_only=False),
                           grouped.column("c_sum").to_numpy(zero_copy_only=False)):
        realized[node_vals[n_pos]] = int(csum)

    # before/after distributions over concepts (only concepts with a schedule)
    before = [N_c[n] for n in node_ids]
    after = [realized.get(n, 0) for n in node_ids]

    gini_before, gini_after = gini(before), gini(after)
    report = {
        "n_concepts": len(node_ids),
        "gini_before": round(gini_before, 4),
        "gini_after": round(gini_after, 4),
        "gini_delta": round(gini_after - gini_before, 4),
        "top2pct_before": round(top_share(before, 0.02), 4),
        "top2pct_after": round(top_share(after, 0.02), 4),
        "mass_before": sum(before),
        "expanded_total": sum(count_of.values()),
        "distinct_kept": sum(1 for c in count_of.values() if c >= 1),
    }

    # realized vs intended keep-rate per concept
    rows = []
    for n in node_ids:
        nc = N_c[n]
        if nc <= 0:
            continue
        realized_rate = realized.get(n, 0) / nc
        intended = m_c[n]
        ratio = realized_rate / intended if intended > 0 else float("inf")
        rows.append({
            "node_id": n, "name": names.get(n, n), "N_c": nc,
            "intended_m": round(intended, 4),
            "realized_rate": round(realized_rate, 4),
            "realized_over_intended": round(ratio, 4),
        })

    # Worst offenders: head concepts (intended<1) whose realized rate most
    # exceeds intent — the coupling failure the audit exists to catch.
    head_offenders = sorted(
        (r for r in rows if r["intended_m"] < 1.0),
        key=lambda r: r["realized_over_intended"], reverse=True,
    )[: args.top_offenders]
    # Tail concepts that under-realized (oversampling not achieved).
    tail_short = sorted(
        (r for r in rows if r["intended_m"] > 1.0),
        key=lambda r: r["realized_over_intended"],
    )[: args.top_offenders]

    report["head_reinflation_offenders"] = head_offenders
    report["tail_undersampled"] = tail_short

    print("\n" + "=" * 66)
    print("STAGE 6a — DISTRIBUTION AUDIT")
    print("=" * 66)
    print(f"concepts:            {report['n_concepts']:,}")
    print(f"Gini  before -> after: {gini_before:.4f} -> {gini_after:.4f} "
          f"(delta {report['gini_delta']:+.4f})")
    print(f"top-2% mass before -> after: {report['top2pct_before']*100:.1f}% -> "
          f"{report['top2pct_after']*100:.1f}%")
    print(f"distinct kept:       {report['distinct_kept']:,}")
    print(f"expanded total:      {report['expanded_total']:,} "
          f"(x{report['expanded_total']/max(1,report['distinct_kept']):.2f})")

    print(f"\n-- head concepts most RE-INFLATED (realized/intended, want ~1.0) --")
    for r in head_offenders[:12]:
        print(f"  {r['name'][:34]:<34} N={r['N_c']:>8,}  "
              f"intended {r['intended_m']:.2f}  realized {r['realized_rate']:.2f}  "
              f"ratio {r['realized_over_intended']:.2f}")

    print(f"\n-- tail concepts UNDER-realized (oversampling shortfall) --")
    for r in tail_short[:12]:
        print(f"  {r['name'][:34]:<34} N={r['N_c']:>8,}  "
              f"intended {r['intended_m']:.2f}  realized {r['realized_rate']:.2f}  "
              f"ratio {r['realized_over_intended']:.2f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[audit] full report -> {args.output}")
    print("\nCalibrate: if head ratios >> 1, lower M_max / raise r_min or reduce")
    print("gamma; re-run build_schedule.py + materialize + this audit until")
    print("realized ~= intended.")


if __name__ == "__main__":
    main()
