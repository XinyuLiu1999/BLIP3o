"""Stage 1b — Category-level QA gate (plan decision #1).

Aggregate ``links.parquet`` into the 11 composition GROUPS + subcategories via
the browser's ``CompositionAnalyzer`` (DAG attribution) and print the group +
subcategory shares plus the frequency-skew headline. This is a **human go/no-go
checkpoint**: if a GROUP is implausibly inflated (the documented homonym /
descriptor leak — ``blue``->butterfly, ``overcast``->fishing-cast), fix
``tags.STOPTAGS`` / the matcher and *re-link* before trusting the counts. The
schedule (Stages 3-4) operates on per-concept node counts, NOT categories — this
gate only validates the link set feeding those counts.

Also emits the Stage-2 per-concept frequency table ``counts.parquet``
(``node_id, N_c``) as a side effect, since it is exactly
``CompositionAnalyzer``'s ``direct_count`` view.

Run (in the browser `wiki` conda env)::

    python concept_rebalancing/qa_gate.py \
        --links concept_rebalancing/runs/blip3o_pretrain/links.parquet \
        --output_dir concept_rebalancing/runs/blip3o_pretrain
"""

import argparse
import json
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from rebalance.linker import DEFAULT_BROWSER_ROOT  # noqa: E402
from rebalance.qa_counts import build_analyzer  # noqa: E402


def write_counts(analyzer, counts_path):
    """Emit counts.parquet (node_id, N_c, node_name) from the analyzer's direct view."""
    analyzer._ensure_built()  # populates analyzer.direct
    node_ids = list(analyzer.direct.keys())
    counts = [analyzer.direct[n] for n in node_ids]
    names = [
        analyzer.taxonomy.nodes[n].name if n in analyzer.taxonomy.nodes else n
        for n in node_ids
    ]
    table = pa.table({
        "node_id": pa.array(node_ids, type=pa.string()),
        "N_c": pa.array(counts, type=pa.int64()),
        "node_name": pa.array(names, type=pa.string()),
    })
    pq.write_table(table, counts_path)
    print(f"[counts] wrote {counts_path}: {len(node_ids)} active concepts")


def print_report(analyzer, report_path):
    overview = analyzer.get_overview()
    skew = analyzer.get_skew()

    print("\n" + "=" * 66)
    print("STAGE 1b — CATEGORY QA GATE")
    print("=" * 66)
    print(f"total tag-mass (links): {overview['total_mass']:,}")
    print(f"distinct linked images: {overview['total_images']:,}")
    print(f"visual share:           {overview['visual_share'] * 100:.1f}%   "
          f"non-visual share: {overview['nonvisual_share'] * 100:.1f}%")
    print("\n-- 11 visual GROUPS (share of tag mass) --")
    for g in overview["groups"]:
        print(f"  {g['label']:<34} {g['share'] * 100:6.2f}%   "
              f"(incl. shared {g['membership_share'] * 100:5.1f}%)")
        for s in g["subcategories"][:5]:
            img_share = s.get("image_share")
            extra = f"  img {img_share * 100:4.1f}%" if img_share is not None else ""
            print(f"        - {s['label']:<28} {s['share'] * 100:5.2f}%{extra}")
    if overview.get("nonvisual"):
        nv = overview["nonvisual"]
        print(f"\n  {nv['label']:<34} {nv['share'] * 100:6.2f}%")
        for s in nv["subcategories"][:6]:
            print(f"        - {s['label']:<28} {s['share'] * 100:5.2f}%")

    print("\n-- frequency skew --")
    print("  " + skew["headline"])
    print(f"  active concepts: {skew['active_nodes']:,}   "
          f"Gini: {skew['gini']:.4f}   "
          f"top-2% mass: {skew['top_shares']['2'] * 100:.1f}%")

    report = {"overview": overview, "skew": skew}
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[qa] full report written to {report_path}")
    print("\nGO/NO-GO: inspect the group shares above. If a GROUP is implausibly")
    print("inflated by a homonym/descriptor leak, fix tags.STOPTAGS / the matcher")
    print("and re-run Stage 1 before building the schedule.")


def main():
    ap = argparse.ArgumentParser(description="Stage 1b: category-level QA gate + Stage 2 counts.")
    ap.add_argument("--links", required=True, help="links.parquet from Stage 1.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--browser_root", default=DEFAULT_BROWSER_ROOT)
    ap.add_argument("--total_images", type=int, default=None,
                    help="True corpus size incl. no-concept samples (else distinct linked).")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    counts_path = os.path.join(args.output_dir, "counts.parquet")
    report_path = os.path.join(args.output_dir, "qa_report.json")

    analyzer, linker = build_analyzer(
        args.links, args.browser_root, total_images=args.total_images
    )
    try:
        write_counts(analyzer, counts_path)
        print_report(analyzer, report_path)
    finally:
        linker.close()


if __name__ == "__main__":
    main()
