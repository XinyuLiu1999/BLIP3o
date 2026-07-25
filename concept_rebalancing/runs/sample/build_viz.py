"""Build a static HTML gallery for the Stage-1 sample: image + tags + linked concepts.

Reads the Stage-1 outputs in this directory and writes index.html referencing
thumbs/ relatively. Concept chips are shaded by match score; abstract/descriptor
nodes are flagged so the linking quality is auditable at a glance.
"""
import os, json, html, collections
import pyarrow.parquet as pq

OUT = os.path.dirname(os.path.abspath(__file__))
EXTRACT = os.path.join(OUT, "extract")

# Descriptor/abstract nodes that name a taxonomy node but carry little visual
# grounding — worth flagging (STOPTAGS candidates), not silently trusting.
ABSTRACT = {
    "atmosphere", "view", "scene", "detail", "background", "manner", "setting",
    "shot", "texture", "expression", "moment", "environment", "design", "content",
    "arrangement", "composition", "quality", "style", "activity", "structure",
}

t2n = pq.read_table(os.path.join(OUT, "tag_to_nodes.parquet")).to_pydict()
st = pq.read_table(os.path.join(OUT, "sample_tags.parquet")).to_pydict()
lk = pq.read_table(os.path.join(OUT, "links.parquet")).to_pydict()
vocab = [l.strip() for l in open(os.path.join(OUT, "vocab.txt")) if l.strip()]

nid2name = {nid: nm for nid, nm in zip(t2n["node_id"], t2n["node_name"])}
# best score per (node) globally, for chip shading
node_best = {}
for nid, sc in zip(t2n["node_id"], t2n["score"]):
    node_best[nid] = max(node_best.get(nid, 0.0), sc)

sample_tags = {k: v for k, v in zip(st["sample_key"], st["tags"])}
sample_nodes = collections.defaultdict(list)
seen = set()
for k, nid in zip(lk["sample_key"], lk["node_id"]):
    if (k, nid) in seen:
        continue
    seen.add((k, nid))
    sample_nodes[k].append(nid)

matched = set(t2n["tag"])
tag2nodes = collections.defaultdict(set)
for t, nid in zip(t2n["tag"], t2n["node_id"]):
    tag2nodes[t].add(nid)

# ---- summary stats ----
cc = collections.Counter(nid2name.get(nid, nid) for nid in lk["node_id"])
npd = sorted(len(v) for v in sample_nodes.values())
n = len(sample_nodes)
median = npd[n // 2]
scores = list(t2n["score"])
frac_hi = sum(1 for s in scores if s >= 0.8) / len(scores)
top = cc.most_common(20)

keys = sorted(sample_tags.keys())

def chip(nid):
    nm = nid2name.get(nid, nid)
    sc = node_best.get(nid, 0.0)
    cls = "chip"
    if nm.lower() in ABSTRACT:
        cls += " abstract"
    elif sc >= 0.9:
        cls += " strong"
    elif sc < 0.72:
        cls += " weak"
    return f'<span class="{cls}" title="score {sc:.2f}">{html.escape(nm)}</span>'

cards = []
for k in keys:
    tags = sample_tags[k]
    nodes = sample_nodes.get(k, [])
    # unmatched (dropped) tags for this sample
    dropped = [t for t in tags if t not in matched]
    tag_html = " ".join(f'<span class="tag">{html.escape(t)}</span>' for t in tags)
    chips = " ".join(chip(nid) for nid in sorted(nodes, key=lambda x: -node_best.get(x, 0)))
    drop_html = ""
    if dropped:
        drop_html = ('<div class="dropped">dropped: '
                     + " ".join(html.escape(t) for t in dropped) + "</div>")
    cards.append(f"""
    <div class="card">
      <img loading="lazy" src="thumbs/{html.escape(k)}.jpg" alt="">
      <div class="body">
        <div class="lbl">tags ({len(tags)})</div>
        <div class="tags">{tag_html}</div>
        <div class="lbl">concepts ({len(nodes)})</div>
        <div class="chips">{chips}</div>
        {drop_html}
      </div>
    </div>""")

top_html = "".join(
    f'<li><span class="bar" style="width:{100*c/top[0][1]:.0f}%"></span>'
    f'<span class="tn">{html.escape(nm)}</span><span class="tc">{c}</span></li>'
    for nm, c in top)

doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Concept Rebalancing — Stage 1 sample (500 images)</title>
<style>
:root{{--bg:#0f1115;--card:#181b22;--edge:#262b36;--txt:#e6e9ef;--mut:#9aa4b2;
--tag:#232834;--strong:#1f6f4a;--weak:#6b4a1f;--abs:#5a2340;--chip:#2a3550;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--txt);
font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
header{{padding:24px 28px;border-bottom:1px solid var(--edge);position:sticky;top:0;
background:linear-gradient(180deg,#0f1115,#0f1115f2);backdrop-filter:blur(6px);z-index:5}}
h1{{margin:0 0 4px;font-size:20px}}
.sub{{color:var(--mut);margin-bottom:16px}}
.stats{{display:flex;gap:28px;flex-wrap:wrap;align-items:flex-start}}
.kpis{{display:flex;gap:20px;flex-wrap:wrap}}
.kpi{{background:var(--card);border:1px solid var(--edge);border-radius:10px;padding:10px 16px;min-width:110px}}
.kpi .v{{font-size:22px;font-weight:650}}
.kpi .k{{color:var(--mut);font-size:12px}}
.top{{flex:1;min-width:280px;max-width:520px}}
.top ol{{list-style:none;margin:6px 0 0;padding:0;columns:2;column-gap:24px}}
.top li{{position:relative;display:flex;align-items:center;gap:8px;padding:2px 0;
break-inside:avoid}}
.top .bar{{position:absolute;left:0;height:16px;background:#2a3550;border-radius:3px;z-index:0;opacity:.5}}
.top .tn{{position:relative;z-index:1}}.top .tc{{position:relative;z-index:1;margin-left:auto;color:var(--mut)}}
.legend{{margin-top:14px;color:var(--mut);font-size:12.5px;display:flex;gap:16px;flex-wrap:wrap;align-items:center}}
.dot{{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px;vertical-align:middle}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px;padding:20px 28px}}
.card{{background:var(--card);border:1px solid var(--edge);border-radius:12px;overflow:hidden;display:flex;flex-direction:column}}
.card img{{width:100%;height:190px;object-fit:cover;background:#000;display:block}}
.body{{padding:12px 13px}}
.lbl{{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.06em;margin:8px 0 5px}}
.lbl:first-child{{margin-top:0}}
.tag{{display:inline-block;background:var(--tag);color:#c7d0dd;border-radius:5px;padding:2px 7px;margin:0 3px 4px 0;font-size:12px}}
.chip{{display:inline-block;background:var(--chip);border-radius:5px;padding:2px 8px;margin:0 3px 4px 0;font-size:12.5px}}
.chip.strong{{background:var(--strong)}}.chip.weak{{background:var(--weak)}}
.chip.abstract{{background:var(--abs)}}
.dropped{{margin-top:8px;color:#7a8494;font-size:11.5px;font-style:italic}}
</style></head><body>
<header>
  <h1>Concept Rebalancing · Stage 1 — tag→concept linking</h1>
  <div class="sub">500-image sample · <code>link_corpus.py</code> · min_similarity = 0.65 · Bamboo taxonomy</div>
  <div class="stats">
    <div class="kpis">
      <div class="kpi"><div class="v">501</div><div class="k">images</div></div>
      <div class="kpi"><div class="v">{len(matched)}/{len(vocab)}</div><div class="k">tags matched ({100*len(matched)//len(vocab)}%)</div></div>
      <div class="kpi"><div class="v">{len(cc)}</div><div class="k">distinct concepts</div></div>
      <div class="kpi"><div class="v">{median}</div><div class="k">median concepts/img</div></div>
      <div class="kpi"><div class="v">{100*frac_hi:.0f}%</div><div class="k">links score ≥0.8</div></div>
    </div>
    <div class="top">
      <div class="lbl">Top 20 concepts by image count</div>
      <ol>{top_html}</ol>
    </div>
  </div>
  <div class="legend">
    <span><span class="dot" style="background:var(--strong)"></span>strong (≥0.90)</span>
    <span><span class="dot" style="background:var(--chip)"></span>ok</span>
    <span><span class="dot" style="background:var(--weak)"></span>weak (&lt;0.72)</span>
    <span><span class="dot" style="background:var(--abs)"></span>abstract/descriptor (STOPTAGS candidate)</span>
    <span>hover a chip for its score · <i>dropped</i> = tag unresolved at 0.65</span>
  </div>
</header>
<div class="grid">
{''.join(cards)}
</div>
</body></html>"""

with open(os.path.join(OUT, "index.html"), "w") as f:
    f.write(doc)
print("wrote", os.path.join(OUT, "index.html"), f"({len(doc)//1024} KB, {len(keys)} cards)")
