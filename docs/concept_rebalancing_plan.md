# Concept Rebalancing for BLIP3o Pretraining — Implementation Plan

Status: **design locked, not yet implemented** · Date: 2026-07-23
Owner: xinyu · Target data: `BLIP3o-Pretrain-Long-Caption-filtered-recaptioned`

---

## 1. Goal

The pretraining corpus is extremely long-tailed at the *concept* level (cf. the
reference: ~2% of categories carry >30% of frequency, ~50% appear <100K times).
We want to **resample the corpus offline** — downsample over-represented head
concepts, fully retain the mid band, and **oversample tail concepts** — then
train BLIP3o on the rebalanced set and measure the effect, especially on
tail-concept generation.

This document is the end-to-end plan connecting three existing pieces:

- **The data** — `BLIP3o-Pretrain-Long-Caption-filtered-recaptioned`, webdataset
  `.tar` shards. Each sample is `{uuid}.{jpg,txt,json,pkl}`. The `.json` sidecar
  already contains a **`tagging_caption`** (VLM comma-separated tags) plus quality
  scores; the `.pkl` is a precomputed feature vector.
- **`semantic_image_browser`** — maps `tagging_caption` → Bamboo taxonomy nodes
  (concepts) via `parse_tags` → `HybridMatcher`, and already computes the
  long-tail skew (concentration curve, Gini) and an 11-group composition view.
- **`BLIP3o/scripts/data_pipeline`** — the offline experiment pipeline
  (`index_tars` → `enrich_index` → `materialize`) that produces a
  `membership` + `shardlist` an experiment trains on via
  `data_args.experiment_dir` (see `blip3o/data/dataset.py`, "experiment mode").

## 2. Why this is feasible (the key alignment)

**Identity is shared with no fuzzy matching.** Everything keys on the UUID stem:

```
webdataset __key__  ==  {uuid}.json filename stem  ==  BLIP3o sample_key  ==  the tagged unit
```

`index_tars.compute_priority` and the tagging both live on `sample_key`, so
concept labels join back to training samples exactly. The `tagging_caption`
format in this corpus is precisely what `semantic_image_browser/backend/tags.py`
was tuned on (count prefixes like `1 male kayaker`, OCR quotes, and the
`high resolution`/`sharp focus`/`photograph` descriptor boilerplate that
`STOPTAGS` drops).

## 3. Locked design decisions

1. **Category-level review is a QA gate, not the sampling granularity.** After
   linking, inspect the 11 composition GROUPS + subcategories to catch gross
   linker errors (homonym/descriptor inflation — the documented `blue`→butterfly,
   `overcast`→fishing-cast class of bug) *before* trusting the counts. The
   **schedule operates on per-concept (Bamboo node) counts**; categories are only
   the sanity checkpoint.
2. **Multi-label reduction = rarest-wins.** A sample carries several concepts.
   Its resampling multiplicity is governed by its **rarest** concept (the one
   demanding the most representation).
3. **Tail oversampling is required.** A pure subset (the current membership-set)
   is insufficient. We generalize membership to an integer **multiplicity per
   sample** (0 = drop, 1 = keep, ≥2 = oversample), which unifies drop /
   downsample / retain / oversample under one abstraction.

## 4. The unifying abstraction: per-sample multiplicity `m`

Every sample gets a real-valued target multiplicity `m ≥ 0`, realized to an
integer count deterministically:

```
h      = hash01(sample_key)          # uniform [0,1), = compute_priority / 2**63
base   = floor(m)
frac   = m - base
count  = base + (1 if h < frac else 0)
```

- `m = 0`      → dropped
- `0 < m < 1`  → probabilistic keep (downsample):  count ∈ {0,1}, E[count]=m
- `m = 1`      → retained
- `m > 1`      → oversampled: `base` guaranteed copies + 1 probabilistic

Determinism comes entirely from `sample_key` (shared sha256 primitive), so the
resampled set is reproducible and re-derivable from the schedule alone.

## 5. Pipeline stages

### Stage 0 — Index the shards (existing code)
`index_tars.py --tar_dir <corpus> --output index.parquet`
→ `(sample_key, tar_path, priority)`. Unchanged.

### Stage 1 — Tag/link the corpus → per-sample concepts (existing linker, new input + a vocab cache)
Run `parse_tags` → `HybridMatcher` (the `semantic_image_browser` linking path)
over every sample's `tagging_caption`. Output: `links.parquet` =
`(sample_key, node_id)` (many-to-many, avg ~1.2 nodes/tag after margin-keep).

**Scale trick (mandatory).** ~640K samples are already on disk (39 GB / 130
shards, ~16K samples/GB) and the full set is far larger; naive per-image
embedding won't scale. Tags repeat massively, so:
1. collect the **unique tag vocabulary** across the corpus,
2. resolve each unique tag → node(s) **once** (lexical-first; GPU embedding only
   for the lexical-miss fallback),
3. join the resolved map back to samples by string.

This is the same HybridMatcher, just amortized over the vocabulary (~1e6 unique
tags) instead of 1e7+ images.

### Stage 1b — Category-level QA gate (decision #1)
Aggregate `links.parquet` into the 11 composition GROUPS (reuse
`composition.py`'s DAG attribution) and eyeball the group + subcategory shares.
**Go/no-go:** if a GROUP is implausibly inflated, fix `STOPTAGS`/matcher and
re-link before proceeding. Do **not** compute rates on a polluted link set.

### Stage 2 — Per-concept frequency table
`counts.parquet` = `(node_id, N_c)` where `N_c` = distinct samples linked to
concept `c`. This is exactly `node_image_counts` / composition.py's `direct_count`.

### Stage 3 — Per-concept multiplicity schedule
Map each concept's count `N_c` to a target multiplicity `m_c`. Piecewise,
continuous at the boundaries, all constants **tunable and calibrated** against
the Stage-6a audit:

```
head  (N_c ≥ N_high) :  m_c = max(r_min, a_head * 2 / log10(N_c))     # < 1  downsample
mid   (N_low ≤ N_c < N_high) :  m_c = 1.0                             # retain
tail  (N_c < N_low)  :  m_c = min(M_max, (N_low / N_c) ** gamma)      # > 1  oversample
```

Starting defaults (calibrate later):
`N_high = 100_000` (from the reference), `N_low = 20_000`,
`a_head = log10(N_high)/2 = 2.5` (⇒ `m_c = 5/log10(N_c)`, so `m_c(1e5)=1`,
`m_c(1e7)≈0.71`), `r_min = 0.1` (floor so heads never vanish),
`gamma = 0.5` (sub-linear so singletons aren't boosted 100×), `M_max = 4` (cap).

The `head` branch is the reference's `N_sample ∝ 2/log(Count)`; the `tail` branch
is its symmetric oversampling extension, which the reference omits because it
never oversamples.

### Stage 4 — Per-sample multiplicity (rarest-wins, decision #2)
Join `links` × `m_c`. For each `sample_key`:

```
m(sample) = max over its concepts c of  m_c          # rarest concept wins
m(sample) = 1.0   if the sample has no content concept (all tags STOPTAGS-dropped)
```

`max` because rarer concepts have larger `m_c`; the rarest concept sets the bar.
Output `sample_multiplicity.parquet` = `(sample_key, m)`.

### Stage 5 — Materialize the resampled membership (extend `materialize.py`)
Realize integer `count` per sample via the Stage-4 hash rule; keep rows with
`count ≥ 1`. Write:

- `membership.parquet` — `(sample_key, count)`  (**replaces** the flat
  `membership.txt`; count carries the multiplicity)
- `shardlist.txt` — unique `tar_path`s over kept samples (unchanged)
- `config.yaml` — as today **plus** `actual_samples = Σ count` (the *expanded*
  total, so `run_experiment.sh`'s `MAX_STEPS = actual_samples / global_batch`
  budgets the extra tail copies correctly), and a `membership_hash`.

### Stage 6 — Train (extend `dataset.py` experiment mode for multiplicity)
`blip3o/data/dataset.py` currently filters HF rows by a membership **set** (one
row = one sample). Add multiplicity expansion:

1. load `shardlist` shards, filter to `count_map.keys()` (unchanged filter),
2. build `count_map = {sample_key: count}` from `membership.parquet`,
3. build `self.index_map = [row_idx repeated count times for each kept row]`,
4. `__len__ = len(index_map)`; `__getitem__(i)` → `index_map[i]` → HF row,
5. shuffle **`index_map`** (seed 42), not the HF dataset, so repeats disperse.

Repeats are cheap **index entries**, not duplicated image bytes (HF rows stay
memory-mapped). No trainer/sampler surgery; works with the existing
`group_by_modality_length` sampler because all modality lengths are constant.

Then train each arm exactly as today: `sbatch run_experiment.sh <experiment_dir>`.

## 6. Evaluation plan

### 6a. Distribution-level (cheap, pre-training — the browser is the instrument)
- **Before/after concentration curve + Gini** (composition.py skew): expect a
  measurable Gini drop and a flatter head.
- **Realized vs intended keep-rate per concept**: for each node, realized
  `Σcount over its samples / N_c` vs the schedule. **This is the primary guard
  against rarest-wins coupling** — a head concept that always co-occurs with tails
  can get re-inflated by oversampling; this audit catches it, and the caps
  (`M_max`, `r_min`) / constants are re-calibrated until realized ≈ intended.
- **Group-level mass shift** across the 11 GROUPS: confirm subjects moved, not
  boilerplate.
- **Budget accounting**: `Σcount`, effective epochs at fixed compute.

### 6b. Model-level (the real question) — controlled ablation
The pipeline makes each arm a different `experiment_dir`, hash-pinned:

- **Arm A — baseline** (full / uniform).
- **Arm B — rebalanced** (this plan).
- **Arm C — baseline subsampled to Arm B's sample count** (critical control:
  isolates *rebalancing* from *dataset size / epoch count*).

Fix total training steps/tokens across arms.

**Metrics (BLIP3o is text→image generation):**
- **Stratified tail eval (sharpest test).** Build an eval prompt suite of Bamboo
  concepts **stratified by original frequency** (head / mid / tail), score
  generation per stratum. Hypothesis: **tail-concept generation improves at ~no
  cost to head.** Report the **floor** (min / 10th-pct per-concept), not just the
  mean. The tagging already gives the concept→frequency map to define strata.
- **Overall quality / alignment:** FID / CLIP-FID vs a held-out reference;
  CLIPScore and/or VLM-judge alignment on a fixed prompt suite.
- **Diversity:** within-prompt LPIPS/feature variance for head concepts — verify
  downsampling didn't collapse variety.
- **External** (secondary): GenEval / DPG-Bench / T2I-CompBench.
- **Targeted VLM-judge preference** on the tail prompt suite (cheap).

## 7. Code inventory — what's new vs reused

| Component | Status |
|---|---|
| `index_tars.py` | reuse as-is |
| Tagging/linking (`parse_tags`+`HybridMatcher`) | reuse; **new**: corpus input + vocab-cache driver script |
| Category QA aggregation | reuse `composition.py` |
| Per-concept counts | reuse (`node_image_counts` logic) |
| Schedule + rarest-wins scorer (Stages 3–4) | **new** small script → `sample_multiplicity.parquet` |
| `materialize.py` | **extend**: emit `(sample_key, count)` + expanded `actual_samples` |
| `enrich_index.py` | optional (only if joining quality-score filters too) |
| `blip3o/data/dataset.py` experiment mode | **extend**: multiplicity `index_map` expansion |
| `run_experiment.sh` | reuse (reads `actual_samples` already) |

Net new code: one scorer script, a `materialize` extension, and a ~15-line
`dataset.py` change. Everything else is existing code on new inputs.

## 8. Open items before coding
- Calibrate schedule constants against a first Stage-6a audit (iterate `M_max`,
  `gamma`, `a_head`, `r_min`).
- Confirm corpus fully downloaded before Stage 0 (`.obs.temp` files were still in
  flight on 2026-07-23).
- Decide whether quality-score gating (`aesthetic`, `nsfw`, `watermark` from the
  `.json`) is applied *before* resampling (recommended: filter first via
  `enrich_index` + a `filter_expression`, then rebalance the survivors).

---

## 9. Detailed end-to-end pipeline walkthrough

This traces the whole system, following one real sample from raw shard to a
training batch, and explains what each stage guarantees.

### The sample
In `shard-000001.tar`:
`cd3cd09d12de4c48a72a24a8de952a5e.{jpg,txt,json,pkl}`. Its `.json` contains,
among quality scores:

```
tagging_caption: "photograph, high resolution, sharp focus, 1 male kayaker,
  adult male, orange helmet, yellow and black drysuit, green life vest,
  intense action, whitewater rapids, turbulent river, red and yellow kayak,
  double-bladed paddle, dynamic splash, water droplets, overhead angle,
  close-up, high contrast, energetic, 'S' logo, 'AQUATEC' text"
```

Its identity everywhere downstream is `sample_key =
cd3cd09d12de4c48a72a24a8de952a5e`.

### Step A — Index (`index_tars.py`)
Every shard is scanned; each unique stem becomes a catalog row
`(sample_key, tar_path, priority)`, where
`priority = sha256(sample_key) mod 2^63`. Our kayaker gets one row pointing at
`shard-000001.tar`. `priority/2^63` is a fixed uniform-random number `h` for this
sample — the single source of randomness reused by *both* the pipeline and the
resampling draw, which is what makes the whole thing reproducible.

### Step B — Tag → concept linking (browser's HybridMatcher, vocab-cached)
`parse_tags` first cleans the caption: it drops OCR quotes (`'S' logo`,
`'AQUATEC' text`), bare numerics, and `STOPTAGS` descriptors — here `photograph`,
`high resolution`, `sharp focus`, `overhead angle`, `close-up`, `high contrast`
all fall out (they are the highest-frequency, least-informative tags and match
misleading homonym nodes). What survives are **content** tags: `male kayaker`
(count-prefix `1 ` stripped), `orange helmet`, `drysuit`, `life vest`,
`whitewater rapids`, `turbulent river`, `kayak`, `double-bladed paddle`,
`water droplets`, …

Each surviving tag is resolved to Bamboo node(s) by the HybridMatcher: a tag that
*names* a node resolves lexically (exact/singularized/count-stripped), and the
embedding only **reranks** the lexical candidates (so homonyms resolve to the
right sense); head-noun compounds keep their general node *and* may gain a
confident distinct concept (`golden retriever puppy` → both `puppy` and
`golden_retriever`). Because we resolve the **unique vocabulary once** and join by
string, this is affordable at corpus scale.

Result for this sample: a set of concept nodes, e.g. `{kayak, rapids/river,
paddle, helmet, life_vest, water_droplet, …}`. Written to `links.parquet` as
`(sample_key, node_id)` pairs.

### Step C — Category QA gate (composition.py)
All links roll up into the 11 visual GROUPS via the Bamboo **DAG** ancestor
closure (not the pruned tree). We *read* the group/subcategory shares to confirm
the linker isn't inflating a category through a homonym or descriptor leak. Our
kayaker contributes to *Activities* (kayaking) and *Objects* (kayak, paddle,
helmet). This is a human checkpoint; if it looks wrong we fix matching and
re-link. It does **not** feed the schedule — the schedule uses per-node counts.

### Step D — Concept counts (`counts.parquet`)
Across the whole corpus we count, per node, how many distinct samples link to it:
`N_c`. Suppose `kayak` is a tail concept (`N_kayak = 8_000`), while `water` /
`river` is a head concept (`N_river = 3_000_000`), and `helmet` is mid
(`N_helmet = 50_000`).

### Step E — Per-concept multiplicity (schedule, Stage 3)
Apply the piecewise schedule:
- `river` (head, 3e6): `m = 5/log10(3e6) = 5/6.48 ≈ 0.77` → downsample.
- `helmet` (mid, 5e4): between `N_low` and `N_high` → `m = 1.0`.
- `kayak` (tail, 8e3 < 20e3): `m = min(4, (20000/8000)^0.5) = min(4, 1.58) = 1.58`
  → oversample.

### Step F — Per-sample multiplicity (rarest-wins, Stage 4)
The sample's concepts have `m ∈ {0.77 (river), 1.0 (helmet), 1.58 (kayak), …}`.
**Rarest-wins = max = 1.58** (the rare `kayak` concept governs). So this sample's
target multiplicity is `m(sample) = 1.58`. Written to
`sample_multiplicity.parquet`.

*(Why max: a sample depicting a rare subject should be preserved/boosted even if
it also contains common ones — otherwise downsampling the common `river` would
throw away scarce `kayak` pixels. The realized-count audit in 6a is the guard
against the opposite failure — `river` riding tail co-occurrences back up.)*

### Step G — Realize to an integer count (`materialize.py`, Stage 5)
Deterministic draw with this sample's fixed `h` (from Step A):
`base = floor(1.58) = 1`, `frac = 0.58`, `count = 1 + (1 if h < 0.58 else 0)`.
So this sample appears **1 or 2 times** depending on its hash — say `count = 2`.
We write `(cd3cd09d…, 2)` to `membership.parquet`, add `shard-000001.tar` to
`shardlist.txt`, and accumulate `2` into `config.yaml.actual_samples`.

A pure head sample (e.g. `m = 0.77`, `base = 0`, `frac = 0.77`) would get
`count ∈ {0,1}` — i.e. dropped with prob 0.23. Same mechanism, whole range.

### Step H — Training-time expansion (`dataset.py` experiment mode, Stage 6)
The trainer loads the shards in `shardlist.txt`, filters HF rows to the
membership keys, then builds `index_map` where each kept row index is repeated
`count` times — our kayaker's row index appears **twice**. `__len__` is the
expanded total (matching `actual_samples`, so `MAX_STEPS` is right). `index_map`
is shuffled (seed 42) so the two copies land in different batches. `__getitem__`
maps a position → row → decodes the jpg, builds the T2I conversation from the
caption, returns tensors. The two copies are identical images seen twice per
epoch — genuine oversampling, at the cost of two ints, not two images.

### Step I — Train the ablation arms
Arm A (baseline), Arm B (this membership), Arm C (baseline subsampled to B's
size) are three `experiment_dir`s handed to the *same* `run_experiment.sh`. Each
is fully described by its `membership.parquet` + `membership_hash`, so runs are
reproducible and comparable.

### Step J — Evaluate
Run 6a (distribution audit — verify Gini dropped and realized≈intended, re-tune
constants if not) and 6b (stratified tail eval + FID/CLIP/diversity across arms).
The rebalancing hypothesis is confirmed if tail-concept generation improves in
Arm B over Arms A and C without a head-concept regression.

### Invariants that make the whole pipeline trust-worthy
- **One identity** (`sample_key`) from shard to batch; no fuzzy joins.
- **One randomness source** (`hash01(sample_key)`) for both catalog priority and
  the resampling draw ⇒ any membership is re-derivable from the schedule.
- **Counts, not weights**, carry multiplicity ⇒ works with the stock trainer and
  step-budget math; drop/downsample/retain/oversample are one formula.
- **Repeats are index entries**, not copied bytes ⇒ oversampling is nearly free.

---

## Appendix — Stage-1b QA audit of the first real run (2026-07-24)

First end-to-end run on `BLIP3o-Pretrain-Long-Caption-filtered-recaptioned`
(`runs/blip3o_pretrain/`: 19,755,541 images, 218,641,343 links, 58,834 active
concepts). We audited `qa_report.json` against the raw parquet, investigated a
systematic linker-quality fix, and — importantly — **measured whether any of it
matters to the schedule.** Verdict up front: **the QA report is arithmetically
correct; the link set has real false-friend errors; those errors are effectively
inert to the rebalancing, so we do NOT relink.**

### A. The report is arithmetically sound
Independently reproduced every group mass, membership mass, the visual/non-visual
split (75.4% / 24.6%), and the skew stats (Gini 0.975, top-2% = 84% of tag mass)
from `links.parquet` + the DAG — all matched to the digit. `total_mass` = link
rows, `total_images` = sample rows. The aggregation code is trustworthy.

### B. But the link set has systemic false-friend mislinks
Bare lexical matches at the 0.98 floor collapse a high-frequency tag onto the
wrong *sense* (the embedding can only rerank, never override). Confirmed by
tracing feeder tags + co-occurrence context. Three failure classes:

1. **Homonym mislinks** (tag names the wrong sense) — the wrong node holds the
   mass while the correct sense sits at `Nc≈0`:

   | tag | links to | should be | mass |
   |---|---|---|---|
   | `table` | "data arranged in rows/columns" | furniture | 759k |
   | `plant` | "buildings for industrial labor" (factory) | botanical | 322k |
   | `glass` | the material | drinking glass | 256k |
   | `female` | "an animal that produces gametes" | person | 1.16M |
   | `car` | "where passengers ride up and down" (elevator) | automobile | 860k |
   | `pool` | pocket billiards | swimming pool | 78k |
   | `nail` | construction nail | fingernail | 75k |
   | `jersey` | Jersey dairy cattle | shirt | 137k |
   | `board` | committee | plank | 12k |

   Asymmetry tell: `male`→person (correct) but `female`→animal (wrong). 1,505
   polysemous linked head tags (`Nc≥3000`) exist — the problem is broad, and it
   touches two top-40 concepts (`table`, `plant`).
2. **Descriptor leaks** — mostly handled by `STOPTAGS` (bare `blue`/`photograph`
   confirmed dropped); residual leaks arrive via compounds (`X in blue`→butterfly).
3. **Taxonomy naming bug** — the automobile node `n02958343` has primary name
   `"car side"`, so the tag `"car"` scores it 0.843 and loses to the elevator node
   named exactly `"car"` (1.000). Bad node metadata; one-field fix.

These distort the composition **report's grouping** (Animals 45% inflated by
`female`; Plants 36% by `hair`/`cloud`/`skin`; Design 23% by `smile`) but that
grouping never feeds the schedule.

### C. A category-free detector: what failed, what worked
Goal: catch mislinks without relying on the 11 hand-curated anchors (which only
expose errors that happen to land in a category). Tested on a 6% corpus sample
(1.23M images) against labeled mislinks + correct controls:

- **Co-occurrence + DAG taxonomic distance — FALSIFIED.** Co-occurrence is a
  *scene* relation (`stripe`+`tiger`), not is-a; it punishes correct links.
- **Co-occurrence + node-embedding coherence — FALSIFIED.** Node vectors are
  built from name+synonyms only, so a surface-string homonym (`car`, `female`)
  embeds next to its misused context and looks coherent.
- **Context-vs-gloss WSD — VALIDATED.** Match a tag's co-occurrence context to
  each candidate *sense's definition* (in `bamboo_V4.json` `id2desc`, currently
  discarded). On a 41-tag labeled set: raw argmax fixed 12/13 wrong, kept 17/28
  correct (11 regressions). Adding a **concreteness guard** (never overturn a
  concrete incumbent for an abstract challenger, via physical-entity ancestry
  `n00001930`) held 12/13 fixes and cut regressions to 8 — of which node-level
  inspection showed ~6 benign co-named sub-senses and **2 genuine** (`male`→animal,
  `hair`→plant). Residual hard core = concrete-vs-concrete same-surface homonyms,
  needing a sense-frequency / corpus-support tie-break on small gloss margins.

Conclusion: a two-guard (concreteness + support) gloss-WSD, run as a
**propose-and-review** Stage-1c (not silent auto-apply), is the right design *if*
link precision ever becomes the goal. It did not, because of D.

### D. Decisive measurement — the mislinks are inert to the schedule
The schedule keys on per-node `N_c` via `m(sample) = max_c m_c` (rarest-wins).
A mislink only matters if the mislinked node *decides* a sample's multiplicity.
Measured on the 6% sample — fraction of samples containing each mislink where it
is the unique rarest concept:

| mislink | band | m_c | decides m |
|---|---|---|---|
| female→animal (1.16M) | HEAD | 0.82 | **0.0%** |
| car→elevator (860k) | HEAD | 0.84 | **0.0%** |
| table→data (759k) | HEAD | 0.85 | **0.0%** |
| plant→factory (322k) | HEAD | 0.91 | 0.2% |
| jersey→cattle (137k) | HEAD | 0.97 | 0.5% |
| pool / nail (75–78k) | MID | 1.00 | 3–4% |
| board→committee (12k) | TAIL | 1.27 | 39% |

Two compounding reasons they don't matter: (1) rarest-wins over ~11 concepts/image
means a correctly-linked rarer tag almost always sets the copy count — the head
mislink is a passenger 0.0–0.5% of the time; (2) head→head is **band-preserving**
(the correct sense is also head, so `m_c` moves <0.2 within the flat head band and
the integer count usually doesn't change). The only per-sample effect is the lone
tail mislink `board→committee` — ~12k images at a mild 1.27×, ~3k extra copies in
a 19.7M corpus. Noise. Rebalancing is robust to link noise **by construction**: it
only cares about frequency band, and the one dangerous mode — a high-frequency tag
mislinking to a genuine *tail* node (spurious junk oversampling) — does not occur
at scale here.

### E. Decisions moving forward
1. **Do not relink; do not build the WSD relinker.** Proven not worth it for the
   rebalancing. Proceed to `build_schedule` → `materialize` → `audit` on the
   current `counts.parquet`.
2. **Keep one cheap guard (Stage 2/3):** assert no high-frequency polysemous tag
   feeds a *tail* (oversampled) node — the only path from link noise to a
   corrupted training mix. **Re-run whenever schedule constants change** (lowering
   `N_low` / raising `M_max` pulls mid mislinks like `pool`/`nail` toward the tail).
   Currently clean.
3. **Evaluation (§6b):** hand-curate the tail-concept eval list rather than
   trusting auto-links for eval buckets — sidesteps link noise for the deliverable.
4. **Zero-cost housekeeping:** footnote the known head mislinks (`table`=data-table,
   `plant`=factory) in the QA report; optionally fix the `n02958343` "car side"
   primary-name bug.

Method note (reproducible): 6% co-occurrence sample = `links.parquet` rows whose
`sample_key` starts with hex `0`; `m_c` from the Stage-3 defaults
(`N_high=100000, N_low=20000, a_head=2.5, r_min=0.1, gamma=0.5, M_max=4`); glosses
+ node vectors from the `wiki` env caches under `semantic_image_browser/data/`.

## Appendix — Schedule calibration against the real N_c distribution (2026-07-24)

Before running Stages 2–5 we measured the actual per-concept count distribution on
`runs/blip3o_pretrain/` and found the plan's *starting* constants (`N_low=20000,
M_max=4`) badly mis-tuned for this corpus. **The example config is not
rebalancing — it is a near-uniform 2× duplication of the whole dataset.** Revised
starting point below. (Stages 2–5 + 6a are pure pyarrow arithmetic — no GPU,
`pyarrow`+`pyyaml` only — so this was calibrated by re-running the reduction, not
guessed.)

### A. The concept distribution is far more bottom-heavy than assumed
58,834 active concepts over 19,755,541 images. Per-concept distinct-sample count
`N_c`:

| p50 | p75 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|
| **8** | 92 | 1,310 | 5,755 | 67,005 | 3,509,499 |

The median concept has **8 samples**. So `N_low=20000` lands at ≈ p97 — it labels
**57,443 / 58,834 concepts (97.6%) as "tail"** to be oversampled. The head is
genuinely small and heavy: the 405 concepts with `N_c ≥ 100000` carry **66% of all
link mass**, confirming `N_high=100000` is well placed.

### B. Rarest-wins turns that mislabeling into blanket duplication
Reduction is *rarest-wins = max* over ~11 concepts/image, so a sample only needs
**one** sub-`N_low` concept to be oversampled. With 97.6% of concepts below the
threshold, almost every image qualifies. Full simulation over all 218,641,343
links (per-sample max `m`, then `Σm`):

| config (`N_low`, `M_max`, `γ`) | samples oversampled | expanded total | pinned at `M_max` |
|---|---|---|---|
| **example** 20000, 4, 0.5 | **69.0%** | **×1.99** | 16.0% |
| 2000, 3, 0.5 | 21.8% | ×1.19 | 4.6% |
| **1000, 3, 0.5 (chosen)** | 13.7% | ×1.10 | 1.8% |

The example ×1.99 is indiscriminate: it copies nearly the whole corpus rather than
concentrating budget on genuinely rare concepts.

### C. Chosen starting constants
```
--n_high 100000 --n_low 1000 --a_head 2.5 --r_min 0.1 --gamma 0.5 --m_max 3.0
```
- **`N_high=100000`, `a_head=2.5`, `r_min=0.1`** — unchanged. Top 405 concepts =
  66% of mass are the true head; `a_head=2.5` keeps `m_c(N_high)=1` (mid boundary
  continuous). Head downsampling stays gentle by the reference `2/log10(N_c)` law
  (biggest concept 3.5M → `m_c=0.76`).
- **`N_low=1000`** (was 20000) — ≈ p88 of concepts. Reserves oversampling for the
  genuinely under-represented (<1000 of 19.7M images); concepts with a few
  thousand samples have enough signal and stay at `m=1`. Cuts oversampled samples
  from 69% → 13.7%.
- **`M_max=3.0`** (was 4) — large singleton population (`p25=2`; every concept with
  `N_c ≤ ~110` hits the cap at `N_low=1000, γ=0.5`), so the cap, not `γ`, governs
  the tail. Lowering to 3 holds pinned samples at 1.8% and expansion at ×1.10.
- **`γ=0.5`** — unchanged (sub-linear; cap governs singletons anyway).

Per-concept mass shift under the chosen config (head/mid/tail link mass, before →
after, ignoring rarest-wins coupling): head 66.3% → 62.3%, mid 32.2% → 34.7%,
tail 1.5% → 3.0% — a real flattening, not blanket duplication.

### D. Still a *starting* point — calibrate with Stage 6a
These reduce indiscriminate oversampling but are not final. Run
`build_schedule → materialize → audit` and check `head_reinflation_offenders`
(a head concept riding tail co-occurrences back up → lower `M_max` / raise `r_min`)
and the Gini drop before training. Head downsampling is deliberately mild; if the
audit shows heads still dominant that is the knob to revisit next. Cross-reference
the previous appendix's guard (§E.2): `N_low=1000` pulls mid mislinks like
`pool`/`nail` (75–78k, still MID) no closer to the tail, but `board→committee`
(12k) remains the lone tail mislink — re-run the guard since a constant changed.

Method note (reproducible): `N_c` from `runs/blip3o_pretrain/counts.parquet`;
per-sample simulation joins `node_multiplicity` onto all `links.parquet` rows
(arrow dictionary-encoded to int32 to stay under the 50 GB cgroup) and takes the
group-max per `sample_key`, adding `m=1` for the 9 no-concept images
(19,755,541 − 19,755,532).

---

## Appendix — Three reductions measured, and why the bottleneck is Stage 1 (2026-07-25)

The previous appendix's calibrated constants were run end-to-end, then two
alternative Stage-4 reductions were implemented and measured against them. **All
three fail for the same structural reason, and it is not a constant that can be
tuned.** This appendix records the measurements, the root cause, the comparison
against the reference paper, and the remaining experiment directions.

### A. Results — three reductions, none flattens the distribution

Baseline Gini (before rebalancing) = **0.9753**.

| arm | Stage-4 rule | Gini after | top-2% mass | corpus | membership_hash |
|---|---|---|---|---|---|
| **B** | `max` (rarest-wins) | 0.9688 (−0.0065) | 82.5% | 22.1M (×1.12) | `67aba1624ad8e4ee` |
| **C** | `geomean` (rarity-weighted) | 0.9703 (−0.0050) | 82.8% | 17.8M (×0.90) | `82b8336a955d931e` |
| **D** | stratified + running dedup | 0.9748 (−0.0005) | 84.0% | 19.2M (×0.97) | `677658aed4707880` |
| **E** | `meaninv` α=0.5 cap=16 *(added 2026-07-25, see the next appendix)* | **0.9506 (−0.0247)** | **75.6%** | 19.8M (×1.00) | `0aadde81cde11204` |

Artifacts: `experiments/rebalanced_{B,C,D}/`, audits at
`runs/blip3o_pretrain/audit_{B,C,D}.json`.

**B (max)** — tail intent is realized *exactly*: bucketing realized/intended by
`N_c` shows the three rarest buckets (44,527 concepts, 76% of all concepts) at
ratio **1.00**, full 3.00× amplification, 100% coverage. The cost is the head:
intended 0.76 → realized 1.07 (`n09436708`), i.e. the head *grew*. Tail-vs-head
exposure ratio 0.0231 → 0.0419 (**1.82× relative improvement**), but the tail
still holds only **4.2%** of head exposure in absolute terms.

**C (geomean)** — fixes the head (worst ratio 1.54 → 1.24, median 1.37 → 1.10;
`n09436708` 1.07 → 0.85) but **dilutes the tail**: `N_c=1` concepts fall to ratio
0.33, because their single sample is averaged against ~11 co-occurring concepts.
Gini gets *worse* than B. 85.9% of samples land in `m ∈ [0.5, 0.9]` — a near-uniform
mild shrink that changes no *relative* frequencies.

**D (stratified, the paper's method)** — barely moves anything: only 2.6% of the
corpus dropped, Gini −0.0005. Cause is threshold/distribution mismatch (§C).

### B. Root cause — per-sample multiplicity cannot decouple head from tail

Measured co-occurrence for the worst head offenders (`scratchpad/coupling.py`):

| concept | `N_c` | `m_c` | own `m_c` governs | overridden | median winner `m` |
|---|---|---|---|---|---|
| n09436708 | 3,505,742 | 0.76 | **0.0%** | 100.0% | 1.00 |
| n13104059 | 3,087,006 | 0.77 | **0.0%** | 100.0% | 1.00 |
| n09224566 | 2,981,401 | 0.77 | **0.0%** | 100.0% | 1.00 |
| n06387980 | 2,293,882 | 0.79 | **0.0%** | 100.0% | 1.00 |

The head's intended `m_c` governs **literally zero** of its samples. Crucially the
override is **84.2% by `m=1` (mid band), only 8.7% by `m>1` (tails)** — so this is
*not* tail oversampling riding along. Globally **81.7% of all samples sit at
exactly `m=1`**: with **12.4 concepts/sample** (median 12, p90 15, max 150) and
6,156 mid-band concepts, essentially every sample touches at least one `m_c=1`
concept, and under a `max` one such concept vetoes all downsampling. Only 923,000
samples (4.67%) have all-head concepts and are downsampleable at all — a hard
ceiling on any max-based rule.

**The general statement:** head and tail concepts live in the *same samples*. Any
rule that assigns one scalar per sample must arbitrate, and every arbitration is a
losing trade — `max` resolves toward the tail (no downsampling), `geomean` toward
the middle (uniform shrink), `dedup` cannot un-draw what a rarer concept already
claimed. **This is a Stage-1 tagging property, not a Stage-4 parameter.**

### C. Comparison against the reference paper

| | paper | ours | verdict |
|---|---|---|---|
| taxonomy | ~285K leaf nodes | Bamboo **298,628** nodes | **aligned** |
| tag source | caption → embedding | **dataset's own free-text tags** | **differs** |
| candidates | top-K = 1000 per sample | ~12.4 tags per sample | differs |
| **diversity sampling** | **yes** — aggregate to a dynamically chosen parent level, keep only highest-scoring child per subtree | **absent** | **key gap** |
| concepts actually linked | — | **58,834 (20% of 298,628)** | 80% of taxonomy unused |
| top-2% of categories | >30% of frequency | **84.4%** | ~3× more skewed |
| categories <100K | ~50% | **99.3%** | very different shape |
| single category >10M | yes | **0** (max 3,509,499) | different shape |

Two conclusions:

1. **`n_head=100000` is imported from a different distribution.** In the paper it
   splits roughly half the categories into the downsample branch; here it captures
   **405 concepts (0.7%)** — which nonetheless carry **66.3% of all exposure**. The
   method idles because the threshold does not match this corpus. This is why D
   moved Gini by 0.0005.
2. **The missing Adaptive Diversity Sampling explains both the redundancy and the
   unused 80% of the taxonomy.** Sample tags cluster semantically —
   `modern architecture / glass skyscraper / commercial building / office tower`
   are four phrases in one subtree. The paper collapses such a cluster to one tag;
   we keep all four. High-frequency nodes get hit repeatedly while ~240K long-tail
   nodes are never linked, which is precisely why our distribution is ~3× more
   skewed than the paper's.

Note the tag sources differ: the paper derives candidates from captions (hence
needs top-K=1000 → diversity sampling), whereas our tags ship with the data. The
redundancy therefore arises from **synonymous/near-duplicate free-text tags**
rather than from top-K clustering — so the fix is subtree dedup *after* the
`tag_to_nodes` mapping, not a literal copy of the paper's algorithm.

### D. Taxonomy is a DAG, not a tree

Two caches coexist and disagree:

- `data/taxonomy_cache_bamboo.pkl` (`TaxonomyTree`) — **tree-ified**: all 298,628
  nodes have exactly one parent (`child_to_parent` values are `str`).
- `data/composition_dag.pkl` (`child2parents`) — **the real DAG**: 298,661 nodes,
  **29,465 (9.9%) have >1 parent**, mean 1.12, max 9.

Any subtree-dedup implementation must handle multi-parent nodes explicitly. Two
defensible policies: treat a multi-parent node as belonging to **all** parent
subtrees (more aggressive collapsing), or assign a deterministic single parent
(reproducible but arbitrary). 90.1% of nodes are unaffected either way.

### E. Remaining experiment directions

**Direction 1 — Stage-1 tag de-redundancy (highest priority).** Target: cut 12.4
concepts/sample to 4–5 orthogonal dimensions and link more of the 298,628-node
taxonomy. Three candidate scorers, to be compared **offline on a few×10K samples
before any full implementation**:

1. subtree dedup on the DAG (same-subtree → keep highest score)
2. rarity (IDF) top-5
3. BM25 top-5

Metrics: concepts linked, independent subtrees per sample, head `N_c` reduction.

Two blockers to resolve first:
- **Are captions available?** `sample_tags.parquet` holds only `sample_key` +
  `tags`. BM25 needs caption text; recovering it from the tars is a different
  cost class.
- **Are BM25 and rarity independent?** Tags are largely verbatim substrings of the
  caption, so BM25's discriminative power may come almost entirely from its IDF
  term — which *is* the rarity signal. Measure the correlation; if |r| > 0.7,
  collapse to one metric.

Caveat worth stating: rarity-top-5 is effectively **pre-emptive head downsampling
at tagging time**, and it changes semantics rather than sampling — deleting
`blue sky` from an image that does contain blue sky removes that concept's
supervision entirely, which is not the same as "sample it less".

**Direction 2 — recalibrate `n_head` (cheapest, ~4 min).** Sweep
`n_head ∈ {1K, 5K, 10K, 50K}` for the stratified arm and plot Gini. Directly
quantifies how much of D's failure is threshold mismatch versus structural
coupling.

**Direction 3 — raise `m_max` (cheap, known ceiling).** 3.0 → 6.0 lets the three
rarest buckets scale linearly to 6×, with expansion rising only ×1.12 → ~×1.15.
Bounded by the same coupling: the tail is still at 4.2% of head exposure.

*(A fourth direction — the paper's evaluation-driven "+20–50% boost for weak
capabilities" — is deferred until eval results exist. `StratifiedConfig.boosts`
and `.segments` already implement it.)*

### F. Code and artifacts from this session

- `rebalance/schedule.py` — added `sample_multiplicity_geomean()` (weighted by
  `|log m_c|` so the mid band abstains instead of vetoing).
- `build_schedule.py` — `--reduction {max,geomean}`; **default stays `max`**.
- `rebalance/stratified.py`, `build_stratified.py` — the paper's method
  (ascending frequency + running dedup), emitting `membership.parquet` directly
  and bypassing Stages 3–5. Supports `--boosts` / `--segments`.
- `tests/test_stratified.py` — 14 tests; vectorised path verified **exactly**
  equal to the scalar reference. Two real bugs were caught by this parity test:
  a tie-break mismatch on equal `N_c` (dedup makes selection order-dependent, so
  the vectorised path had to adopt the reference's `(N_c, node_id)` key) and a
  non-deterministic `argpartition` on equal priorities.
- Vectorised geomean verified against the scalar reference (max abs diff 8.9e-16).

**Pre-existing failure, untouched:** `test_multiplicity.py::test_realize_boundaries`
fails on a float edge case — `1.58 - 1 == 0.5800000000000001 > 0.58`, so
`realize_count(1.58, 0.58)` returns 2 rather than the expected 1.

**Environment note:** this pod has **no cgroup memory limit** (`limit_in_bytes` is
the unlimited sentinel; 503 GiB host, 455 GiB available, `failcnt=0`) — earlier
appendices' "50 GB cgroup" constraint does not apply. CPU is capped by **CFS quota
at 8 cores** (`cfs_quota_us=800000 / cfs_period_us=100000`) but `cpuset` is
unrestricted, so `nproc` and `sched_getaffinity` both report 255. Set
`OMP_NUM_THREADS=8` for these stages or thread pools oversubscribe 255-wide
against 8 cores' worth of quota.

---

## Appendix — The three directions tested; two premises refuted, and a working scheme (2026-07-25)

The previous appendix proposed three directions and named Stage-1 tag
de-redundancy the highest-priority one. All three were run. **Directions 1 and 2
rest on premises that measurement does not support**, and the fix turned out to
be neither a threshold nor a tagging change but the *reduction statistic* —
which the previous appendix had concluded was structurally hopeless.

Method: `links.parquet` was dictionary-encoded to int32 once and cached as
`.npy`, making every arm a pure-numpy pass (~2 min instead of a full
`build_schedule → materialize → audit` cycle). The harness reproduces the
published B/C/D Gini and top-2% **exactly** (`scratchpad/reproduce.py`), so the
new numbers are comparable to the old table.

### A. Correction 1 — the corpus has no tag redundancy to remove (Direction 1)

Direction 1 assumed ~12.4 concepts/sample are semantically redundant —
"`modern architecture / glass skyscraper / commercial building / office tower`
are four phrases in one subtree". Three measurements say otherwise:

| source of redundancy | predicted | **measured** |
|---|---|---|
| top-K fan-out (`top_k=3`) | inflates node count | **1.05 nodes/tag**; 94.9% of tags emit exactly 1 node |
| tags → nodes overall | amplification | **0.89×** — 12.50 tags/sample → **11.07 nodes/sample** (tags *collapse*) |
| subtree clustering (DAG, walk up 1/2/3 levels) | 12.4 → 4–5 | **11.03 → 11.50 / 11.21 / 11.08** — collapse **0.96–1.00×** |

Subtree dedup has nothing to collapse: a sample's ~11 concepts already span ~11
independent subtrees, and walking up *increases* the group count because 9.9% of
nodes are multi-parent. Inspection of concrete samples
(`scratchpad/inspect_samples.py`) confirms it — one sample's concepts are
`decay, sky, way, sign, trim, building, paint, corner, facade, path, power_line`,
which live under `natural_process`, `matter`, `action`, `indication`, `artifact`
and `location` respectively. These are genuinely distinct concepts, not synonyms.

**Direction 1 is withdrawn.** Its blockers are moot, but for the record: captions
*are* available (the json sidecar carries `short_caption` / `medium_caption` /
`long_caption` beside `tagging_caption`), so a BM25 arm was never blocked. It is
simply aimed at a problem this corpus does not have.

What the inspection *did* reveal is the real structure, and it is more useful
than redundancy: **every sample pairs a few ultra-common concepts with a few
genuinely rare ones** — `chiavari chair` (N_c=1,270) beside `chair` (516,436) and
`floor` (923,174); `Gingerbread house` (2,108) beside `hair` (2,273,271). The
per-sample rarity signal is real and strong. `max` and `geomean` were simply
discarding it.

### B. Correction 2 — `n_head` was inert because `a_head` neutralized it (Direction 2)

The n_head sweep produced **byte-identical results at 1K, 5K, 10K, 50K and 100K**
— a 100× threshold change with zero effect. Not threshold mismatch: the head
branch never binds. `a_head=2.5` was chosen so `m_c(n_high)=1` for *continuity*,
which forces the keep-rate to **start at 1.0 at the threshold** and decay only to
0.76 at N_c=3.5M. Mean head keep-rate is **0.932** at `n_head=1e5` and **0.996**
at `n_head=1e3` — lowering the threshold only adds concepts that are kept anyway.

Continuity is an aesthetic property, not a requirement. Giving it up and sweeping
both constants together (`scratchpad/sweep_ahead.py`):

| `n_head`, `a_head`, `r_min` | Gini | top-2% | corpus |
|---|---|---|---|
| 100000, 2.5, 0.1 (published D) | 0.9748 | 0.840 | 19.2M |
| 100000, 0.5, 0.1 | 0.9743 | 0.837 | 18.8M |
| 10000, 1.0, 0.1 | 0.9641 | 0.772 | 10.8M |
| 1000, 0.25, 0.02 | **0.9438** | 0.730 | **3.1M** |

So D *can* be pushed far past its published 0.9748 — but only by discarding 84%
of the corpus. This exposed the flaw in the whole published table: **every arm
changes the distribution and the corpus size at once**, so its Gini values were
never comparable to each other.

### C. The controlled experiment: matched budget

Fixing the training budget at 19,755,532 samples for every arm and judging on
four metrics jointly — Gini alone is gameable (see §D) — gives the real ranking:

| arm | Gini | top-2% | distinct | concepts covered | max repeat |
|---|---|---|---|---|---|
| uniform (control) | 0.9753 | 0.844 | 19.8M | 58,834 | 1 |
| `max` (published B) | 0.9688 | 0.825 | 17.9M | 58,834 | 3 |
| `geomean` (published C) | 0.9703 | 0.828 | 18.5M | 58,834 | — |
| 1/min(N_c)^0.5 | 0.9360 | 0.743 | 11.3M | 58,834 | 48 |
| **mean(1/N_c^0.5)** | **0.9488** | **0.755** | 14.0M | **58,834** | 135 |
| mean(1/N_c^1.0) | **0.8643** | 0.675 | 4.1M | **58,834** | 2,116 |

**The previous appendix's central claim — "per-sample multiplicity cannot
decouple head from tail; this is a Stage-1 tagging property, not a Stage-4
parameter" — is false.** A per-sample scalar *can* do it. `max` and `geomean`
failed for a fixable reason: both reduce the **schedule** `m_c`, which has already
destroyed the signal (mid band flattened to exactly 1.0, tail clipped at
`m_max`). With ~11 concepts/sample nearly every sample touches a mid-band
concept, so `max` is vetoed to 1.0 and `geomean` averages toward 1.0.

Reducing the **raw `N_c`** instead keeps the full dynamic range. Gini falls
0.9753 → 0.9488, i.e. **−0.0265 versus the published best of −0.0065 (4× larger)**,
and rarest-decile exposure rises **24×** (0.00006 → 0.00144).

### D. Why Gini alone is not the objective

Single-extreme rules (`1/max(N_c)^α`) reach lower Gini but degenerately:

| arm | Gini | concepts covered | **starved to zero** | max repeat |
|---|---|---|---|---|
| `1/max(N_c)^1.25` | 0.9579 | 54,306 | 4,528 | 30,837 |
| `1/max(N_c)^2.0` | 0.9440 | 43,418 | **15,416** | **2,532,915** |

`1/max^2.0` "wins" on Gini by repeating one image 2.5M times and starving 15,416
concepts out of the corpus entirely. Any future arm must report **coverage and
max-repeat alongside Gini**. The mean statistic is what avoids this: a mean lies
strictly between the per-concept extremes, so no single ultra-rare concept can
blow a sample's weight up — which is why `meaninv` holds full 58,834-concept
coverage at every α tested, with a max repeat of 135 rather than 2.5M.

### E. Chosen scheme

```
--reduction meaninv --alpha 0.5 --cap 16.0     # budget defaults to corpus size
```

    w_s = mean over the sample's concepts of  1 / N_c ** alpha
    m_s = w_s * budget / sum(w),  capped at `cap`, clipped mass redistributed

- **`alpha=0.5`** — the knee of the curve. α tunes flatness against how much the
  budget concentrates: 0.25 → Gini 0.9681 (17.6M distinct), 0.5 → 0.9488 (14.0M),
  0.75 → 0.9067 (8.7M), 1.0 → 0.8643 (4.1M). Beyond 0.5 the distinct-sample count
  falls faster than Gini improves.
- **`cap=16`** — costs almost nothing (0.9488 → 0.9506) and bounds worst-case
  duplication 8×. Uncapped max repeat is 135; p99.9 is 17 either way.
- **No thresholds, no `n_high`/`n_low`/`gamma`/`r_min`, no taxonomy walk.** The
  whole piecewise schedule is bypassed — this reduction reads `counts.parquet`
  directly.

Verification (`scratchpad/validate_final.py`, `scratchpad/parity.py`):
- **Pipeline parity**: `build_schedule.py --reduction meaninv` was run at full
  scale and checked against the scratchpad harness that chose the constants —
  all 19,755,532 common keys match with max abs diff **2.2e-06** (float ordering
  in the cap-redistribution loop), correlation **1.0000000000**. Realized
  metrics agree: **Gini 0.9506, top-2% 0.7565, 14.02M distinct, 58,834 covered,
  max repeat 16**. Artifacts at
  `runs/blip3o_pretrain/meaninv/sample_multiplicity.parquet`.
- **Materialized** as arm **E** — `experiments/rebalanced_E/`: 14,020,894
  distinct samples, expanded total 19,756,464 (×1.41 vs distinct), 3,909 tars,
  `membership_hash 0aadde81cde11204`. The 689-sample gap from the harness
  prediction is the expected difference between `materialize`'s real
  sha256-derived per-sample draw and the harness's numpy RNG.
- **Seed-stable**: Gini 0.9488 ± 0.00000 over 5 seeds; coverage exactly 58,834 every time.
- **Drops redundancy, not content**: of the 29.0% of samples dropped, the median
  *rarest* concept has N_c=26,767 (their rarest concept is still common), versus
  N_c=5,105 for kept samples — 5× rarer. The scheme discards images whose every
  concept is well-covered and keeps those carrying genuine long-tail content.

### F. Still open — this is a distribution result, not a model result

Everything above is Stage-6a arithmetic. **No model has been trained on any of
these arms**, so "flatter" is not yet known to mean "better". The ablation in §6b
remains the deciding experiment, and α=0.5 is the recommended starting arm rather
than a proven optimum. Two specific risks worth holding in mind:

1. **29% of the corpus is dropped** at the budget-matched setting. §E argues the
   dropped samples are the redundant ones, but only training shows whether that
   costs head-concept quality.
2. **Repeats up to 16×** of some rare-concept images could induce memorization —
   the reason for preferring `cap=16` over uncapped, and worth watching in the
   ablation.

Reproduce the chosen arm end-to-end:

```bash
python concept_rebalancing/build_schedule.py \
    --links  concept_rebalancing/runs/blip3o_pretrain/links.parquet \
    --index  concept_rebalancing/runs/blip3o_pretrain/index.parquet \
    --counts concept_rebalancing/runs/blip3o_pretrain/counts.parquet \
    --output_dir concept_rebalancing/runs/blip3o_pretrain/meaninv \
    --reduction meaninv --alpha 0.5 --cap 16.0

python concept_rebalancing/materialize_rebalanced.py \
    --sample_multiplicity concept_rebalancing/runs/blip3o_pretrain/meaninv/sample_multiplicity.parquet \
    --index concept_rebalancing/runs/blip3o_pretrain/index.parquet \
    --output_dir experiments/rebalanced_E --name rebalanced_E

python concept_rebalancing/audit.py \
    --links concept_rebalancing/runs/blip3o_pretrain/links.parquet \
    --node_multiplicity concept_rebalancing/runs/blip3o_pretrain/node_multiplicity.parquet \
    --membership experiments/rebalanced_E/membership.parquet \
    --output concept_rebalancing/runs/blip3o_pretrain/audit_E.json
```

`audit_E.json` confirms the headline through the real pipeline:
**Gini 0.9753 → 0.9506 (−0.0247), top-2% 84.4% → 75.6%**, 14,020,894 distinct,
expanded ×1.41.

Read `audit.py`'s realized-vs-intended columns with care for this arm: they
compare against the Stage-3 `m_c` that `meaninv` bypasses, so they describe the
arm rather than grading it. Both directions are informative anyway:

- "Head re-inflated" offenders (`label` 2.42×, `food photography` 2.29×,
  `packaging`, `illustration`) all sit at N_c 100–250K — **mid-band, not the
  true head**. They gain because they co-occur with genuinely rare concepts.
  The real head (`sky`, `building` at ~3.5M) does not appear on the list.
- "Tail under-realized" concepts land at realized ≈ 1.0 against an intended
  3.5–4.0×. This is by design: α=0.5 does not push singletons to 4× copies; it
  raises the **rarest decile's share of total exposure 24×** (0.00006 → 0.00144),
  which is the outcome the multiplicity schedule was only proxying for.

The `Calibrate:` hint the script prints ("lower M_max / raise r_min / reduce
gamma") does not apply — none of those constants exist in this arm. For a
like-for-like read of any arm, `scratchpad/core.py` + `sweep.py:metrics` is the
instrument used above.

### G. Code and artifacts from this session

- `rebalance/schedule.py` — added `sample_weight_meaninv()`.
- `build_schedule.py` — added `per_sample_meaninv()` and
  `--reduction meaninv --alpha --budget --cap`. **Default stays `max`**; the new
  path reduces raw `N_c` and bypasses Stage 3.
- `tests/test_schedule.py` — 5 new tests (no-concept, rare-sample ranking,
  no-single-concept-veto, bounded-by-extremes, α-monotonicity).
- `tests/run_all.py` — **now includes `test_stratified.py`**, which existed but
  was never being run.
- `rebalance/multiplicity.py` — **fixed** the pre-existing `realize_count` float
  bug the last appendix flagged (`1.58 - 1 == 0.5800000000000001`); the fraction
  is now rounded to 12 dp. All tests pass.
- `scratchpad/` — `core.py` (cached-link harness + metrics), `reproduce.py`
  (exact B/C/D reproduction), `sweep.py`, `sweep_ahead.py`, `diag_strat.py`,
  `redundancy.py`, `inspect_samples.py`, `matched.py`, `invfreq_sweep.py`,
  `final_tune.py`, `validate_final.py`, and their `.log`/`.json` results.

---

## Appendix — Stage-1b guard re-run under `meaninv`: the mislinks are no longer inert (2026-07-25)

The 2026-07-24 Stage-1b audit concluded false-friend mislinks were **inert to the
schedule** and that we should not relink. That conclusion was correct *for `max`*
and its §E.2 explicitly required re-running the guard "whenever schedule
constants change". `meaninv` changes more than a constant — it changes what the
reduction reads — so the guard was re-run. **It now fires.** The fix is cheap,
measured, and implemented; the "do not relink" decision still stands.

### A. Why the old inertness argument does not carry over

It rested on two properties of `max` that `meaninv` does not have:

1. *"A mislink only matters if it **decides** the sample's `m`"* — under `max`
   only the single rarest concept speaks, so a head mislink was a passenger
   0.0–0.5% of the time. **A mean has no passenger seat**: every concept
   contributes to every sample it appears in.
2. *"head→head is band-preserving"* — true only for a piecewise schedule with a
   flat head band. `meaninv` has no bands; it reads raw `N_c`, so a wrong-sense
   node with a different count shifts the weight continuously.

Measured upper bound on the disturbance (remove the mislinked concept entirely,
`scratchpad/false_friends.py`): **median 3.4–6.2% weight shift**, p95 ≈ 7–11%,
across `female`(1.16M), `table`(759K), `car`(860K), `glass`, `plant`, `board`.
That is the *most extreme possible* correction — a relink to the correct
same-band sense moves far less — and it sits within the ~1/11 share a single
concept can hold. **These head mislinks remain acceptable and are not fixed.**

The real exposure is elsewhere.

### B. The guard fires: two failure modes in the tail

`meaninv` concentrates weight exactly where §E.2 pointed. Every sample behind
each flagged node was reviewed by hand (`scratchpad/tail_guard_review.py`;
~2.2K images total, so this was minutes, not a project):

| node | `N_c` | verdict | evidence |
|---|---|---|---|
| `seal` | **12** | **MISLINK** | all 12 are **wax sealing** (`wax`, `stamp`, `candle`, `writing`); not one animal |
| `bank` | 335 | **MISLINK** | ATMs, Chase/Woori/Bank of Ceylon; co-occurs `building`,`signage`,`facade`; no riverbank |
| `pitcher` | 254 | **MISLINK** | **251/254 also link `pitcher (container)`** — the baseball sense misfires on jugs |
| `mint` | 613 | MIXED | real mint candy, but `packaging`(222)/`coin`(124), incl. the US Mint building |
| `jersey` | 107 | MIXED | garment + Jersey cattle + the island (`guernsey`, Channel Islands maps) |
| `pool` | 812 | **CORRECT** | swimming/mineral pools (`water`,`terrace`,`resort`) — **not** billiards |

Note the last row corrects the earlier audit's `pool`→billiards entry: at this
node the linked sense is the right one.

**Without the guard those mislinks were being amplified**, which is precisely the
"spurious junk oversampling" mode §E.2 was written to prevent:

| node | `N_c` | mean copies | max copies |
|---|---|---|---|
| `seal` | 12 | **7.25** | 9 |
| `pitcher` | 254 | 2.90 | 10 |
| `bank` | 335 | 2.63 | **16 (the cap)** |

**Second failure mode — split surface forms (new, and more systemic).** 456 tail
nodes share a base name with a far more common node. This is a *taxonomy
granularity* artifact, not a sense error, and it was completely harmless under
`max` — but under `1/sqrt(N_c)` the near-empty twin dominates:

| base name | tail twin | head twin | weight ratio |
|---|---|---|---|
| `surface` | **N=1** | 513,825 | **717×** |
| `building` | **N=10** | 3,509,499 | **592×** |
| `eye` | 8 | 404,148 | 225× |
| `flower` | 57 | 641,185 | 106× |

### C. The fix is essentially free

`rebalance/tail_guard.py` excludes the 3 confirmed mislinks plus the 456 split
forms from the weight computation (the *samples* are retained — only the noisy
concept stops contributing). Excluded: **456 of 58,834 nodes (0.78%), carrying
0.0268% of links.**

| arm | Gini | top-2% | distinct | covered | max repeat |
|---|---|---|---|---|---|
| E, no guard | 0.9506 | 0.7565 | 14,021,583 | 58,834 | 16 |
| **E + guard** | **0.9507** | 0.7567 | 14,041,868 | 58,826 | 16 |

**Gini moves by 0.0001** — the guard is not trading flattening for correctness,
it is nearly free. Coverage drops by 8 concepts, all near-empty split twins of a
head node (e.g. `surface` N=1), i.e. noise rather than lost supervision.

Enabled by default; `--no_tail_guard` reproduces the pre-guard numbers. Pass
`--node_names tag_to_nodes.parquet` so split-form detection can run (without it
only the 3 confirmed mislinks are excluded).

### D. The delivered arm

`experiments/rebalanced_E_guarded/` — **this is the arm to train**, not the
un-guarded `rebalanced_E`:

```
name: rebalanced_E_guarded      distinct_samples: 14,040,822
actual_samples: 19,756,413      num_shards: 3,909
membership_hash: f454a3aa73badd9e
```

Audited end-to-end through the real pipeline (`audit_E_guarded.json`):
**Gini 0.9753 → 0.9507 (−0.0246), top-2% mass 84.4% → 75.7%**, ×1.41 expansion —
matching the scratchpad prediction of 0.9507 exactly.

`actual_samples` (19.76M) is the expanded total and the number `MAX_STEPS` must
budget against; `distinct_samples` (14.0M) is the image count. Using the latter
would under-train by 29%.

**Trainer wiring is now in place and tested.** `blip3o/train/train.py` calls
`rebalanced_dataset.register()` when `--dataset_cls rebalanced` is passed —
previously `get_dataset_cls` only knew `'mix'` and `register()` was never called
anywhere, so any rebalanced run would have died at startup with
`Unknown dataset class rebalanced`. B/C/D were never trained, so this path had
never executed.

A CPU-only smoke test on real shards (`scratchpad/smoke_dataset.py`) confirms the
expansion end-to-end: 10,087 raw rows → 6,733 kept (67%, tracking the corpus-wide
71%) → 8,793 expanded instances, every row repeated exactly its `count`, max 16,
and `lengths`/`modality_lengths` sized to the **expanded** length — had the
subclass inherited the base versions, the length-grouped sampler would have
silently emitted only the unexpanded rows and dropped ~29% of the epoch.
`tests/test_rebalanced_dataset.py` pins these invariants (8 tests, and they run
with or without torch installed).

**Set `--num_loading_workers` low.** The HF `filter` pickles the membership key
set into every worker: measured **491 MB and 6.4 s per worker** for this
membership. At `train.py`'s default of 32 that is ~15.7 GB and ~3.5 min of pure
pickling before step 0. Use 4–8.

Caveat on the smoke test: the real training env (`blip3o-next`, under
`/user/liuxinyu/miniforge3`) does not exist on the CPU pod, so it ran in
`data-juicer` (torch + datasets present). It validates the rebalancing logic, not
the full trainer stack — first startup on a GPU node is still the real check.

### E. Standing recommendation

1. **Still do not relink, and still do not build the WSD relinker.** The head
   mislinks cost 3–6% weight at the upper bound. Unchanged from §E.1.
2. **The guard is now a permanent Stage-2/3 step, not a one-off check.** Re-run
   `tail_guard.audit()` whenever `alpha`, `cap`, or the reduction changes — the
   failure mode scales with how much weight the tail receives, and `alpha > 0.5`
   makes it strictly worse.
3. **`mint` and `jersey` stay in** (excluding them would drop genuine
   supervision) but are recorded in `REVIEWED_MIXED` — keep them off any
   hand-curated tail eval list, per §E.3.
4. **Re-review if `alpha` rises.** At α=0.75 or 1.0 the tail's share of exposure
   roughly doubles and then quadruples, so nodes that are currently borderline
   would need the same scrutiny `seal` got.

---

## 附录 — 最终采样方案的完整中文说明(2026-07-25)

本节用中文完整描述**最终确定并已落盘**的采样逻辑,是前面几个英文附录的结论汇总,
不引入新结论。所有数字均来自实际产出物(`audit_E_guarded.json`、
`experiments/rebalanced_E_guarded/config.yaml`),非估算。

### 一、一句话概括

对每张图,计算它所含概念的**逆频率均值**作为权重,归一化到训练预算后取整,得到该图
被训练的次数(0 到 16 次)。稀有概念多的图被多次训练,全是常见概念的图被丢弃或只训一次。

```
--reduction meaninv --alpha 0.5 --cap 16.0
```

### 二、公式与逐步计算

**第 1 步:统计每个概念的样本数 `N_c`**(Stage 2,`counts.parquet`)
全库 58,834 个概念,`N_c` 从 1 到 3,509,499。分布极度倾斜:中位数仅 8,而前 405 个
概念(0.7%)占了全部曝光的 66%。

**第 2 步:算每张图的权重**

```
w(图) = mean over 该图的每个概念 c ( 1 / N_c ^ 0.5 )
```

即对该图的每个概念取 `1/√N_c`,再求**算术平均**。举例(真实样本):

| 概念 | `N_c` | `1/√N_c` |
|---|---|---|
| chair | 516,436 | 0.0014 |
| floor | 923,174 | 0.0010 |
| dress | 853,481 | 0.0011 |
| **chiavari chair** | **1,270** | **0.0281** |
| **ballroom** | **2,726** | 0.0192 |
| ...(共 12 个概念) | | |

这张婚礼图因为含 `chiavari chair`(全库仅 1,270 张)而获得高权重,尽管它同时含
`chair`、`floor` 这些超常见概念。

**第 3 步:归一化到预算并截断**

```
m(图) = w(图) × 预算 / Σw        # 预算默认 = 原语料大小 19,755,541
m(图) = min(m(图), 16)           # 截断,被削掉的量重新分配给未截断的图
```

截断后重新分配是迭代进行的(最多 50 轮),保证 `Σm` 始终等于预算。实测
`Σm = 19,755,541`,与预算完全一致。

**第 4 步:取整为实际训练次数**(Stage 5,`materialize_rebalanced.py`)

```
h     = sha256(图的 key) / 2^63        # 该图自身决定的固定随机数 ∈ [0,1)
count = floor(m) + (1 if h < m - floor(m) else 0)
```

用图自己的哈希而非全局随机数,好处是**同一个 schedule 永远复现同一个数据集**,
无需存状态。`count = 0` 的图被丢弃。

**第 5 步:训练时展开**(Stage 6,`rebalanced_dataset.py`)
把每行的索引重复 `count` 次生成 `index_map`,再打乱(种子 42),使同一张图的多份副本
落在不同 batch。**重复的是索引条目,不是图片字节**,HF 行保持内存映射,无额外存储开销。

### 三、为什么用「均值」,而不是 max 或 geomean

这是整个工作最关键的一点。前一版方案用的是 `max`(rarest-wins,取最稀有概念的
`m_c`),失败原因有两层:

**第一层:reduce 的输入错了。** `max` 和 `geomean` 作用在 Stage-3 的分段 schedule
`m_c` 上,而 `m_c` 已经把中间频段全部压平成 1.0、把尾部截在 `m_max`。信号在被 reduce
之前就已经丢失了。`meaninv` 直接读**原始 `N_c`**,保留了完整动态范围。

**第二层:极值统计量会被单个概念绑架。**

- 用 `max`:每张图平均含 11 个概念,几乎必然碰到某个 `m_c=1` 的中频概念,而一个这样的
  概念就能否决全部降采样。实测头部概念的意图 `m_c` **对其 100% 的样本都不起作用**。
- 用 `1/min(N_c)`(最稀有概念):走向另一个极端,单个超稀有概念能让权重爆炸。
- 用**均值**:数学上均值严格落在各概念 `1/N_c` 的最小值与最大值之间,所以既不会被
  中频概念一票否决,也不会被单个极稀有概念绑架。这就是为什么 `meaninv` 能在
  α 从 0.25 扫到 1.0 的全过程中始终保持 58,834 个概念**零饿死**。

对比(固定预算 19.7M):

| 方案 | Gini | 概念覆盖 | 饿死概念 | 最大重复 |
|---|---|---|---|---|
| 不做处理(uniform) | 0.9753 | 58,834 | 0 | 1 |
| `max`(旧 B 臂) | 0.9688 | 58,834 | 0 | 3 |
| `1/max(N_c)^2.0` | 0.9440 | 43,418 | **15,416** | **2,532,915** |
| **`meaninv` α=0.5** | **0.9507** | **58,834** | **0** | **16** |

第三行是重要的反面教材:它 Gini 看起来最好,但代价是把一张图重复 250 万次、饿死
15,416 个概念。**所以 Gini 绝不能单独作为目标,必须同时看概念覆盖和最大重复次数。**

### 四、两个参数怎么定的

**`alpha = 0.5`** —— 控制「压平程度」与「预算集中度」的取舍。α 越大越平,但预算会集中
到越少的图上:

| α | Gini | 保留的不重复图片 |
|---|---|---|
| 0.25 | 0.9681 | 17.6M |
| **0.5(选定)** | **0.9506** | **14.0M** |
| 0.75 | 0.9067 | 8.7M |
| 1.0 | 0.8643 | 4.1M |

选 0.5 是因为超过它之后,不重复图片数下降得比 Gini 改善得更快。**α=0.5 是推荐起点,
不是已证明的最优值** —— 只有训练结果能定最优。

**`cap = 16`** —— 无截断时最大重复 135 次。截到 16 只让 Gini 从 0.9488 变到 0.9506
(几乎无损),却把最坏情况的重复压低 8 倍,降低记忆化风险。

**已删除的参数:** `n_high`、`n_low`、`gamma`、`r_min`、`m_max` 全部不再需要 ——
`meaninv` 没有分段、没有阈值、不走分类树。

### 五、链接噪声防护(guard)

`meaninv` 把权重集中在尾部,这使**链接错误第一次变得危险**(在 `max` 下是惰性的)。
`rebalance/tail_guard.py` 默认排除两类节点:

1. **已人工确认的误链** 3 个 —— `seal`(N=12,12 张全是火漆封蜡而非海豹,无 guard 时
   平均被复制 **7.25 次**)、`bank`(N=335,全是银行)、`pitcher`(N=254,棒球义误触发)。
2. **同名分裂节点** 456 个 —— 同一个词同时存在高频和近乎空的两个节点,例如
   `building` 有 N=3,509,499 和 **N=10** 两个,后者权重是前者的 **592 倍**。这是分类树
   粒度问题而非词义错误。

共排除 456 个节点(占 0.78%,链接质量 0.0268%),**Gini 仅变化 0.0001** —— 修复几乎免费。
被排除的概念只是不再贡献权重,**图片本身仍通过其他概念保留**。

`pool` 经查是正确词义(游泳池,非台球),`mint`/`jersey` 词义混杂但保留(排除会丢真实监督),
均已记录在代码里避免日后重复讨论。

### 六、最终数据集实况

**`experiments/rebalanced_E_guarded/`** ← 训这个,不是 `rebalanced_E`(后者无 guard)

```
distinct_samples : 14,040,822   (原语料的 71.1%)
actual_samples   : 19,756,413   ← MAX_STEPS 按这个算
num_shards       : 3,909
membership_hash  : f454a3aa73badd9e
```

| 指标 | 前 | 后 |
|---|---|---|
| Gini | 0.9753 | **0.9507**(−0.0246) |
| top-2% 概念占比 | 84.4% | **75.7%** |
| 概念覆盖 | 58,834 | **58,834(零饿死)** |
| 总训练量 | 19,755,541 | 19,756,413(**×1.000,等算力**) |

复制次数分布:1 份 **78.06%**、2 份 14.66%、3 份 3.75%、>10 份 0.40%,
`max=16, mean=1.407, p90=2, p99=7`。**近八成图片只训一次**,重复集中在很窄的尾部。

丢弃的 28.9% 是冗余而非独特内容:被丢图片的**最稀有概念**中位数 `N_c=26,767`
(连它们最罕见的概念都仍很常见),而保留图片是 5,105(稀有 5 倍)。

### 七、训练命令与注意事项

```bash
--dataset_cls rebalanced \
--experiment_dir experiments/rebalanced_E_guarded \
--num_loading_workers 4
```

- **`MAX_STEPS` 必须按 `actual_samples`(19,756,413)算**,用 `distinct_samples`
  (14,040,822)会少训 29%。
- **`--num_loading_workers` 要设小**:HF `filter` 会把 membership key set 序列化进
  每个 worker,实测 **491 MB / 6.4 秒每 worker**。默认 32 意味着 step 0 前就要烧掉
  ~15.7 GB 内存和 ~3.5 分钟。
- 接线已完成:`train.py` 在 `--dataset_cls rebalanced` 时自动调用 `register()`,无需手改。

### 八、必须说明的局限

1. **没有训练过任何模型。** 以上全部是分布层面(Stage 6a)的算术结果。「分布更平」
   **尚未被证明**等于「模型更好」,§6b 的对照实验才是决定性的。
2. **α=0.5 是起点不是最优。** 同理,`cap=16` 也是。
3. **丢掉了 28.9% 的语料。** §六给出的证据支持「丢的是冗余」,但只有训练能验证这是否
   损害了头部概念的质量。
4. **部分图片重复达 16 次**,存在记忆化风险 —— 这也是选 `cap=16` 而非不截断的原因,
   训练时值得关注。
5. **冒烟测试未在真实训练环境跑过。** `blip3o-next` 环境在 GPU 节点上,CPU pod 用
   `data-juicer` 代跑,验证的是重平衡逻辑而非完整训练栈。

## 附录 — 训练环境落地与基础设施阻塞(2026-07-26)

本节记录把重平衡数据集接到真实训练栈时踩到的问题,补上 §八局限 5 所说的
「冒烟测试未在真实训练环境跑过」。**截至本节写作时训练仍未成功启动**,
阻塞在节点侧而非代码侧。

### 一、`datasets==2.16.1` 的 webdataset builder 无法并行

`num_proc>1` 时必然失败:

```
TypeError: cannot pickle 'ExFileObject' object
```

根因在 `packaged_modules/webdataset/webdataset.py:51` —— `_split_generators`
把**活的 tar 迭代器**(已打开的文件句柄)放进 `gen_kwargs`,而 `num_proc>1` 时
`datasets` 必须 pickle `gen_kwargs` 分发给 worker,打开的 `ExFileObject` 不可
pickle。用 8 个 shard + `num_proc=4` 即可复现,与并发数无关。

最坏的一点:它**流式跑完整个语料才崩,且缓存目录零字节**。首次全量构建跑到
19,164,907 / 19.7M(约 97%)才失败,3 小时白费。

**解法**(不动第三方库,升级 `datasets` 会牵动 `transformers 4.51.3`):把并行
放到 `datasets` 外面 —— `scripts/build_arrow_cache_parallel.py` 每个子进程用
`num_proc=1` 建一批 shard,多进程并发。实测 3909 shard / 62 chunk / 24 并发,
**36.9 分钟完成,零失败**,且每个 chunk 独立落盘,中断可续。

### 二、分块缓存必须以相同 chunk_size 读回

分块缓存按各自的 `data_files` 做 key。一次性传全部 3909 个 shard 会**完全
miss**,退回单进程重建(约 6 小时,8 卡全程空转)。`rebalanced_dataset.py` 已改为
按 `BLIP3O_ARROW_CHUNK_SIZE`(默认 64)分块加载再 `concatenate_datasets`。
**建缓存与训练必须用同一个值。**

已验证这个改动不改变训练结果:同一份 shardlist 下,单次加载与分块加载的
`__key__` **逐条同序**,`index_map` 与解引用后的**训练样本序列完全一致**
(shuffle 用固定 seed 42,`count_map` 按 `__key__` 查表而非行号)。

### 三、缓存与采样策略解耦

`load_dataset` 只收 `data_files`,`membership.parquet` 不参与缓存 key。因此
**改 alpha / cap / reduction / guard 名单、乃至退回 uniform,都不需要重建缓存**,
只需重跑 `materialize_rebalanced.py` 换 `experiment_dir`。实测三种不同倍数策略
共用同一份缓存,零重建。

代价是缓存必须是**全量**语料(996G):被丢弃的 28.9% 也要解码,否则 `filter`
无从下手。为省这部分空间而只解码在册样本,会牺牲跨实验臂的复用性,不值得。

### 四、`save_to_disk` 合并加速尝试 —— 失败,已放弃

动机是消掉每次启动约 14 分钟的分块元数据解析(每个 rank 各付一遍)。
`scripts/consolidate_arrow_cache.py` 合并成一份 `save_to_disk`:

- 合并结果**正确**:`rows=19,755,541`(与 §二预算逐位一致),`pkl` 列已丢弃
- 但 `load_from_disk` 耗时 **776.7s(12.9 分钟)**,对比分块的约 14 分钟,
  **几乎没有改善**。140 分钟写盘 + 911G 空间换来零收益。

原因:`num_shards=2000` 设得过高,`load_from_disk` 要逐个打开 2000 个 shard 建
内存映射,开销随 shard 数线性增长,把收益吃光。小样本测出的 0.028s 是 8 个
shard 的结果,**不能外推**。

顺带记录一个独立的坑:`save_to_disk` 不传 `num_shards` 时会先跑
`_estimate_nbytes()`(`arrow_dataset.py:1453`),对 19.7M 行是**单线程全表扫描且
无进度输出**,实测卡 37 分钟仍未开始写盘,极易误判为死锁。显式传 `num_shards`
可跳过。

**结论:就用分块缓存训练。** 启动多花约 14 分钟,对 154,346 步的训练可忽略。
`BLIP3O_ARROW_CONSOLIDATED` 保留为可选开关但**默认不启用**(无实测收益,不应让
训练走未验证路径)。合并产物可删。

### 五、conda 环境不能只装在容器 overlay 上

`/root/miniconda3` 位于容器**本地可写层**(overlay),不在 `/cephfs`。在 CPU pod
里 `conda create` 建的 env 只存在于该 pod 的 overlay,GPU 容器**用同一镜像启动
也看不到**。镜像相同 ≠ overlay 内容相同。

症状是 `torchrun: command not found` 而**无任何 conda 报错** —— 因为
`source ... && conda activate` 在 `source` 失败时短路且不报错,脚本一路跑到
torchrun 才崩(exit 127),平台重试 4 次全部同样失败。

区分办法:镜像自带的 env 时间戳更早(`data-juicer` 1/04、`qwen` 6/05),
`blip3o-next` 是 7/25 —— 属于事后写入本 pod 的内容。

**两条解法**:平台的「加载存档」把家目录恢复进 GPU 容器(`/root` 76G,其中
`miniconda3` 71G;实测这条路可行,env 与 torchrun 均正常);或把 env 装到
`/cephfs` 共享。`run_rebalanced_single_node.sh` 已加前置检查,缺 conda 时**立即
报错并提示 `CONDA_ROOT` 可覆盖**,不再跑到第 87 行才崩。

### 六、A800-SXM4 节点 Error 802 —— 当前阻塞项

环境和脚本都已就绪(conda 正常、torchrun 正常、8 卡识别、`MAX_STEPS` 算对
154,346),但 CUDA 初始化失败:

```
Error 802: system not yet initialized
→ ValueError: ProcessGroupNCCL is only supported with GPUs, no GPUs found!
```

诊断证据:

```
torch.cuda.is_available() -> False     # 但 device_count() -> 8
nvidia-smi -q | grep Fabric:
    Fabric State  : N/A
    Fabric Status : N/A
    GPU Fabric GUID : N/A
```

`nvidia-smi` 列出设备只说明驱动能读元数据。A800-SXM4 靠 NVSwitch 互联,必须由
宿主机 `nvidia-fabricmanager` 完成 fabric 初始化(State 应为 `Completed`)才能
创建 CUDA context。全部 `N/A` 说明 fabricmanager 未参与 —— 未启动,或与驱动
575.57.08 版本不匹配。**该层在容器内不可修。**

处置:优先换驱动 535 重开节点(fabricmanager 需与驱动主版本一致;换 535 无需重装
环境,torch 2.3.0+cu121 与 flash-attn cu123 wheel 在 535 上同样可跑);若仍
`N/A` 则报集群管理员。

### 七、若改用 8×A10-24G

可跑但需改两处,且要接受慢很多:

1. **`bs=16` 必然 OOM。** A10 是 24GB(A800 的 30%)。ZeRO-1 只切分优化器状态,
   参数与梯度每卡各存完整一份:约 2.2B 参数下静态占用约 13GB,只剩约 11GB 给
   激活值,`bs=16` + `model_max_length 2048` 远超。改 `PER_DEVICE_BS=2` +
   `GRAD_ACCUM=8`,global_batch 仍为 128,**训练数学等价**,`max_steps` 不变。
2. **A10 无 NVLink**,8 卡走 PCIe,`zero1.json` 里 `1e9` 的 bucket 会造成明显
   停顿,建议降到 `2e8`。

吞吐粗估仅 A800 的 0.2–0.3 倍(算力 125 vs 312 TFLOPS,互联 ~32 vs ~400GB/s),
154,346 步耗时约 3–5 倍。**适合冒烟验证,不适合跑完整预训练。**

### 八、其他已确认的落地细节

- **flash-attn 不要源码编译。** PyPI 无预编译包,`setup.py` 首行 `import torch`
  在 pip 构建隔离环境里直接 `ModuleNotFoundError`,且编译需 nvcc。用官方 wheel:
  `flash_attn-2.6.2+cu123torch2.3cxx11abiFALSE-cp311`。cu123 配 cu121 torch 是
  对的(只发 cu118/cu123 两档,12.x minor 间 ABI 兼容);ABI 变体由
  `torch._C._GLIBCXX_USE_CXX11_ABI`(本环境为 `False`)决定。**装 wheel 不需要
  GPU 也不需要 nvcc**,CPU 节点可装,`import` 才需要 GPU。
- **flash-attn 非硬依赖。** 全仓仅两处默认参数(`train.py:82`、`builder.py:8`),
  无顶层 `import`,CPU 上调试可改 `sdpa`/`eager`。
- **caption 用 `long_caption`,不是 `txt`。** recaptioned 数据的真 caption 在
  `.json` 的 `long_caption`,tar 内 `.txt` 是旧 caption。抽查 4 个 shard 各 200
  条,800/800 非空。注意 `dataset.py:354` 的 `meta.get(caption_key, txt)` 在字段
  缺失时会**静默回退到旧 `.txt`**,拼错字段名不报错。
- **`pkl` 列(36MB/shard)训练全程不读**,`keep_cols` 会丢弃,但那是在缓存建好
  *之后*。
- **`deepspeed==0.14.4` 可在无 GPU/无 nvcc 节点安装**(`DS_BUILD_OPS` 默认 0,
  CUDA 算子运行时 JIT)。
- **SHM 默认 4GB 不够。** 8 卡 DDP + DeepSpeed 用 `/dev/shm` 传张量,建议 ≥128GB,
  否则易在训练中途 `Bus error`。
- **checkpoint 落在 `/cephfs`**(非 overlay),容器销毁不丢。`save_steps 5000` 意味
  首个存档要 64 万样本之后,首次跑建议先调小验证。脚本**未设**
  `--resume_from_checkpoint`,崩溃重启会从 step 0 开始。
