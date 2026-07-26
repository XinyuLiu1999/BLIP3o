# Concept Rebalancing for BLIP3o Pretraining

Offline resampling of the pretraining corpus at the **concept** (Bamboo node)
level — downsample over-represented head concepts, retain the mid band,
oversample tail concepts — then train BLIP3o on the rebalanced set and measure
the effect on tail-concept generation.

This is the implementation of `docs/concept_rebalancing_plan.md`. It is a
**separate** package: it *reuses* the existing `scripts/data_pipeline` and
`semantic_image_browser` code but does not modify either. The one trainer-side
change is delivered as a subclass + a monkeypatch (`rebalanced_dataset.py`), so
`blip3o/data/dataset.py` is untouched.

## The unifying idea — per-sample multiplicity `m`

Every sample gets a real target multiplicity `m ≥ 0`, realized to an integer copy
count from the sample's own deterministic hash (`rebalance/multiplicity.py`):

```
h     = priority / 2**63           # priority = sha256(sample_key) mod 2**63  (index_tars)
count = floor(m) + (1 if h < m-floor(m) else 0)
#   m=0 drop · 0<m<1 downsample · m=1 retain · m>1 oversample
```

The draw `h` is the **same** number `index_tars.py` already stores as `priority`,
so any membership is re-derivable from the schedule alone.

## Pipeline

| Stage | Script | Output | Needs GPU / browser env? |
|---|---|---|---|
| 0 Index shards | `scripts/data_pipeline/index_tars.py` (reused) | `index.parquet` | no |
| 1 Tag→concept link | `link_corpus.py` | `sample_tags.parquet`, `vocab.txt`, `tag_to_nodes.parquet`, `links.parquet` | **yes** (resolve step) |
| 1b QA gate + 2 counts | `qa_gate.py` | `qa_report.json`, `counts.parquet` | **yes** |
| 2–4 Schedule | `build_schedule.py` | `counts.parquet`, `node_multiplicity.parquet`, `sample_multiplicity.parquet` | no |
| 5 Materialize | `materialize_rebalanced.py` | `membership.parquet`, `shardlist.txt`, `config.yaml` | no |
| 6a Audit | `audit.py` | `audit_*.json` | no |
| 6 Train | `rebalanced_dataset.py` + `scripts/data_pipeline/run_experiment.sh` | model | (training env) |

Everything except Stage 1 and 1b is pure arithmetic over parquet — cheap to
re-run while **calibrating** the schedule constants against the Stage-6a audit.

### Environments

- **Stage 1 / 1b (GPU)** — the browser's `wiki` conda env
  (`/root/miniconda3/envs/wiki/bin/python`: torch+cuda, sentence-transformers,
  lancedb) plus its prebuilt caches under
  `/cephfs/liuxinyu/semantic_image_browser/data/` (`bamboo_V4.json`,
  `taxonomy_cache_bamboo.pkl`, `embedding_index_bamboo.lance`). Point at a
  different checkout with `--browser_root`.
- **Stages 2–6a** — any env with `pyarrow` + `pyyaml`.
- **Stage 6 (train)** — the BLIP3o training env.

## Walkthrough (example paths)

```bash
CORPUS=/cephfs/liuxinyu/BLIP3o-Pretrain-Long-Caption-filtered-recaptioned
RUN=concept_rebalancing/runs/blip3o_pretrain
WIKI=/root/miniconda3/envs/wiki/bin/python

# Stage 0 — index the shards (existing pipeline)
python scripts/data_pipeline/index_tars.py --tar_dir $CORPUS --output $RUN/index.parquet

# Stage 1 — tag/link (GPU; vocab-cached)
$WIKI concept_rebalancing/link_corpus.py --tar_dir $CORPUS --output_dir $RUN --num_workers 32

# Stage 1b — QA gate (inspect groups; go/no-go) + Stage 2 counts
$WIKI concept_rebalancing/qa_gate.py --links $RUN/links.parquet --output_dir $RUN \
    --total_images $(python -c "import pyarrow.parquet as pq;print(pq.read_metadata('$RUN/index.parquet').num_rows)")

# Stages 2–4 — schedule (re-run freely while calibrating constants)
python concept_rebalancing/build_schedule.py \
    --links $RUN/links.parquet --index $RUN/index.parquet --output_dir $RUN \
    --n_high 100000 --n_low 20000 --gamma 0.5 --m_max 4.0

# Stage 5 — materialize the rebalanced arm (an experiment_dir)
python concept_rebalancing/materialize_rebalanced.py \
    --sample_multiplicity $RUN/sample_multiplicity.parquet \
    --index $RUN/index.parquet --output_dir experiments/rebalanced_B --name rebalanced_B

# Stage 6a — distribution audit (Gini drop + realized≈intended)
python concept_rebalancing/audit.py \
    --links $RUN/links.parquet --node_multiplicity $RUN/node_multiplicity.parquet \
    --membership experiments/rebalanced_B/membership.parquet \
    --counts_names $RUN/counts.parquet --output $RUN/audit_B.json

# Stage 6 — train (see "Trainer wiring")
sbatch scripts/data_pipeline/run_experiment.sh experiments/rebalanced_B
```

### Ablation arms (plan §6b)

- **A baseline** — full/uniform (materialize with all `m=1`, or the existing flat pipeline).
- **B rebalanced** — this schedule.
- **C baseline subsampled to B's sample count** — isolates rebalancing from dataset size.

Each arm is one `experiment_dir`, hash-pinned by `membership_hash`.

## Trainer wiring (Stage 6)

`materialize_rebalanced.py` writes `config.yaml` with `dataset_cls: rebalanced`
and a `membership.parquet` carrying integer counts, and the trainer expands those
counts into repeated **index entries** (not duplicated bytes).

**This is already wired up** — `blip3o/train/train.py` calls
`rebalanced_dataset.register()` when `--dataset_cls rebalanced` is passed, so no
manual edit is needed. Just run with
`--dataset_cls rebalanced --experiment_dir experiments/rebalanced_E`.

> **Startup cost — set `--num_loading_workers` low.** The HF `filter` that
> restricts shards to the membership pickles the key set into *every* worker:
> measured **491 MB and 6.4 s per worker** for a 14.0M-sample membership. At the
> `train.py` default of 32 that is **~15.7 GB and ~3.5 min of pure pickling**
> before training starts. Use `--num_loading_workers 4` (or 8) for a rebalanced
> run — the filter is I/O-light, so the parallelism buys little here.
`run_experiment.sh` already reads `actual_samples` (the expanded total) for
`MAX_STEPS`, so the tail copies are budgeted correctly. The repeats are shuffled
via the index map (seed 42) so copies land in different batches; the HF rows stay
memory-mapped.

## Calibration loop

The schedule constants (`N_high`, `N_low`, `a_head`, `r_min`, `gamma`, `M_max`)
are **starting defaults**. Iterate: `build_schedule` → `materialize` → `audit`,
and tune until the Stage-6a report shows realized ≈ intended per concept (watch
`head_reinflation_offenders` — a head concept riding tail co-occurrences back up
means lower `M_max` / raise `r_min` / reduce `gamma`).

## Tests

Pure-math and offline-integration tests (no GPU, no corpus):

```bash
python concept_rebalancing/tests/test_multiplicity.py
python concept_rebalancing/tests/test_schedule.py
python concept_rebalancing/tests/test_pipeline_offline.py     # runs build_schedule + materialize
```

## Layout

```
concept_rebalancing/
  rebalance/
    multiplicity.py   # hash01 + realize (shared with the trainer)
    schedule.py       # per-concept schedule + rarest-wins
    linker.py         # load taxonomy+embedding, resolve tag vocab (Stage 1)
    qa_counts.py      # links.parquet-backed store for CompositionAnalyzer
  link_corpus.py            # Stage 1
  qa_gate.py                # Stage 1b + Stage 2 counts
  build_schedule.py         # Stages 2–4
  materialize_rebalanced.py # Stage 5
  audit.py                  # Stage 6a
  rebalanced_dataset.py     # Stage 6 (trainer subclass + register())
  tests/
```


---

# 中文详解：端到端工作流（Concept Rebalancing）

本文档用中文完整讲解 `concept_rebalancing/` 这个包做的事情：**为什么做、怎么做、
每一阶段的输入输出、以及一个真实样本如何从原始 tar 分片一路走到训练 batch。**
它是 `docs/concept_rebalancing_plan.md`（英文设计文档）的落地实现说明，配合
`README.md`（英文操作手册）一起看。

---

## 0. 要解决的问题

预训练语料在**概念层级**上极度长尾：约 2% 的概念占了 >30% 的出现频次，约 50% 的
概念出现不到 10 万次。直接训练会让模型偏向头部概念（如「天空」「人」），而尾部
概念（如「皮划艇 kayak」）几乎学不到。

目标：**离线地对语料重采样**——把过度出现的头部概念**降采样**、中间段**原样保留**、
尾部概念**过采样**，然后训练 BLIP3o，重点观察尾部概念的生成是否变好、且不伤害头部。

关键前提（为什么可行）：**全流程共享同一个身份，无需任何模糊匹配**。

```
webdataset __key__  ==  {uuid}.json 文件名  ==  BLIP3o sample_key  ==  被打标签的单元
```

打标签（tagging）和索引（index）都建立在 `sample_key` 上，所以概念标签能**精确**地
join 回训练样本。

---

## 1. 核心抽象：每样本乘数 `m`（multiplicity）

整个系统只有一个核心概念：给每个样本一个实数目标乘数 `m ≥ 0`，再用样本自身的
哈希**确定性地**取整成拷贝数 `count`。代码在 `rebalance/multiplicity.py`：

```
h     = priority / 2**63          # priority = sha256(sample_key) mod 2**63，与 index_tars 完全一致
base  = floor(m)
frac  = m - base
count = base + (1 if h < frac else 0)
```

- `m = 0`      → 丢弃（count=0）
- `0 < m < 1`  → 概率保留（降采样）：count ∈ {0,1}，期望 E[count]=m
- `m = 1`      → 原样保留（count=1）
- `m > 1`      → 过采样：base 份必留 + 1 份按概率

**为什么这套设计好**：
1. 随机数 `h` 就是 `index_tars.py` 早已写进 `index.parquet` 的 `priority`，所以整个
   重采样结果**只靠 schedule 就能复现**，不用再读 tar。
2. 用「拷贝数 count」而不是「权重」承载乘数，天然兼容现成 trainer 的步数预算
   （`MAX_STEPS = actual_samples / global_batch`）。
3. 丢弃/降采样/保留/过采样是**同一个公式**。

---

## 2. 完整流程与各阶段

| 阶段 | 脚本 | 产物 | 是否需 GPU/浏览器环境 |
|---|---|---|---|
| 0 索引分片 | `scripts/data_pipeline/index_tars.py`（复用） | `index.parquet` | 否 |
| 1 打标签→概念链接 | `link_corpus.py` | `sample_tags.parquet`、`vocab.txt`、`tag_to_nodes.parquet`、`links.parquet` | **是**（resolve 步） |
| 1b 类别 QA 门禁 + 2 计数 | `qa_gate.py` | `qa_report.json`、`counts.parquet` | **是** |
| 2–4 生成 schedule | `build_schedule.py` | `counts/node_multiplicity/sample_multiplicity.parquet` | 否 |
| 5 物化 membership | `materialize_rebalanced.py` | `membership.parquet`、`shardlist.txt`、`config.yaml` | 否 |
| 6a 分布审计 | `audit.py` | `audit_*.json` | 否 |
| 6 训练 | `rebalanced_dataset.py` + `run_experiment.sh` | 模型 | 训练环境 |

除了阶段 1、1b 需要 GPU，其余都是对 parquet 的纯算术，**很便宜**，方便在标定
（calibration）时反复重跑。

### 环境说明
- **阶段 1 / 1b（GPU）**：浏览器的 `wiki` conda 环境
  （`/root/miniconda3/envs/wiki/bin/python`：torch+cuda、sentence-transformers、
  lancedb），并复用 `semantic_image_browser/data/` 下已构建好的缓存
  （`bamboo_V4.json`、`taxonomy_cache_bamboo.pkl`、`embedding_index_bamboo.lance`）。
  用 `--browser_root` 可指向别的 checkout。
- **阶段 2–6a**：任何有 `pyarrow` + `pyyaml` 的环境。
- **阶段 6（训练）**：BLIP3o 训练环境。

---

## 3. 跟着一个真实样本走一遍

以 `shard-000001.tar` 里的样本
`cd3cd09d12de4c48a72a24a8de952a5e.{jpg,txt,json,pkl}` 为例。它的 `.json` 里有
`tagging_caption`：

```
"photograph, high resolution, sharp focus, 1 male kayaker, adult male,
 orange helmet, ... whitewater rapids, turbulent river, red and yellow kayak,
 double-bladed paddle, ... 'S' logo, 'AQUATEC' text"
```

它在下游各处的身份始终是 `sample_key = cd3cd09d12de4c48a72a24a8de952a5e`。

### 阶段 0 — 索引（`index_tars.py`，复用）
扫描每个分片，为每个唯一 stem 生成一行 `(sample_key, tar_path, priority)`，其中
`priority = sha256(sample_key) mod 2^63`。我们的皮划艇样本得到一行，指向
`shard-000001.tar`。`priority/2^63` 就是这个样本固定的均匀随机数 `h`——**索引与
重采样共用同一个随机源**，这正是全程可复现的根基。

### 阶段 1 — 打标签→概念链接（`link_corpus.py`）
这一步把 `tagging_caption` 里的标签映射到 Bamboo 分类树的概念节点。**关键的规模
技巧**（`links.parquet` 的注释和设计文档都强调）：标签海量重复，所以

1. 先扫遍全语料，收集**唯一标签词表**（`vocab.txt`）；
2. 用 `HybridMatcher` 把每个唯一标签解析成节点**一次**（`tag_to_nodes.parquet`）；
3. 再按字符串 join 回每个样本（`links.parquet`）。

这就把「对 1e7+ 张图逐一 embedding」摊薄成「对 ~1e6 个唯一标签解析一次」。

`link_corpus.py` 分三小步，各自可断点续跑（`--resume`）：
- **扫描（CPU，多进程）**：读每个 tar 里的 `.json`，用浏览器的
  `parse_tags` 清洗标签——丢掉 OCR 引号（`'S' logo`）、纯数字、以及 `STOPTAGS`
  描述词（`photograph`、`high resolution`、`sharp focus`、`close-up`……这些是
  高频但无信息、且会误配同形异义节点的词）。产物：`sample_tags.parquet`
  `(sample_key, tags)` 和 `vocab.txt`。本步无需 GPU（`--skip_resolve` 可只做这步）。
- **解析（GPU，一次）**：`rebalance/linker.py` 的 `VocabLinker` 加载 Bamboo 分类树
  + 已构建的节点向量库，用 `HybridMatcher.match` 解析整个词表。词法优先（一个
  标签若「命名」了某节点就走词法命中），embedding 只对候选**重排**以消歧同形词
  （`bat`→动物蝙蝠而非棒球棒）；复合词头名可同时给出通用节点与置信的具体概念
  （`golden retriever puppy` → `puppy` 和 `golden_retriever` 两个）。产物：
  `tag_to_nodes.parquet` `(tag, node_id, node_name, score)`。
- **join**：每个样本的节点集 = 其所有标签命中的节点并集。产物：`links.parquet`
  `(sample_key, node_id)`（多对多）。

对我们的样本，存活的是**内容**标签：`male kayaker`（去掉计数前缀 `1 `）、
`orange helmet`、`drysuit`、`life vest`、`whitewater rapids`、`turbulent river`、
`kayak`、`double-bladed paddle`、`water droplets`……解析成一组概念节点，例如
`{kayak, rapids/river, paddle, helmet, life_vest, water_droplet, …}`。

### 阶段 1b — 类别 QA 门禁（`qa_gate.py`，人工 go/no-go）
把 `links.parquet` 按 11 个视觉大类（GROUP）+ 子类聚合，打印各类占比和长尾偏斜
（Gini、top-2% 占比）。这是**人工检查点**：如果某个大类被明显灌水（典型的同形词/
描述词泄漏，如 `blue`→蝴蝶、`overcast`→抛竿），就先修 `tags.STOPTAGS`/匹配器再
重跑阶段 1，**绝不在被污染的链接集上算比率**。

实现上复用了浏览器的 `CompositionAnalyzer`（它做的 DAG 归属和偏斜统计正是我们要
的）。`CompositionAnalyzer` 只通过 3 个方法访问数据库，所以我用
`rebalance/qa_counts.py` 里的 `LinksCountStore` 基于 `links.parquet` 在内存里实现
这 3 个方法——**不需要 DuckDB，也不改浏览器一行代码**。

注意：**这一步不喂给 schedule**——schedule 用的是逐节点计数。QA 门禁只保证喂给
计数的链接集是干净的。此步还顺带产出 `counts.parquet`（阶段 2）。

对我们的样本：它给 *Activities*（皮划艇运动）和 *Objects*（kayak、paddle、helmet）
贡献计数。看起来合理即放行。

### 阶段 2 — 逐概念频次表（`counts.parquet`）
`N_c` = 链接到概念 `c` 的**去重样本数**。因为阶段 1 已对每个样本按节点去重，
`links.parquet` 里一条 = 一个 (样本, 概念)，所以直接 group-by 计数即得去重样本数。
假设 `N_kayak = 8000`（尾部）、`N_river = 3_000_000`（头部）、`N_helmet = 50_000`
（中段）。

### 阶段 3 — 逐概念乘数 schedule（`build_schedule.py`，`rebalance/schedule.py`）
分段、边界连续、常数全部可调（并需对着阶段 6a 审计标定）：

```
head  (N_c ≥ N_high)         : m_c = max(r_min, a_head * 2 / log10(N_c))    # < 1 降采样
mid   (N_low ≤ N_c < N_high) : m_c = 1.0                                    # 保留
tail  (N_c < N_low)          : m_c = min(M_max, (N_low / N_c) ** gamma)     # > 1 过采样
```

默认起始值：`N_high=100000`、`N_low=20000`、`a_head=2.5`（使 `m_c(N_high)=1`，
即两个边界都连续）、`r_min=0.1`（头部下限，永不消失）、`gamma=0.5`（次线性，
避免把单例样本放大 100 倍）、`M_max=4`（过采样上限）。head 分支就是参考文献的
`N_sample ∝ 2/log(Count)`；tail 分支是其对称的过采样扩展。

对样本内三个概念：
- `river`（头部 3e6）：`m ≈ 5/log10(3e6) ≈ 0.77` → 降采样
- `helmet`（中段 5e4）：`m = 1.0` → 保留
- `kayak`（尾部 8e3）：`m = min(4, (20000/8000)^0.5) = 1.58` → 过采样

产物：`node_multiplicity.parquet` `(node_id, N_c, m_c)`。

### 阶段 4 — 逐样本乘数（rarest-wins，最稀者胜）
一个样本有多个概念，取**最大的 m_c**作为该样本的乘数：

```
m(sample) = max over its concepts c of m_c          # 越稀有 m_c 越大 → 稀有者定标准
m(sample) = 1.0   若样本没有任何内容概念（标签全被 STOPTAGS 丢掉）
```

我们的样本 `m ∈ {0.77(river), 1.0(helmet), 1.58(kayak)}` → **max = 1.58**（稀有的
kayak 主导）。为什么取 max：一张描绘稀有主体的图应当被保留/加强，即便它同时含常见
概念——否则为降采样常见的 `river` 就会丢掉稀缺的 `kayak` 像素。（相反方向的失败
——`river` 借尾部共现被重新抬高——由阶段 6a 的「实际 vs 意图」审计来兜底。）

产物：`sample_multiplicity.parquet` `(sample_key, m)`。传入 `--index` 时，语料里
**没有任何概念**的样本也会被纳入并赋 `m=1.0`。

### 阶段 5 — 物化重平衡后的 membership（`materialize_rebalanced.py`）
用阶段 4 的哈希规则把 `m` 取整成 `count`，保留 `count ≥ 1` 的样本。`h` 来自
`index.parquet` 里存的 `priority`。

我们的样本：`base=floor(1.58)=1`，`frac=0.58`，`count = 1 + (1 if h<0.58 else 0)`，
即出现 **1 或 2 次**（取决于其固定哈希）。纯头部样本（如 `m=0.77`）则 `base=0`、
`frac=0.77`，`count∈{0,1}`——按 0.23 概率被丢。同一机制覆盖全区间。

产物（写进一个 `experiment_dir`）：
- `membership.parquet` — `(sample_key, count)`（取代旧的扁平 `membership.txt`，
  count 承载乘数）
- `shardlist.txt` — 保留样本涉及的去重 tar 路径
- `config.yaml` — 含 `actual_samples = Σcount`（**展开后**的总量，让
  `run_experiment.sh` 的 `MAX_STEPS` 把过采样的额外拷贝算进预算）+ 可复现的
  `membership_hash`（按 sample_key 排序后哈希，与输入行序无关）。

若上游已做质量过滤（把过滤后的 `index.parquet` 传进来），不在 index 里的样本
会在这里自动被排除——即「先按质量分过滤、再对幸存者重平衡」（设计文档 open item #3）。

### 阶段 6 — 训练期展开（`rebalanced_dataset.py`）
现成 `dataset.py` 的 experiment 模式是用 membership **集合**过滤 HF 行（一行=一个
样本）。我用**子类** `LazySupervisedRebalancedDataset` 增加乘数展开，**不改**核心
`dataset.py`：

1. 读 `shardlist` 分片，过滤到 `count_map` 的键（与原逻辑一致）；
2. 从 `membership.parquet` 建 `count_map = {sample_key: count}`；
3. 建 `index_map = [行号按 count 重复]`；
4. `__len__ = len(index_map)`；`__getitem__(i)` → `index_map[i]` → HF 行；
5. 打乱 **`index_map`**（seed 42）而非 HF 数据集，让同一样本的多份拷贝散到不同
   batch。

我们皮划艇样本的行号在 `index_map` 里出现**两次**；`__len__` 是展开后的总数
（与 `actual_samples` 一致，故 `MAX_STEPS` 正确）。**重复的是 index 条目（两个
整数），不是图像字节**（HF 行仍是内存映射）——所以过采样几乎零成本。子类只重写
`__init__`/`__len__`/`__getitem__`，图像解码、构造 T2I 对话、tokenize 全部复用父类。

启用方式（README 里给了 3 行）：在 `train.py` 启动处调用
`rebalanced_dataset.register()`，它 monkeypatch `get_dataset_cls`，使
`--dataset_cls rebalanced` 生效——无需改核心函数。

### 阶段 6a — 分布审计（`audit.py`，训练前、便宜、纯算术）
这是防「rarest-wins 耦合」的主要兜底（设计文档 §6a）。给定 links、schedule、
物化后的 membership，报告：
- **前后集中度 + Gini**：期望 Gini 下降、头部变平。
- **逐概念「实际 vs 意图」保留率**：对每个节点，实际 = `Σcount over 其样本 / N_c`，
  与 schedule 的 `m_c` 对比。头部概念若总与尾部共现，可能被过采样重新抬高——
  报告列出最严重的 `head_reinflation_offenders` 和过采样不足的 `tail_undersampled`。
- **预算核算**：展开总量、固定算力下的有效 epoch。

**标定循环**：`build_schedule → materialize → audit` 反复迭代，调
`M_max`/`r_min`/`gamma`/`a_head` 等，直到「实际 ≈ 意图」。若头部 ratio 远大于 1，
就降 `M_max`、升 `r_min` 或减 `gamma`。

---

## 4. 消融实验（设计文档 §6b）

每个实验臂就是一个 `experiment_dir`，由 `membership_hash` 钉死、可复现可比较：
- **臂 A 基线**：全量/均匀。
- **臂 B 重平衡**：本方案。
- **臂 C 基线降采样到 B 的样本量**：关键对照，把「重平衡」与「数据量/epoch 数」
  解耦。

跨臂固定总训练步数/ token。模型层评测（BLIP3o 是文生图）：**按原始频次分层的尾部
生成评测**（头/中/尾）是最锋利的测试，报告**下界**（每概念 10 分位/最小值）而非
均值；再加 FID/CLIP-FID、CLIPScore/VLM 评审对齐、头部概念的多样性（LPIPS）等。
假设：**臂 B 尾部生成变好，且头部几乎不退化。**

---

## 5. 让整条链路可信的不变量

- **单一身份**（`sample_key`）：从分片到 batch，无模糊 join。
- **单一随机源**（`hash01(sample_key)`）：索引优先级与重采样抽样共用 ⇒ 任何
  membership 都能仅凭 schedule 复现。
- **用拷贝数而非权重**承载乘数 ⇒ 兼容现成 trainer 与步数预算；丢/降/留/过采样
  一个公式。
- **重复是 index 条目而非复制字节** ⇒ 过采样几乎免费。

---

## 6. 一句话总览命令（示例路径见英文 README.md）

```
index_tars.py            → index.parquet                （阶段 0，复用）
link_corpus.py           → links.parquet                （阶段 1，GPU）
qa_gate.py               → qa_report.json + counts       （阶段 1b/2，GPU，人工放行）
build_schedule.py        → node/sample_multiplicity      （阶段 2–4，纯算术，可反复调参）
materialize_rebalanced.py→ membership.parquet + config    （阶段 5）
audit.py                 → audit_*.json                  （阶段 6a，标定用）
register() + run_experiment.sh                            （阶段 6，训练）
```

测试（无需 GPU/语料）：
`python concept_rebalancing/tests/run_all.py`
（覆盖哈希取整、schedule 分段与边界、以及 build_schedule+materialize 的离线集成）。
```
