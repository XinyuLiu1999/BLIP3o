#!/bin/bash
# ============================================================
# 单节点 8xA800 概念重平衡训练 (rebalanced_E_guarded)
# 用法: bash scripts/run_rebalanced_single_node.sh
# 方案文档: docs/concept_rebalancing_plan.md 附录(2026-07-25)
# ============================================================

# === CONDA ENVIRONMENT ===
# CONDA_ROOT is overridable because the env may live somewhere else on a fresh
# GPU container. Fail loudly here rather than 80 lines later with an opaque
# "torchrun: command not found" -- `source ... && conda activate` short-circuits
# silently when the install is absent.
CONDA_ROOT=${CONDA_ROOT:-/root/miniconda3}
CONDA_ENV=${CONDA_ENV:-blip3o-next}

if [ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    echo "ERROR: no conda at $CONDA_ROOT (this container may not have it --"
    echo "       it lives on the node's local overlay, not on /cephfs)."
    echo "       Set CONDA_ROOT=/path/to/conda and re-run."
    exit 1
fi
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || { echo "ERROR: cannot activate env '$CONDA_ENV'"; exit 1; }

if ! command -v torchrun >/dev/null; then
    echo "ERROR: torchrun not on PATH after activating '$CONDA_ENV'."
    echo "       python: $(command -v python || echo none)"
    exit 1
fi
echo "conda env: $CONDA_PREFIX"
echo "torchrun : $(command -v torchrun)"

# DeepSpeed probes for a CUDA toolkit at *import* time (fp_quantizer's
# is_compatible -> installed_cuda_version), and dies with
# "MissingCUDAException: CUDA_HOME does not exist" if $CUDA_HOME/bin/nvcc is
# absent. The container ships no /usr/local/cuda, so nvcc 12.1 was installed
# into the conda env (`conda install -c nvidia cuda-nvcc=12.1`, matching torch's
# cu121). Point CUDA_HOME at the env so that probe succeeds.
export CUDA_HOME=${CUDA_HOME:-$CONDA_PREFIX}
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "ERROR: no nvcc at $CUDA_HOME/bin/nvcc -- deepspeed will fail on import."
    echo "       Fix: conda install -y -c nvidia cuda-version=12.1 cuda-nvcc=12.1 \\"
    echo "                                       cuda-cudart-dev=12.1 cuda-cccl=12.1"
    exit 1
fi

# nvcc alone is NOT enough. zero1.json declares "optimizer": AdamW, so DeepSpeed uses
# its own FusedAdam and JIT-compiles it inside accelerator.prepare() -- i.e. ~40 min in,
# after the dataset is up. That compile needs CUDA headers, and cuda-nvcc ships only the
# compiler: cuda_runtime.h comes from cuda-cudart-dev, while ATen's CUDAContextLight.h
# also pulls cusparse.h/cublas_v2.h, which exist only in torch's pip wheels
# (site-packages/nvidia/*/include) and are symlinked into $CONDA_PREFIX/include.
# Without them one rank dies with "cc1plus: fatal error: cuda_runtime.h: No such file or
# directory" and the other seven with "ImportError: fused_adam.so: cannot open shared
# object file". Fail here instead, in one second.
for _h in cuda_runtime.h cusparse.h cublas_v2.h; do
    if [ ! -e "$CUDA_HOME/include/$_h" ]; then
        echo "ERROR: missing $CUDA_HOME/include/$_h -- the fused_adam JIT build will fail."
        echo "       Fix: conda install -y -c nvidia cuda-cudart-dev=12.1"
        echo "            for d in \$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/*/include; do"
        echo "                for f in \$d/*; do ln -sn \"\$f\" \"\$CONDA_PREFIX/include/\$(basename \$f)\" 2>/dev/null; done"
        echo "            done"
        exit 1
    fi
done

# cuda-cudart-dev's libcudart.so symlink targets its own patch release, so a mismatched
# cuda-cudart leaves it dangling and the link step fails with "cannot find -lcudart".
# -e is false on a dangling symlink, which is exactly the case we want to catch.
if [ ! -e "$CUDA_HOME/lib/libcudart.so" ] && [ ! -e "$CUDA_HOME/lib64/libcudart.so" ]; then
    echo "ERROR: $CUDA_HOME/lib/libcudart.so is missing or a dangling symlink."
    echo "       Fix: conda install -y -c nvidia cuda-cudart=12.1.105"
    exit 1
fi

GPUS_PER_NODE=${GPUS_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}
MASTER_PORT=${MASTER_PORT:-29501}
if [ "$GPUS_PER_NODE" -eq 0 ]; then
    echo "ERROR: no GPU detected. Run this on the 8xA800 node."
    exit 1
fi

# ============================================================
# USER CONFIG
# ============================================================
AR_BACKBONE=Qwen/Qwen3-0.6B
DIFFUSION=Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers
LR=${LR:-5e-5}

# Overridable so a smoke test can be run without touching the real run's state.
# RUN_NAME drives output_dir, which train.py globs for "checkpoint-*" and silently
# resumes from -- a 2-GPU trial that saves into the production dir would make the
# next 8-GPU run continue from it. Always pass a different RUN_NAME for trials.
RUN_NAME=${RUN_NAME:-"Pretrain-rebalanced-E-guarded"}

REPO_DIR=/cephfs/liuxinyu/BLIP3o
EXPERIMENT_DIR=${EXPERIMENT_DIR:-${REPO_DIR}/experiments/rebalanced_E_guarded}
DATA_CACHE_DIR=/cephfs/liuxinyu/.cache/blip3o-data
LOCAL_DIR="/cephfs/liuxinyu/BLIP3o-Pretrain-results/${RUN_NAME}"

# tar 内的 .txt 是重标注前的旧 caption, 真 caption 在 json 里 (medium/long/short 三档均存在)。
# 用 medium_caption 而非 long_caption: 16 卡 `--dataset_cls mix` 基线用的就是它, 而本次 run
# 要和那个基线对照。同一批 tar 上换字段会让 caption 长度 (medium 465 字符 vs long 677 字符)
# 混进"重平衡效果"里, 结论就不干净了。
CAPTION_KEY=${CAPTION_KEY:-medium_caption}

# --- 步数预算 ---
# 由 --num_train_epochs 驱动, 不写死步数: HF 在 max_steps<0 时走 epoch 分支
# (trainer.py:5298 epoch_based = max_steps < 0), 步数 = len_dataloader // grad_accum,
# 即 floor(样本数/global_batch)。样本数取自数据集本身 (rebalanced 的 index_map 长度 =
# actual_samples 19,756,413, 含重复拷贝; 不是 distinct_samples 14,040,822, 否则少训 29%)。
#
# 这里刻意不再用写死的样本数去算 MAX_STEPS: 那个常数一旦和 EXPERIMENT_DIR 指向的数据集
# 对不上 (例如 128-shard 冒烟子集只有 535,870 个样本), 就会安静地训错步数。epoch 驱动
# 永远跟着实际数据集走。冒烟/压测要截断时用 MAX_STEPS=30 覆盖, 正值会盖过 epoch。
NUM_EPOCHS=${NUM_EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:--1}
PER_DEVICE_BS=${PER_DEVICE_BS:-16}
# 2 而非 1: 16 卡基线是 16x16=256 的 global batch 配 LR 5e-5。8 卡上累积两次才能还原同一个
# global batch 和同样的 LR 语义; micro-batch 仍是 16, 显存不变 (实测峰值已达 75.4/80 GB,
# 没有加 per-device batch 的余地)。步数由 epoch 推导, 会自动减半, 样本预算不受影响。
GRAD_ACCUM=${GRAD_ACCUM:-2}
GLOBAL_BATCH=$((GPUS_PER_NODE * PER_DEVICE_BS * GRAD_ACCUM))
SAVE_STEPS=${SAVE_STEPS:-5000}

# 仅用于日志核对, 不参与控制: 从实验目录的 config.yaml 读真实样本数, 预告预期步数,
# 这样启动一秒后就能确认预算, 不必等一小时数据集加载完才看到。
ACTUAL_SAMPLES=$(sed -n 's/^actual_samples: *//p' "$EXPERIMENT_DIR/config.yaml" 2>/dev/null)

# Throughput knobs, exposed for benchmarking. Defaults reproduce the original run.
# GRAD_CKPT=False trades memory for speed and is usually the biggest single win when
# the 80GB cards have headroom; DL_WORKERS=1 is a likely starvation point given every
# sample costs a jpg decode + tokenization.
GRAD_CKPT=${GRAD_CKPT:-True}
DL_WORKERS=${DL_WORKERS:-1}
# modality_lengths is a constant [128]*N (rebalanced_dataset.py), so length grouping
# sorts a 19.7M list to no effect. Kept on by default to match the original run.
GROUP_BY_MODALITY=${GROUP_BY_MODALITY:-True}
# Report peak GPU memory in the final metrics -- needed to know if batch size can grow.
SKIP_MEM_METRICS=${SKIP_MEM_METRICS:-True}

# ============================================================
# ENVIRONMENT (对齐 ~/.bashrc 的本集群路径)
# ============================================================
export HF_HOME=/cephfs/liuxinyu/.cache/data_juicer/models
export HF_HUB_CACHE=/cephfs/liuxinyu/.cache/data_juicer/models
export HF_DATASETS_CACHE=/cephfs/liuxinyu/.cache/data_juicer/datasets
export TORCH_HOME=/cephfs/liuxinyu/.cache/torch/hub
export HF_ENDPOINT=https://hf-mirror.com

# GPU 节点通常无外网; 权重已预下载到 HF_HOME, 全程离线读缓存
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export WANDB_API_KEY=e4bb266d6e5a159a1280afa4a476720e92a6dbe7
export WANDB_PROJECT='BLIP3o-Pretrain'

# 单机 8 卡: 走本机 NVLink/PCIe, 关掉 IB 避免容器内握手失败
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false

# === LOGGING ===
LOG_DIR="${LOCAL_DIR}/logs"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/train.log") 2>&1

echo "============================================"
echo "=== Rebalanced Single-Node Run ==="
echo "  HOSTNAME:        $(hostname)"
echo "  GPUS_PER_NODE:   $GPUS_PER_NODE"
echo "  RUN_NAME:        $RUN_NAME"
echo "  OUTPUT_DIR:      $LOCAL_DIR"
echo "  EXPERIMENT_DIR:  $EXPERIMENT_DIR"
echo "  CAPTION_KEY:     $CAPTION_KEY"
echo "  GLOBAL_BATCH:    $GLOBAL_BATCH  (${GPUS_PER_NODE} x ${PER_DEVICE_BS} x ${GRAD_ACCUM})"
if [ "$MAX_STEPS" -lt 0 ]; then
    echo "  步数控制:        --num_train_epochs $NUM_EPOCHS (步数由数据集长度推导)"
    if [ -n "$ACTUAL_SAMPLES" ]; then
        echo "  ACTUAL_SAMPLES:  $ACTUAL_SAMPLES  (来自 $(basename $EXPERIMENT_DIR)/config.yaml)"
        echo "  预期步数:        ~$((ACTUAL_SAMPLES * NUM_EPOCHS / GLOBAL_BATCH))  <- 与训练日志里的 Total optimization steps 核对"
    else
        echo "  ACTUAL_SAMPLES:  (config.yaml 未提供, 无法预告步数)"
    fi
else
    echo "  步数控制:        --max_steps $MAX_STEPS (正值, 盖过 epoch)"
fi
echo "  LR:              $LR"
echo "============================================"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
free -h
echo "============================================"

torchrun \
    --standalone \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$MASTER_PORT \
    ${REPO_DIR}/blip3o/train/train.py \
    --deepspeed ${REPO_DIR}/scripts/zero1.json \
    --dataset_cls 'rebalanced' \
    --experiment_dir ${EXPERIMENT_DIR} \
    --data_cache_dir ${DATA_CACHE_DIR} \
    --num_loading_workers 4 \
    --caption_key ${CAPTION_KEY} \
    --num_image_tokens 65536 \
    --num_scale_tokens 3 \
    --load_embeddings_from_vision True \
    --model_name_or_path $AR_BACKBONE \
    --diffusion_name_or_path $DIFFUSION \
    --version "qwen_1_5" \
    --dispatch_batches False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --group_by_modality_length ${GROUP_BY_MODALITY} \
    --image_aspect_ratio square \
    --mm_patch_merge_type flat \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir ${LOCAL_DIR} \
    --num_train_epochs ${NUM_EPOCHS} \
    --max_steps ${MAX_STEPS} \
    --per_device_train_batch_size ${PER_DEVICE_BS} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --save_strategy "steps" \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit 5 \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr":1e-5}' \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing ${GRAD_CKPT} \
    --dataloader_num_workers ${DL_WORKERS} \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend inductor \
    --dataloader_drop_last True \
    --skip_memory_metrics ${SKIP_MEM_METRICS}

EXIT_CODE=$?
echo ">>> Training finished with exit code: $EXIT_CODE"
if [ $EXIT_CODE -ne 0 ]; then
    echo "=== Post-mortem diagnostics ==="
    dmesg -T 2>/dev/null | tail -100 || echo "dmesg not accessible"
    free -h
    nvidia-smi
fi
exit $EXIT_CODE
