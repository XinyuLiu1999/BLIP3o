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
    echo "       Fix: conda install -y -c nvidia cuda-nvcc=12.1"
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
LR=5e-5
RUN_NAME="Pretrain-rebalanced-E-guarded"

REPO_DIR=/cephfs/liuxinyu/BLIP3o
EXPERIMENT_DIR=${REPO_DIR}/experiments/rebalanced_E_guarded
DATA_CACHE_DIR=/cephfs/liuxinyu/.cache/blip3o-data
LOCAL_DIR="/cephfs/liuxinyu/BLIP3o-Pretrain-results/${RUN_NAME}"

# recaptioned 数据的真 caption 在 json 的 long_caption 字段, tar 内 .txt 是旧 caption
CAPTION_KEY=long_caption

# --- 步数预算 ---
# 必须按 actual_samples=19,756,413 (含重复拷贝) 算, 不是 distinct_samples=14,040,822,
# 否则少训 29%。
ACTUAL_SAMPLES=19756413
PER_DEVICE_BS=16
GRAD_ACCUM=1
GLOBAL_BATCH=$((GPUS_PER_NODE * PER_DEVICE_BS * GRAD_ACCUM))
MAX_STEPS=$((ACTUAL_SAMPLES / GLOBAL_BATCH))

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
echo "  EXPERIMENT_DIR:  $EXPERIMENT_DIR"
echo "  CAPTION_KEY:     $CAPTION_KEY"
echo "  GLOBAL_BATCH:    $GLOBAL_BATCH"
echo "  ACTUAL_SAMPLES:  $ACTUAL_SAMPLES"
echo "  MAX_STEPS:       $MAX_STEPS"
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
    --group_by_modality_length True \
    --image_aspect_ratio square \
    --mm_patch_merge_type flat \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir ${LOCAL_DIR} \
    --max_steps ${MAX_STEPS} \
    --per_device_train_batch_size ${PER_DEVICE_BS} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 5 \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr":1e-5}' \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend inductor \
    --dataloader_drop_last True

EXIT_CODE=$?
echo ">>> Training finished with exit code: $EXIT_CODE"
if [ $EXIT_CODE -ne 0 ]; then
    echo "=== Post-mortem diagnostics ==="
    dmesg -T 2>/dev/null | tail -100 || echo "dmesg not accessible"
    free -h
    nvidia-smi
fi
exit $EXIT_CODE
