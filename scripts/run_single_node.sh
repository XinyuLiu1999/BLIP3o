#!/bin/bash
# ============================================================
# 单节点 BLIP3O 训练脚本 (本集群版本)
# 直接在一台 GPU 节点上执行: bash scripts/run_single_node.sh
# ============================================================

# === CONDA ENVIRONMENT ===
source /root/miniconda3/etc/profile.d/conda.sh && conda activate blip3o-next

# === SINGLE NODE SETUP ===
GPUS_PER_NODE=${GPUS_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}
MASTER_PORT=${MASTER_PORT:-29501}

if [ "$GPUS_PER_NODE" -eq 0 ]; then
    echo "ERROR: no GPU detected. This script must run on a GPU node."
    exit 1
fi

# ============================================================
# USER CONFIG — edit these
# ============================================================
AR_BACKBONE=Qwen/Qwen3-0.6B
DIFFUSION=Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers
LR=5e-5
RUN_NAME="Pretrain-single-node"

REPO_DIR=/cephfs/liuxinyu/BLIP3o
DATA_DIR=/cephfs/liuxinyu/BLIP3o-Pretrain-Long-Caption-filtered-recaptioned
DATA_CACHE_DIR=/cephfs/liuxinyu/.cache/blip3o-data
LOCAL_DIR="/cephfs/liuxinyu/BLIP3o-Pretrain-results/${RUN_NAME}"

# 本集群数据是 recaptioned 的, 真正要用的 caption 在 .json 的 long_caption 字段,
# tar 内的 .txt 是旧 caption。可选: long_caption / medium_caption / short_caption
CAPTION_KEY=long_caption

# ============================================================
# ENVIRONMENT VARIABLES (对齐 ~/.bashrc 的本集群路径)
# ============================================================
export HF_HOME=/cephfs/liuxinyu/.cache/data_juicer/models
export HF_HUB_CACHE=/cephfs/liuxinyu/.cache/data_juicer/models
export HF_DATASETS_CACHE=/cephfs/liuxinyu/.cache/data_juicer/datasets
export TORCH_HOME=/cephfs/liuxinyu/.cache/torch/hub
export HF_ENDPOINT=https://hf-mirror.com

# 本集群 GPU 节点通常连不上外网。模型权重必须预先下好放进 HF_HOME，
# 训练时全程离线读缓存，避免 torchrun 起来后卡在下载上。
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export WANDB_API_KEY=e4bb266d6e5a159a1280afa4a476720e92a6dbe7
export WANDB_PROJECT='BLIP3o-Pretrain'

# 单节点 NCCL: 只走本机 GPU 互联, 关掉 IB 避免容器内握手失败
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false

# === LOGGING ===
LOG_DIR="${LOCAL_DIR}/logs"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/train.log") 2>&1

# === NODE INFO ===
echo "============================================"
echo "=== Single-Node Run ==="
echo "  HOSTNAME:      $(hostname)"
echo "  GPUS_PER_NODE: $GPUS_PER_NODE"
echo "  MASTER_PORT:   $MASTER_PORT"
echo "  AR_BACKBONE:   $AR_BACKBONE"
echo "  DIFFUSION:     $DIFFUSION"
echo "  RUN_NAME:      $RUN_NAME"
echo "  LR:            $LR"
echo "  DATA_DIR:      $DATA_DIR"
echo "  CAPTION_KEY:   $CAPTION_KEY"
echo "============================================"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
echo "=== Resource limits ==="
echo "  ulimit -n (open files):     $(ulimit -n)"
echo "  vm.max_map_count:           $(cat /proc/sys/vm/max_map_count)"
free -h
cgroup_limit=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo 'N/A')
echo "  cgroup memory limit:        $cgroup_limit"
echo "============================================"

# === RUN TRAINING ===
echo ">>> Starting torchrun on $GPUS_PER_NODE GPU(s)..."

torchrun \
    --standalone \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$MASTER_PORT \
    ${REPO_DIR}/blip3o/train/train.py \
    --deepspeed ${REPO_DIR}/scripts/zero1.json \
    --data_dir ${DATA_DIR} \
    --data_cache_dir ${DATA_CACHE_DIR} \
    --caption_key ${CAPTION_KEY} \
    --num_image_tokens 65536 \
    --num_scale_tokens 3 \
    --load_embeddings_from_vision True \
    --model_name_or_path $AR_BACKBONE \
    --diffusion_name_or_path $DIFFUSION \
    --version "qwen_1_5" \
    --dataset_cls 'mix' \
    --dispatch_batches False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --group_by_modality_length True \
    --image_aspect_ratio square \
    --mm_patch_merge_type flat \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir ${LOCAL_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit -1 \
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
    echo "--- memory after crash ---"
    free -h
    nvidia-smi
fi
exit $EXIT_CODE
