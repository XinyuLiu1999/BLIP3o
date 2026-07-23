#!/bin/bash
# ============================================================
# 多节点 BLIP3O 训练脚本
# 提交给集群平台，每个节点都会执行此脚本。
# 平台自动注入: RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
# ============================================================

# === CONDA ENVIRONMENT ===
source /user/liuxinyu/miniforge3/etc/profile.d/conda.sh && conda activate blip3o-next

# === PLATFORM-PROVIDED VARIABLES (with safe defaults) ===
WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29501}
GPUS_PER_NODE=${GPUS_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}

# ============================================================
# USER CONFIG — edit these
# ============================================================
AR_BACKBONE=Qwen/Qwen3-0.6B
DIFFUSION=Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers
LR=5e-5
RUN_NAME="Pretrain-0408-original-caption-19M"
LOCAL_DIR="/backup/user/liuxinyu/BLIP3o-Pretrain-results/${RUN_NAME}"

# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================
export HF_DATASETS_CACHE=/user/liuxinyu/.cache/data_juicer/datasets
export HF_HOME=/user/liuxinyu/.cache/data_juicer/models
export TORCH_HOME=/user/liuxinyu/.cache/torch/hub
export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_CACHE=/user/liuxinyu/.cache/data_juicer/models
export HF_HUB_CACHE=/user/liuxinyu/.cache/data_juicer/models
export HF_TOKEN=HF_TOKEN_REMOVED
export WANDB_API_KEY='e4bb266d6e5a159a1280afa4a476720e92a6dbe7'
export WANDB_PROJECT='BLIP3o-Pretrain'

# === LOGGING ===

LOG_DIR="$(dirname "$LOCAL_DIR")/${RUN_NAME}/logs"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/train_rank${RANK}.log") 2>&1

# === RESOLVE MASTER IP ===
MASTER_IP=$(getent hosts "$MASTER_ADDR" | awk '{print $1}' | head -1)
if [ -z "$MASTER_IP" ]; then
    MASTER_IP="$MASTER_ADDR"
fi

# === NODE INFO ===
echo "============================================"
echo "=== Node Info (RANK=$RANK) ==="
echo "  MASTER_ADDR:   $MASTER_ADDR"
echo "  MASTER_IP:     $MASTER_IP"
echo "  MASTER_PORT:   $MASTER_PORT"
echo "  GPUS_PER_NODE: $GPUS_PER_NODE"
echo "  WORLD_SIZE:    $WORLD_SIZE"
echo "  RANK:          $RANK"
echo "  HOSTNAME:      $(hostname)"
echo "  AR_BACKBONE:   $AR_BACKBONE"
echo "  DIFFUSION:     $DIFFUSION"
echo "  RUN_NAME:      $RUN_NAME"
echo "  LR:            $LR"
echo "============================================"

# === RUN TRAINING ===
echo ">>> [RANK=$RANK] Starting torchrun..."

echo "=== Resource limits on $(hostname) (RANK=$RANK) ==="
echo "  ulimit -n (open files):     $(ulimit -n)"
echo "  ulimit -v (virtual mem KB): $(ulimit -v)"
echo "  ulimit -m (RSS KB):         $(ulimit -m)"
echo "  vm.max_map_count:           $(cat /proc/sys/vm/max_map_count)"
echo "  free -h:"
free -h
echo "  /proc/meminfo MemAvailable: $(grep MemAvailable /proc/meminfo)"
cgroup_limit=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo 'N/A')
if [ "$cgroup_limit" != "N/A" ] && [ "$cgroup_limit" != "max" ]; then
    cgroup_limit_gb=$(awk "BEGIN {printf \"%.1f GB\", $cgroup_limit/1024/1024/1024}" 2>/dev/null)
    echo "  cgroup memory limit:        $cgroup_limit ($cgroup_limit_gb)"
else
    echo "  cgroup memory limit:        $cgroup_limit"
fi
echo "============================================"

torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$WORLD_SIZE \
    --node_rank=$RANK \
    --master_addr=$MASTER_IP \
    --master_port=$MASTER_PORT \
    /user/liuxinyu/BLIP3o/blip3o/train/train.py \
    --deepspeed /user/liuxinyu/BLIP3o/scripts/zero1.json \
    --data_dir /backup/user/liuxinyu/BLIP3o-Pretrain-Long-Caption-filtered-recaptioned \
    --data_cache_dir /backup/user/liuxinyu/.cache \
    --data_arrow_dir /backup/user/liuxinyu/.cache/webdataset/default-47e78d0b13c2e4f1/0.0.0/b802d95c473c6c5bd395aae2b78ce3bd599a784d43c7d164510277fdc676c1f8 \
    --caption_key txt \
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
echo ">>> [RANK=$RANK] Training finished with exit code: $EXIT_CODE"
if [ $EXIT_CODE -ne 0 ]; then
    echo "=== Post-mortem diagnostics ==="
    dmesg -T 2>/dev/null | tail -100 || echo "dmesg not accessible"
    echo "--- memory after crash ---"
    free -h
fi
exit $EXIT_CODE