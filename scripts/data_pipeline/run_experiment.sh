#!/bin/bash
#SBATCH --job-name=blip3o_exp
#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --time=96:00:00

# Usage: sbatch scripts/run_experiment.sh experiments/baseline_1M

set -e

EXPERIMENT_DIR=${1:?"Usage: $0 <experiment_dir>"}

conda activate your_env

export WANDB_API_KEY='your wandb key'
export HF_HOME=/your/hf/home/

# Read actual_samples from config.yaml to compute max_steps
ACTUAL_SAMPLES=$(python -c "
import yaml
cfg = yaml.safe_load(open('${EXPERIMENT_DIR}/config.yaml'))
print(cfg['actual_samples'])
")

NODES=4
GPUS_PER_NODE=8
BATCH_PER_GPU=16
GRAD_ACCUM=1
GLOBAL_BATCH_SIZE=$((NODES * GPUS_PER_NODE * BATCH_PER_GPU * GRAD_ACCUM))
MAX_STEPS=$((ACTUAL_SAMPLES / GLOBAL_BATCH_SIZE))

AR_BACKBONE=Qwen/Qwen3-0.6B
DIFFUSION=Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers
LR=5e-5
RUN_NAME=$(basename ${EXPERIMENT_DIR})
LOCAL_DIR="models/${RUN_NAME}"

echo "============================================"
echo "Experiment:    ${EXPERIMENT_DIR}"
echo "Run name:      ${RUN_NAME}"
echo "Samples:       ${ACTUAL_SAMPLES}"
echo "Global batch:  ${GLOBAL_BATCH_SIZE}"
echo "Max steps:     ${MAX_STEPS}"
echo "============================================"

srun torchrun --nproc_per_node=${GPUS_PER_NODE} --nnodes=$SLURM_NNODES \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$HOSTNAME:29501 \
    blip3o/train/train.py \
    --deepspeed scripts/zero1.json \
    --num_image_tokens 65536 \
    --num_scale_tokens 3 \
    --load_embeddings_from_vision True \
    --model_name_or_path $AR_BACKBONE \
    --diffusion_name_or_path $DIFFUSION \
    --version "qwen_1_5" \
    --dataset_cls 'mix' \
    --experiment_dir ${EXPERIMENT_DIR} \
    --data_cache_dir /fsx/sfr/data/cache/webdataset \
    --num_loading_workers 32 \
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
    --per_device_train_batch_size ${BATCH_PER_GPU} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr":1e-5}' \
    --logging_steps 5 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend inductor \
    --dataloader_drop_last True
