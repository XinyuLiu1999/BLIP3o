#!/bin/bash

export CONDA_ROOT=/fsx/home/jiuhai.chen/miniconda3
source $CONDA_ROOT/etc/profile.d/conda.sh
conda  activate  diff_clip_2

export WANDB_API_KEY='d8075df78a873149bb390d22e6fc2c6de539e365'

export HF_HOME=/fsx/sfr/data/jiuhai





srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$HOSTNAME:29501 blip3o/train/train_mem.py \
    --deepspeed ./scripts/zero1.json \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --version qwen \
    --data_type "mix" \
    --gen_vision_tower eva-clip-E-14-plus \
    --gen_projector_type mlp2x_gelu \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir {OUTPUT_FOLDER}\
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.003 \
    --lr_scheduler_type "cosine" \
    --model_max_length 512 \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --gen_pooling early_pool2d_4 \
    --n_query 64 \
    --n_und_query 0 \
    --report_to wandb \
    --run_name blip3o_qwen_vl_7b




