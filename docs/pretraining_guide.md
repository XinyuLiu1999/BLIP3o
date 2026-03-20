# BLIP3o-NEXT Pre-training Guide

This document provides step-by-step instructions for running BLIP3o-NEXT pre-training from scratch.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Data Preparation](#data-preparation)
5. [Configuration](#configuration)
6. [Launching Pre-training](#launching-pre-training)
7. [Training Architecture and Objectives](#training-architecture-and-objectives)
8. [Monitoring and Checkpoints](#monitoring-and-checkpoints)
9. [Post-pretraining: SFT and GRPO](#post-pretraining-sft-and-grpo)
10. [Troubleshooting](#troubleshooting)

---

## 1. Overview

BLIP3o-NEXT is a multimodal generative model that combines three components:

| Component | Model | Parameters | Role |
|---|---|---|---|
| **AR Backbone** | Qwen3-0.6B | ~0.6B | Language modeling and discrete image token prediction |
| **Diffusion Decoder** | SANA 1.5 | ~1.6B | High-fidelity image generation from AR hidden states |
| **Vision Tokenizer** | TA-Tok (SigLIP-2) | - | Encodes images into 256 discrete tokens for AR supervision |

Pre-training jointly optimizes two loss objectives:
- **Cross-Entropy Loss**: Next-token prediction over text and discrete image tokens.
- **Diffusion Loss**: The AR model's hidden states condition the SANA diffusion model to denoise VAE-encoded image latents.

---

## 2. Prerequisites

### Hardware Requirements

| Setup | Specification |
|---|---|
| **Production** (recommended) | 4 nodes × 8 GPUs (32 GPUs total), SLURM cluster |
| **Debug / Small-scale** | 1 node × 4+ GPUs |
| **GPU Type** | NVIDIA GPUs with BF16 support (A100, H100 recommended) |
| **GPU Memory** | 40GB+ per GPU (80GB recommended for batch size 16) |

### Software Requirements

- Linux OS
- CUDA 12.1
- Python 3.11
- Conda package manager
- SLURM workload manager (for multi-node training)

### Model Weights (downloaded automatically from HuggingFace)

- `Qwen/Qwen3-0.6B` — AR backbone
- `Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers` — Diffusion decoder (SANA transformer + VAE + scheduler)
- `google/siglip2-so400m-patch14-384` — Vision encoder (loaded by TA-Tok)

---

## 3. Environment Setup

### Step 1: Create Conda Environment

```bash
conda create -n blip3o-next python=3.11 -y
conda activate blip3o-next
```

### Step 2: Install Dependencies

```bash
pip install --upgrade pip setuptools
pip install -r requirements.txt
pip install -e .
```

This installs the following key packages:

| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.3.0+cu121 | PyTorch with CUDA 12.1 |
| `transformers` | 4.51.3 | HuggingFace Transformers (Qwen3 support) |
| `deepspeed` | 0.14.4 | Distributed training (ZeRO optimization) |
| `diffusers` | 0.34.0 | SANA diffusion model and VAE |
| `flash_attn` | 2.6.2 | Flash Attention 2 for efficient attention |
| `accelerate` | 0.28.0 | HuggingFace Accelerate |
| `datasets` | 2.16.1 | Dataset loading (WebDataset format) |
| `wandb` | latest | Experiment tracking |
| `einops` | 0.8.1 | Tensor operations |

### Step 3: Set Environment Variables

```bash
# HuggingFace cache directory (where model weights are downloaded)
export HF_HOME=/path/to/your/huggingface/cache

# Weights & Biases API key for experiment logging (optional)
export WANDB_API_KEY='your_wandb_api_key'
```

### Step 4: Verify Installation

```bash
python -c "
import torch
import transformers
import deepspeed
import diffusers
import flash_attn
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPUs: {torch.cuda.device_count()}')
print(f'Transformers: {transformers.__version__}')
print(f'DeepSpeed: {deepspeed.__version__}')
print(f'Diffusers: {diffusers.__version__}')
print(f'Flash Attention: {flash_attn.__version__}')
"
```

---

## 4. Data Preparation

### 4.1 Pre-training Data Sources

The official pre-training data is hosted on HuggingFace:

| Dataset | Size | Link |
|---|---|---|
| Detailed Captions | 27M image-text pairs | [BLIP3o/BLIP3o-Pretrain-Long-Caption](https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-Long-Caption) |
| Short Captions | 5M image-text pairs | [BLIP3o/BLIP3o-Pretrain-Short-Caption](https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-Short-Caption) |

### 4.2 Data Format

The training code loads data in **WebDataset** (`.tar`) format via the HuggingFace `datasets` library. Each tar file must contain samples with these fields:

| Field | Type | Description |
|---|---|---|
| `jpg` (or `png`) | Image file | The image (renamed to `image` during loading) |
| `txt` | Text file | Caption/text associated with the image |

A `type` column is automatically added as `"T2I"` (text-to-image) during loading.

### 4.3 Configuring the Data Path

**Important:** The data path is currently **hardcoded** in `blip3o/data/dataset.py` (line 175). You must edit it before training:

```python
# blip3o/data/dataset.py, line 175
# Change this:
train_dataset = load_dataset(
    "webdataset",
    data_files='/fsx/home/jiuhai.chen/soda/overfit.tar',  # <-- CHANGE THIS
    split="train",
    num_proc=1,
    cache_dir='/fsx/sfr/data/jiuhai/webdataset'  # <-- CHANGE THIS
)
```

Replace with your own paths:

```python
train_dataset = load_dataset(
    "webdataset",
    data_files='/your/path/to/data/*.tar',  # glob pattern for all tar files
    split="train",
    num_proc=1,
    cache_dir='/your/cache/directory'
)
```

### 4.4 Data Processing Pipeline

During training, the data undergoes these transformations:

1. **Image for vision encoder** (understanding): Processed by `SiglipImageProcessor` — resized and normalized for the SigLIP-2 encoder (384×384 input).
2. **Target image for diffusion** (generation): Resized to 1024×1024, center-cropped, and normalized to [-1, 1] range for the SANA VAE.
3. **Text**: Tokenized using the Qwen3 tokenizer with chat template formatting:
   - For T2I tasks: `user: "Please generate image based on the following caption: {caption}"` → `assistant: "<image>"`
   - The `<image>` token is wrapped with `<im_start>` and `<im_end>` boundary tokens in the assistant response.

### 4.5 Combining Multiple Data Sources

To add multiple datasets (e.g., both long and short captions), duplicate the loading block in `dataset.py` and append to `list_data_dict`:

```python
# Load first dataset
dataset_1 = load_dataset("webdataset", data_files='/path/to/long_captions/*.tar', split="train")
dataset_1 = dataset_1.rename_column("jpg", "image")
dataset_1 = dataset_1.add_column('type', len(dataset_1) * ['T2I'])
dataset_1 = dataset_1.remove_columns([col for col in dataset_1.column_names if col not in ["image", "txt", "type"]])
list_data_dict.append(dataset_1)

# Load second dataset
dataset_2 = load_dataset("webdataset", data_files='/path/to/short_captions/*.tar', split="train")
dataset_2 = dataset_2.rename_column("jpg", "image")
dataset_2 = dataset_2.add_column('type', len(dataset_2) * ['T2I'])
dataset_2 = dataset_2.remove_columns([col for col in dataset_2.column_names if col not in ["image", "txt", "type"]])
list_data_dict.append(dataset_2)
```

The datasets are automatically concatenated and shuffled (seed=42).

---

## 5. Configuration

### 5.1 DeepSpeed Configuration

Two DeepSpeed configs are provided in `scripts/`:

| Config | File | Description |
|---|---|---|
| **ZeRO Stage 1** | `scripts/zero1.json` | Partitions optimizer states. Recommended for pre-training. |
| **ZeRO Stage 2** | `scripts/zero2.json` | Partitions optimizer states + gradients. Lower memory, more communication. |

ZeRO Stage 1 (`zero1.json`) is the default and recommended setting:
- AllGather optimized with 1e9 bucket size
- Overlapped communication enabled
- BF16 auto-enabled
- AdamW optimizer (betas: [0.9, 0.999], eps: 1e-8)

### 5.2 Training Hyperparameters

| Parameter | Default Value | Description |
|---|---|---|
| `learning_rate` | 5e-5 | Peak learning rate |
| `lr_scheduler_type` | `cosine_with_min_lr` | LR schedule with minimum LR |
| `min_lr` | 1e-5 | Minimum learning rate for cosine scheduler |
| `warmup_ratio` | 0.01 | Fraction of steps for LR warmup (1%) |
| `per_device_train_batch_size` | 16 | Batch size per GPU |
| `gradient_accumulation_steps` | 1 | Gradient accumulation steps |
| `num_train_epochs` | 1 | Number of training epochs |
| `model_max_length` | 2048 | Maximum sequence length |
| `weight_decay` | 0.0 | Weight decay |
| `bf16` | True | Use BF16 mixed precision |
| `tf32` | True | Enable TF32 for matmul |
| `gradient_checkpointing` | True | Enable gradient checkpointing |
| `torch_compile` | True | Enable `torch.compile` |
| `torch_compile_backend` | `inductor` | Compile backend |
| `attn_implementation` | `flash_attention_2` | Use Flash Attention 2 |

### 5.3 Model-Specific Parameters

| Parameter | Value | Description |
|---|---|---|
| `num_image_tokens` | 65536 | Vocabulary size for discrete image tokens |
| `num_scale_tokens` | 3 | Number of scale/quality tokens |
| `load_embeddings_from_vision` | True | Initialize image token embeddings from the vision tokenizer's codebook |
| `mm_vision_select_layer` | -2 | Use second-to-last layer of vision encoder |
| `mm_use_im_start_end` | True | Wrap image tokens with `<im_start>` / `<im_end>` |
| `mm_patch_merge_type` | flat | Flatten vision patches into sequence |
| `image_aspect_ratio` | square | Pad images to square before processing |
| `version` | qwen_1_5 | Chat template version for tokenization |
| `dataset_cls` | mix | Dataset class to use (maps to `LazySupervisedMixDataset`) |

### 5.4 What Gets Trained vs. Frozen

The default `mm_tunable_parts=mm_language_model` setting along with hardcoded freeze logic produces this training configuration:

| Component | Trainable | Notes |
|---|---|---|
| Qwen3 Language Model | Yes | Full LM (embeddings, attention, FFN, LM head) |
| Vision Tokenizer (TA-Tok) | No | Always frozen and set to eval mode |
| SANA Transformer | No | Explicitly frozen |
| SANA VAE | No | Frozen (used only for encoding target images) |
| Diffusion Connector | Yes | Projects AR hidden states → SANA conditioning (Linear→GELU→Linear→RMSNorm, hidden_size → 2304) |
| SANA Caption Projection | Yes | The only trainable part of the SANA pipeline |

To also train the vision tower, set `--mm_tunable_parts mm_language_model,mm_vision_tower`.

---

## 6. Launching Pre-training

### Option A: Multi-Node SLURM (Recommended for Production)

Edit `scripts/run.sh` to set your environment, data paths, and W&B key, then submit:

```bash
sbatch scripts/run.sh
```

The script is configured for 4 nodes × 8 GPUs (32 GPUs total) with a 96-hour time limit.

**Note:** There is a bug in `scripts/run.sh` line 34 — the `$` sign is missing before `DIFFUSION`:
```bash
# Bug (line 34):
--diffusion_name_or_path  DIFFUSION
# Should be:
--diffusion_name_or_path  $DIFFUSION
```

### Option B: Single-Node Debug

For quick debugging on a single node with 4 GPUs:

```bash
bash scripts/debug.sh
```

Key differences from production:
- Batch size 1 (instead of 16)
- No W&B logging (`--report_to none`)
- 4 dataloader workers (instead of 1)
- Cosine scheduler without min LR floor
- 3% warmup (instead of 1%)

### Option C: Custom Launch

For a custom setup, run `torchrun` directly:

```bash
torchrun \
    --nproc_per_node=NUM_GPUS \
    --nnodes=1 \
    --master_port=29509 \
    blip3o/train/train.py \
    --deepspeed scripts/zero1.json \
    --num_image_tokens 65536 \
    --num_scale_tokens 3 \
    --load_embeddings_from_vision True \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --diffusion_name_or_path Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers \
    --version "qwen_1_5" \
    --dataset_cls 'mix' \
    --dispatch_batches False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --group_by_modality_length True \
    --image_aspect_ratio square \
    --mm_patch_merge_type flat \
    --bf16 True \
    --output_dir models/Pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
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
```

### Option D: Multi-Node Without SLURM

For multi-node setups without SLURM, use `torchrun` with rendezvous:

```bash
# On each node, run:
torchrun \
    --nproc_per_node=8 \
    --nnodes=NUM_NODES \
    --node_rank=NODE_RANK \
    --rdzv_backend=c10d \
    --rdzv_endpoint=MASTER_HOSTNAME:29501 \
    blip3o/train/train.py \
    [... same arguments as Option C ...]
```

Replace `MASTER_HOSTNAME` with the hostname of node 0, `NUM_NODES` with the total number of nodes, and `NODE_RANK` with each node's rank (0, 1, 2, ...).

---

## 7. Training Architecture and Objectives

### 7.1 Forward Pass

The training forward pass (defined in `blip3o/model/language_model/blip3o_qwen.py`) proceeds as follows:

1. **Input Preparation** (`prepare_inputs_labels_for_multimodal`):
   - Text tokens are embedded via `embed_tokens`.
   - Images are passed through the vision tokenizer (TA-Tok) to produce 256 discrete indices.
   - Discrete indices are offset by `image_start_token_id` and embedded via the same `embed_tokens` lookup.
   - A scale token (pool scale) is prepended to the image token sequence.
   - Image embeddings replace `<image>` placeholders in the input sequence.
   - Labels are set to the discrete image token indices for the image region.

2. **AR Forward Pass**:
   - The combined text + image-token embedding sequence is passed through the Qwen3 transformer.
   - Hidden states and logits are produced.

3. **Cross-Entropy Loss** (line 104–111):
   - Standard shifted next-token prediction loss over the full vocabulary (text + image tokens).

4. **Diffusion Loss** (lines 115–166, only when `target_images` is present):
   - The SANA VAE encodes the target image (1024×1024) into latents.
   - Random noise and timesteps are sampled (uniform weighting scheme).
   - Noisy latents are constructed: `(1 - σ) * latents + σ * noise`.
   - Hidden states between `<im_start>` and `<im_end>` are extracted (730 tokens).
   - These hidden states are projected through the diffusion connector (with 10% mask dropout).
   - SANA transformer predicts the denoised output.
   - Loss = weighted MSE between prediction and target (`noise - latents`), using SD3-style loss weighting.

5. **Total Loss**: `loss = cross_entropy_loss + diffusion_loss`

### 7.2 Token Vocabulary Extension

During initialization (`initialize_vision_tokenizer` in `blip3o_arch.py`), the tokenizer is extended with:

| Token Type | Count | Format | Purpose |
|---|---|---|---|
| Image boundary | 2 | `<im_start>`, `<im_end>` | Mark image token regions |
| Scale tokens | 3 | `<S0>`, `<S1>`, `<S2>` | Indicate image scale/quality |
| Image tokens | 65,536 | `<I0>` through `<I65535>` | Discrete image token vocabulary |

When `load_embeddings_from_vision=True`, image token embeddings are initialized from the TA-Tok codebook rather than random averages.

### 7.3 Diffusion Connector

The diffusion connector bridges the AR model and the SANA diffusion model:

```
AR hidden states (hidden_size) → Linear → GELU → Linear → RMSNorm → (2304-dim output)
```

The RMSNorm is initialized with weight = √5.5. The 2304-dim output conditions the SANA transformer via cross-attention (`encoder_hidden_states`).

---

## 8. Monitoring and Checkpoints

### 8.1 Logging

- **W&B**: Set `--report_to wandb` and export `WANDB_API_KEY`. Training metrics (cross-entropy loss, diffusion loss) are logged every `--logging_steps 5` steps.
- **Console**: The trainer prints `Cross-entropy loss X, Diffusion loss Y` at every training step on rank 0.
- **Disable logging**: Use `--report_to none` (as in `debug.sh`).

### 8.2 Checkpoints

| Parameter | Value | Behavior |
|---|---|---|
| `--save_strategy steps` | - | Save by step count |
| `--save_steps 1000` | - | Save every 1000 steps |
| `--save_total_limit 1` | - | Keep only the latest checkpoint |
| `--output_dir models/Pretrain` | - | Checkpoint directory |

Checkpoints are saved in HuggingFace format at `models/Pretrain/checkpoint-XXXX/`.

### 8.3 Resuming from Checkpoint

Training automatically resumes if checkpoints exist in the output directory:

```python
# blip3o/train/train.py, lines 237-240
if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()
```

Simply re-run the same training command — it will detect and resume from the latest checkpoint.

---

## 9. Post-pretraining: SFT and GRPO

### 9.1 Supervised Fine-Tuning (SFT)

After pre-training completes, run instruction tuning using `scripts/sft.sh`:

1. Edit `scripts/sft.sh` and set `AR_BACKBONE` to your pretrained checkpoint path:
   ```bash
   AR_BACKBONE=/path/to/models/Pretrain  # your pretrained checkpoint
   ```

2. Submit:
   ```bash
   sbatch scripts/sft.sh
   ```

SFT data sources:
- [BLIP3o-60k](https://huggingface.co/datasets/BLIP3o/BLIP3o-60k)
- [ShareGPT-4o-Image](https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-4o-Image)

### 9.2 GRPO (Reinforcement Learning)

GRPO requires a separate environment due to PyTorch version conflicts:

```bash
cd trl
conda create -n grpo python=3.11 -y
conda activate grpo
pip install -r requirements.txt
cd ..
pip install -e .
```

Then run GRPO training via `trl/train_grpo.py`.

---

## 10. Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|---|---|---|
| `CUDA out of memory` | Batch size too large | Reduce `--per_device_train_batch_size` or use ZeRO Stage 2 (`scripts/zero2.json`) |
| `NCCL timeout` | Slow inter-node communication | The trainer sets a 52-week NCCL timeout; check network configuration |
| `FileNotFoundError` on data path | Hardcoded data path in `dataset.py` | Update the `data_files` path in `blip3o/data/dataset.py:175` |
| Flash Attention errors | Incompatible GPU or CUDA version | Ensure CUDA 12.1 and an Ampere+ GPU (A100, H100) |
| `$DIFFUSION` not expanded in `run.sh` | Missing `$` in the script | Fix line 34: change `DIFFUSION` to `$DIFFUSION` |
| `torch.compile` errors | Incompatible operations | Disable with `--torch_compile False` |
| Model downloads fail | No internet or HF rate limit | Pre-download models to `HF_HOME` or use `huggingface-cli download` |

### Memory Optimization Tips

1. **Reduce batch size**: Start with `--per_device_train_batch_size 1` and increase.
2. **Use ZeRO Stage 2**: Switch `--deepspeed scripts/zero2.json` for more memory savings.
3. **Gradient checkpointing** is enabled by default (`--gradient_checkpointing True`).
4. **Reduce `model_max_length`**: Lower from 2048 if your captions are short.
5. **Increase gradient accumulation**: Use `--gradient_accumulation_steps 4` with smaller batch size to maintain effective batch size.

### Verifying Training is Working

After launching, look for these indicators in the logs:

1. `"Loading vision tower"` — TA-Tok vision tokenizer loaded
2. `"Load vision embeddings from vision tower"` — Image token embeddings initialized from codebook
3. `"Total parameters: ~XXXMB"` and `"Trainable parameters: ~XXXMB"` — Parameter counts printed
4. A table of all parameters with their trainable status
5. `"Cross-entropy loss X, Diffusion loss Y"` — Both losses being computed (confirms dual-objective training is active)

---

## Quick-Start Checklist

- [ ] Create conda environment with Python 3.11
- [ ] Install requirements (`pip install -r requirements.txt && pip install -e .`)
- [ ] Set `HF_HOME` environment variable
- [ ] Download or prepare WebDataset `.tar` files with image + text pairs
- [ ] Edit `blip3o/data/dataset.py:175` to point to your data
- [ ] Fix the `$DIFFUSION` bug in `scripts/run.sh:34` (if using SLURM)
- [ ] Set `WANDB_API_KEY` (or use `--report_to none`)
- [ ] Launch with `sbatch scripts/run.sh` (SLURM) or `bash scripts/debug.sh` (single-node)
- [ ] Monitor logs for both cross-entropy and diffusion loss values
