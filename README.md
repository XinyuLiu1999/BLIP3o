# 🌌 BLIP3-o

BLIP3-o is a unified multimodal model that combines the reasoning and instruction following strength of autoregressive models with the generative power of diffusion models. Unlike prior works that diffuse VAE features or raw pixels, BLIP3-o diffuses semantically rich **CLIP image features**, enabling a powerful and efficient architecture for both image understanding and generation.

## 📖 [Arxiv](http://arxiv.org/abs/2505.09568)

## ✨ Highlights

- **Fully Open-Source:** Fully open-source training data (Pretraining and Instruction Tuning), training recipe, model weights, code.
- **Unified Architecture:** for both image understanding and generation.
- **CLIP Feature Diffusion:** Directly diffuses semantic vision features for stronger alignment and performance.
- **State-of-the-art performance:** across a wide range of image understanding and generation benchmarks.


<!-- <p align="center">
  <img src="figure/arch.png" alt="BLIP3-U Overview Figure" width="700"/>
</p>

*Figure: Overview of the BLIP3-U architecture. We use Flow Matching Loss to predict the ground truth CLIP embeddings. At inference, the autoregressive model first generates a sequence of visual tokens from the given conditioning, and those visual tokens are then passed to a diffusion transformer that decodes them into the final image.* -->


---

## Demo


You can try out BLIP3-o in your browser using our interactive [Demo](https://blip3o.salesforceresearch.ai/). (Due to high usage, we’re currently updating the demo and will have it back online shortly.)


Install package for tranining
```Shell
conda create -n blip3o python=3.11 -y
conda activate blip3o
pip install --upgrade pip  
pip install -r requirements.txt
```

## Model Checkpoint

BLIP3o-4B [4B](https://huggingface.co/BLIP3o/BLIP3o-Model)

BLIP3o-8B [8B](https://huggingface.co/BLIP3o/BLIP3o-Model)



## CLIP + Diffusion (Encoder + Decoder)
We also provide two CLIP + Diffusion: 

[EVA-CLIP + SDXL](https://huggingface.co/BLIP3o/BLIP3o-Model)

[SigLIP + SANA](https://huggingface.co/BLIP3o/BLIP3o-Model)



## Supported Tasks

- **Text → Text**  
- **Image → Text** (Image Understanding) 
- **Text → Image** (Image Generation)  
- **Image → Image** (Image Editing)  
- **Multitask Training** (Image generation and undetstanding mix training)


## Supported Image Generation Methods

- **CLIP + MSE**  
- **CLIP + Flow Matching** 
- **VAE + Flow Matching** 
- **Transfusion, LMFusion** 



## Supported Autoregressive Backbones

- **Qwen-2.5-VL**  
- **LLaMA 3**


## Supported Dataset Format

- **Webdataset**  
- **Json**


## Data Loading

Most of our training data use Huggingface datasets to load **WebDataset**. To download the datasets:

[Pretrain](https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain)

[BLIP3o-60k](https://huggingface.co/datasets/BLIP3o/BLIP3o-60k)

<!-- ---

## 🔍 Image Understanding Performance

BLIP3-U achieves strong performance on standard benchmarks for image understanding.

**Table: Results on image understanding benchmarks. Best results are highlighted in bold.**

| Model             | VQAv2 | GQA  | MMBench | SEED | POPE | MM-Vet | MME-P   | MME-C   | MMMU | RWQA | TEXTVQA |
|------------------|-------|------|---------|------|------|--------|---------|---------|------|------|---------|
| EMU2 Chat 34B     | -     | **65.1** | -       | 62.8 | -    | 48.5   | -       | -       | 34.1 | -    | 66.6    |
| Chameleon 7B      | -     | -    | 19.8    | 27.2 | 19.4 | 8.3    | 202.7   | -       | 22.4 | 39.0 | 0.0     |
| Chameleon 34B     | -     | -    | 32.7    | -    | 59.8 | 9.7    | 604.5   | -       | 38.8 | 39.2 | 0.0     |
| Seed-X 17B        | 63.4  | 49.1 | 70.1    | 66.5 | 84.2 | 43.0   | 1457.0  | -       | 35.6 | -    | -       |
| VILA-U 7B         | 79.4  | 60.8 | 66.6    | 57.1 | 85.8 | 33.5   | 1401.8  | -       | 32.2 | 46.6 | 48.3    |
| LLaVAFusion 16B   | -     | -    | -       | 72.1 | -    | -      | 1603.7  | 367.8   | 41.7 | 60.0 | -       |
| Show-o 1.3B       | 69.4  | 58.0 | -       | -    | 80.0 | -      | 1097.2  | -       | 27.4 | -    | -       |
| EMU3 8B           | 75.1  | 60.3 | 58.5    | 68.2 | 85.2 | 37.2   | 1243.8  | 266.1   | 31.6 | 57.4 | 64.7    |
| MetaMorph 8B      | -     | -    | 75.2    | 71.8 | -    | -      | -       | -       | 41.8 | 58.3 | 60.5    |
| TokenFlow-XL 14B  | 77.6  | 62.7 | 76.8    | 72.6 | **87.8** | 48.2   | 1551.1  | 371.1   | 43.2 | 56.6 | 77.6    |
| Janus 1.3B        | 77.3  | 59.3 | 75.5    | 68.3 | 87.0 | 34.3   | 1338.0  | -       | 30.5 | -    | -       |
| Janus Pro 7B      | -     | 62.0 | 79.2    | 72.1 | 87.4 | 50.0   | 1567.1  | -       | 41.0 | -    | -       |
| **BLIP3-o 8B (Ours)** | **83.1** | 60.5 | **83.5** | **77.5** | 87.5 | **66.6** | **1682.6** | **647.1** | **50.6** | **69.0** | **83.1** | -->


<!-- ---

<!-- ## 🖼️ Image Generation Performance

We evaluate the image generation capability of BLIP3-U on text-conditional generation tasks. The model produces diverse and high-fidelity samples that align well with textual prompts.

| Model              | GenEval | DPG-Bench |
|-------------------|---------|-----------|
| GPT-4o            | 0.84    | -         |
| Chameleon 7B      | 0.39    | -         |
| Seed-X 17B        | 0.51    | -         |
| LLaVAFusion 16B   | 0.63    | -         |
| Show‑o 1.3B       | 0.68    | 67.27     |
| EMU3 8B           | 0.66    | 80.60     |
| TokenFlow‑XL 14B  | 0.63    | 73.38     |
| Janus 1.3B        | 0.61    | 79.68     |
| JanusFlow 1.3B        | 0.63    | 80.09     |
| **BLIP3-o 8B (Ours)** | **0.84** | **82.60**         |

*Table: Image generation results for image generation.* -->



![BLIP3-o Overview Figure](figure/image.png)
*Figure: Qualitative results of BLIP3-o.*


<!-- ---

## 🧠 Novel Capabilities

Below, we highlight three key applications that showcase the model’s versatility beyond standard image generation and understanding benchmarks.

### 🧩 Reasoning-Based Generation

BLIP3-U supports **reasoning-aware image generation**, enabling the model to generate images that require understanding complex textual instructions, abstract prompts, or multi-step inference. Unlike traditional models that rely on shallow keyword matching, BLIP3-U utilizes its unified multimodal architecture to handle:

<!--
![BLIP3-U Overview Figure](figure/reasoning.png)
*Figure: Qualitative results of Reasoning-Based image generation.*
-->

<!-- ### ✏️ Image Editing

Through conditioning on existing images and natural language prompts, BLIP3-U enables **semantic image editing**. This includes:

- Object insertion, deletion, or replacement.
- Style or mood adjustments (e.g., “make it look like a winter night”).
- Context-aware modifications while preserving background and structure.

TODO.

### 🔁 Multi-turn dialogue

A unified model that jointly supports image understanding and generation naturally enables in-context learning scenarios. Previously generated images can serve as context for subsequent tasks, enabling iterative image editing, visual dialogue, and step‑by‑step visual reasoning without mode switching or external pipelines.

TODO.

--- --> 
