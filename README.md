# CLIP-Flow: A Unified Model for Image Understanding and Generation

CLIP-Flow is a unified multimodal foundation model that combines autoregressive large language models with diffusion-based generation techniques. Unlike existing approaches that diffuse VAE features, CLIP-Flow leverages CLIP image features as the diffusion target, enabling both efficient generation and strong performance across understanding and generation tasks. This repository contains the code, models, and experiments presented in our research paper.

## Overview

CLIP-Flow bridges the gap between vision understanding and generation by:

- Unifying the modeling of images and text under a single architecture.
- Diffusing semantically meaningful CLIP features rather than raw pixels or VAE latents.
- Demonstrating competitive performance in both zero-shot image understanding and high-quality image generation.

<p align="center">
  <img src="fig/overall_arch.png" alt="CLIP-Flow Overview Figure" width="600"/>
</p>

*Figure: Overview of the CLIP-Flow architecture. The model learns to autoregressively generate images and text by diffusing CLIP features and decoding them through a shared transformer.*

---

## 🔍 Image Understanding Performance

CLIP-Flow achieves strong performance on standard benchmarks for image understanding. The model is evaluated in zero-shot and few-shot settings, demonstrating competitive accuracy with significantly fewer parameters than traditional vision-language models.

**Table: Results on image understanding benchmarks. Best results are highlighted in bold.**

| Model             | VQAv2 | GQA  | MMBench | SEED | POPE | MM-Vet | MME-P   | MME-C   | MMMU | RWQA | TEXTVQA |
|------------------|-------|------|---------|------|------|--------|---------|---------|------|------|---------|
| EMU2 Chat 34B     | -     | 65.1 | -       | 62.8 | -    | 48.5   | -       | -       | 34.1 | -    | 66.6    |
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
| Pisces 8B         | 82.1  | **64.8** | 73.9 | -    | 86.3 | 50.0   | 1582.8  | 324.3   | 41.2 | 63.0 | 66.2    |
| **CLIP-Flow 8B** | **83.1** | 60.5 | **83.5** | **77.5** | 87.5 | **66.6** | **1682.6** | **647.1** | **50.6** | **69.0** | **83.1** |



*Table: Zero-shot classification accuracy across multiple datasets.*

CLIP-Flow's ability to perform competitively without task-specific fine-tuning highlights the power of using semantically rich diffusion targets and unified modeling.

---

## 🖼️ Image Generation Performance

We evaluate the image generation capability of CLIP-Flow on both unconditional and text-conditional generation tasks. The model produces diverse and high-fidelity samples that align well with textual prompts.

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
| Janus Pro 7B  | **0.80** | **84.19** |
| **CLIP-Flow 8B** | 0.79 | -         |

*Table: Image generation results. FID = Fréchet Inception Distance (lower is better), IS = Inception Score (higher is better).*

CLIP-Flow’s performance indicates that generating in CLIP-space leads to semantically aligned and visually coherent outputs.

---

## 🔧 Installation

```bash
git clone https://github.com/your-username/CLIP-Flow.git
cd CLIP-Flow
pip install -r requirements.txt
