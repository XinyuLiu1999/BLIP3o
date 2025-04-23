# CLIP-Flow: A Unified Model for Image Understanding and Generation

CLIP-Flow is a unified multimodal foundation model that combines autoregressive large language models with diffusion-based generation techniques. Unlike existing approaches that diffuse VAE features, CLIP-Flow leverages CLIP image features as the diffusion target, enabling both efficient generation and strong performance across understanding and generation tasks. This repository contains the code, models, and experiments presented in our research paper.

## Overview

CLIP-Flow bridges the gap between vision understanding and generation by:

- Unifying the modeling of images and text under a single architecture.
- Diffusing semantically meaningful CLIP features rather than raw pixels or VAE latents.
- Demonstrating competitive performance in both zero-shot image understanding and high-quality image generation.

<p align="center">
  <img src="assets/clip-flow-overview.png" alt="CLIP-Flow Overview Figure" width="600"/>
</p>

*Figure: Overview of the CLIP-Flow architecture. The model learns to autoregressively generate images and text by diffusing CLIP features and decoding them through a shared transformer.*

---

## 🔍 Image Understanding Performance

CLIP-Flow achieves strong performance on standard benchmarks for image understanding. The model is evaluated in zero-shot and few-shot settings, demonstrating competitive accuracy with significantly fewer parameters than traditional vision-language models.

| Dataset        | Metric    | CLIP-Flow | Baseline A | Baseline B |
|----------------|-----------|-----------|------------|------------|
| ImageNet       | Top-1 (%) | 76.5      | 75.1       | 72.8       |
| CIFAR-100      | Top-1 (%) | 85.2      | 83.9       | 80.5       |
| Oxford Flowers | Top-1 (%) | 94.6      | 93.1       | 91.2       |

*Table: Zero-shot classification accuracy across multiple datasets.*

CLIP-Flow's ability to perform competitively without task-specific fine-tuning highlights the power of using semantically rich diffusion targets and unified modeling.

---

## 🖼️ Image Generation Performance

We evaluate the image generation capability of CLIP-Flow on both unconditional and text-conditional generation tasks. The model produces diverse and high-fidelity samples that align well with textual prompts.

| Dataset    | FID ↓ | IS ↑ | CLIP-Flow | Baseline A | Baseline B |
|------------|-------|------|-----------|------------|------------|
| CIFAR-10   | 3.21  | 9.22 | ✅         | 3.85       | 8.74       |
| MS-COCO    | 10.5  | —    | ✅         | 12.7       | —          |
| CelebA-HQ  | 6.83  | —    | ✅         | 8.32       | —          |

*Table: Image generation results. FID = Fréchet Inception Distance (lower is better), IS = Inception Score (higher is better).*

CLIP-Flow’s performance indicates that generating in CLIP-space leads to semantically aligned and visually coherent outputs.

---

## 🔧 Installation

```bash
git clone https://github.com/your-username/CLIP-Flow.git
cd CLIP-Flow
pip install -r requirements.txt
