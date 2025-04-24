# 🌌 BLIP3-U

BLIP3-U is a unified vision-language foundation model that combines the reasoning and instruction following strength of large language models with the generative power of diffusion models. Unlike prior works that diffuse VAE features or raw pixels, BLIP3-U diffuses semantically rich **CLIP image features**, enabling a powerful and efficient architecture for both image understanding and generation.

## ✨ Highlights

- **Unified Architecture** for both image understanding and generation.
- **CLIP Feature Diffusion**: Directly diffuses semantic vision features for stronger alignment and performance.
- **State-of-the-art performance** across a wide range of benchmarks.
- **Supports reasoning-based generation, semantic editing, and interleaved outputs.**

<p align="center">
  <img src="figure/overall_arch.png" alt="BLIP3-U Overview Figure" width="600"/>
</p>

*Figure: Overview of the BLIP3-U architecture. We use Flow Matching Loss to predict the ground truth CLIP embeddings. At inference, the autoregressive model first generates a sequence of visual tokens from the given conditioning, and those visual tokens are then passed to a diffusion transformer that decodes them into the final image.*


---

## 🚀 Demo

You can try out BLIP-3u in your browser using our interactive [Gradio demo](https://c15a85dd865a925007.gradio.live/).



---

## 🔍 Image Understanding Performance

BLIP3-U achieves strong performance on standard benchmarks for image understanding.

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
| **BLIP3-U 8B** | **83.1** | 60.5 | **83.5** | **77.5** | 87.5 | **66.6** | **1682.6** | **647.1** | **50.6** | **69.0** | **83.1** |


---

## 🖼️ Image Generation Performance

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
| Janus Pro 7B  | **0.80** | **84.19** |
| **BLIP3-U 8B** | 0.81 | 81.60         |

*Table: Image generation results for image generation.*



![BLIP3-U Overview Figure](figure/img_eval.png)
*Figure: Qualitative results of BLIP3-U.*


---

## 🧠 Novel Capabilities

Below, we highlight three key applications that showcase the model’s versatility beyond standard image generation and understanding benchmarks.

### 🔍 Reasoning-Based Generation

BLIP3-U supports **reasoning-aware image generation**, enabling the model to generate images that require understanding complex textual instructions, abstract prompts, or multi-step inference. Unlike traditional models that rely on shallow keyword matching, BLIP3-U utilizes its unified multimodal architecture to handle:


![BLIP3-U Overview Figure](figure/reasoning.png)
*Figure: Qualitative results of Reasoning-Based image generation.*

### ✏️ Image Editing

Through conditioning on existing images and natural language prompts, BLIP3-U enables **semantic image editing**. This includes:

- Object insertion, deletion, or replacement.
- Style or mood adjustments (e.g., “make it look like a winter night”).
- Context-aware modifications while preserving background and structure.

TODO.

### 🔁 Multi-turn dialogue

A unified model that jointly supports image understanding and generation naturally enables in-context learning scenarios. Previously generated images can serve as context for subsequent tasks, enabling iterative image editing, visual dialogue, and step‑by‑step visual reasoning without mode switching or external pipelines.

TODO.

---
