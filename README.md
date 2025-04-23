# CLIP-Flow: A Unified Model for Image Understanding and Generation

CLIP-Flow is a unified multimodal foundation model that combines autoregressive large language models with diffusion-based generation techniques. Unlike existing approaches that diffuse VAE features, CLIP-Flow leverages CLIP image features as the diffusion target, enabling both efficient generation and strong performance across understanding and generation tasks. This repository contains the code, models, and experiments presented in our research paper.

## Overview

CLIP-Flow bridges the gap between vision understanding and generation by:

- Unifying the modeling of images and text under a single architecture.
- Diffusing semantically meaningful CLIP features rather than raw pixels or VAE latents.
- Demonstrating competitive performance in both zero-shot image understanding and high-quality image generation.

<p align="center">
  <img src=fig/overall_arch.png" alt="CLIP-Flow Overview Figure" width="600"/>
</p>

*Figure: Overview of the CLIP-Flow architecture. The model learns to autoregressively generate images and text by diffusing CLIP features and decoding them through a shared transformer.*

---

## 🔍 Image Understanding Performance

CLIP-Flow achieves strong performance on standard benchmarks for image understanding. The model is evaluated in zero-shot and few-shot settings, demonstrating competitive accuracy with significantly fewer parameters than traditional vision-language models.

**Table: Results on image understanding benchmarks. Best results are highlighted in bold.**

<table>
  <thead>
    <tr>
      <th><b>Model</b></th>
      <th><b>VQAv2</b></th>
      <th><b>GQA</b></th>
      <th><b>MMBench</b></th>
      <th><b>SEED</b></th>
      <th><b>POPE</b></th>
      <th><b>MM-Vet</b></th>
      <th><b>MME-P</b></th>
      <th><b>MME-C</b></th>
      <th><b>MMMU</b></th>
      <th><b>RWQA</b></th>
      <th><b>TEXTVQA</b></th>
    </tr>
  </thead>
  <tbody>
    <tr><td>EMU2 Chat 34B</td><td>-</td><td>65.1</td><td>-</td><td>62.8</td><td>-</td><td>48.5</td><td>-</td><td>-</td><td>34.1</td><td>-</td><td>66.6</td></tr>
    <tr><td>Chameleon 7B</td><td>-</td><td>-</td><td>19.8</td><td>27.2</td><td>19.4</td><td>8.3</td><td>202.7</td><td>-</td><td>22.4</td><td>39.0</td><td>0.0</td></tr>
    <tr><td>Chameleon 34B</td><td>-</td><td>-</td><td>32.7</td><td>-</td><td>59.8</td><td>9.7</td><td>604.5</td><td>-</td><td>38.8</td><td>39.2</td><td>0.0</td></tr>
    <tr><td>Seed-X 17B</td><td>63.4</td><td>49.1</td><td>70.1</td><td>66.5</td><td>84.2</td><td>43.0</td><td>1457.0</td><td>-</td><td>35.6</td><td>-</td><td>-</td></tr>
    <tr><td>VILA-U 7B</td><td>79.4</td><td>60.8</td><td>66.6</td><td>57.1</td><td>85.8</td><td>33.5</td><td>1401.8</td><td>-</td><td>32.2</td><td>46.6</td><td>48.3</td></tr>
    <tr><td>LLaVAFusion 16B</td><td>-</td><td>-</td><td>-</td><td>72.1</td><td>-</td><td>-</td><td>1603.7</td><td>367.8</td><td>41.7</td><td>60.0</td><td>-</td></tr>
    <tr><td>Show-o 1.3B</td><td>69.4</td><td>58.0</td><td>-</td><td>-</td><td>80.0</td><td>-</td><td>1097.2</td><td>-</td><td>27.4</td><td>-</td><td>-</td></tr>
    <tr><td>EMU3 8B</td><td>75.1</td><td>60.3</td><td>58.5</td><td>68.2</td><td>85.2</td><td>37.2</td><td>1243.8</td><td>266.1</td><td>31.6</td><td>57.4</td><td>64.7</td></tr>
    <tr><td>MetaMorph 8B</td><td>-</td><td>-</td><td>75.2</td><td>71.8</td><td>-</td><td>-</td><td>-</td><td>-</td><td>41.8</td><td>58.3</td><td>60.5</td></tr>
    <tr><td>TokenFlow-XL 14B</td><td>77.6</td><td>62.7</td><td>76.8</td><td>72.6</td><td><b>87.8</b></td><td>48.2</td><td>1551.1</td><td>371.1</td><td>43.2</td><td>56.6</td><td>77.6</td></tr>
    <tr><td>Janus 1.3B</td><td>77.3</td><td>59.3</td><td>75.5</td><td>68.3</td><td>87.0</td><td>34.3</td><td>1338.0</td><td>-</td><td>30.5</td><td>-</td><td>-</td></tr>
    <tr><td>Janus Pro 7B</td><td>-</td><td>62.0</td><td>79.2</td><td>72.1</td><td>87.4</td><td>50.0</td><td>1567.1</td><td>-</td><td>41.0</td><td>-</td><td>-</td></tr>
    <tr><td>Pisces 8B</td><td>82.1</td><td><b>64.8</b></td><td>73.9</td><td>-</td><td>86.3</td><td>50.0</td><td>1582.8</td><td>324.3</td><td>41.2</td><td>63.0</td><td>66.2</td></tr>
    <tr style="background-color:#e6f4ea;"><td><b>CLIP-Flow 8B</b></td><td><b>83.1</b></td><td>60.5</td><td><b>83.5</b></td><td><b>77.5</b></td><td>87.5</td><td><b>66.6</b></td><td><b>1682.6</b></td><td><b>647.1</b></td><td><b>50.6</b></td><td><b>69.0</b></td><td><b>83.1</b></td></tr>
  </tbody>
</table>


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
