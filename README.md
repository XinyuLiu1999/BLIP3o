# 🌌 BLIP3-o

BLIP3-o is a unified multimodal model that combines the reasoning and instruction following strength of autoregressive models with the generative power of diffusion models. Unlike prior works that diffuse VAE features or raw pixels, BLIP3-o diffuses semantically rich **CLIP image features**, enabling a powerful and efficient architecture for both image understanding and generation.

## 📖 [Arxiv](http://arxiv.org/abs/2505.09568)

## Update

- [2025/05/16] 🔥 We’ve published a dataset of 20 million images with detailed captions [BLIP3o Pretrain Long Caption](https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-Long-Caption) and 3 million images with short caption [BLIP3o Pretrain Short Caption](https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-Short-Caption). All images and their captions are compressed into tar archives, **no separate image url downloads or manual unzipping required**. 




- [2025/05/16] 🔥 We’ve reorganized and cleaned up the repository to ensure a clear, well-structured codebase. Please give the training and inference scripts a try, and feel free to leave an issue if you run into any problems. We apologize for any confusion caused by our original codebase release.




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



You can try out BLIP3-o in your browser using our interactive [Demo](https://blip3o.salesforceresearch.ai/). 



Install package for tranining
```Shell
conda create -n blip3o python=3.11 -y
conda activate blip3o
pip install --upgrade pip  setuptools
pip install -r requirements.txt
```

## Model Checkpoint

BLIP3o-4B [4B](https://huggingface.co/BLIP3o/BLIP3o-Model)

BLIP3o-8B [8B](https://huggingface.co/BLIP3o/BLIP3o-Model)

## Inference

You can  download our chekpoint

```Shell
python -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='BLIP3o/BLIP3o-Model', repo_type='model'))"
```

and run the inference code

```Shell
python inference.py  /HF_model/checkpoint/path/
```
## Training
We include two scripts: **slurm.sh** for multi-node training on Slurm clusters, and **run.sh** for debugging.

For both **slurm.sh** and **run.sh**, you need to import huggingface home **HF_HOME**, training data folder **IMG_FOLDER** and output model save folder **OUTPUT_FOLDER**. 


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
  
We suggest to use Qwen-2.5-VL as the backbone, we are fixing some tokenizer issues for LLama3.

## Supported Dataset Format

- **Webdataset**  
- **Json**


## Data Loading

Most of our training data use Huggingface datasets to load **WebDataset**. To download the datasets:

[Pretrain](https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-Long-Caption)

You can download the datasets by
```Shell
python -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='BLIP3o/BLIP3o-Pretrain', repo_type='dataset'))"
```
And load them directly with HuggingFace WebDataset
```Shell
train_dataset = load_dataset("webdataset", data_files=data_files, split="train", num_proc=128)
```

[BLIP3o-60k](https://huggingface.co/datasets/BLIP3o/BLIP3o-60k)



![BLIP3-o Overview Figure](figure/image.png)
*Figure: Qualitative results of BLIP3-o.*


