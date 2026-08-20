# Implicit Cultural Alignment Reward Model

This repository provides the core architecture and evaluation pipeline for the **Implicit Cultural Alignment Reward Model**. Built upon the Phi-3.5-vision architecture and optimized with DeepSpeed and LoRA, this framework is designed to evaluate and align Multimodal Large Language Models (MLLMs) with nuanced cultural expectations.

## 📌 Repository Structure

- `eval/`: Contains evaluation scripts, including the batch inference pipeline (`batch_inference.py`) for pairwise comparison.
- `llava_reward/`: The core model architecture, custom loss functions, and dataset loading utilities.
- `data/`: Directory for input datasets and human annotations.
- `script/`: Shell scripts for distributed training and evaluation configurations.

## 📊 Dataset

This project utilizes the **CulturalFrames** dataset to evaluate cultural biases and human expectations in text-to-image generation and multimodal understanding. 

You can access and download the official dataset from Hugging Face:
* **Dataset Link**: [CulturalFrames on Hugging Face](https://huggingface.co/datasets/mair-lab/CulturalFrames)

Please ensure the dataset is downloaded and properly formatted as `./data/test_dataset.json` before running the evaluation.

## 💾 Model Weights

To maintain a lightweight and clean Git history, the fine-tuned LoRA adapters and model checkpoints are hosted on Hugging Face instead of GitHub. 

* **Model Repository**: [Bensonch/phi35_cultural_reward](https://huggingface.co/Bensonch/phi35-cultural-ultra-reward) *(Note: Please update the URL if the repository name differs)*

### Downloading the Weights

You can easily pull the required weights directly into your local `checkpoints` directory using the Hugging Face CLI:

```bash
hf download Bensonch/phi35_cultural_reward --local-dir ./checkpoints
